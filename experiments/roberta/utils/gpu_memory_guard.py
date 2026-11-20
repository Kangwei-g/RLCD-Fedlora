"""Dynamic GPU VRAM reservation: keep a stable per-GPU footprint and block other jobs."""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

import torch

_GLOBAL_GUARD: Optional["GpuMemoryGuard"] = None


def parse_gpu_memory_reserves(spec: str) -> List[tuple[str, float]]:
    """Parse 'cuda:0:75' or 'cuda:0:75,cuda:1:75' -> [(device, gb), ...]."""
    text = str(spec or "").strip()
    if not text:
        return []
    items = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        tokens = part.split(":")
        if len(tokens) != 3:
            raise ValueError(
                f"Invalid --gpu_memory_reserve segment {part!r}; "
                "expected cuda:0:75 (device:gb)."
            )
        device = f"{tokens[0]}:{tokens[1]}"
        items.append((device, float(tokens[2])))
    return items


class GpuMemoryGuard:
    """
    Reserve target GB per GPU for this process.

    Modes:
      - idle: pad process VRAM up to target_gb (block other users between rounds)
      - compute: keep only a small baseline hold, leave headroom for training/inference

    With persistent_reservation disabled, entering compute returns deleted hold
    blocks to CUDA. With it enabled, the blocks remain in PyTorch's caching
    allocator: model allocations reuse the pool while other processes cannot.
    """

    def __init__(
        self,
        targets: Dict[str, float],
        training_headroom_gb: float = 72.0,
        slack_gb: float = 0.5,
        watchdog_interval_sec: float = 0.5,
        persistent_reservation: bool = False,
    ):
        self.targets = dict(targets)
        self.training_headroom_gb = float(training_headroom_gb)
        self.slack_gb = float(slack_gb)
        self.watchdog_interval_sec = float(watchdog_interval_sec)
        self.persistent_reservation = bool(persistent_reservation)
        self._holds: Dict[str, List[torch.Tensor]] = {}
        self._lock = threading.Lock()
        self._mode = "idle"
        self._stop = threading.Event()
        self._watchdog: Optional[threading.Thread] = None

    @classmethod
    def from_spec(
        cls,
        spec: str,
        training_headroom_gb: float = 72.0,
        slack_gb: float = 0.5,
        persistent_reservation: bool = False,
    ) -> "GpuMemoryGuard":
        return cls(
            {d: gb for d, gb in parse_gpu_memory_reserves(spec)},
            training_headroom_gb=training_headroom_gb,
            slack_gb=slack_gb,
            persistent_reservation=persistent_reservation,
        )

    def _device_used_gb(self, device: str) -> float:
        free, total = torch.cuda.mem_get_info(device)
        return (total - free) / (1024 ** 3)

    def _process_reserved_gb(self, device: str) -> float:
        return torch.cuda.memory_reserved(device) / (1024 ** 3)

    def _free_gb(self, device: str) -> float:
        free, _ = torch.cuda.mem_get_info(device)
        return free / (1024 ** 3)

    def _hold_gb(self, device: str) -> float:
        return sum(t.numel() * 4 for t in self._holds.get(device, [])) / (1024 ** 3)

    def _baseline_hold_gb(self, device: str) -> float:
        """Minimum hold kept even during GPU compute."""
        if self.persistent_reservation:
            # Cached allocator blocks already keep the VRAM owned by this process.
            # Keeping live placeholder tensors here only steals compute headroom.
            return 0.0
        return max(0.0, self.targets[device] - self.training_headroom_gb)

    def _persistent_pool_target_gb(self, device: str) -> float:
        # Keep allocator rounding below the hard per-process limit.
        return max(0.0, self.targets[device] - self.slack_gb)

    def _log(self, msg: str, logger=None) -> None:
        if logger is not None:
            logger.info(msg)
        else:
            print(msg, flush=True)

    def _allocate_hold_unlocked(self, device: str, gb: float, logger=None) -> float:
        remaining_gb = float(gb)
        chunk_limit_gb = 1.0 if self.persistent_reservation else remaining_gb
        allocated_gb = 0.0
        torch.cuda.set_device(device)
        while remaining_gb >= 0.01:
            chunk_gb = min(chunk_limit_gb, remaining_gb)
            n_elems = int(chunk_gb * (1024 ** 3)) // 4
            if n_elems <= 0:
                break
            try:
                buf = torch.empty(n_elems, dtype=torch.float32, device=device)
            except RuntimeError as exc:
                self._log(
                    f"GPU memory hold FAILED on {device} after "
                    f"{allocated_gb:.2f}/{gb:.2f} GB: {exc}",
                    logger=logger,
                )
                break
            self._holds.setdefault(device, []).append(buf)
            actual_gb = buf.numel() * 4 / (1024 ** 3)
            allocated_gb += actual_gb
            remaining_gb -= actual_gb
        return allocated_gb

    def _clear_holds_unlocked(self, device: str) -> None:
        self._holds.pop(device, None)

    def _set_hold_unlocked(
        self,
        device: str,
        hold_gb: float,
        logger=None,
        release_cached: bool = False,
    ) -> None:
        self._clear_holds_unlocked(device)
        if release_cached:
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()
        if hold_gb >= 0.1:
            actual = self._allocate_hold_unlocked(device, hold_gb, logger=logger)
            if actual >= 0.1:
                self._log(
                    f"GPU memory hold on {device}: {actual:.2f} GB "
                    f"(mode={self._mode}, process reserved "
                    f"~{self._process_reserved_gb(device):.2f} GB, "
                    f"GPU used ~{self._device_used_gb(device):.2f} GB, "
                    f"target {self.targets[device]:.1f} GB)",
                    logger=logger,
                )

    def _reclaim_idle_unlocked(self, device: str, logger=None) -> None:
        if device not in self.targets:
            return
        if self.persistent_reservation:
            target = self._persistent_pool_target_gb(device)
            used_gb = self._process_reserved_gb(device)
        else:
            target = self.targets[device]
            used_gb = self._device_used_gb(device)
        need_gb = target - used_gb
        if need_gb < 0.1:
            return
        hold_gb = min(need_gb, max(0.0, self._free_gb(device) - self.slack_gb))
        if hold_gb >= 0.1:
            self._allocate_hold_unlocked(device, hold_gb, logger=logger)

    def start_watchdog(self) -> None:
        if self._watchdog is not None and self._watchdog.is_alive():
            return

        def _loop() -> None:
            while not self._stop.is_set():
                if self._mode == "idle":
                    with self._lock:
                        if self._mode == "idle":
                            for device in self.targets:
                                self._reclaim_idle_unlocked(device)
                self._stop.wait(self.watchdog_interval_sec)

        self._watchdog = threading.Thread(
            target=_loop, name="gpu-memory-guard", daemon=True
        )
        self._watchdog.start()

    def stop_watchdog(self) -> None:
        self._stop.set()
        if self._watchdog is not None:
            self._watchdog.join(timeout=2.0)

    def claim_startup(self, device: str, logger=None) -> None:
        if device not in self.targets:
            return
        with self._lock:
            self._mode = "idle"
            self._clear_holds_unlocked(device)
            target = self.targets[device]
            if self.persistent_reservation:
                torch.cuda.set_device(device)
                total_bytes = torch.cuda.get_device_properties(device).total_memory
                fraction = min(1.0, target * (1024 ** 3) / float(total_bytes))
                torch.cuda.set_per_process_memory_fraction(fraction, device=device)
                pool_target = self._persistent_pool_target_gb(device)
                hold_gb = max(
                    0.0,
                    pool_target - self._process_reserved_gb(device),
                )
                hold_gb = min(
                    hold_gb,
                    max(0.0, self._free_gb(device) - self.slack_gb),
                )
                self._log(
                    f"Persistent GPU pool enabled on {device}: "
                    f"pool target={pool_target:.2f} GB, "
                    f"hard process limit={target:.2f} GB",
                    logger=logger,
                )
            else:
                hold_gb = min(
                    target,
                    max(0.0, self._free_gb(device) - self.slack_gb),
                )
            if hold_gb >= 0.1:
                self._set_hold_unlocked(device, hold_gb, logger=logger)

    def claim_startup_all(self, logger=None) -> None:
        if not torch.cuda.is_available():
            return
        for device in self.targets:
            self.claim_startup(device, logger=logger)

    def enter_compute(self, devices=None, logger=None) -> None:
        """Shrink holds but keep a baseline so footprint does not drop to near-zero."""
        if not torch.cuda.is_available():
            return
        with self._lock:
            self._mode = "compute"
            devs = list(devices) if devices is not None else list(self.targets.keys())
            for device in devs:
                if device not in self.targets:
                    continue
                baseline = self._baseline_hold_gb(device)
                if abs(self._hold_gb(device) - baseline) < 0.1:
                    continue
                self._set_hold_unlocked(
                    device,
                    baseline,
                    logger=logger,
                    release_cached=not self.persistent_reservation,
                )

    def enter_idle(self, devices=None, logger=None) -> None:
        """Pad VRAM back to target footprint between rounds."""
        if not torch.cuda.is_available():
            return
        with self._lock:
            self._mode = "idle"
            devs = list(devices) if devices is not None else list(self.targets.keys())
            for device in devs:
                if device in self.targets:
                    self._reclaim_idle_unlocked(device, logger=logger)

    # Backward-compatible aliases used by older call sites.
    def release_all(self, devices=None, logger=None) -> None:
        self.enter_compute(devices, logger=logger)

    def reclaim_all(self, devices=None, logger=None) -> None:
        self.enter_idle(devices, logger=logger)


def init_gpu_memory_guard(
    spec: str,
    logger=None,
    early_claim: bool = True,
    training_headroom_gb: float = 72.0,
    start_watchdog: bool = True,
    persistent_reservation: bool = False,
) -> Optional[GpuMemoryGuard]:
    global _GLOBAL_GUARD
    text = str(spec or "").strip()
    if not text:
        _GLOBAL_GUARD = None
        return None
    guard = GpuMemoryGuard.from_spec(
        text,
        training_headroom_gb=training_headroom_gb,
        persistent_reservation=persistent_reservation,
    )
    if early_claim:
        guard.claim_startup_all(logger=logger)
    if start_watchdog:
        guard.start_watchdog()
    _GLOBAL_GUARD = guard
    return guard


def get_gpu_memory_guard() -> Optional[GpuMemoryGuard]:
    return _GLOBAL_GUARD


def apply_gpu_memory_reserve(
    spec: str,
    logger=None,
    training_headroom_gb: float = 72.0,
    persistent_reservation: bool = False,
) -> Optional[GpuMemoryGuard]:
    return init_gpu_memory_guard(
        spec,
        logger=logger,
        early_claim=True,
        training_headroom_gb=training_headroom_gb,
        start_watchdog=True,
        persistent_reservation=persistent_reservation,
    )
