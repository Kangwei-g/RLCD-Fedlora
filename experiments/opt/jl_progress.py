"""Plain-text progress reports; no model, metric or random-state dependencies."""

from datetime import datetime, timedelta
import logging
from pathlib import Path
import threading

_logger = logging.getLogger("jl.progress")
_logger.propagate = False
_lock = threading.Lock()
ROUND = "\u8f6e"
TRAINING = "\u8bad\u7ec3\u8fdb\u5ea6"
FINAL_TEST_NOTE = "\u6700\u7ec8\u6d4b\u8bd5\u53ca\u4fdd\u5b58\u53e6\u8ba1"


def configure_progress(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        for handler in list(_logger.handlers):
            _logger.removeHandler(handler)
            handler.close()
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        _logger.addHandler(handler)
        _logger.setLevel(logging.INFO)


def emit_progress(message, logger=None):
    with _lock:
        _logger.info(message)
        if logger is not None and logger is not _logger:
            logger.info(message)
        print(message, flush=True)


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def progress_message(completed, total, step_seconds, elapsed_seconds, *,
                     label="", unit=ROUND, step_name=ROUND, heading=TRAINING,
                     note="", now=None):
    if total < 1 or not 0 <= completed <= total:
        raise ValueError("Progress requires 0 <= completed <= total and total > 0")
    fields = [f"[{heading}]"]
    if label:
        fields.append(label)
    fields += [f"\u5df2\u5b8c\u6210 {completed}/{total} {unit}",
               f"\u672c{step_name}\u8017\u65f6 {format_duration(step_seconds)}"]
    if completed:
        average = max(0, elapsed_seconds) / completed
        remaining = average * (total - completed)
        finish_at = (now or datetime.now()) + timedelta(seconds=remaining)
        fields += [f"\u5e73\u5747\u6bcf{step_name} {format_duration(average)}",
                   f"\u9884\u8ba1\u5269\u4f59 {format_duration(remaining)}",
                   f"\u9884\u8ba1\u5b8c\u6210\u65f6\u95f4 {finish_at:%Y-%m-%d %H:%M:%S}"]
    else:
        fields += ["\u9884\u8ba1\u5269\u4f59 \u5f85\u4f30\u7b97",
                   "\u9884\u8ba1\u5b8c\u6210\u65f6\u95f4 \u5f85\u4f30\u7b97"]
    if note:
        fields.append(note)
    return " | ".join(fields)


def report_progress(completed, total, step_seconds, elapsed_seconds, *, logger=None, **kwargs):
    message = progress_message(completed, total, step_seconds, elapsed_seconds, **kwargs)
    emit_progress(message, logger=logger)
    return message
