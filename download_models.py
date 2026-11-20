#!/usr/bin/env python3
"""Download official model files explicitly and record their revision/checksums."""
import argparse
import hashlib
import json
from pathlib import Path

MODELS = {
    "roberta": ("FacebookAI/roberta-base", "main"),
    "opt": ("facebook/opt-125m", "27dcfa74d334bc871f3234de431e71c6eeba5dd6"),
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", choices=MODELS, required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    repo, revision = MODELS[args.backbone]
    resolved = HfApi().model_info(repo, revision=revision).sha
    target = args.output_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo, revision=resolved, local_dir=str(target),
                      allow_patterns=["*.json", "*.txt", "*.safetensors", "pytorch_model*.bin", "LICENSE*"])
    files = [dict(name=str(path.relative_to(target)), sha256=sha256(path))
             for path in sorted(target.iterdir())
             if path.is_file() and path.name != "jl_download_manifest.json"]
    manifest = dict(repo=repo, revision=resolved, files=files, status="verified")
    (target / "jl_download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(str(target))


if __name__ == "__main__":
    main()
