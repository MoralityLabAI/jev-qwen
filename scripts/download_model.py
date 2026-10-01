"""Fetch the backbone checkpoint into the Hugging Face cache without loading it.

    python scripts/download_model.py

Needs no GPU. Prints the resolved commit so `model.revision` in configs/base.yaml can be pinned.
"""

from pathlib import Path

import _bootstrap  # noqa: F401  (must run before any HTTPS client is created)
from huggingface_hub import snapshot_download

from jevq.config import ROOT, load_yaml


def main() -> None:
    model_cfg = load_yaml(ROOT / "configs" / "base.yaml")["model"]
    path = Path(snapshot_download(model_cfg["id"], revision=model_cfg.get("revision")))
    size_gb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**30
    print(f"model: {model_cfg['id']}")
    print(f"snapshot: {path}")
    print(f"commit: {path.name}")
    print(f"size_gb: {size_gb:.2f}")


if __name__ == "__main__":
    main()
