"""Adapter-path resolution shared by scripts that build bundles outside `harness.run`."""

from __future__ import annotations

from ..harness import adapter_path


def adapter_for(cfg: dict):
    """The variant's adapter directory (or None), resolved like `harness.run` does."""
    return adapter_path(cfg)
