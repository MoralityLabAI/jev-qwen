"""YAML config loading: base.yaml <- variant <- suite <- command-line overrides."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_override(cfg: dict, assignment: str) -> None:
    """Apply one `a.b.c=value` override in place; the value is parsed as YAML."""
    dotted, sep, raw = assignment.partition("=")
    if not sep:
        raise ValueError(f"override must look like key.path=value, got {assignment!r}")
    node = cfg
    keys = dotted.strip().split(".")
    for key in keys[:-1]:
        if not isinstance(node.get(key), dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = yaml.safe_load(raw)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def resolve(variant: str | Path, suite: str | Path, overrides: list[str] | None = None) -> dict:
    """Build the fully resolved run config. Every value that shapes a run lives in the result."""
    cfg = load_yaml(ROOT / "configs" / "base.yaml")
    cfg = deep_merge(cfg, load_yaml(resolve_path(variant)))
    cfg = deep_merge(cfg, {"suite": load_yaml(resolve_path(suite))})
    for assignment in overrides or []:
        apply_override(cfg, assignment)
    return cfg
