"""Unified xbench record schema: one record per (arm, suite), one item row per decision.

    results/xbench/<suite>/<arm>/record.json   arm, suite, claim label, metrics, provenance
    results/xbench/<suite>/<arm>/items.jsonl   one row per decision

Item rows share these fields (others are suite-specific and allowed):
    item_id, family, depth (or None), gold, pred, correct, options (list) or None,
    probs (list, aligned with options) or None, unsafe (bool or None),
    over_refusal (bool or None), passes, emitted_tokens, latency_s (or None)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..config import ROOT
from ..hardware import describe
from ..records import git_state

SCHEMA = "jevq.xbench.record.v1"
CLAIM_LABELS = ("live_model_run", "deterministic_replay", "post_hoc_projection", "control_plane_threshold_eval")
RESULTS = ROOT / "results" / "xbench"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def arm_info(
    arm_id: str,
    *,
    neural: bool,
    heuristic_analogue: bool = False,
    params_total: int | None = None,
    params_trainable: int | None = None,
    trained_on: str | None = None,
    checkpoint: str | None = None,
    extra: dict | None = None,
) -> dict:
    info = {
        "arm_id": arm_id,
        "neural": neural,
        "heuristic_analogue": heuristic_analogue,
        "params_total": params_total,
        "params_trainable": params_trainable,
        "trained_on": trained_on,
        "checkpoint": checkpoint,
        "checkpoint_sha256": file_sha256(checkpoint) if checkpoint and Path(checkpoint).is_file() else None,
    }
    info.update(extra or {})
    return info


def suite_info(suite_id: str, sources: Iterable[str | Path], split: str, extra: dict | None = None) -> dict:
    info = {
        "suite_id": suite_id,
        "split": split,
        "sources": [{"path": str(p), "sha256": file_sha256(p)} for p in sources],
    }
    info.update(extra or {})
    return info


def write_record(
    suite_id: str,
    arm: dict,
    suite: dict,
    items: list[dict],
    metrics: dict,
    claim_label: str,
    *,
    notes: list[str] | None = None,
    root: Path = RESULTS,
    include_hardware: bool = True,
) -> Path:
    if claim_label not in CLAIM_LABELS:
        raise ValueError(f"unknown claim label {claim_label!r}")
    out = root / suite_id / arm["arm_id"]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "items.jsonl", "w", encoding="utf-8") as fh:
        for row in items:
            fh.write(json.dumps(row) + "\n")
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "claim_label": claim_label,
        "arm": arm,
        "suite": suite,
        "n_items": len(items),
        "metrics": metrics,
        "notes": notes or [],
        "items_sha256": file_sha256(out / "items.jsonl"),
    }
    if include_hardware:
        record["hardware"] = describe(include_torch=False)
    with open(out / "record.json", "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
        fh.write("\n")
    return out / "record.json"


def write_not_applicable(suite_id: str, arm_id: str, reason: str, root: Path = RESULTS) -> Path:
    """A registered (arm, suite) cell that cannot run: the reason is the record (SPEC section 2)."""
    out = root / suite_id / arm_id
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "arm": {"arm_id": arm_id},
        "suite": {"suite_id": suite_id},
        "not_applicable": reason,
    }
    with open(out / "record.json", "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
        fh.write("\n")
    return out / "record.json"
