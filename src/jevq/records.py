"""Machine-readable run records."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT

RECORD_VERSION = 1


def _git_executable() -> str | None:
    found = shutil.which("git")
    if found:
        return found
    # GitHub Desktop ships a git that is not on PATH.
    pattern = os.path.join(os.environ.get("LOCALAPPDATA", ""), "GitHubDesktop", "app-*", "resources", "app", "git", "cmd", "git.exe")
    candidates = sorted(glob.glob(pattern))
    return candidates[-1] if candidates else None


def git_state() -> dict:
    git = _git_executable()
    if git is None:
        return {"commit": None, "dirty": None, "error": "git executable not found"}
    try:
        commit = subprocess.run([git, "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=20)
        status = subprocess.run([git, "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"commit": None, "dirty": None, "error": str(exc)}
    if commit.returncode != 0:
        return {"commit": None, "dirty": None, "error": commit.stderr.strip()}
    return {"commit": commit.stdout.strip(), "dirty": bool(status.stdout.strip())}


def new_run_id(variant_name: str, suite_name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{variant_name}_{suite_name}"


def sha256_of(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def make_record(
    run_id: str,
    cfg: dict,
    model_info: dict,
    dataset_info: dict,
    hardware: dict,
    scores: dict,
    wall_time_s: float,
    files: dict,
) -> dict:
    return {
        "record_version": RECORD_VERSION,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "model": model_info,
        "variant": cfg["variant"],
        "dataset": dataset_info,
        "seed": cfg["seed"],
        "hyperparameters": cfg,
        "hardware": hardware,
        "scores": scores,
        "wall_time_s": wall_time_s,
        "files": files,
    }


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False)
        fh.write("\n")


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
