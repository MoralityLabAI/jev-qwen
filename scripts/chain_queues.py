"""Run queues one after another through vramorama, holding one lease at a time.

    python scripts/chain_queues.py \
        --stage 3G jevq-u7l configs/queues/u7l.yaml scripts/u7_report.py \
        --stage 8G jevq-u8 configs/queues/u8.yaml scripts/u8_report.py \
        --stage 11G jevq-u7m configs/queues/u7m.yaml scripts/u7_report.py

Each stage is VRAM OWNER QUEUE [REPORT ...]. For each stage in order:
1. wait until `vramorama leases` lists no lease owned by OWNER, so a lease requested elsewhere for
   the same queue finishes first;
2. if every step of QUEUE is already done, skip it; otherwise request a lease of VRAM and run
   `run_queue.py --queue QUEUE` under it, blocking until that job exits (output appended to
   results/queue/<queue name>.out);
3. run the REPORT scripts.
The next stage's lease is requested only after the previous job has exited. --dry-run prints each
stage's state and requests nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401
from jevq.config import ROOT, load_yaml, resolve_path

VRAMORAMA = shutil.which("vramorama") or str(Path(os.environ["LOCALAPPDATA"]) / "Programs" / "vramorama" / "vramorama.exe")
POLL_SECONDS = 60


def log(msg: str) -> None:
    print(f"[chain {datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def owner_leases(owner: str) -> list[dict] | None:
    """Leases held or requested by OWNER; None if vramorama could not be read."""
    try:
        out = subprocess.run([VRAMORAMA, "leases", "--json"], capture_output=True, text=True, timeout=120, check=True).stdout
        return [lease for lease in json.loads(out)["leases"] if lease.get("owner") == owner]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
        log(f"could not read vramorama leases: {exc!r}")
        return None


def queue_state(queue_path: Path) -> tuple[str, int, int]:
    queue = load_yaml(queue_path)
    status_path = ROOT / "results" / "queue" / queue["name"] / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    names = [step["name"] for step in queue["steps"]]
    return queue["name"], sum(status.get(n, {}).get("state") == "done" for n in names), len(names)


def wait_for_owner(owner: str) -> None:
    announced = False
    while (leases := owner_leases(owner)) is None or leases:
        if leases and not announced:
            log(f"waiting for {owner}'s lease(s) {[lease['id'] for lease in leases]} to end")
            announced = True
        time.sleep(POLL_SECONDS)


def run_stage(vram: str, owner: str, queue_path: Path, reports: list[str], dry_run: bool) -> None:
    name, done, total = queue_state(queue_path)
    log(f"stage {owner}: queue {name} {done}/{total} steps done")
    if dry_run:
        log(f"  dry run: leases now {owner_leases(owner)}")
        return
    wait_for_owner(owner)
    name, done, total = queue_state(queue_path)
    if done < total:
        out_path = ROOT / "results" / "queue" / f"{name}.out"
        cmd = [VRAMORAMA, "run", "--vram", vram, "--owner", owner, "--",
               sys.executable, "-u", "scripts/run_queue.py", "--queue", str(queue_path.relative_to(ROOT))]
        log(f"requesting {vram} for {name} ({done}/{total} done); output to {out_path}")
        env = {**os.environ, "VRAMORAMA_OWNER": owner, "PYTHONUNBUFFERED": "1"}
        with out_path.open("a", encoding="utf-8") as out:
            code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT).returncode
        name, done, total = queue_state(queue_path)
        log(f"stage {owner}: exit {code}, queue {name} {done}/{total} steps done")
    else:
        log(f"stage {owner}: already complete")
    for report in reports:
        code = subprocess.run([sys.executable, "-u", report], cwd=ROOT).returncode
        log(f"report {report}: exit {code}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", nargs="+", action="append", required=True, metavar="ARG",
                        help="VRAM OWNER QUEUE [REPORT ...]")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    stages = []
    for stage in args.stage:
        if len(stage) < 3:
            parser.error(f"--stage needs VRAM OWNER QUEUE, got {stage}")
        stages.append((stage[0], stage[1], resolve_path(stage[2]), stage[3:]))
    log(f"pid {os.getpid()}, {len(stages)} stages")
    for vram, owner, queue_path, reports in stages:
        run_stage(vram, owner, queue_path, reports, args.dry_run)
    log("all stages finished")


if __name__ == "__main__":
    main()
