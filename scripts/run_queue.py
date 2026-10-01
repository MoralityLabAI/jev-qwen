"""Run a list of GPU jobs one after another, each only when the shared GPU has room.

    python scripts/run_queue.py --queue configs/queues/m2.yaml

Before every step it waits (nvidia-smi only) for free VRAM, then runs the step as its own
process so a crash costs one step, not the queue. Steps already marked done in the status
file are skipped, so the same command resumes an interrupted queue. Output of every step
goes to results/queue/<queue name>/<step>.log; progress is in status.json next to it.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime

import _bootstrap  # noqa: F401
from jevq.config import ROOT, load_yaml, resolve_path
from jevq.records import write_json
from wait_for_gpu import wait_until_free


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue", required=True)
    args = parser.parse_args()
    queue = load_yaml(resolve_path(args.queue))
    out_dir = ROOT / "results" / "queue" / queue["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "status.json"
    status = load_yaml(status_path) if status_path.exists() else {}  # JSON is valid YAML

    for step in queue["steps"]:
        name = step["name"]
        if status.get(name, {}).get("state") == "done":
            print(f"[queue] {name}: already done, skipping", flush=True)
            continue
        for attempt in range(1, queue["retries"] + 2):
            ready = wait_until_free(
                queue["min_free_mib"],
                queue["stable_seconds"],
                queue["timeout_minutes"],
                min_free_commit_gb=queue.get("min_free_commit_gb", 0.0),
            )
            if not ready:
                status[name] = {"state": "gpu_timeout", "attempts": attempt}
                write_json(status_path, status)
                raise SystemExit(f"[queue] {name}: GPU never became free")
            started = datetime.now().isoformat(timespec="seconds")
            print(f"[queue] {name}: attempt {attempt} started {started}", flush=True)
            t0 = time.perf_counter()
            with open(out_dir / f"{name}.log", "a", encoding="utf-8") as log:
                log.write(f"\n===== attempt {attempt} at {started} =====\n")
                log.flush()
                code = subprocess.run(
                    [sys.executable, "-u", *step["args"]], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
                ).returncode
            status[name] = {
                "state": "done" if code == 0 else "failed",
                "exit_code": code,
                "attempts": attempt,
                "started": started,
                "seconds": round(time.perf_counter() - t0, 1),
            }
            write_json(status_path, status)
            print(f"[queue] {name}: exit {code} after {status[name]['seconds']} s", flush=True)
            if code == 0:
                break
        else:
            raise SystemExit(f"[queue] {name}: failed {queue['retries'] + 1} times; stopping the queue")
    print("[queue] all steps done", flush=True)


if __name__ == "__main__":
    main()
