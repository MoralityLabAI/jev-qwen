"""Run a list of GPU jobs one after another, each only when the shared GPU has room.

    python scripts/run_queue.py --queue configs/queues/m2.yaml

Before every step it waits (nvidia-smi only) for free VRAM, then runs the step as its own
process so a crash costs one step, not the queue. Steps already marked done in the status
file are skipped, so the same command resumes an interrupted queue. Output of every step
goes to results/queue/<queue name>/<step>.log; progress is in status.json next to it.

`keep_awake: true` in the queue file asks Windows not to idle-sleep while the queue runs, and
only while the machine is on mains power. It is off by default: a laptop that sleeps
mid-step stalls the queue (the 2026-10-01 dev run lost 6.6 h that way) but keeps its battery.
It does not override closing the lid or choosing Sleep, and lapses when the queue exits.
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

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def on_mains_power() -> bool:
    import ctypes

    class PowerStatus(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", ctypes.c_ubyte),
            ("BatteryFlag", ctypes.c_ubyte),
            ("BatteryLifePercent", ctypes.c_ubyte),
            ("SystemStatusFlag", ctypes.c_ubyte),
            ("BatteryLifeTime", ctypes.c_ulong),
            ("BatteryFullLifeTime", ctypes.c_ulong),
        ]

    status = PowerStatus()
    return bool(ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status))) and status.ACLineStatus == 1


def set_keep_awake(enabled: bool) -> bool:
    """Request (or release) 'system required' for this thread. Returns whether it is now held."""
    if sys.platform != "win32":
        return False
    import ctypes

    want = enabled and on_mains_power()
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if want else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)
    return want


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue", required=True)
    args = parser.parse_args()
    queue = load_yaml(resolve_path(args.queue))
    out_dir = ROOT / "results" / "queue" / queue["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "status.json"
    status = load_yaml(status_path) if status_path.exists() else {}  # JSON is valid YAML
    keep_awake = bool(queue.get("keep_awake", False))

    try:
        run_steps(queue, out_dir, status_path, status, keep_awake)
    finally:
        set_keep_awake(False)


def run_steps(queue: dict, out_dir, status_path, status: dict, keep_awake: bool) -> None:
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
            awake = set_keep_awake(keep_awake)  # re-checked per step: the charger may have changed
            suffix = " (keeping the system awake)" if awake else ""
            print(f"[queue] {name}: attempt {attempt} started {started}{suffix}", flush=True)
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
                "kept_awake": awake,
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
