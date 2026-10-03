"""Block until another queue has finished every step (done) or stopped (a failed step).

    python scripts/wait_queue.py --queue configs/queues/m3.yaml --timeout-minutes 720

Lets a second queue start only after the first has released the GPU, so two queues never load
models at the same time. Exit 0 once every step of the other queue is done, 1 otherwise.
"""

import argparse
import json
import sys
import time

import _bootstrap  # noqa: F401
from jevq.config import ROOT, load_yaml, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--timeout-minutes", type=float, default=720)
    args = parser.parse_args()
    queue = load_yaml(resolve_path(args.queue))
    status_path = ROOT / "results" / "queue" / queue["name"] / "status.json"
    names = [step["name"] for step in queue["steps"]]
    deadline = time.monotonic() + args.timeout_minutes * 60
    while time.monotonic() < deadline:
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        states = [status.get(name, {}).get("state") for name in names]
        if all(state == "done" for state in states):
            print(f"queue {queue['name']}: all {len(names)} steps done", flush=True)
            return
        # A failed final attempt stops that queue; give it a few minutes to record it.
        if any(state == "failed" for state in states) and status.get(names[states.index("failed")], {}).get("attempts", 0) > queue["retries"]:
            print(f"queue {queue['name']}: stopped on a failed step; continuing anyway", flush=True)
            return
        time.sleep(30)
    print(f"timed out waiting for {queue['name']}", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
