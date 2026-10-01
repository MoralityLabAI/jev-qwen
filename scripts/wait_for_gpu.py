"""Block until the GPU has enough free memory, without touching it.

    python scripts/wait_for_gpu.py --min-free-mib 11000 --stable-seconds 60 --timeout-minutes 240

The card is shared with other jobs. This only reads nvidia-smi; it never creates a CUDA
context and never stops anything. Exit code 0 = free for the whole stability window,
1 = timed out.
"""

import argparse
import subprocess
import sys
import time


def free_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    total, used = (int(part) for part in out.stdout.strip().splitlines()[0].split(","))
    return total - used


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-free-mib", type=int, default=11000)
    parser.add_argument("--stable-seconds", type=int, default=60, help="free memory must hold this long")
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--timeout-minutes", type=float, default=240)
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout_minutes * 60
    free_since = None
    while time.monotonic() < deadline:
        free = free_mib()
        now = time.monotonic()
        if free >= args.min_free_mib:
            free_since = free_since or now
            if now - free_since >= args.stable_seconds:
                print(f"GPU free: {free} MiB available", flush=True)
                return
        else:
            if free_since is not None:
                print(f"GPU busy again: {free} MiB free", flush=True)
            free_since = None
        time.sleep(args.poll_seconds)
    print(f"timed out after {args.timeout_minutes} minutes; last free = {free_mib()} MiB", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
