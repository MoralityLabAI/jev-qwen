"""Wait (up to --hours) for a Hugging Face login made with `hf auth login`, then run push_hf.py.

    python scripts/push_hf_when_ready.py --repo AlephFunk/jev-qwen --hours 36

The token is never read or printed here; huggingface_hub uses the stored login.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--hours", type=float, default=36)
    parser.add_argument("--attempts", type=int, default=3, help="upload attempts; push_hf.py skips adapters already on the Hub")
    parser.add_argument("--wait", type=float, default=300, help="seconds between attempts")
    args = parser.parse_args()
    from huggingface_hub import get_token

    deadline = time.time() + args.hours * 3600
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] waiting for a Hugging Face login", flush=True)
    while get_token() is None:
        if time.time() > deadline:
            print("no login before the deadline; nothing uploaded", flush=True)
            return
        time.sleep(30)
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] login found; uploading", flush=True)
    for attempt in range(1, args.attempts + 1):
        code = subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / "push_hf.py"), "--repo", args.repo], cwd=ROOT).returncode
        print(f"[{datetime.now():%Y-%m-%d %H:%M}] attempt {attempt}: exit {code}", flush=True)
        if code == 0:
            return
        time.sleep(args.wait)


if __name__ == "__main__":
    main()
