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
    for attempt in range(1, 4):
        code = subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / "push_hf.py"), "--repo", args.repo], cwd=ROOT).returncode
        print(f"[{datetime.now():%Y-%m-%d %H:%M}] attempt {attempt}: exit {code}", flush=True)
        if code == 0:
            return
        time.sleep(300)


if __name__ == "__main__":
    main()
