"""Publish the adapters, aggregate run records and the paper PDF to the Hugging Face Hub.

    python scripts/push_hf.py --repo AlephFunk/jev-qwen [--private] [--dry-run]

Needs a write token stored by `hf auth login` (this script never reads or prints it).
Uploaded:
- adapters/<run>/{adapter_config.json, adapter_model.safetensors, train_record.json} for every
  run under paths.checkpoints;
- results/xbench/**/record.json (aggregate records only; items.jsonl holds held-out eval items of
  other projects and is never uploaded);
- results/latency/*h1*.json;
- paper.pdf;
- README.md (docs/hf_model_card.md).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from jevq.config import ROOT, load_yaml, resolve_path

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def planned_files() -> list[tuple[Path, str]]:
    """(local path, path in repo) for everything this script uploads."""
    base = load_yaml(resolve_path("configs/base.yaml"))
    checkpoints = resolve_path(base["paths"]["checkpoints"])
    files: list[tuple[Path, str]] = []
    for run in sorted(p for p in checkpoints.iterdir() if (p / "adapter" / "adapter_config.json").exists()):
        files += [(run / "adapter" / name, f"adapters/{run.name}/{name}") for name in ADAPTER_FILES]
        if (run / "train_record.json").exists():
            files.append((run / "train_record.json", f"adapters/{run.name}/train_record.json"))
    for record in sorted((ROOT / "results" / "xbench").glob("*/*/record.json")):
        files.append((record, record.relative_to(ROOT).as_posix()))
    for latency in sorted((ROOT / "results" / "latency").glob("*h1*.json")):
        files.append((latency, latency.relative_to(ROOT).as_posix()))
    files.append((ROOT / "paper" / "main.pdf", "paper.pdf"))
    files.append((ROOT / "docs" / "hf_model_card.md", "README.md"))
    assert not any(p.name == "items.jsonl" for p, _ in files)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = planned_files()
    total = sum(p.stat().st_size for p, _ in files)
    print(f"{len(files)} files, {total / 2**20:.0f} MiB -> {args.repo} ({'private' if args.private else 'public'})")
    if args.dry_run:
        for local, remote in files:
            print(f"  {remote}  <-  {local}")
        return

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    print(f"logged in as {api.whoami()['name']}")
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    # One commit per adapter folder, then one for records, paper and card: on a slow or flaky
    # connection a failure loses one batch, and a rerun skips adapters already on the Hub.
    present = set(api.list_repo_files(args.repo))
    batches: dict[str, list[tuple[Path, str]]] = {}
    for local, remote in files:
        key = "/".join(remote.split("/")[:2]) if remote.startswith("adapters/") else "records, paper, card"
        batches.setdefault(key, []).append((local, remote))
    for key, batch in batches.items():
        if key.startswith("adapters/") and all(remote in present for _, remote in batch):
            print(f"skip {key}: already on the Hub", flush=True)
            continue
        operations = [CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)) for local, remote in batch]
        commit = api.create_commit(args.repo, operations=operations, commit_message=f"Publish {key}")
        print(f"pushed {key}: {commit.commit_url}", flush=True)


if __name__ == "__main__":
    main()
