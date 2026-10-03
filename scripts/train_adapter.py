"""Train a LoRA adapter (V1, V1c; V2b later through `run_overrides`).

    python scripts/train_adapter.py --config configs/train/v1_lora.yaml
    python scripts/train_adapter.py --config configs/train/v1c_lora.yaml --set seed=1
    python scripts/train_adapter.py --config configs/train/v1_lora.yaml --set max_steps=6 --set name=probe

Writes <paths.checkpoints>/<name>_s<seed>/{adapter/, train_record.json, train_log.jsonl}, where
paths.checkpoints is outside OneDrive (configs/base.yaml). An interrupted run resumes from
<name>_s<seed>/resume.pt when started again with the same config. Evaluate with a variant file
whose `adapter:` is the adapter directory relative to paths.checkpoints.
"""

import argparse

import _bootstrap  # noqa: F401
from jevq.training import load_train_config, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="training-config override")
    args = parser.parse_args()
    record = train(load_train_config(args.config, args.set))
    final = record["final"] or {}
    print(
        f"done: {record['steps']} steps in {record['wall_time_s'] / 60:.1f} min; "
        f"final loss {final.get('loss')}; val choice accuracy {final.get('val_choice_accuracy')}; "
        f"trainable params {record['model'].get('trainable_params')}"
    )


if __name__ == "__main__":
    main()
