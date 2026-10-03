"""Training rows for J-V1-rmp / J-V2b-rmp (SPEC section 2): RMP train-region items, choice format.

Rows come from `research_gym.interp.bundle.train_example` (train hash region only) at indices
`index_offset...`, disjoint from the few-shot examples (indices from 0). Every fingerprint is
checked against the S1 evaluation items. Prompts use the same few-shot prefix as S1 evaluation.
"""

from __future__ import annotations

from . import s1_rmp
from .foreign import LOOPED_TRANSFORMERS, import_from


def build_rmp_encoded(tokenizer, data_cfg: dict, seed: int):
    from ..records import sha256_of
    from ..training import Encoded

    bundle = import_from(LOOPED_TRANSFORMERS, "research_gym.interp.bundle")
    config = s1_rmp.rmp_bundle_config()
    families = data_cfg.get("families") or list(s1_rmp.FAMILIES)
    offset = int(data_cfg.get("index_offset", 1000))
    held_out = {r["fingerprint"] for family in s1_rmp.FAMILIES for r in s1_rmp.load_items(family, 1000)}

    def encode(family: str, example: dict, shots: list[dict]) -> Encoded:
        options, labels = s1_rmp.options_for(family)
        gold = s1_rmp.gold_value(family, example["target_token"])
        prompt = s1_rmp.render_choice_prompt(family, shots, example["semantic"])
        option_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in labels]
        index = options.index(gold)
        ids = tokenizer.encode(prompt)
        return Encoded(ids + [option_ids[index]], len(ids), [option_ids[index]], "choice", len(options), index, option_ids)

    train, val, used = [], [], []
    for family in families:
        shots = s1_rmp.train_shots(family)
        shot_ids = {s["fingerprint"] for s in shots}
        for split, start, count, out in (
            ("train", offset, int(data_cfg["n_per_family"]), train),
            ("val", offset + 100_000, int(data_cfg["n_val_per_family"]), val),
        ):
            index, taken = start, 0
            while taken < count:
                example = bundle.train_example(family, index, config).to_dict()
                index += 1
                if example["fingerprint"] in held_out or example["fingerprint"] in shot_ids:
                    raise RuntimeError(f"train-region example collides with an evaluation item: {example['example_id']}")
                out.append(encode(family, example, shots))
                used.append([family, split, example["fingerprint"]])
                taken += 1
    import random

    random.Random(f"{seed}|rmp-train-order").shuffle(train)
    info = {
        "source": "rmp train region (research_gym.interp.bundle.train_example)",
        "families": families,
        "n_train": len(train),
        "n_val": len(val),
        "index_offset": offset,
        "n_eval_fingerprints_checked": len(held_out),
        "rows_sha256": sha256_of(used),
        "formats": {"choice": len(train)},
    }
    return train, val, info
