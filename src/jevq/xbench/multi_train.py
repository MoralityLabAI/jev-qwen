"""Training rows for J-multi (docs/ubench/SPEC.md): one adapter on the training splits of the
utility suites, each in its own label space and in its evaluation prompt format.

`data.sources` lists the parts; each has a `kind` and an optional `upsample`:
- `dev`: the jev-qwen train split (S4), built exactly as V1's (`training.build_items`);
- `rmp`: RMP train-region rows (S1), built exactly as J-V1-rmp's (`rmp_train`);
- `s2`: commit/veto `train` split of both packs (validation: `val_seen`);
- `s7`: hermes-lite registered `train_rows`, shuffled shortlist, a new shuffle per repeat.
No part reads an evaluation split; the never-train list is in docs/xbench/SPEC.md section 9.
"""

from __future__ import annotations

import random

from .jrunner import ChoiceItem


def encode_choice(tokenizer, item: ChoiceItem):
    from ..training import Encoded

    option_ids = []
    for label in item.labels:
        tokens = tokenizer.encode(label, add_special_tokens=False)
        if len(tokens) != 1:
            raise RuntimeError(f"label {label!r} is not a single token")
        option_ids.append(tokens[0])
    index = item.options.index(item.gold)
    ids = tokenizer.encode(item.prompt)
    return Encoded(ids + [option_ids[index]], len(ids), [option_ids[index]], "choice", len(item.options), index, option_ids)


def build_multi_encoded(tokenizer, data_cfg: dict, seed: int, n_shots: int, label_ids):
    from ..records import sha256_of
    from ..training import build_items, encode
    from . import s2_commit_veto as s2
    from . import s7_routing as s7
    from .rmp_train import build_rmp_encoded

    train, val, parts = [], [], {}
    for source in data_cfg["sources"]:
        kind, upsample = source["kind"], int(source.get("upsample", 1))
        if kind == "dev":
            tr_items, va_items, info = build_items(source, seed, n_shots)
            tr = [encode(tokenizer, item, label_ids) for item in tr_items]
            va = [encode(tokenizer, item, label_ids) for item in va_items]
        elif kind == "rmp":
            tr, va, info = build_rmp_encoded(tokenizer, source, seed)
        elif kind == "s2":
            items = s2.train_choice_items("train")
            tr = [encode_choice(tokenizer, item) for item in items]
            va = [encode_choice(tokenizer, item) for item in s2.train_choice_items("val_seen")]
            info = {"n_rows": len(items), "rows_sha256": sha256_of(sorted(i.item_id for i in items))}
        elif kind == "s7":
            # Each copy is a fresh shortlist shuffle, so upsampling happens here, not by repetition.
            items = [item for repeat in range(upsample) for item in s7.train_choice_items(repeat)]
            tr, va = [encode_choice(tokenizer, item) for item in items], []
            info = {"n_rows": len(items) // upsample, "rows_sha256": sha256_of(sorted({i.item_id.split("#")[0] for i in items}))}
        elif kind in ("u4_decide", "u4_repair"):
            # SPEC-U5 J-u4: the U4 train items (the Decision-TRM's training data), zero-shot prompts;
            # validation is a fixed 100-item subset of the U4 val pool. U4/U5 test items are never used.
            from . import u5_campsite as u5

            decide, repair = u5.training_rows("train")
            vdecide, vrepair = u5.training_rows("val", limit=100)
            if kind == "u4_decide":
                tr = [encode_choice(tokenizer, item) for item in decide]
                va = [encode_choice(tokenizer, item) for item in vdecide]
            else:
                tr = [u5.encode_generate(tokenizer, p, t) for p, t in repair]
                va = [u5.encode_generate(tokenizer, p, t) for p, t in vrepair]
            info = {"n_rows": len(decide), "rows_sha256": sha256_of(sorted(item.item_id for item in decide))}
        elif kind in ("real_decide", "real_repair"):
            # SPEC-U6 J-real: real J-V0 / Bonsai-8B proposals on the training puzzles; the U5 proposal
            # set (the test) is never used. Validation: a fixed 100-item subset of the real val proposals.
            from . import u6_campsite as u6

            decide, repair = u6.training_rows("realtrain")
            vdecide, vrepair = u6.training_rows("realval", limit=u6.VAL_LIMIT)
            if kind == "real_decide":
                tr = [encode_choice(tokenizer, item) for item in decide]
                va = [encode_choice(tokenizer, item) for item in vdecide]
            else:
                from . import u5_campsite as u5

                tr = [u5.encode_generate(tokenizer, p, t) for p, t in repair]
                va = [u5.encode_generate(tokenizer, p, t) for p, t in vrepair]
            info = {"n_rows": len(decide), "rows_sha256": sha256_of(sorted(item.item_id for item in decide))}
        else:
            raise ValueError(f"unknown source kind {kind!r}")
        if kind != "s7":
            tr = tr * upsample
        kept = ("n_rows", "rows_sha256", "formats", "families")
        parts[kind] = {"n_train": len(tr), "n_val": len(va), "upsample": upsample, **{k: v for k, v in info.items() if k in kept}}
        train += tr
        val += va
    random.Random(f"{seed}|multi-train-order").shuffle(train)
    data_info = {"source": "multi (SPEC-U1)", "parts": parts, "n_train": len(train), "n_val": len(val),
                 "formats": {fmt: sum(e.fmt == fmt for e in train) for fmt in sorted({e.fmt for e in train})}}
    return train, val, data_info
