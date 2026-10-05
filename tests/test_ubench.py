"""SPEC-U1: training rows come from training splits only, encode correctly, and the U1 arms stay
out of the frozen v1 pools and coverage matrix."""

import sys
from pathlib import Path

import pytest

from jevq.xbench.foreign import HERMES_LITE, HERMES_SKILLS

needs_hs = pytest.mark.skipif(not HERMES_SKILLS.exists(), reason="Hermes-Skills not on this machine")
needs_hl = pytest.mark.skipif(not HERMES_LITE.exists(), reason="hermes-lite not on this machine")


@needs_hs
def test_s2_training_rows_never_touch_eval_splits():
    from jevq.xbench import s2_commit_veto as s2

    with pytest.raises(ValueError):
        s2.train_choice_items("holdout_seen")
    train = s2.train_choice_items("train")
    eval_ids = {i.item_id for pack in ("csig", "near_miss") for i in s2.choice_items(pack)}
    shot_ids = {s["id"] for pack in ("csig", "near_miss") for s in s2.pick_shots(s2.load(pack)["train"])}
    assert len(train) == 234 and not {i.item_id for i in train} & (eval_ids | shot_ids)
    assert {i.meta["split"] for i in train} == {"train"}


@needs_hl
def test_s7_training_rows_are_registered_train_rows_with_shuffles():
    from jevq.xbench import s7_routing as s7

    a, b = s7.train_choice_items(0), s7.train_choice_items(1)
    assert len(a) == len(b) == 54
    assert sum(i.gold == "ABSTAIN" for i in a) == 22 and all(i.gold in i.options for i in a)
    assert [i.options for i in a] != [i.options for i in b]  # a fresh shuffle per copy
    held = {c["query"] for c in s7._jsonl(s7.REGISTERED / "held_cases.jsonl")}
    assert not any(q in i.prompt for i in a for q in held)


def test_encode_choice_uses_the_item_label_space():
    from jevq.xbench.jrunner import ChoiceItem
    from jevq.xbench.multi_train import encode_choice

    class Tok:
        def encode(self, text, add_special_tokens=True):
            return [ord(c) for c in text] if len(text) > 2 else [1000 + ord(text[-1])]

    item = ChoiceItem("x", "Which?", [" A", " B", " C"], ["a", "b", "c"], "b")
    enc = encode_choice(Tok(), item)
    assert enc.option_ids == [1065, 1066, 1067] and enc.answer_index == 1
    assert enc.targets == [1066] and enc.input_ids[-1] == 1066 and enc.n_prompt == 6


def test_u1_arms_stay_out_of_v1_pools_and_coverage():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import xbench_report as report
    from jevq.xbench import coverage

    assert not {"J-multi", "J-multi-loop"} & set(coverage.ARMS)
    record = {"arm": {"arm_id": "J-multi", "neural": True}, "metrics": {"accuracy": 1.0, "n": 1}, "_items": []}
    registered, addenda = report.split_addenda({"s4": {"J-multi": record, "J-multi-loop-r2": record}})
    assert not registered.get("s4") and set(addenda["s4"]) == {"J-multi", "J-multi-loop-r2"}
