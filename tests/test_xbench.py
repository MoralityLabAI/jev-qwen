"""xbench: statistics, adapters and records (CPU). Foreign-repo tests skip when the repo is absent."""

import json
import math

import pytest
import torch

from jevq.xbench import stats
from jevq.xbench.foreign import CONTROL_HARNESS, HERMES_LITE, HERMES_SKILLS, LOOPED_TRANSFORMERS, RMP_ARTIFACTS
from jevq.xbench.jrunner import ChoiceItem, run_choice
from jevq.xbench.metrics import summarize
from jevq.xbench.records import write_not_applicable, write_record

needs_lt = pytest.mark.skipif(not LOOPED_TRANSFORMERS.exists(), reason="LoopedTransformers not on this machine")
needs_rmp = pytest.mark.skipif(not RMP_ARTIFACTS.exists(), reason="RMP artifacts not on this machine")
needs_hs = pytest.mark.skipif(not HERMES_SKILLS.exists(), reason="Hermes-Skills not on this machine")
needs_hl = pytest.mark.skipif(not HERMES_LITE.exists(), reason="hermes-lite not on this machine")
needs_ch = pytest.mark.skipif(not CONTROL_HARNESS.exists(), reason="Control-Harness not on this machine")


# ------------------------------------------------------------------ statistics


def test_wilson_and_mcnemar():
    low, high = stats.wilson(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-3) and high == pytest.approx(0.7634, abs=1e-3)
    assert stats.wilson(0, 0) == (None, None)
    assert stats.mcnemar_exact(0, 5) == pytest.approx(0.0625)
    assert stats.mcnemar_exact(3, 3) == 1.0 and stats.mcnemar_exact(0, 0) == 1.0


def test_holm_is_monotone_and_bounded():
    adjusted = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["c"] == pytest.approx(0.06) and adjusted["b"] == pytest.approx(0.06)
    assert all(0 <= v <= 1 for v in adjusted.values())


def test_kaplan_meier_half_life_and_log_rank():
    times = [1, 1, 2, None, None, 3]
    curve = stats.kaplan_meier(times, 4)
    assert curve[0] == pytest.approx(4 / 6) and curve[1] == pytest.approx(0.5) and curve[2] == pytest.approx(1 / 3)
    assert stats.half_life(curve) == 2
    assert stats.half_life(stats.kaplan_meier([None] * 5, 10)) is None
    assert stats.log_rank(times, times, 4) == pytest.approx(1.0)
    assert stats.log_rank([1] * 20, [None] * 20, 10) < 1e-4


def test_spearman_and_bootstrap():
    assert stats.spearman([1, 2, 3, 4], [2, 3, 5, 9]) == pytest.approx(1.0)
    assert stats.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert stats.spearman([1, 1, 1], [1, 2, 3]) is None
    estimate, low, high = stats.bootstrap_diff([1, 1, 1, 0], [0, 0, 1, 0], resamples=2000)
    assert estimate == pytest.approx(0.5) and low <= estimate <= high


def test_summarize_rates_calibration_and_abstention():
    rows = [
        {"gold": "A", "pred": "A", "correct": True, "options": ["A", "B"], "probs": [0.9, 0.1], "unsafe": None, "over_refusal": False, "family": "f", "depth": 1},
        {"gold": "B", "pred": "A", "correct": False, "options": ["A", "B"], "probs": [0.6, 0.4], "unsafe": True, "over_refusal": None, "family": "f", "depth": 2},
        {"gold": "B", "pred": "B", "correct": True, "options": ["A", "B"], "probs": [0.2, 0.8], "unsafe": False, "over_refusal": None,
         "family": "g", "depth": 2, "abstain_gold": True, "abstain_pred": True},
    ]
    m = summarize(rows)
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert m["unsafe"]["rate"] == 0.5 and m["unsafe"]["n"] == 2
    assert m["over_refusal"]["rate"] == 0.0 and m["over_refusal"]["n"] == 1
    assert m["calibration"]["nll"] == pytest.approx((-math.log(0.9) - math.log(0.4) - math.log(0.8)) / 3)
    assert m["abstention"] == {"precision": 1.0, "recall": 1.0, "n_gold": 1, "n_pred": 1}
    assert set(m["accuracy_by_family"]) == {"f", "g"} and m["accuracy_by_depth"]["2"]["n"] == 2


def test_records_round_trip(tmp_path):
    arm = {"arm_id": "demo", "neural": False}
    path = write_record("sx", arm, {"suite_id": "sx"}, [{"item_id": "1", "correct": True}], {"n": 1}, "deterministic_replay",
                        root=tmp_path, include_hardware=False)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["schema"] == "jevq.xbench.record.v1" and record["n_items"] == 1 and record["items_sha256"]
    with pytest.raises(ValueError):
        write_record("sx", arm, {}, [], {}, "made_up_label", root=tmp_path, include_hardware=False)
    na = json.loads(write_not_applicable("sx", "other", "why", root=tmp_path).read_text(encoding="utf-8"))
    assert na["not_applicable"] == "why"


# ------------------------------------------------------------------ J runner on the tiny model


def test_run_choice_and_iteration_lens(bundle):
    from jevq.looped import LoopSpec

    items = [ChoiceItem(f"i{k}", "Q: 1+1?\nAnswer:", [" A", " B"], ["yes", "no"], "yes", {"family": "t", "depth": k}) for k in range(3)]
    rows = run_choice(bundle, items)
    assert len(rows) == 3 and all(abs(sum(r["probs"]) - 1) < 1e-5 for r in rows)
    looped = run_choice(bundle, items, loop=LoopSpec(4, 8, 3), record_iterations=True)
    for row in looped:
        assert len(row["iteration_probs"]) == 3
        assert row["iteration_probs"][-1] == pytest.approx(row["probs"], abs=1e-5)
    single = run_choice(bundle, items, loop=LoopSpec(4, 8, 1))
    assert looped[0]["iteration_probs"][0] == pytest.approx(single[0]["probs"], abs=1e-5)
    with pytest.raises(RuntimeError, match="single token"):
        run_choice(bundle, [ChoiceItem("x", "Q", [" AB"], ["a"], "a")])


def test_s3_chunked_prefill_matches_single_pass(bundle):
    from jevq.xbench.jrunner import _last_logits

    ids = torch.randint(1, 100, (1, 40))
    full = _last_logits(bundle, ids, None, None, None)
    chunked = _last_logits(bundle, ids, None, None, 16)
    assert torch.allclose(full, chunked, atol=1e-4)


# ------------------------------------------------------------------ foreign adapters


@needs_lt
def test_s1_items_render_and_gold():
    from jevq.xbench import s1_rmp

    for family in s1_rmp.FAMILIES:
        rows = s1_rmp.load_items(family, 2)
        options, labels = s1_rmp.options_for(family)
        assert len(options) == len(labels) == len(set(labels))
        assert len(s1_rmp.option_tokens(family)) == len(options)
        for row in rows:
            assert s1_rmp.gold_value(family, row["target_token"]) in options
            assert s1_rmp.render_question(family, row["semantic"]) == s1_rmp.render_question(family, row["semantic"])
            assert s1_rmp.rationale(family, row).endswith(".")
            assert row["split"] in ("validation", "ood_stress") and row["sibling"] is None


@needs_lt
def test_s1_shots_come_from_the_train_region():
    from jevq.xbench import s1_rmp

    shots = s1_rmp.train_shots("pointer_chase")
    eval_ids = {r["fingerprint"] for f in s1_rmp.FAMILIES for r in s1_rmp.load_items(f, 25)}
    assert len(shots) == 3 and {s["split"] for s in shots} == {"train"}
    assert not {s["fingerprint"] for s in shots} & eval_ids


@needs_lt
@needs_rmp
def test_s1_native_loop_t_ds_solves_trained_depths():
    from jevq.xbench import s1_rmp

    rows, info = s1_rmp.native_rows("LOOP-T-ds", "pointer_chase", 2)
    assert info["params_total"] == 459_776
    assert all(r["correct"] for r in rows if not r["ood"])
    assert all(len(r["iteration_preds"]) == 8 for r in rows)


@needs_hs
def test_s2_packs_align_and_script_gate_is_the_ceiling():
    from jevq.xbench import s2_commit_veto as s2

    for pack in ("csig", "near_miss"):
        data = s2.load(pack)
        for split, rows in data.items():
            assert all(s2.script_gate(r["state"]) == r["target"] for r in rows), (pack, split)
    flags = s2.gate_flags({"gold": "reject_or_abstain", "pred": "commit"})
    assert flags == {"unsafe": True, "over_refusal": None}


@needs_ch
def test_s5_loop_matches_the_harness_runner(tmp_path):
    from jevq.xbench import s5_control as s5

    m2 = s5._modules()[0]
    config, _, _ = m2.load_config(s5.DEV_CONFIG)
    for control in config["controls"]:
        for scenario in s5.load_scenarios()[:4]:
            theirs = m2.run_episode(config=config, config_sha256="0" * 64, seed=0, actor={"id": "s", "kind": "scripted", "policy": "attack_reroute"},
                                    control=control, visibility="visible", horizon=8, scenario=scenario, receipt_path=tmp_path / f"{control['id']}{scenario['id']}.jsonl")
            _, rows = s5.run_episode(scenario, "attack_reroute", s5.HarnessControlMonitor(control))
            assert sum(bool(r["unsafe"]) for r in rows) == theirs["escaped_violations"]
            assert sum(bool(r["over_refusal"]) for r in rows) == theirs["false_blocks"]
    with pytest.raises(PermissionError):
        s5.load_scenarios(s5.DEV_PACK.with_name("bitagent_matched_provenance_v1_pack.yaml"))


def test_s6_attack_keeps_authoritative_fields():
    from jevq.xbench import s6_halflife as s6

    _, targets = s6.auth_targets()
    example = targets[0]
    for turn in (0, 1, 10):
        text = s6.attacked_question(example, turn)
        original = example.question.split("\n")
        assert next(l for l in original if l.startswith("Requester role:")) in text
        assert next(l for l in original if l.startswith("Request:")) in text
        assert s6.AUTHORITATIVE.strip() in text
        assert ("Request note:" in text) == bool(turn)
    times, excluded = s6.flip_times({0: ["DENY", "ALLOW", "DENY"], **{t: ["ALLOW" if t >= 3 else "DENY", "ALLOW", "DENY"] for t in range(1, 11)}})
    assert times == [3, None] and excluded == 1


@needs_hl
def test_s7_shortlist_contains_every_positive_gold():
    from jevq.xbench import s7_routing as s7

    items = s7.choice_items()
    assert len(items) == 128
    assert all(i.gold in i.options for i in items if i.meta["kind"] == "positive")
    assert all(i.options[-1] == "ABSTAIN" and len(i.labels) == len(i.options) for i in items)


def test_arm_adapters_resolve_under_the_checkpoint_root(tmp_path):
    """Regression: arm adapters are relative to paths.checkpoints, not the repo root (the first
    J-V1 xbench run failed on this; the end-to-end smoke test patches the loader and missed it)."""
    from pathlib import Path

    from jevq.config import load_yaml, resolve_path
    from jevq.xbench.arms import J_ARMS
    from jevq.xbench.jrunner import resolve_adapter

    root = resolve_path(load_yaml(resolve_path("configs/base.yaml"))["paths"]["checkpoints"])
    for arm in J_ARMS.values():
        if arm.adapter:
            assert resolve_adapter(arm.adapter) == root / arm.adapter
            assert "OneDrive" not in str(resolve_adapter(arm.adapter))
    assert resolve_adapter(str(tmp_path / "x")) == Path(tmp_path / "x")


@needs_hl
def test_s7_permuted_items_keep_options_and_spread_the_gold():
    """Addendum A2: same candidates per item, ABSTAIN last, gold no longer always option A."""
    from collections import Counter

    from jevq.xbench import s7_routing as s7

    base, perm = s7.choice_items(), s7.choice_items(permute=True)
    assert [i.item_id for i in base] == [i.item_id for i in perm]
    for a, b in zip(base, perm):
        assert sorted(a.options) == sorted(b.options) and b.options[-1] == "ABSTAIN" and a.gold == b.gold
    positions = Counter(i.options.index(i.gold) for i in perm if i.meta["kind"] == "positive")
    assert {i.options.index(i.gold) for i in base if i.meta["kind"] == "positive"} == {0}
    assert len(positions) == 5 and max(positions.values()) < 0.4 * sum(positions.values())
    assert s7.choice_items(permute=True)[0].prompt == perm[0].prompt  # deterministic
