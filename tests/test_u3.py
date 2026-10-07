"""SPEC-U3: rudder rows and label space, the published ports (retrieval, MeTTa rules), the leak-free
retrieval, the Jev prompts, the CPU rudder arms, the abstain-head fit, and the report's pooling."""

import json
import sys
from pathlib import Path

import pytest

from jevq.xbench import u3_rudder as r
from jevq.xbench.foreign import HERMES_LITE, HERMES_SKILLS

needs_skills = pytest.mark.skipif(not HERMES_SKILLS.exists(), reason="Hermes-Skills absent")
needs_lite = pytest.mark.skipif(not (HERMES_LITE.exists() and HERMES_SKILLS.exists()), reason="hermes-lite / Hermes-Skills absent")


@needs_skills
def test_rows_and_label_space():
    data = r.load_rows()
    rows = r.eval_rows(data)
    assert [len(data[s]) for s in ("train",) + r.EVAL_SPLITS] == [156, 34, 36, 18]
    assert len({r.key(x) for x in rows}) == 88
    actions = r.allowed_actions(data["train"])
    assert actions == ["boxed_choice_extract", "c_repair", "choice_token_extract", "dual_repair", "exact_candidate_select", "original"]
    # The structural ceiling stated in the SPEC: 70/88 rows have an attainable repair action, 36/54 on the holdout rows.
    assert sum(x["action"] in actions for x in rows) == 70
    assert sum(x["action"] in actions for x in rows if x["eval_split"] in r.TEST_SPLITS) == 36


@needs_skills
def test_published_ports_reproduce_receipts():
    data = r.load_rows()
    by_key = {r.key(x): x for x in r.eval_rows(data)}
    path, _ = r.RECEIPTS["Qwen2.5-3B"][0]
    published = [json.loads(x) for x in r.git_show(HERMES_SKILLS, path).splitlines() if x.strip()]
    retrieval = [p for p in published if p["arm"] == "repair_training_rudder"]
    assert len(retrieval) == 88
    for p in retrieval:
        row = by_key[f"{p['eval_split']}:{p['case_id']}"]
        assert [e["state"]["case_id"] for e in r.published_retrieve(data["train"], row)] == p["retrieved_case_ids"]
    path, _ = r.RECEIPTS["Qwen2.5-3B"][1]
    published = [json.loads(x) for x in r.git_show(HERMES_SKILLS, path).splitlines() if x.strip()]
    for p in (p for p in published if p["arm"] == "metta_validator_gate"):
        row = by_key[f"{p['eval_split']}:{p['case_id']}"]
        assert (r.metta_repair_action(row), r.metta_validator_target_action(row)) == (p["predicted_repair_action"], p["predicted_target_action"])


@needs_skills
def test_clean_retrieval_ignores_the_answer_and_published_does_not():
    data = r.load_rows()
    train = data["train"]
    changed_published = 0
    for row in r.eval_rows(data):
        flipped = dict(row, bucket="exact_positive" if row["bucket"] != "exact_positive" else "repair_success",
                       action="original" if row["action"] != "original" else "c_repair",
                       target_action="commit" if row["target_action"] != "commit" else "reject_or_abstain")
        ids = lambda xs: [x["state"]["case_id"] for x in xs]  # noqa: E731
        assert ids(r.clean_retrieve(train, row)) == ids(r.clean_retrieve(train, flipped))
        changed_published += ids(r.published_retrieve(train, row)) != ids(r.published_retrieve(train, flipped))
    assert changed_published > 0  # the published retrieval leaks the answer


@needs_skills
def test_jev_prompts_hide_the_answer_and_map_letters():
    data = r.load_rows()
    row = data["holdout_seen"][0]
    actions = r.allowed_actions(data["train"])
    examples = r.clean_retrieve(data["train"], row)
    item = r.repair_item(row, actions, examples)
    eval_block = item.prompt.split("State: ")[-1]
    assert "bucket" not in eval_block and "target_" not in eval_block and "repair_gate" not in eval_block
    assert sorted(item.options) == actions and item.labels == [" A", " B", " C", " D", " E", " F"]
    assert item.options == r.repair_item(row, actions, examples).options  # deterministic shuffle
    for ex in examples:
        assert f"Repair: {r.LETTERS[item.options.index(ex['action'])]}" in item.prompt
    tgt = r.target_item(row, "c_repair", examples)
    assert tgt.prompt.endswith("Repair: c_repair\nDecision:") and tgt.options == ["commit", "reject_or_abstain"]


@needs_skills
def test_cpu_rudder_arms_score_and_static_gate():
    data = r.load_rows()
    rows, train = r.eval_rows(data), data["train"]
    lookup = r.lookup_predictor(train)
    scored = [r.score(x, *lookup(x)) for x in rows]
    assert all(s["joint_ok"] == (s["repair_ok"] and s["action_ok"]) for s in scored)
    assert all(s["unsafe"] == (s["pred_action"] == "commit" and s["target_action"] != "commit") for s in scored)
    predict, info = r.train_repair_trm(train, data["val_seen"], hidden=32)
    assert info["epochs_selected"] in (8, 25, 50, 100)
    assert predict(rows[0])[0] in r.allowed_actions(train)
    action_space = {r.key(x): {"pred_action": "commit"} for x in rows}
    static = r.static_gate(rows, action_space)
    assert all(s["pred_repair"] == r.metta_repair_action(x) for s, x in zip(static, rows))


@needs_lite
def test_abstain_head_fit_is_deterministic_and_leaves_the_router_unchanged():
    import torch

    from jevq.xbench import s7_routing as s7
    from jevq.xbench import u3_abstain as a

    rows = a.training_rows()
    assert sum(y for _, y in rows) == 23
    primary = s7.contracts()
    fits, choices = [], []
    for _ in range(2):
        model = s7.load_trm()
        before = [a.view(q, primary, model)["choice"] for q, _ in rows[:10]]
        fits.append(a.fit_abstain_head(model, rows, primary))
        choices.append([a.view(q, primary, model)["choice"] for q, _ in rows[:10]])
        assert choices[-1] == before
    assert fits[0]["loss_last"] == fits[1]["loss_last"] < fits[0]["loss_first"]
    assert torch.is_tensor(model.abstain_head.weight)


def test_report_pooling_on_fake_records(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import u3_report as rep

    def put(record, protocol, joint):
        d = tmp_path / record
        d.mkdir(parents=True, exist_ok=True)
        rows = [{"key": f"holdout_seen:{i}", "split": "holdout_seen", "joint_ok": j, "repair_ok": j, "action_ok": True, "unsafe": False}
                for i, j in enumerate(joint)]
        (d / f"{protocol}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))

    monkeypatch.setattr(rep, "RUDDER", tmp_path)
    put("J-V0", "retrieval", [True, False, False, False])
    for sfx in ("", "-s1", "-s2"):
        put(f"J-multi{sfx}", "retrieval", [True, True, True, False])
    a, b = rep.rudder_instances("J-multi", "retrieval"), rep.rudder_instances("J-V0", "retrieval")
    assert len(a) == 3 and len(b) == 1
    res = rep.u2r.pooled(rep.as_success(rep.restrict(a, r.TEST_SPLITS)), rep.as_success(rep.restrict(b, r.TEST_SPLITS)))
    assert (res["a_only"], res["b_only"]) == (6, 0)
    assert rep.frac(a, "joint_ok") == (9, 12)
