"""SPEC-U2: the joint set is deterministic and balanced, gate labels follow the registered rule,
end-to-end scoring, and pipeline composition with pooled tests."""

import json
import sys
from pathlib import Path

import pytest

from jevq.xbench import s2_commit_veto as s2
from jevq.xbench import u2_joint as u
from jevq.xbench.foreign import HERMES_LITE, HERMES_SKILLS

needs_both = pytest.mark.skipif(not (HERMES_LITE.exists() and HERMES_SKILLS.exists()), reason="hermes-lite / Hermes-Skills absent")


@needs_both
def test_joint_set_is_deterministic_balanced_and_rule_labelled():
    a, b = u.build_items(), u.build_items()
    assert a == b and len(a) == 600
    joint = [i for i in a if i["kind"] == "joint"]
    assert len(joint) == 360 and sum(i["gold_gate"] == "commit" for i in joint) == 180
    assert all(s2.script_gate(i["gate_state"]) == i["gold_gate"] for i in joint)
    zero = [i for i in joint if i["gate_state"]["reward_delta"] == 0]
    assert zero and all(i["gold_gate"] == "reject_or_abstain" for i in zero)
    from jevq.xbench import s7_routing as s7
    registered = {r["query"] for r in s7._jsonl(s7.REGISTERED / "held_cases.jsonl")} | {r["query"] for r in s7._jsonl(s7.REGISTERED / "train_rows.jsonl")}
    assert not {i["query"] for i in a} & registered


def test_end_to_end_scoring():
    joint = {"kind": "joint", "gold_skill": "g", "forbidden": "", "gold_gate": "reject_or_abstain"}
    assert u.end_to_end(joint, "g", "reject_or_abstain") == {"success": True, "route_ok": True, "unsafe": False}
    assert u.end_to_end(joint, "g", "commit") == {"success": False, "route_ok": True, "unsafe": True}
    assert u.end_to_end(joint, "other", "reject_or_abstain") == {"success": False, "route_ok": False, "unsafe": False}
    negative = {"kind": "negative", "gold_skill": "", "forbidden": "f", "gold_gate": None}
    assert u.end_to_end(negative, "ABSTAIN", None)["success"] and u.end_to_end(negative, "f", None)["unsafe"]
    confusable = {"kind": "confusable", "gold_skill": "c", "forbidden": "", "gold_gate": None}
    assert u.end_to_end(confusable, "c", None)["success"] and not u.end_to_end(confusable, "g", None)["success"]


def test_pipeline_composition_and_pooled_tests(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import u2_report as rep

    items = [{"item_id": f"u2.joint.{k}", "kind": "joint", "gold_skill": "g", "forbidden": "", "gold_gate": "commit"} for k in range(4)]

    def put(arm, route_preds=None, gate_preds=None):
        d = tmp_path / arm
        d.mkdir(parents=True, exist_ok=True)
        if route_preds is not None:
            (d / "route.jsonl").write_text("".join(json.dumps({"item_id": i["item_id"], "pred": p}) + "\n" for i, p in zip(items, route_preds)))
        if gate_preds is not None:
            (d / "gate.jsonl").write_text("".join(json.dumps({"item_id": i["item_id"], "pred": p}) + "\n" for i, p in zip(items, gate_preds)))

    monkeypatch.setattr(rep, "RUNS", tmp_path)
    put("J-V0", ["g", "x", "x", "x"])
    put("TRM-cv", gate_preds=["commit"] * 4)
    for sfx in ("", "-s1", "-s2"):
        put(f"J-multi{sfx}", ["g", "g", "g", "x"], ["commit"] * 4)
    a, c = rep.instances(items, "A"), rep.instances(items, "C")
    assert len(a) == 1 and len(c) == 3
    r = rep.pooled(c, a)
    assert (r["a_only"], r["b_only"]) == (6, 0) and r["p_one_sided"] == pytest.approx(1 / 64)
    assert rep.rate(c, "success") == (9, 12)
    assert rep.instances(items, "B") == []
