"""SPEC-U4: deterministic disjoint items, the projection inputs equal the visible puzzle, the Hermes-Skills
repair source loads, outcomes and flow policies, prompts, grid parsing, and the report's pooling.
Module behaviour is checked on train items only (test outcomes are first computed by the registered run)."""

import json
import sys
from pathlib import Path

import pytest

from jevq.xbench.foreign import HERMES_LITE, HERMES_SKILLS

needs_both = pytest.mark.skipif(not (HERMES_LITE.exists() and HERMES_SKILLS.exists()), reason="hermes-lite / Hermes-Skills absent")


@needs_both
def test_items_deterministic_balanced_disjoint():
    from jevq.xbench import u4_campsite as u

    test = u.build_items("test")
    assert len(test) == 480 and all(sum(i["kind"] == k for i in test) == 60 for k in u.TYPES)
    assert u.items_hash() == "02e802f76cdaabf2"
    hashes = {p: {t.hash for t, _ in u.puzzles(p)} for p in ("test", "train", "val")}
    assert not (hashes["test"] & hashes["train"]) and not (hashes["test"] & hashes["val"]) and not (hashes["train"] & hashes["val"])
    for item in test:
        assert u.verify(item, item["candidate"])["official_pass"] == (item["kind"] == "correct")
        assert u.verify(item, item["gold"])["official_pass"]


@needs_both
def test_modules_outcomes_and_policies_on_train_items():
    from jevq.xbench import u4_campsite as u

    train = u.build_items("train")
    drop = next(i for i in train if i["kind"] == "drop_tent")
    assert u.project(drop, "c_repair") is not None
    bad = next(i for i in train if i["kind"] == "bad_shape")
    assert u.project(bad, "c_repair") is None and u.committed(bad, "c_repair") == bad["candidate"]
    assert u.outcome(bad, "c_repair")["unsafe"]
    rejected = u.outcome(bad, "reject")
    assert (rejected["success"], rejected["unsafe"], rejected["rejected"]) == (False, False, True)
    correct = next(i for i in train if i["kind"] == "correct")
    assert u.best_action(correct) == "commit" and u.flow_policy(correct, "dual_repair_if_any_sig_fail") == "commit"
    swap = next(i for i in train if i["kind"] == "swap_rect")
    assert u.signatures(swap) == (True, True) and u.flow_policy(swap, "c_repair_if_c_fail") == "commit"
    assert len(u.features(drop)) == len(u.features(bad))


@needs_both
def test_prompts_and_parsing():
    from jevq.xbench import u4_campsite as u

    item = next(i for i in u.build_items("test") if i["kind"] == "drop_tent")
    prompt, options = u.decide_prompt(item)
    assert sorted(options) == sorted(u.ACTIONS) and prompt.endswith("Answer:")
    assert u.grid_text(item["gold"]) not in prompt.split("Answer:")[-1]  # the gold grid of the item is never shown
    rp = u.repair_prompt(item)
    assert rp.endswith("Repaired:\n") and u.grid_text(item["gold"]) not in rp.split("Repaired:")[-2] + rp.split("Repaired:")[-1]
    n = len(item["task"]["grid"])
    text = u.grid_text(item["gold"]) + "\n\nextra"
    assert u.parse_grid(text, n) == item["gold"]
    assert u.parse_grid("not a grid", n) is None
    done = u.repair_items([item])[0].done
    assert not done(u.grid_text(item["gold"][:-1])) and done(u.grid_text(item["gold"]) + "\n")


def test_report_pooling(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import u4_report as rep

    def put(record, success):
        d = tmp_path / "decide"
        d.mkdir(parents=True, exist_ok=True)
        rows = [{"item_id": f"u4.test.correct.{k:03d}", "kind": "correct", "action": "commit", "success": s, "unsafe": not s, "rejected": False}
                for k, s in enumerate(success)]
        (d / f"{record}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    monkeypatch.setattr(rep, "RUNS", tmp_path)
    put("J-V0", [True, False, False])
    for sfx in ("", "-s1", "-s2"):
        put(f"J-multi{sfx}", [True, True, False])
    a, b = rep.instances("decide", "J-multi"), rep.instances("decide", "J-V0")
    res = rep.u2r.pooled(a, b)
    assert (res["a_only"], res["b_only"]) == (3, 0) and rep.rate(a, "success") == (6, 9)
