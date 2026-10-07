"""SPEC-U5: disjoint pools (U4 unchanged), large puzzles, reject paths never lower success, the capped
projection worker, zero-shot prompts and training rows, proposal prompts, and the report's pooling.
Module behaviour is checked on U4 train items only."""

import json
import sys
from pathlib import Path

import pytest

from jevq.xbench.foreign import HERMES_LITE, HERMES_SKILLS

needs_both = pytest.mark.skipif(not (HERMES_LITE.exists() and HERMES_SKILLS.exists()), reason="hermes-lite / Hermes-Skills absent")


@needs_both
def test_pools_disjoint_u4_unchanged_and_large_puzzles():
    from jevq.xbench import u4_campsite as u4
    from jevq.xbench import u5_campsite as u5

    assert u4.items_hash("test") == "02e802f76cdaabf2"
    assert len(u4.build_items("u5test")) == 480 and u4.items_hash("u5test") == "05e8f23e50a6feb8"
    hashes = {p: {t.hash for t, _ in u4.puzzles(p)} for p in u4.POOLS}
    names = list(hashes)
    assert all(not (hashes[a] & hashes[b]) for i, a in enumerate(names) for b in names[i + 1:])
    large = u5.large_puzzles()
    assert [len(t.grid) for t, _ in large].count(8) == 60 and [len(t.grid) for t, _ in large].count(10) == 60
    assert all(u4.camp().verify_candidate(t, g)["official_pass"] for t, g in large)
    assert len(u5.proposal_puzzles()) == len(u4.puzzles("u5std")) + 120


@needs_both
def test_reject_paths_never_lower_success(monkeypatch):
    from jevq.xbench import u4_campsite as u4
    from jevq.xbench import u5_campsite as u5

    items = [i for k in u4.TYPES for i in [x for x in u4.build_items("train") if x["kind"] == k][:5]]
    cache = {i["item_id"]: {m: {"grid": u4.project(i, m), "status": "ok"} for m in ("c_repair", "dual_repair")} for i in items}
    monkeypatch.setattr(u5, "projections", lambda set_name: cache)
    for policy in u5.POLICIES:
        res = {mode: [u5.outcome("t", i, u5.with_reject("t", i, u4.flow_policy(i, policy), mode)) for i in items] for mode in u5.REJECT_MODES}
        succ = {m: sum(r["success"] for r in v) for m, v in res.items()}
        unsafe = {m: sum(r["unsafe"] for r in v) for m, v in res.items()}
        assert succ["plain"] == succ["noop-reject"] == succ["verify-reject"]
        assert unsafe["verify-reject"] == 0 <= unsafe["noop-reject"] <= unsafe["plain"]
        assert all(r["rejected"] for r, i in zip(res["noop-reject"], items) if i["kind"] in ("swap_rect", "bad_shape"))


@needs_both
def test_capped_projection_matches_in_process():
    from jevq.xbench import u4_campsite as u4
    from jevq.xbench import u5_campsite as u5

    item = next(i for i in u4.build_items("train") if i["kind"] == "tree_mutation")
    source = u4.git_show(u4.HERMES_SKILLS, u4.REPAIR_SCRIPT)
    out = u5._capped(item, "dual_signature_projection", source, cap=60)
    assert out["status"] == "ok" and out["grid"] == u4.project(item, "dual_repair")


@needs_both
def test_zero_shot_prompts_training_rows_and_proposals():
    from jevq.xbench import u4_campsite as u4
    from jevq.xbench import u5_campsite as u5

    decide, repair = u5.training_rows("val", limit=20)
    assert len(decide) == len(repair) == 20 and all(d.gold in d.options for d in decide)
    assert all(p.endswith("Repaired:\n") and t.endswith("\n\n") for p, t in repair)
    assert "Answer: " not in decide[0].prompt  # zero-shot: no answered shots
    task, gold = u5.large_puzzles()[0]
    prompt = u5.propose_prompt(task)
    assert prompt.endswith("Solution:\n") and u4.grid_text(gold) not in prompt
    assert u5.proposal_budget(task) == 2 * 64 + 16


def test_report_pooling(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import u5_report as rep

    def put(record, success):
        d = tmp_path / "u5" / "decide" / "proposals"
        d.mkdir(parents=True, exist_ok=True)
        rows = [{"item_id": f"p{k}", "kind": "std", "success": s, "unsafe": not s, "rejected": False} for k, s in enumerate(success)]
        (d / f"{record}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    monkeypatch.setattr(rep, "U5_RUNS", tmp_path / "u5")
    put("Decision-TRM", [True, False, False])
    for sfx in ("", "-s1", "-s2"):
        put(f"J-u4{sfx}", [True, True, False])
    a, b = rep.instances("decide", "proposals", "J-u4"), rep.instances("decide", "proposals", "Decision-TRM")
    assert len(a) == 3 and len(b) == 1
    res = rep.u2r.pooled(a, b)
    assert (res["a_only"], res["b_only"]) == (3, 0)
