"""SPEC-U6: disjoint real-proposal pools, set registration with U5, the forgetting test's interval
level, and the report's arm routing. Nothing here runs a proposer or reads U5 test outcomes."""

import sys
from pathlib import Path

import pytest

from jevq.xbench.foreign import HERMES_LITE, HERMES_SKILLS

needs_both = pytest.mark.skipif(not (HERMES_LITE.exists() and HERMES_SKILLS.exists()), reason="hermes-lite / Hermes-Skills absent")


@needs_both
def test_real_pools_disjoint_and_registered():
    from jevq.xbench import u4_campsite as u4
    from jevq.xbench import u5_campsite as u5
    from jevq.xbench import u6_campsite as u6

    rt, rv = u6.puzzles("realtrain"), u6.puzzles("realval")
    assert len(rt) == 598 and len(rv) == 119
    hashes = lambda ps: {t.hash for t, _ in ps}  # noqa: E731
    held = hashes(u5.proposal_puzzles()) | hashes(u4.puzzles("test")) | hashes(u4.puzzles("u5test"))
    assert not (hashes(rt) & hashes(rv)) and not (hashes(rt) & held) and not (hashes(rv) & held)
    assert all(u4.camp().verify_candidate(t, g)["official_pass"] for t, g in rt + rv)
    assert "realtrain" in u5.EXTRA_SETS and u5.projection_path("realtrain").parts[-3] == "u6"
    assert u5.projection_path("proposals").parts[-3] == "u5"


def test_noninferiority_level_and_arm_routing():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import u6_report as rep

    a = {f"i{k}": {"correct": k % 10 != 0} for k in range(200)}
    b = {f"i{k}": {"correct": k % 12 != 0} for k in range(200)}
    pairs = [(a, b, sorted(a))]
    wide, narrow = rep.noninferiority(pairs, level=0.99), rep.noninferiority(pairs, level=0.95)
    assert wide["low"] <= narrow["low"] and wide["high"] >= narrow["high"] and wide["estimate"] == narrow["estimate"]
    assert rep.path_for("repair", "proposals", "J-real-s1").parts[-4] == "u6"
    assert rep.path_for("repair", "proposals", "J-u4").parts[-4] == "u5"
    assert rep.path_for("decide", "u4test", "Decision-TRM").parts[-3] == "u4"
