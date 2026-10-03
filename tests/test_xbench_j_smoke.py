"""End-to-end smoke of the J-arm path (`run.run_j`) on the tiny random model, every suite but S3.

The real arms load the 4B model; here the bundle, the label-token lookup and the record root are
patched so every branch of run_j executes on CPU. Quality is meaningless; only plumbing is checked.
"""

import functools
import json


import pytest

from jevq.xbench import jrunner, run
from jevq.xbench.arms import JArm
from jevq.xbench.foreign import CONTROL_HARNESS, HERMES_LITE, HERMES_SKILLS, LOOPED_TRANSFORMERS
from jevq.xbench.records import write_not_applicable, write_record

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in (LOOPED_TRANSFORMERS, HERMES_SKILLS, HERMES_LITE, CONTROL_HARNESS)),
    reason="needs the sibling Morality Lab repositories",
)


class TolerantTokenizer:
    """Wraps the fake tokenizer: unknown characters become '?'."""

    def __init__(self, base):
        self.base = base
        self.pad_token_id, self.eos_token_id = base.pad_token_id, base.eos_token_id

    def encode(self, text, add_special_tokens=False):
        safe = "".join(ch if ch in self.base.char_to_id or ch == " " else "?" for ch in text)
        return self.base.encode(safe)

    def decode(self, ids, **kwargs):
        return self.base.decode(ids)


@pytest.fixture()
def patched(monkeypatch, tmp_path, tiny_model, tokenizer):
    from jevq.modeling import make_bundle

    bundle = make_bundle(tiny_model, TolerantTokenizer(tokenizer), {"id": "tiny"})
    vocab = tiny_model.config.vocab_size

    table: dict[str, int] = {}

    def fake_label_ids(tok, labels):
        for label in labels:
            table.setdefault(label, len(table) % vocab)
        ids = [table[label] for label in labels]
        assert len(set(ids)) == len(ids)
        return ids

    monkeypatch.setattr(jrunner, "load_j_bundle", lambda adapter=None, model_cfg=None: bundle)
    monkeypatch.setattr(jrunner, "label_ids", fake_label_ids)
    monkeypatch.setattr(run, "write_record", functools.partial(write_record, root=tmp_path, include_hardware=False))
    monkeypatch.setattr(run, "write_not_applicable", functools.partial(write_not_applicable, root=tmp_path))
    # Small S1 and S6 so the test stays quick.
    from jevq.xbench import s1_rmp, s6_halflife

    original = s1_rmp.load_items
    monkeypatch.setattr(s1_rmp, "load_items", lambda family, per_depth=25: original(family, 1)[::4])
    original_targets = s6_halflife.auth_targets
    monkeypatch.setattr(s6_halflife, "auth_targets", lambda seed=0: (lambda s, t: (s, t[:3]))(*original_targets(seed)))
    monkeypatch.setattr(run, "s4_items", lambda: run.s4_items.__wrapped__()[:16] if hasattr(run.s4_items, "__wrapped__") else original_s4()[:16])
    return tmp_path


original_s4 = run.s4_items


def _records(root):
    return {p.parent.relative_to(root).as_posix(): json.loads(p.read_text(encoding="utf-8")) for p in root.rglob("record.json")}


def test_run_j_choice_arm_writes_every_suite(patched, monkeypatch):
    monkeypatch.setitem(run.J_ARMS, "T-plain", JArm("T-plain", None, suites=("s1", "s2", "s4", "s5", "s6", "s7")))
    run.run_j("T-plain", ["s1", "s2", "s4", "s5", "s6", "s7"])
    records = _records(patched)
    for key in ("s1/T-plain", "s2/T-plain", "s4/T-plain", "s5/T-plain", "s6/T-plain", "s6c/T-plain", "s7/T-plain"):
        assert key in records, key
        assert records[key]["claim_label"] == "live_model_run"
    assert records["s2/T-plain"]["metrics"]["unsafe"]["n"] > 0
    assert "episodes" in records["s5/T-plain"]["metrics"]
    assert records["s6/T-plain"]["metrics"]["survival"] is not None


def test_run_j_looped_arm_splits_iterations(patched, monkeypatch):
    monkeypatch.setitem(run.J_ARMS, "T-loop", JArm("T-loop", None, loop_iters=2, suites=("s1", "s2", "s5", "s7")))
    monkeypatch.setattr(run.J_ARMS["T-loop"].__class__, "loop", property(lambda self: __import__("jevq.looped", fromlist=["LoopSpec"]).LoopSpec(4, 8, self.loop_iters) if self.loop_iters else None))
    run.run_j("T-loop", ["s1", "s2", "s5", "s7"])
    records = _records(patched)
    for suite in ("s1", "s2", "s5", "s7"):
        assert f"{suite}/T-loop-r1" in records and f"{suite}/T-loop-r2" in records, suite
    r1 = [json.loads(l) for l in (patched / "s1" / "T-loop-r1" / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(len(row["iteration_preds"]) == 1 for row in r1)


def test_run_j_cot_arm(patched, monkeypatch):
    monkeypatch.setitem(run.J_ARMS, "T-cot", JArm("T-cot", None, readout="cot", suites=("s1", "s2")))
    monkeypatch.setitem(run.COT_TOKENS, "s1", 4)
    run.run_j("T-cot", ["s1", "s2"])
    records = _records(patched)
    assert records["s1/T-cot"]["metrics"]["emitted_tokens_mean"] <= 4
    assert "not_applicable" in records["s2/T-cot"]
