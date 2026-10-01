"""Correctness of the layer-schedule driver against the stock HF forward."""

import math

import pytest
import torch

from jevq.flops import FlopModel, estimate
from jevq.instrument import ConvergenceHalt, StepRecorder
from jevq.looped import LoopSpec, build_schedule, forward_hidden, greedy_generate


@pytest.fixture()
def ids():
    torch.manual_seed(1)
    return torch.randint(1, 100, (2, 24))


def driver_logits(model, ids, loop=None, **kwargs):
    result = forward_hidden(model.model, input_ids=ids, loop=loop, **kwargs)
    return model.lm_head(result.hidden), result


@torch.no_grad()
def test_no_loop_reproduces_stock_forward(tiny_model, ids):
    stock = tiny_model(input_ids=ids, use_cache=False).logits
    mine, result = driver_logits(tiny_model, ids)
    assert torch.equal(stock, mine)
    assert result.layer_applications == 8 and result.n_iters == 1


@torch.no_grad()
def test_single_iteration_loop_is_identity(tiny_model, ids):
    stock = tiny_model(input_ids=ids, use_cache=False).logits
    for start, end in [(0, 4), (2, 6), (4, 8), (0, 8), (3, 4)]:
        mine, _ = driver_logits(tiny_model, ids, LoopSpec(start, end, 1))
        assert torch.equal(stock, mine), (start, end)


@torch.no_grad()
def test_stock_cached_forward_is_close(tiny_model, ids):
    # The cached path takes a different numerical route through the DeltaNet conv state.
    cached = tiny_model(input_ids=ids, use_cache=True).logits
    mine, _ = driver_logits(tiny_model, ids)
    assert torch.allclose(cached, mine, atol=1e-4)


@torch.no_grad()
def test_loop_changes_output_and_counts_applications(tiny_model, ids):
    base, _ = driver_logits(tiny_model, ids)
    looped, result = driver_logits(tiny_model, ids, LoopSpec(2, 6, 3))
    assert result.n_iters == 3
    assert result.layer_applications == 8 + 2 * 4
    assert not torch.allclose(base, looped, atol=1e-3)
    assert torch.isfinite(looped).all()


def test_schedule_layout():
    assert build_schedule(4) == [(0, 0), (1, 0), (2, 0), (3, 0)]
    assert build_schedule(4, LoopSpec(1, 3, 2)) == [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (3, 0)]
    with pytest.raises(ValueError):
        build_schedule(4, LoopSpec(2, 5, 2))
    with pytest.raises(ValueError):
        build_schedule(4, LoopSpec(1, 3, 0))


@torch.no_grad()
def test_recorder_rows(tiny_model, ids):
    lens = lambda h: tiny_model.lm_head(tiny_model.model.norm(h))  # noqa: E731
    recorder = StepRecorder(lens_fn=lens, record_layers=True)
    recorder.begin_forward(gen_step=0)
    forward_hidden(tiny_model.model, input_ids=ids[:1], loop=LoopSpec(4, 8, 4), recorder=recorder)

    spans = [r for r in recorder.rows if r["kind"] == "span_iter"]
    layers = [r for r in recorder.rows if r["kind"] == "layer"]
    assert [r["iter"] for r in spans] == [0, 1, 2, 3]
    assert len(layers) == 8 + 3 * 4
    assert spans[0]["cos_first_last"] == 1.0 and spans[0]["lens_kl_prev"] is None
    assert spans[1]["cos_prev2_last"] is None and spans[2]["cos_prev2_last"] is not None
    for row in spans:
        for key in ("norm_last", "norm_mean", "cos_prev_last", "rel_delta_last", "lens_entropy", "dt_s"):
            assert math.isfinite(row[key]), key
        assert row["gen_step"] == 0
    for row in spans[1:]:
        assert row["lens_kl_prev"] >= -1e-6


@torch.no_grad()
def test_recorder_does_not_change_the_result(tiny_model, ids):
    plain, _ = driver_logits(tiny_model, ids, LoopSpec(2, 6, 3))
    recorder = StepRecorder(record_layers=True)
    recorder.begin_forward()
    recorded, _ = driver_logits(tiny_model, ids, LoopSpec(2, 6, 3), recorder=recorder)
    assert torch.equal(plain, recorded)


@torch.no_grad()
def test_halting(tiny_model, ids):
    loop = LoopSpec(2, 6, 5)
    always, result = driver_logits(tiny_model, ids, loop, halt_fn=ConvergenceHalt(1e9))
    assert result.n_iters == 1 and result.halted_early
    base, _ = driver_logits(tiny_model, ids)
    assert torch.equal(always, base)

    never, result = driver_logits(tiny_model, ids, loop, halt_fn=ConvergenceHalt(0.0))
    full, _ = driver_logits(tiny_model, ids, loop)
    assert result.n_iters == 5 and not result.halted_early
    assert torch.equal(never, full)

    recorder = StepRecorder()
    recorder.begin_forward()
    forward_hidden(tiny_model.model, input_ids=ids, loop=loop, recorder=recorder, halt_fn=ConvergenceHalt(1e9))
    assert [r["halted"] for r in recorder.rows] == [True]


@torch.no_grad()
def test_uncached_greedy_matches_stock_generate(tiny_model, ids):
    prompt = ids[:1]
    stock = tiny_model.generate(
        input_ids=prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=6, do_sample=False, pad_token_id=0
    )[0, prompt.shape[1] :].tolist()
    mine, forwards = greedy_generate(tiny_model.model, tiny_model.lm_head, prompt, 6, lambda new: False)
    assert mine == stock
    assert len(forwards) == 6


def test_flops_scale_with_schedule(tiny_model):
    fm = FlopModel.from_text_model(tiny_model.model, tiny_model.lm_head)
    vanilla = estimate(fm, build_schedule(8), prompt_tokens=50, new_tokens=0, uncached=True)
    looped = estimate(fm, build_schedule(8, LoopSpec(4, 8, 3)), prompt_tokens=50, new_tokens=0, uncached=True)
    assert looped["flops_cached_equiv"] > vanilla["flops_cached_equiv"]
    assert vanilla["flops_actual"] == vanilla["flops_cached_equiv"]

    generated = estimate(fm, build_schedule(8), prompt_tokens=50, new_tokens=8, uncached=True)
    assert generated["flops_actual"] > generated["flops_cached_equiv"] > vanilla["flops_cached_equiv"]
    cached = estimate(fm, build_schedule(8), prompt_tokens=50, new_tokens=8, uncached=False)
    assert cached["flops_actual"] == cached["flops_cached_equiv"]
