"""Layer-schedule forward driver for Qwen3.5 text models (variants V2+).

`forward_hidden` mirrors `Qwen3_5TextModel.forward` (transformers 5.17.0,
models/qwen3_5/modeling_qwen3_5.py, the block around the decoder-layer loop) but
walks an explicit schedule: prefix layers once, a configurable span of layers
`n_iters` times, suffix layers once. It calls the model's own submodules and the
same mask helpers, so with `loop=None` (or `n_iters=1`) it must reproduce the
stock forward exactly. tests/test_looped.py asserts that.

The driver never uses a KV / recurrent-state cache: every call recomputes the
full sequence. That keeps re-applied layers correct without per-iteration cache
slots. See notes/003-compromises.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask


@dataclass(frozen=True)
class LoopSpec:
    """Run layers [start, end) `n_iters` times in total. n_iters=1 is the vanilla model."""

    start: int
    end: int
    n_iters: int = 1

    @property
    def span(self) -> int:
        return self.end - self.start

    def validate(self, num_layers: int) -> None:
        if not (0 <= self.start < self.end <= num_layers):
            raise ValueError(f"loop span [{self.start}, {self.end}) is outside 0..{num_layers}")
        if self.n_iters < 1:
            raise ValueError(f"n_iters must be >= 1, got {self.n_iters}")


def build_schedule(num_layers: int, loop: LoopSpec | None = None) -> list[tuple[int, int]]:
    """The static (layer_idx, iter_idx) order the driver executes when nothing halts early."""
    if loop is None:
        return [(i, 0) for i in range(num_layers)]
    loop.validate(num_layers)
    schedule = [(i, 0) for i in range(loop.start)]
    for it in range(loop.n_iters):
        schedule += [(i, it) for i in range(loop.start, loop.end)]
    schedule += [(i, 0) for i in range(loop.end, num_layers)]
    return schedule


@dataclass
class ForwardResult:
    hidden: torch.Tensor  # (batch, seq, hidden), after the final norm
    n_iters: int  # span iterations actually executed (1 for the vanilla model)
    layer_applications: int
    halted_early: bool


def forward_hidden(
    text_model,
    input_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    loop: LoopSpec | None = None,
    recorder=None,
    halt_fn: Callable[[dict], bool] | None = None,
) -> ForwardResult:
    """Full-sequence forward through `text_model` (a Qwen3_5TextModel) following `loop`.

    `halt_fn` receives the per-iteration stats dict (see instrument.span_stats) after each
    span iteration and returns True to stop iterating; `loop.n_iters` is then the maximum.
    """
    cfg = text_model.config
    num_layers = cfg.num_hidden_layers
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("specify exactly one of input_ids or inputs_embeds")
    if loop is not None:
        loop.validate(num_layers)

    if inputs_embeds is None:
        inputs_embeds = text_model.embed_tokens(input_ids)
    batch_size, seq_len = inputs_embeds.shape[:2]

    # Same position-id layout as the stock forward: 4 rows = text, temporal, height, width.
    position_ids = torch.arange(seq_len, device=inputs_embeds.device).view(1, 1, -1).expand(4, batch_size, -1)
    text_position_ids = position_ids[0]
    position_ids = position_ids[1:]

    mask_kwargs = {
        "config": cfg,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": None,
        "position_ids": text_position_ids,
    }
    masks = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
    }

    hidden = inputs_embeds
    position_embeddings = text_model.rotary_emb(hidden, position_ids)
    time_layers = recorder is not None and recorder.record_layers
    applications = 0

    def run_layer(i: int, h: torch.Tensor, it: int) -> torch.Tensor:
        nonlocal applications
        t0 = recorder.now() if time_layers else 0.0
        out = text_model.layers[i](
            h,
            position_embeddings=position_embeddings,
            attention_mask=masks[cfg.layer_types[i]],
            position_ids=text_position_ids,
            past_key_values=None,
            use_cache=False,
        )
        applications += 1
        if time_layers:
            recorder.on_layer(i, it, h, out, t0)
        return out

    start, end = (loop.start, loop.end) if loop is not None else (num_layers, num_layers)

    def tail(h: torch.Tensor) -> torch.Tensor:
        """Layers after the span, for the recorder's lens. Not counted as model computation."""
        for i in range(end, num_layers):
            h = text_model.layers[i](
                h,
                position_embeddings=position_embeddings,
                attention_mask=masks[cfg.layer_types[i]],
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
            )
        return h

    for i in range(start):
        hidden = run_layer(i, hidden, 0)

    n_iters, halted = 1, False
    if loop is not None:
        n_iters = 0
        for it in range(loop.n_iters):
            span_in = hidden
            t0 = recorder.now() if recorder is not None else 0.0
            for i in range(start, end):
                hidden = run_layer(i, hidden, it)
            n_iters += 1
            stats = recorder.on_span_iter(it, span_in, hidden, t0, tail_fn=tail) if recorder is not None else None
            if halt_fn is not None and it + 1 < loop.n_iters:
                if stats is None:
                    from .instrument import span_stats

                    stats = span_stats(span_in, hidden)
                halted = bool(halt_fn(stats))
                if recorder is not None:
                    recorder.on_halt_decision(halted)
                if halted:
                    break

    for i in range(end, num_layers):
        hidden = run_layer(i, hidden, 0)

    hidden = text_model.norm(hidden)
    return ForwardResult(hidden=hidden, n_iters=n_iters, layer_applications=applications, halted_early=halted)


@torch.no_grad()
def greedy_generate(
    text_model,
    lm_head,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    stop_fn: Callable[[list[int]], bool],
    loop: LoopSpec | None = None,
    recorder=None,
    halt_fn: Callable[[dict], bool] | None = None,
) -> tuple[list[int], list[ForwardResult]]:
    """Uncached greedy decoding for batch size 1: one full forward per emitted token."""
    if input_ids.shape[0] != 1:
        raise ValueError("greedy_generate supports batch size 1 only")
    ids = input_ids
    new_tokens: list[int] = []
    forwards: list[ForwardResult] = []
    for step in range(max_new_tokens):
        if recorder is not None:
            recorder.begin_forward(gen_step=step)
        result = forward_hidden(text_model, input_ids=ids, loop=loop, recorder=recorder, halt_fn=halt_fn)
        next_id = lm_head(result.hidden[:, -1]).argmax(dim=-1, keepdim=True)
        result.hidden = None  # keep only the bookkeeping
        forwards.append(result)
        ids = torch.cat([ids, next_id], dim=-1)
        new_tokens.append(int(next_id))
        if stop_fn(new_tokens):
            break
    return new_tokens, forwards
