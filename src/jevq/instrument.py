"""Per-step instrumentation for the layer-schedule driver.

One row per span iteration (and optionally per layer application) with:
hidden-state norms, cosine similarity / relative change against the previous
state, cosine against the first iteration and against two iterations back
(period-2 oscillation), logit-lens entropy and KL (through the remaining layers
by default), watched-token probabilities, halting decision, and wall-clock
time. All statistics are computed in float32.
"""

from __future__ import annotations

import time
from typing import Callable

import torch
import torch.nn.functional as F


@torch.no_grad()
def span_stats(h_prev: torch.Tensor, h_cur: torch.Tensor) -> dict:
    """State-change statistics between two (batch, seq, hidden) tensors.

    `*_last` is the final sequence position (the readout position); `*_mean` averages
    over all positions.
    """
    a, b = h_prev.float(), h_cur.float()
    norm_a = a.norm(dim=-1)
    norm_b = b.norm(dim=-1)
    cos = F.cosine_similarity(a, b, dim=-1)
    rel = (b - a).norm(dim=-1) / norm_a.clamp_min(1e-8)
    return {
        "norm_last": norm_b[:, -1].mean().item(),
        "norm_mean": norm_b.mean().item(),
        "cos_prev_last": cos[:, -1].mean().item(),
        "cos_prev_mean": cos.mean().item(),
        "rel_delta_last": rel[:, -1].mean().item(),
        "rel_delta_mean": rel.mean().item(),
    }


class StepRecorder:
    """Collects instrumentation rows across one or more forward passes."""

    def __init__(
        self,
        lens_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        record_layers: bool = False,
        sync_cuda: bool = False,
        lens_mode: str = "tail",
        watch_ids: list[int] | None = None,
    ):
        # lens_fn maps hidden states (batch, 1, hidden) -> logits (batch, 1, vocab).
        # lens_mode "tail" first runs the layers after the span, so the lens after iteration r is
        # exactly the model's output had the loop stopped at r. "direct" skips them: cheap, but
        # mid-stack states are not in the output basis and the result is close to uniform.
        # watch_ids: token ids whose renormalised probabilities are stored per iteration
        # (the option labels, for the choice readout).
        if lens_mode not in ("tail", "direct"):
            raise ValueError(f"unknown lens_mode {lens_mode!r}")
        self.lens_fn = lens_fn
        self.lens_mode = lens_mode
        self.watch_ids = watch_ids
        self.record_layers = record_layers
        self.sync_cuda = sync_cuda
        self.rows: list[dict] = []
        self._ctx: dict = {}
        self._outputs: list[torch.Tensor] = []  # last-position span outputs, this forward
        self._prev_logp: torch.Tensor | None = None

    def now(self) -> float:
        if self.sync_cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    def begin_forward(self, **ctx) -> None:
        """Reset per-forward state; `ctx` (e.g. gen_step) is copied onto every row."""
        self._ctx = ctx
        self._outputs = []
        self._prev_logp = None

    def on_layer(self, layer_idx: int, iter_idx: int, h_in: torch.Tensor, h_out: torch.Tensor, t0: float) -> None:
        row = {"kind": "layer", "layer": layer_idx, "iter": iter_idx, "dt_s": self.now() - t0}
        row.update(span_stats(h_in, h_out))
        row.update(self._ctx)
        self.rows.append(row)

    @torch.no_grad()
    def on_span_iter(
        self, iter_idx: int, h_in: torch.Tensor, h_out: torch.Tensor, t0: float, tail_fn: Callable | None = None
    ) -> dict:
        # dt_s is taken before any lens work, so it times the span iteration alone.
        row = {"kind": "span_iter", "iter": iter_idx, "dt_s": self.now() - t0, "halted": None}
        row.update(span_stats(h_in, h_out))

        last = h_out[:, -1].float()
        row["cos_first_last"] = (
            F.cosine_similarity(self._outputs[0], last, dim=-1).mean().item() if self._outputs else 1.0
        )
        row["cos_prev2_last"] = (
            F.cosine_similarity(self._outputs[-2], last, dim=-1).mean().item() if len(self._outputs) >= 2 else None
        )
        self._outputs.append(last.clone())

        if self.lens_fn is not None:
            source = tail_fn(h_out) if self.lens_mode == "tail" and tail_fn is not None else h_out
            logits = self.lens_fn(source[:, -1:]).float()[:, -1]
            if self.watch_ids is not None:
                watch = torch.tensor(self.watch_ids, device=logits.device)
                row["lens_watch_probs"] = torch.softmax(logits[0, watch], dim=-1).tolist()
            logp = F.log_softmax(logits, dim=-1)
            p = logp.exp()
            row["lens_entropy"] = -(p * logp).sum(-1).mean().item()
            row["lens_top1"] = int(logp[0].argmax())
            row["lens_top1_prob"] = p[0].max().item()
            if self._prev_logp is not None:
                prev = self._prev_logp
                row["lens_kl_prev"] = (prev.exp() * (prev - logp)).sum(-1).mean().item()
                row["lens_top1_changed"] = bool(prev[0].argmax() != logp[0].argmax())
            else:
                row["lens_kl_prev"] = None
                row["lens_top1_changed"] = None
            self._prev_logp = logp

        row.update(self._ctx)
        self.rows.append(row)
        return row

    def on_halt_decision(self, halted: bool) -> None:
        self.rows[-1]["halted"] = halted


class ConvergenceHalt:
    """Non-learned halting rule: stop once the readout position stops moving.

    A placeholder that exercises the halting seam for V4; the learned controller
    replaces it. Halts when rel_delta_last drops below `threshold`.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, stats: dict) -> bool:
        return stats["rel_delta_last"] < self.threshold
