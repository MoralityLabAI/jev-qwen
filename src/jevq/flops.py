"""Analytic FLOP estimates for the text stack.

Matmul-dominated estimate, 2 FLOPs per multiply-accumulate. Counted: every
weight matrix in each decoder layer (once per token), the quadratic term of
full-attention layers, the Gated DeltaNet state update, and the LM head for
positions that are actually decoded. Not counted: norms, activations, the
depthwise conv, embedding lookups. Treat the numbers as comparable across
variants on the same backbone, not as absolute hardware cost.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlopModel:
    layer_matmul_params: tuple[int, ...]  # per decoder layer, weights with ndim >= 2
    layer_types: tuple[str, ...]
    attn_dim: int  # num_attention_heads * head_dim
    deltanet_state: int  # linear_num_value_heads * key_head_dim * value_head_dim
    lm_head_params: int

    @classmethod
    def from_text_model(cls, text_model, lm_head) -> "FlopModel":
        cfg = text_model.config
        params = tuple(sum(p.numel() for p in layer.parameters() if p.ndim >= 2) for layer in text_model.layers)
        return cls(
            layer_matmul_params=params,
            layer_types=tuple(cfg.layer_types),
            attn_dim=cfg.num_attention_heads * cfg.head_dim,
            deltanet_state=cfg.linear_num_value_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim,
            lm_head_params=lm_head.weight.numel(),
        )

    def layer_pass(self, layer_idx: int, n_tokens: int) -> float:
        """One application of one layer to a full sequence of `n_tokens`."""
        flops = 2.0 * self.layer_matmul_params[layer_idx] * n_tokens
        if self.layer_types[layer_idx] == "full_attention":
            # QK^T and AV over a causal context: ~2 * attn_dim * T^2 in total.
            flops += 2.0 * self.attn_dim * n_tokens * n_tokens
        else:
            # Delta-rule state decay, read, write and query: ~4 passes over the state per token.
            flops += 8.0 * self.deltanet_state * n_tokens
        return flops

    def sequence_pass(self, schedule: list[tuple[int, int]], n_tokens: int, n_decoded: int) -> float:
        """Every scheduled layer application over `n_tokens`, plus the LM head on `n_decoded` positions."""
        flops = sum(self.layer_pass(i, n_tokens) for i, _ in schedule)
        return flops + 2.0 * self.lm_head_params * n_decoded


def estimate(
    fm: FlopModel,
    schedule: list[tuple[int, int]],
    prompt_tokens: int,
    new_tokens: int,
    uncached: bool,
) -> dict:
    """FLOPs for one example.

    `flops_cached_equiv` is what an implementation with a KV / recurrent-state cache needs:
    each token passes through each scheduled layer application once. This is the number to
    compare across variants. `flops_actual` is what the code path really executed; for the
    uncached driver that re-runs the whole prefix for every emitted token.

    `new_tokens` = 0 means a single-pass readout (one decoded position, nothing emitted).
    """
    decoded = max(new_tokens, 1)
    # The last emitted token is never fed back, so a cached decode processes prompt + new - 1 inputs.
    total_inputs = prompt_tokens + max(new_tokens - 1, 0)
    cached = fm.sequence_pass(schedule, total_inputs, decoded)
    if uncached and new_tokens > 1:
        actual = sum(fm.sequence_pass(schedule, prompt_tokens + t, 1) for t in range(new_tokens))
    else:
        actual = cached
    return {"flops_cached_equiv": cached, "flops_actual": actual}
