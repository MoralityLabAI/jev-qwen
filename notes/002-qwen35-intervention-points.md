# 002 - Qwen3.5 implementation and the recurrence intervention point

Inspected: `transformers==5.17.0`, `models/qwen3_5/modeling_qwen3_5.py` and
`configuration_qwen3_5.py` (local copy in the bluebeam venv), plus the model card at
https://huggingface.co/Qwen/Qwen3.5-4B-Base. The checkpoint itself has not been downloaded, so
the 4B-specific numbers below are from the model card, not from the loaded config.

## Architecture (model card)

- Causal LM with a vision encoder; about 4B parameters; hidden size 2560; 32 layers.
- Layout: 8 x (3 x [Gated DeltaNet -> FFN] + 1 x [Gated Attention -> FFN]).
- Gated DeltaNet (linear attention): 32 value heads, 16 query/key heads, head dim 128.
- Gated Attention: 16 query heads, 4 key/value heads, head dim 256.
- FFN intermediate size 9216; vocabulary 248,320, embeddings tied to the output layer.
- "Pre-trained only": no chat template. Trained with multi-token prediction (an `mtp` head is
  in the checkpoint).

Parameter estimate for the text stack from those dimensions: about 113M per DeltaNet layer,
97M per attention layer, 0.64B embeddings, about 4.1B in total. One macro-block is about 436M.

## Code structure

- `Qwen3_5ForCausalLM`: text-only wrapper. `AutoModelForCausalLM` maps the `qwen3_5` config to
  it ("VLM compatibility") and it ignores `mtp.*` and `model.visual.*` checkpoint keys. This is
  what `jevq.modeling.load_model` uses, so the vision tower never reaches VRAM.
- `model.model` is a `Qwen3_5TextModel`: `embed_tokens`, `layers` (ModuleList of
  `Qwen3_5DecoderLayer`), `norm`, `rotary_emb`.
- `config.layer_types[i]` is `"linear_attention"` unless `(i + 1) % 4 == 0`, then
  `"full_attention"`. So macro-block k is layers `[4k, 4k + 4)`.
- `Qwen3_5DecoderLayer.forward`: pre-norm token mixer (DeltaNet or attention) with a residual,
  then pre-norm MLP with a residual. Input and output are both the residual stream, shape
  (batch, seq, hidden), so any contiguous span of layers can be fed its own output.
- The layer loop lives in `Qwen3_5TextModel.forward`. Each layer receives the mask for its
  type: a causal mask for attention, a 2D padding mask (or None) for DeltaNet.
- Caching is by layer index: attention layers append K/V to `past_key_values`, DeltaNet layers
  store a conv state and a recurrent state per `layer_idx`.

## Candidate intervention points

| Option | How | Verdict |
|--------|-----|---------|
| A. External layer-schedule driver | Re-implement the ~20-line layer loop outside the model and call the model's own layers in the order prefix -> span x n -> suffix | **Chosen** |
| B. Unrolled weight-tied stack | Replace `layers` with a longer list that repeats the span (shared modules, distinct layer indices) and edit `num_hidden_layers` / `layer_types` | Keeps stock `generate` and caching, but mutates the model object and config, makes depth static, and complicates saving and PEFT |
| C. Subclass / patch `Qwen3_5TextModel.forward` | Edit generated HF code | The file is auto-generated from a modular source and wrapped in output-capturing decorators; fragile across versions |
| D. Loop a single layer or only the token mixer | Finer-grained than a macro-block | Possible with option A by setting the span; not the default because a lone layer is not the architecture's repeating unit |

Why A:

- The vanilla model object and weights are never touched. V0 is literally the same object.
- Depth is a runtime argument, which V4 (halting) needs, and `inputs_embeds` is accepted,
  which V3 (latent feedback) needs.
- Instrumentation sits in our code, not in hooks on HF internals.
- Correctness is testable: with no loop, or `n_iters=1`, the driver must reproduce the stock
  forward. `tests/test_looped.py` asserts exact equality on a tiny random-weight Qwen3.5 for
  five spans, closeness to the cached path, and equal greedy decodes against stock `generate`.
  `scripts/check_model_load.py` repeats the comparison on the real weights.

## Proposed default span

Layers 12-15 (macro-block 3): three DeltaNet layers then one attention layer, mid-stack. The
choice of *which* block is not settled by reading code; `configs/sweeps/loop_span.yaml` sweeps
blocks 1, 3, 4 and 6 at 1-6 iterations to pick it empirically (U3 in EXPERIMENT.md).

## Semantics to keep in mind

- The loop is over depth, applied to the whole sequence each time (Universal-Transformer
  style). A DeltaNet layer inside the span re-runs its left-to-right state scan from an empty
  state on every iteration, now reading the previous iteration's output.
- Rotary position embeddings and masks are computed once and reused across iterations.
- The driver runs without any cache, so decoding costs one full forward per emitted token. See
  `003-compromises.md`.
