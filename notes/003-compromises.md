# 003 - Compromises and deviations

Every place where the implementation is narrower than the research question. The question
itself (EXPERIMENT.md section 1) has not been changed.

| # | Compromise | Why | Effect on conclusions | Removal path |
|---|------------|-----|-----------------------|--------------|
| C1 | The schedule driver uses no KV / recurrent-state cache; decoding recomputes the full prefix per token | Re-applied layers would need a cache slot per (layer, iteration); wrong caching would silently corrupt results | Wall-clock for `generate` under the schedule driver is not comparable with stock `generate`. Compare V2 latency with V0-driver, and compute with `flops_cached_equiv` | Give each (layer, iteration) its own cache index |
| C2 | Batch size 1 everywhere | DeltaNet state and padding interact; left-padding is easy to get subtly wrong | Throughput numbers are single-stream only | Add padded batching with a test against unbatched results |
| C3 | Logit lens inside the loop is the direct lens (`lm_head(norm(h))`), skipping the suffix layers | Cheap enough to record on every step | It tracks state change, not the model's eventual answer. The answer-level effect of iterations comes from comparing runs at different `n_iters` | Optional tail-lens that runs the suffix per iteration |
| C4 | FLOPs are analytic and matmul-only | Portable and deterministic | Good for ratios between variants, not absolute cost | Profile on device |
| C5 | With halting, per-example FLOPs use the rounded mean iteration count | Halting is not a Milestone 1 feature | Slight error only when iterations vary within one example | Sum per forward |
| C6 | All tasks are synthetic and templated | Cheap, seeded, exact ground truth, controllable difficulty | Says nothing yet about natural-language decision tasks | Add an external benchmark slice in the larger suite |
| C7 | `choice` scores label letters (A-H) after "Answer:" rather than scoring option text | One forward pass with a pre-enumerated output space, matching the documented Jev readout | Sensitive to label-position bias in a base model; options are shuffled per example and few-shot answers vary | Add option-text likelihood scoring as a second single-pass-per-option readout |
| C8 | Few-shot prompting, because the backbone is a base model without a chat template | The instruction is to use Qwen3.5-4B-Base | V0 accuracy reflects in-context learning ability, not instruction following | None needed; V1 measures the adapted model |
| C9 | The checkpoint's MTP head and vision tower are dropped at load | Text-only experiment; saves VRAM | Unknown whether MTP matters for "fewer decoding steps" (U5) | Load the full model class and inspect |
| C10 | Halting is a non-learned convergence rule | Only there to exercise the seam and the logging | No claim about V4 can be made from it | The learned controller (V4) |
| C11 | Adapter variants are rejected by the harness (`NotImplementedError`) | V1 training is the next milestone | None yet | Wire PEFT loading into `make_variant` |
| C12 | (Resolved 2026-10-01.) Tests first ran in another project's venv (`.venvs\bluebeam`), read-only | Creating this project's venv needed download approval | None: all 37 tests now pass in `.venvs\jev-qwen` with the same versions | Done |
| C13 | Tests use a character-level fake tokenizer and a tiny random model | No downloads; exercises the real HF layer code | Tokenizer-specific behaviour (single-token labels, newline stop) is verified only when the real tokenizer loads; `label_token_ids` fails loudly if a label is not one token | First real run |
