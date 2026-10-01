# EXPERIMENT: Jev-style decision models on a Qwen3.5-4B-Base backbone

Status: Milestone 1 in progress. Last updated 2026-10-01.

This is an experimental reproduction *inspired by* public descriptions of Jev. Nothing here
assumes that Qwen3.5-4B matches Jev's architecture or size. Every statement below is tagged as
one of:

- **[JEV-OFFICIAL]** stated by TypeSafe in its own announcement
- **[JEV-3P]** reported by a third party (reverse engineering, press, security research)
- **[HYP]** our hypothesis
- **[IMPL]** our implementation choice

Sources and the evidence behind each Jev claim are in `notes/001-jev-public-facts.md`.

---

## 1. Research question

Can a ~4B-parameter pretrained base model be trained or adapted into a system with the useful
properties attributed to Jev-style models:

1. **fast structured inference**: a short typed answer from one (or few) forward passes instead
   of token-by-token reasoning;
2. **compact hidden computation**: extra recurrent or latent computation that replaces emitted
   reasoning tokens;
3. **strong accuracy on tasks where a conventional autoregressive LLM spends many reasoning
   tokens to produce a short answer**;

and, for each added mechanism, does the additional hidden computation produce meaningful state
evolution, or does it converge, repeat, or drift?

### Framing caveat (read before the hypotheses)

Property 1 is what Jev is publicly documented to do. Property 2 is **not** a documented Jev
property. TypeSafe says only "a new model architecture" and "a parallel sampler"; it says nothing
about recurrence, looping, or latent reasoning. One third-party reverse-engineering write-up
reports finding no evidence of iterative refinement or recurrence **[JEV-3P]**. So in this
project:

- the single-pass typed readout (`choice`) is the part that reproduces documented Jev behaviour;
- recurrence (V2), latent reasoning (V3) and learned halting (V4) are **our hypotheses** about how
  a small model might buy back the reasoning it loses by not emitting tokens. If they work, that
  is a result about Qwen3.5-4B, not evidence about how Jev works.

The research question is unchanged by this; the caveat fixes what a positive result would mean.

---

## 2. Public facts about Jev we rely on

| # | Claim | Tag | Used for |
|---|-------|-----|----------|
| F1 | Jev returns typed decisions (choice, score, yes/no probability), not text; outputs are defined in advance | JEV-OFFICIAL | `choice` readout, task design |
| F2 | "Generates all outputs in a single query"; a "parallel sampler", not sequential decoding | JEV-OFFICIAL | single-pass readout; later parallel-question variant |
| F3 | Trained with "Reinforcement Learning for Calibrated Decisions" (RLCD); answers carry calibrated probabilities | JEV-OFFICIAL | calibration metrics (ECE, Brier, NLL) in every `choice` result |
| F4 | End-to-end 70-500 ms; claimed 40x-200x faster than LLMs at similar quality on "System One" tasks | JEV-OFFICIAL (vendor-measured) | motivates compute/latency accounting; not a target we can verify |
| F5 | Output cardinality up to 255 | JEV-OFFICIAL | option-set size limits |
| F6 | Shared state encoded once; independent question branches that cannot attend to each other; latency flat up to ~100 questions | JEV-3P (behavioural tests) | few-shot block as shared prefix; future parallel-branch variant |
| F7 | No autoregressive decoding, no diffusion, no detectable recurrence or latent loop | JEV-3P (one author's inference) | the framing caveat above |
| F8 | Vulnerable to evidence-style prompt injection at rates comparable to non-reasoning LLMs; enabling reasoning in comparison models was the strongest defence observed | JEV-3P (Check Point) | control tasks; H7 |

Not relied on: any parameter count, backbone, or tokenizer lineage. Third-party sources
disagree with each other on these and TypeSafe states none of them.

---

## 3. Unknowns

About Jev (cannot be resolved from public material):

- parameter count, backbone, whether it is a dense or MoE model;
- what "new architecture" means beyond the readout;
- the RLCD objective, reward and data;
- whether any hidden iterative computation exists.

About our setup (resolved by running things):

- U1. Does vanilla Qwen3.5-4B-Base do these tasks at all few-shot, in either readout?
- U2. How far apart are `choice` and `generate` accuracy on the vanilla model?
- U3. Which macro-blocks tolerate re-application without training, if any?
- U4. Do Gated DeltaNet layers behave differently from Gated Attention layers under re-application?
  (A DeltaNet layer already carries a recurrent state along the sequence; looping it in depth
  re-runs that scan on its own output.)
- U5. Is the checkpoint's multi-token-prediction head (dropped by the text-only loader) relevant?
- U6. Real VRAM and latency on the RTX 5080 Laptop GPU (estimates in section 9).

---

## 4. Hypotheses

Each is written so that a dev-suite result can contradict it. Thresholds are in section 8.

- **H1 (readout).** On short-answer decision tasks, the single-pass `choice` readout of an
  adapted model reaches the accuracy of its own `generate` readout at a small fraction of the
  compute. *This is the documented Jev property transplanted to our backbone.*
- **H2 (adapter is a strong baseline).** LoRA on the task distribution (V1) captures most of the
  gain available from post-training. Every later variant must beat V1 at matched trainable
  parameters and training tokens, not V0.
- **H3 (untrained recurrence).** Re-applying a block of the *vanilla* model does not improve
  accuracy; hidden states either settle toward a fixed point or drift, and the answer
  distribution moves away from the vanilla one roughly monotonically with iterations.
- **H4 (trained recurrence).** With an adapter trained while the loop is active, accuracy of the
  single-pass readout on multi-step tasks improves with `n_iters`, and the gain grows with
  difficulty. Extra depth substitutes for reasoning steps.
- **H5 (latent reasoning).** k latent hidden-state updates with no emitted tokens recover most
  of the accuracy that the same model gets from emitting a reasoning trace, at lower latency.
- **H6 (halting).** A learned halting controller uses more iterations on harder examples and
  matches fixed-maximum accuracy with fewer mean iterations.
- **H7 (control).** Extra hidden computation does not raise the unsafe-allow rate on
  authorization tasks. Stronger form: trained recurrence lowers it under override attempts,
  recovering some of the robustness that explicit reasoning gives (F8).
- **H8 (distillation).** Teacher reasoning traces improve V3/V4 sample efficiency relative to
  answer-only supervision.

---

## 5. Architecture variants

One mechanism per variant. Nothing in V2-V5 is combined until its own ablation exists.

| Variant | What changes vs. vanilla | Trained? | Status |
|---------|--------------------------|----------|--------|
| **V0** baseline | nothing; unmodified HF forward/generate | no | harness ready, awaiting weights |
| **V0-driver** | same function through our layer-schedule driver (`n_iters: 1`) | no | implemented, verified on tiny model |
| **V1** adapter | LoRA / QLoRA on the task distribution | yes | planned |
| **V2a** zero-shot loop | one macro-block run `n_iters` times, vanilla weights | no | implemented (Milestone 1) |
| **V2b** trained loop | V2a + LoRA trained with the loop active | yes | planned |
| **V3** latent reasoning | k hidden-state feedback steps before the readout, no tokens emitted | yes | planned |
| **V4** learned halting | controller picks `n_iters` per input | yes | seam implemented, controller not |
| **V5** distillation | V3/V4 trained on teacher traces | yes | only if V3/V4 show signal |

Readout is an orthogonal axis: every variant is scored with both `choice` (single pass) and
`generate` (greedy decoding), so "fewer tokens" and "more hidden compute" are never confounded.

**[IMPL] V2 intervention point.** Qwen3.5-4B's text stack is 32 layers arranged as
8 x (3 Gated DeltaNet + 1 Gated Attention), each followed by an FFN. The natural recurrent unit
is one macro-block of four layers, e.g. layers 12-15. V2 runs: layers 0..start-1 once, the span
`n_iters` times feeding its output back as its input over the whole sequence, then layers
end..31 once. The loop is implemented as an external driver that calls the model's own layer
modules in a different order; the HF model object and weights are never modified, so the
baseline is always the same object. Details: `notes/002-qwen35-intervention-points.md`.

**[HYP] V3 sketch.** Continuous-thought feedback: after the prompt, append k positions whose
input embedding is the previous position's final hidden state (through a small learned
projection), then read out. The driver already accepts `inputs_embeds`, which is the seam.

**[HYP] V4 sketch.** A small head on the span output at the readout position emits a halt
probability per iteration (ACT/PonderNet-style objective). The driver already takes a
`halt_fn` and records every decision; a non-learned convergence rule exercises that path today.

**Proposed addition, not in the original plan (needs your yes/no):** a calibration-trained
readout, "V1c": V1 trained with a proper scoring rule on the option distribution. It is the
closest analogue of the documented RLCD objective (F3) and costs nothing architecturally. It is
listed here rather than silently added to the sequence.

---

## 6. Training plan

Parameter-efficient first; all hyperparameters in `configs/`.

1. **Data.** The task generators are seeded and split-aware. Train on a `train` split of the
   same generators (disjoint questions from `test`), difficulties 1-3 only. Difficulties 4-5 are
   never trained on: they measure extrapolation.
2. **V1.** LoRA rank 16 on the attention, DeltaNet and FFN projections
   (`q_proj,k_proj,v_proj,o_proj`, `in_proj_qkv,in_proj_z,out_proj`, `gate_proj,up_proj,down_proj`),
   bf16 base weights if VRAM allows, otherwise 4-bit NF4 (QLoRA) and a matching 4-bit V0.
   Loss: cross-entropy on the answer label (`choice`) plus answer tokens (`generate`).
3. **V2b.** Same LoRA budget, restricted to the looped span, trained at a fixed `n_iters`, then
   at randomly sampled `n_iters` to test depth generalisation. Equal trainable parameters and
   equal training tokens to V1.
4. **V3, V4.** Each trained from the vanilla checkpoint with its own adapter, compared with V1
   and with each other before any combination.
5. **V5.** Only if V3 or V4 beats V1 on some axis. Teacher traces from a stronger reasoning
   model; the student is trained to match answers with no emitted trace.
6. Three seeds per trained variant. Zero-shot variants are deterministic.

---

## 7. Evaluation plan

Harness: `scripts/run_eval.py` (one variant x one suite) and `scripts/sweep_loop.py`.
Suites: `smoke` (96 examples per readout, minutes) and `dev` (1000 examples per readout).

Task classes (all synthetic, seeded, with an independent ground-truth check in the tests):

| Class | Task | Difficulty knob |
|-------|------|-----------------|
| arithmetic / algorithmic | `arith_chain`, `var_trace` | number of sequential operations |
| compact logic | `order_chain`, `bool_eval` | chain length, nesting depth |
| tool call / structured action | `tool_select` | distractor tools, indirect phrasing, routing rules |
| short planning | `graph_hops` | shortest-path length |
| short answer, many steps | `relation_hops` | relation hops before the lookup |
| control (authorization) | `auth_gate` | rule complexity; override attempts from difficulty 3 |

Recorded per example: correctness, output tokens, forward passes, span iterations, layer
applications, latency, peak VRAM, estimated FLOPs (cache-equivalent and as-executed), and for
`choice` the option probabilities. Aggregated overall, per class, per task, per difficulty.
Calibration (ECE, Brier, NLL) is reported for `choice`.

Control metrics: `unsafe_allow_rate` (allowed when policy says deny), split by whether the
request carried an override note; `over_refusal_rate`; `invalid_decision_rate`.

Per-step instrumentation for schedule-driver runs (`steps.jsonl`): hidden-state norms, cosine to
the previous state, to the first iteration and to two iterations back, relative residual
change, direct logit-lens entropy and KL to the previous iteration, halting decision, wall
clock per iteration.

Comparison rules:

- V2+ are compared with V0-driver (same code path, uncached) for latency and with V0 for
  accuracy; FLOPs are compared on the cache-equivalent number.
- Trained variants are compared with V1 at matched trainable parameters and training tokens.
- "Robustness to difficulty" is the accuracy curve over difficulty, including untrained
  difficulties 4-5.

---

## 8. Falsification criteria

Evaluated on the dev suite (n = 1000 per readout; about +/-3 points at 95% for overall accuracy,
about +/-7 points per difficulty level). "Points" are percentage points of accuracy.

- **H1 is false** if, for V1, `choice` accuracy is more than 5 points below `generate` accuracy
  overall, or if matching it needs more than 25% of `generate`'s cache-equivalent FLOPs.
- **H2 is false** if V1 gains less than 10 points over V0 on trained difficulties.
- **H3 is false** if some zero-shot (span, n_iters > 1) beats V0-driver by more than 3 points on
  dev. That would be a surprising and cheap positive result and gets replicated before anything else.
- **H4 is false** if V2b at its best `n_iters` is within 3 points of V1 at matched budget on the
  multi-step classes, or if its accuracy does not increase with `n_iters` at difficulty 4-5.
- **H4's mechanism is unsupported** even if accuracy improves, when iterations after the first
  show `rel_delta_last` < 0.02 and logit-lens KL near zero: the gain then comes from the adapter,
  not from recurrence. The `n_iters: 1` ablation of the same adapter settles it.
- **H5 is false** if V3 with k latent steps is more than 5 points below the same model emitting
  a reasoning trace, or not faster.
- **H6 is false** if mean iterations do not increase with difficulty, or accuracy drops more
  than 2 points versus fixed-maximum iterations.
- **H7 is false** if any recurrent variant raises `unsafe_allow_rate` by more than 3 points over
  its non-recurrent counterpart. The stronger form is false if the with-override-note rate does
  not fall.
- **The project's main claim fails** if no variant in V2-V4 beats V1 on any of accuracy at equal
  compute, compute at equal accuracy, or difficulty extrapolation. That outcome is reported as
  the result.

---

## 9. Hardware and expected VRAM

Detected: RTX 5080 Laptop GPU, 16 GB VRAM; Ryzen AI 9 365; 31 GB RAM. Details and current
contention in `notes/000-environment.md`.

Estimates from the published config (to be replaced by `scripts/check_model_load.py` output):

| Configuration | Expected VRAM |
|---------------|---------------|
| Text-only weights, bf16 (about 4.1B params) | about 8.3 GB |
| V0 / V2a inference, bf16, prompts under 2k tokens | 9-10 GB |
| Same in 4-bit NF4 | about 4 GB |
| V1 / V2b LoRA training, bf16, seq <= 1024, batch 1-2, gradient checkpointing | 11-14 GB |
| QLoRA 4-bit training | 6-9 GB |

Looping adds no weights. One extra iteration of a 4-layer macro-block adds about 12.5% of the
stack's per-token compute; at inference it adds almost no memory, in training it adds one
macro-block of stored activations per iteration unless checkpointed.

All of this needs the GPU to be otherwise idle. bf16 training does not fit beside another job.

---

## 10. Milestones

- **M1.** Vanilla model loads; baseline suite run and saved; minimal recurrent-block prototype
  that is correct, measurable and easy to ablate. Log: `notes/004-milestone-1-log.md`.
- **M2.** V0 on dev; zero-shot loop sweep (U3, U4, H3); choose the span for V2b.
- **M3.** V1 (LoRA) with train/test splits and difficulty extrapolation.
- **M4.** V2b trained loop versus V1 at matched budget (H4).
- **M5+.** V3, V4, then V5 if warranted.
