# 005 - Milestone 2 log

Goal: V0 on the dev suite in all three readouts; zero-shot loop sweep; choose the span for
trained recurrence. Decisions taken at the start (2026-10-01): add the reasoning-trace readout
before the dev baseline (approved), and V1c is an approved variant.

## 2026-10-01

### Added before running anything

- `generate_cot` readout; every task generator now supplies a one-line worked solution. The
  smoke examples are unchanged (fingerprint pinned in `tests/test_tasks.py`).
- Tail lens (C3), paired latency benchmark (C14), resumable job queue that waits for VRAM and
  for commit memory. 49 tests pass.

### Probe: reasoning traces on smoke (run `20261001T071055Z_v0_baseline_smoke`, commit f9fae0d)

Same 96 examples as the Milestone 1 baseline. Vanilla model, stock driver.

| Readout | Accuracy | By difficulty 1 / 2 / 3 | Tokens emitted | Forward passes | Est. FLOPs (cache-equiv.) |
|---------|----------|-------------------------|----------------|----------------|---------------------------|
| `choice` | 76/96 = 0.792 | .81 / .81 / .75 | 0 | 1 | 2.6e12 |
| `generate` | 76/96 = 0.792 | .91 / .84 / .63 | 2.8 | 2.8 | 1.9e12 |
| `generate_cot` | 89/96 = 0.927 | 1.0 / .94 / .84 | 31.7 | 31.7 | 3.0e12 |

- One line of emitted reasoning is worth 13.5 points to the vanilla model on smoke
  (13 examples; the 95% interval on a 96-example difference is wide, roughly +/-9). By task the
  gain is concentrated where steps chain: `arith_chain` .67 -> 1.0, `var_trace` .58 -> .92,
  `relation_hops` .67 -> 1.0. `graph_hops` stays at .50: the model writes the breadth-first
  layers in the right format but gets them wrong, six of the seven errors.
- The model follows the trace format without any tuning; no run hit the token budget.
- **This gap is the target.** A hidden-computation variant succeeds to the extent that its
  single-pass `choice` accuracy moves from 0.79 toward 0.93 without emitting the trace.

### Finding that affects a falsification criterion (decision needed, not yet changed)

H1's efficiency bound is written in FLOPs ("`choice` must need at most 25% of
`generate_cot`'s cache-equivalent FLOPs"). Measured, `choice` costs 88% of `generate_cot`'s
FLOPs here, and no readout could get near 25%:

- the few-shot prompt is 375 tokens on average and the trace is 32, so decoding is about 8%
  of the tokens processed;
- a generated token costs the same FLOPs as a prompt token. What decoding costs is
  *sequential* forward passes: 32 of them against 1. Mean latency was 7.5 s against 0.28 s
  (separate runs on a shared GPU, so the ratio is rough).

So FLOPs cannot express the advantage the hypothesis is about; the documented Jev claim is
about latency. Proposed restatement, awaiting a yes/no: "H1 is false if `choice` is more than
5 points below `generate_cot`, or if it is not at least 5x faster in paired latency." Until
decided, EXPERIMENT.md keeps the original wording with this issue noted next to it.

### Queue

`configs/queues/m2.yaml`, launched detached at 18:25 (survives session ends; progress in
`results/queue/m2/status.json`, output in `results/queue/m2.out`). Steps: paired latency;
V0 on dev (`choice`, `generate`, 1000 examples each); loop sweep over four macro-blocks at
1-6 iterations; V0 reasoning traces on dev.

The sweep and the reasoning traces run on the first 10 examples per (task, difficulty), 400
examples, a prefix of the dev set (C17). The first attempt at the probe died with
`os error 1455` (paging file too small). A 10 GB commit floor then stalled the queue (commit
sat at 1-3 GB because the page file only grows on demand), so the floor was lowered to 2 GB
with more retries, and the queue was relaunched at about 18:40.

## 2026-10-02: Milestone 2 results

All seven queue steps finished between 22:03 and 07:03 (commit 0f6b635 for every run).
`dev_v0` failed once on `os error 1455` and passed on the retry.

**The dev run's wall-clock is not meaningful.** It took 7.5 h because the laptop went to
sleep during one example (`var_trace-test-*`, `generate`), which logged 23,840 s. Accuracy is
unaffected. The mean latency in that record is contaminated; the medians are not. Latency also
varied 0.3-2.1 s per example across the run as other jobs came and went, so only the paired
benchmark below says anything about speed.

The 400-example subset used by the sweep and the trace run was checked against the full dev
run: all 400 ids have identical answers, options and worked solutions.

### V0 on dev (run `20261001T141159Z_v0_baseline_dev`, 1000 examples per readout)

| Readout | Accuracy | d1 | d2 | d3 | d4 | d5 |
|---------|----------|----|----|----|----|----|
| `choice` | 0.713 | .88 | .75 | .70 | .67 | .58 |
| `generate` | 0.667 | .91 | .78 | .68 | .53 | .45 |

- `choice` is well calibrated out of the box: ECE 0.032, mean confidence 0.70 against accuracy
  0.71, 96% of next-token mass on valid labels. (Jev's third-party ECE was 0.031 on an MMLU
  sample; different tasks, so this is not a comparison, but it means V1c's job is to keep
  calibration through fine-tuning, not to create it.)
- `choice` degrades more gracefully with difficulty than `generate` (d5: .58 vs .45). The
  direct-answer readout collapses on long chains: `arith_chain` d5 0.00, `graph_hops` d4-d5
  .04/.00, `relation_hops` d5 .12.
- `choice` has a position bias: it picks letter A on 35% of four-option items against a 26%
  base rate, and over-refuses on `auth_gate` (21% of ALLOW cases answered DENY, against 3% for
  `generate`). No unsafe allows in either readout.

### Same 400 examples, three readouts (the H1 / research-question comparison)

| Readout | Accuracy | d1 | d2 | d3 | d4 | d5 | Tokens emitted | Median latency* |
|---------|----------|----|----|----|----|----|----------------|-----------------|
| `choice` | 0.725 | .88 | .80 | .71 | .62 | .61 | 0 | 0.57 s |
| `generate` | 0.655 | .90 | .80 | .65 | .50 | .42 | 2.8 | 0.73 s |
| `generate_cot` | **0.875** | .97 | .97 | .91 | .80 | .71 | 41.9 | 4.73 s |

\* Different runs at different times on a shared card; the 8x ratio is indicative only.

- Emitted reasoning is worth 15 points over single-pass `choice` on the same examples.
  Paired: 83 examples only `generate_cot` gets right, 23 only `choice` gets right (McNemar,
  p far below 0.001). This gap is real and is the target for V2-V4.
- Where the gap is: `arith_chain` .62 -> 1.00, `var_trace` .62 -> .94, `relation_hops`
  .50 -> .94. Where it is not: `bool_eval` .74 -> .78, `graph_hops` .54 -> .46 and
  `order_chain` 1.00 -> .88 (reasoning hurts on these two). So the benefit of emitted reasoning
  is task-specific: large for sequential state tracking, absent for graph search.
- 1.25% of traces hit the 192-token budget.

### Zero-shot loop sweep (choice readout, 400 examples, tail lens)

Accuracy by iteration count (`n_iters=1` is the vanilla model; all four spans agree on it, as
they must):

| Span | 1 | 2 | 3 | 4 | 5 | 6 | Prediction changed vs vanilla at 6 |
|------|---|---|---|---|---|---|-------------------------------------|
| 4-7 | .725 | .667 | .623 | .578 | .458 | .378 | 64% |
| 12-15 | .725 | .725 | .690 | .662 | .675 | .623 | 41% |
| 16-19 | .725 | .708 | .685 | .680 | .665 | .627 | 31% |
| 24-27 | .725 | .715 | .713 | .698 | .698 | .682 | 17% |

**H3 holds.** No span beats vanilla at any iteration count; the best case is a tie (12-15 at
two iterations). The damage is ordered by depth: re-running an early block is catastrophic, a
late block is nearly harmless and also nearly inert.

Task-level blips at 12-15 (`relation_hops` .50 -> .60, `auth_gate` .84 -> .96, d4 .62 -> .70,
all at two or three iterations) are within the noise for 50 examples per task (about +/-14
points) and are not claimed.

**What the extra computation does to the state.** It neither settles nor oscillates. After
the second pass, every span adds an almost constant vector on each pass:

| Span | Absolute step per pass, passes 3-6 (approx.) | Norm at pass 1 -> 6 | cos to previous state at 6 | cos to state 2 passes back at 6 |
|------|----------------------------------------------|---------------------|-----------------------------|---------------------------------|
| 4-7 | 3.1-3.3 | 7.4 -> 20.7 | .994 | .978 |
| 12-15 | 3.3-3.5 | 10.5 -> 20.7 | .990 | .960 |
| 16-19 | 8.2-9.1 | 17.3 -> 50.4 | .994 | .971 |
| 24-27 | 13.1-15.8 | 35.9 -> 86.8 | .991 | .962 |

(Step = relative change x previous norm, from per-iteration means.) Relative change falls to
about 0.2 per pass only because the norm keeps growing; cosine to the previous state
approaches 0.99 while cosine to the first pass falls steadily (12-15: 1.00 -> .74). The block
re-applies the same push in a nearly fixed direction: drift, not convergence and not new
computation. Lens entropy rises (12-15: 0.90 -> 1.52 nats), so the model becomes less
certain as the state drifts.

**At the answer level the drift mostly turns into option-letter bias.** Share of predictions
on the most-chosen letter for four-option items, pass 1 -> 6: span 4-7 .35 -> .60 (letter D
falls from 48 predictions to 1), span 16-19 .35 -> .55 (towards C), span 12-15 .35 -> .35
(stable), span 24-27 .35 -> .35. On two-option items, span 4-7 picks A 96% of the time at
six passes.

**Control (H7 warning, small n).** Unsafe-allow rate on the 22 DENY cases, pass 1 -> 6:
span 4-7 0 -> 73% (16/22), span 12-15 0 -> 27%, spans 16-19 and 24-27 stay at 0-5%. The
rate is similar with and without the override note (span 4-7 at six passes: 10/14 with,
6/8 without), and for span 4-7 it coincides with the collapse onto letter A. So untrained
recurrence does corrupt authorization decisions, but by scrambling the readout, not by making
the model more persuadable. H7 is stated for trained variants; this is a baseline it has to
beat, and a reason to track position bias in every recurrent run.

### Paired latency (`results/latency/20261001T140802Z_choice.json`)

Choice readout, 24 examples x 5 rounds, variants interleaved on the same example; two other
processes were on the GPU throughout.

| Variant | Median | Ratio to V0 through the driver (p10-p90) | Layer applications |
|---------|--------|------------------------------------------|--------------------|
| V0 through the driver | 578 ms | 1 | 32 |
| V0 stock | 584 ms | 1.006 (0.95-1.07) | 32 |
| Loop 12-15 x2 | 653 ms | 1.122 (1.07-1.16) | 36 |

Measured cost of one extra pass over a macro-block matches the layer count (36/32 = 1.125).
The driver adds no overhead over the stock forward. Absolute times are slow because the
DeltaNet layers use the reference PyTorch kernels (no `flash-linear-attention`).

### Decisions taken

- **Span for V2b: layers 12-15** (unchanged default, now with evidence). It is the only span
  that costs nothing at two passes, it keeps the letter distribution balanced (its
  perturbation is not just position bias), it moves the state substantially, and it leaves 16
  layers to read the state out. Span 24-27 is safer but changes so little that training would
  have little room to use it. V2b starts at two passes.
- Position bias becomes a tracked metric for every `choice` run (share of the most-chosen
  letter), because it is the main failure mode of untrained loops and would also mask or fake
  calibration gains.

### Open

- H1 efficiency metric (FLOPs vs paired latency): awaiting a decision; see above.
