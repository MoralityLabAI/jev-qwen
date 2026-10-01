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
`os error 1455` (paging file too small); the queue now also waits for 10 GB of free commit
memory. At launch the card had 12.6 GB free but commit was at 1.0-1.5 GB, so the queue was
waiting on memory held by other jobs, not on the GPU.
