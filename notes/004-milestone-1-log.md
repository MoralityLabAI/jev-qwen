# 004 - Milestone 1 log

Goal: Qwen3.5-4B-Base loads; baseline suite run and saved; minimal recurrent-block prototype
that is correct, measurable and easy to ablate.

## 2026-10-01

### Done

- Environment inspected (`000-environment.md`).
- Public Jev material collected and tiered (`001-jev-public-facts.md`). Finding that shapes
  the design: recurrence and latent reasoning are not documented Jev properties.
- Qwen3.5 implementation read in transformers 5.17.0; intervention point chosen
  (`002-qwen35-intervention-points.md`).
- Recurrent-block prototype: `src/jevq/looped.py`. Configurable span and iteration count,
  optional halting callback, no model or weight modification.
- Instrumentation: `src/jevq/instrument.py` (per-iteration norms, cosines, relative change,
  logit-lens entropy and KL, halting decision, timing).
- Evaluation harness, eight task generators in six classes, smoke and dev suites, run records.
- Tests: 35 passing on CPU with a tiny random-weight Qwen3.5 (same HF layer code as the 4B
  model). What they establish:
  - driver with no loop == stock forward, bit-exact; `n_iters=1` == stock for five spans;
  - driver output is within 1e-4 of the stock cached path;
  - uncached greedy decode == stock `generate` (greedy, cached);
  - `n_iters=3` over a 4-layer span executes 8 + 2*4 layer applications and changes the output;
  - recording does not change results; halting stops when told to and logs the decision;
  - every task's ground truth is recomputed independently (expression eval, BFS, chain
    following, policy re-derivation); generators are deterministic per seed;
  - a harness run writes a complete record; identity-schedule predictions equal stock ones.

### Not done, and why

- **Model not loaded, baseline not run.** Two blockers, neither worked around:
  1. The weights are not on disk (about 10 GB from Hugging Face) and this project has no venv
     (about 3 GB of wheels). Downloads need explicit approval.
  2. The GPU was occupied by other jobs for the whole session (14-16 GB of 16 GB in use).
- So nothing about the real model is measured yet: load status, VRAM, latency, baseline
  accuracy, tokenizer label check, and driver equivalence on real weights are all pending.
  The VRAM figures in EXPERIMENT.md are estimates from the published config.

### Later the same day (downloads approved)

- Project venv created at `%USERPROFILE%\.venvs\jev-qwen`; all 37 tests pass in it.
- Checkpoint downloaded: 9.33 GB, Hub commit `1001bb4d826a52d1f399e183466143f4da7b741b`, now
  pinned in `configs/base.yaml`. The download needed the OS trust store (antivirus TLS
  scanning); see `000-environment.md`.
- Real tokenizer checked on CPU: option labels ` A`..` H` are single tokens; `encode` adds no
  special tokens; the newline stop rule fires on ` 12\n`; longest smoke prompt is 681 tokens;
  answers are at most 4 tokens. The tokenizer ships a chat template even though the model card
  calls the model pretrained-only; we do not use it.
- Loader now streams weights to the device and caps the process at 70% of VRAM, so on the
  shared card an overflow is an OOM in our process rather than a slowdown for every job.
- `scripts/run_milestone1.py` does steps 2-5 below in one process with one model load. It is
  queued behind `scripts/wait_for_gpu.py` (needs 11 GB free for 90 s). The GPU had 2.5-5 GB
  free whenever checked, so **the model is still not loaded and nothing below is measured.**

### Milestone 1 result (12:24-12:31 local, `results/milestone1_summary.json`)

A first attempt at 12:23 died three seconds after start with Windows fault `0xC000070A` in
`ntdll.dll`, before any Python output. The Application log shows the same fault killing other
projects' Python jobs four times in the preceding hour, with 1 GB of RAM free. The immediate
retry ran cleanly; nothing was changed in between.

**Load.** `Qwen3_5ForCausalLM`, bf16, 4,205,751,296 parameters, 26.9 s, 8,022 MB VRAM after
load, 8,240 MB peak across all runs. (Estimate was 8.3 GB of weights and 9-10 GB peak.)

**Driver on real weights.** Max absolute logit difference against the stock forward: 0.0 with no
loop and 0.0 with a one-iteration loop over layers 12-15. Two iterations move the logits by up
to 6.6 (largest logit 25.6). On the smoke suite the identity-driver run reproduced all 192
stock predictions and identical option probabilities.

**Smoke suite, 96 examples per readout (one example is about 1 point; 95% interval about +/-8).**

| Variant | choice acc | generate acc | choice ECE | layer applications (choice) |
|---------|-----------|--------------|------------|-----------------------------|
| V0 stock | 76/96 = 0.792 | 76/96 = 0.792 | 0.102 | 32 |
| V0 through the driver | 0.792 | 0.792 | 0.102 | 32 |
| V2a loop 12-15 x2, untrained | 73/96 = 0.760 | 75/96 = 0.781 | 0.071 | 36 |

V0 by task (choice / generate): arith_chain .67/.67, var_trace .58/.67, order_chain 1.0/.92,
bool_eval .83/.92, tool_select 1.0/1.0, graph_hops .67/.50, relation_hops .67/.67,
auth_gate .92/1.0. By difficulty 1/2/3: choice .81/.81/.75, generate .91/.84/.63.

Control (12 examples): no unsafe allow in any variant or readout; one over-refusal in `choice`
for V0 and V2a alike. Too few examples to say anything about H7.

**What the second iteration does to the state (choice readout, 96 examples).**

| | first pass | second pass |
|---|-----------|-------------|
| relative change at the readout position | 0.72 | 0.41 |
| cosine to the state before the pass | 0.83 | 0.955 |
| hidden-state norm at the readout position | 10.5 | 12.9 |

So the re-applied block still moves the state a lot (not a fixed point after one extra pass) and
inflates its norm by about 24%. At the answer level: 10 of 96 `choice` predictions changed
(3 wrong->right, 6 right->wrong), mean total-variation shift in option probabilities 0.069.
Direction matches H3 (untrained recurrence does not help) but the difference is 3 examples and
is not significant. More iterations and other spans are the M2 sweep.

**Things this run showed about the setup itself.**

- U1 is answered: the vanilla base model does these tasks few-shot in both readouts; no prompt
  rework needed before comparing variants.
- The direct logit lens at layer 16 is uninformative: entropy 11.3 nats against a maximum of
  12.4. Compromise C3 has to be fixed (tail lens) before lens numbers mean anything.
- Latency is not comparable across these runs. The same first-pass computation took 41 ms per
  span in the identity run and 66 ms in the V2 run a few minutes later, because other jobs came
  back onto the GPU. See C14.
- `generate` emitted 2.75 tokens on average, and `choice` cost *more* estimated FLOPs than
  `generate` (2.6e12 vs 1.9e12) because listing the options lengthens the prompt. There is no
  reasoning-trace arm yet, so the comparison the research question cares about, hidden
  computation against emitted reasoning tokens, is not in the harness. See C15.
- `revision_resolved` is null in records when loading offline; the requested (pinned) revision
  is recorded.

### Exact next steps, in order (as planned before the run; steps 1-5 are now done)

1. `scripts\setup_env.ps1` (once approved), then `pytest tests -q` in the new venv.
2. With the GPU idle: `python scripts\check_model_load.py`. Pass criteria: no missing keys,
   `driver_matches_stock: True`, max logit difference at numerical-noise level. Pin
   `model.revision` in `configs/base.yaml` to the reported commit.
3. `python scripts\run_eval.py --variant configs\variants\v0_baseline.yaml --suite evals\suites\smoke.yaml`
4. Same with `v0_driver_identity.yaml`; predictions must match step 3.
5. Same with `v2_loop_mid_r2.yaml`: first real recurrent run, with `steps.jsonl`.
6. Then M2: V0 on `dev`, and `scripts\sweep_loop.py`.

If V0 few-shot accuracy on smoke is near chance in both readouts, fix the prompt format before
any variant comparison (U1 in EXPERIMENT.md).
