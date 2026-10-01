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

### Exact next steps, in order

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
