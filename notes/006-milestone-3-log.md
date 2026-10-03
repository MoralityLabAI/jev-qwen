# 006 - Milestone 3 log: V1 and V1c (LoRA)

Goal: how much of the gap to emitted reasoning comes from post-training alone (H2), whether
the adapted model's single-pass readout catches up with its own reasoning readout (H1), and
whether a calibration-aware loss keeps calibration through fine-tuning (V1c).

## 2026-10-02: design (implementation choices, tagged [IMPL])

**Data.** [IMPL] A `train` and a `val` split from the same seeded generators as the
benchmarks, difficulties 1-3 only, 100 questions per task and difficulty (2,400) plus 4 for
validation (96). Every question of the smoke and dev suites (1,000+ shots and tests) is
excluded, and train and val exclude each other. Difficulties 4 and 5 are never trained on.

**Formats.** [IMPL] Each training question is rendered in exactly one of the three readout
formats, rotating continuously so the totals balance (800 each) and every task sees every
format. The prompt is the same four-shot prefix the harness uses on the dev suite, so the
adapter is evaluated in the format it was trained in. Training on all three formats is what
makes H1 testable: it compares readouts of one adapted model, and an adapter trained only on
direct answers could suppress the reasoning readout and hand `choice` an unfair win.

**Targets.** Encoded separately from the prompt, as at inference (prompt tokens fixed, model
produces the continuation): `choice` is the one label token, `generate` is ` answer\n\n`,
`generate_cot` is ` worked solution\nA: answer\n\n`. The worked solutions are the generators'
template traces (C16). The trailing blank line was added after the probe (below): in the
few-shot blocks every answer is followed by the `\n\n` token, and with it the separately
encoded targets match the blocks' tokenization exactly, including at the prompt boundary.

**Losses.** Mean over target tokens per sequence, so every example weighs the same.
- V1: token cross-entropy over the full vocabulary for every format (plain SFT).
- V1c: identical except that on `choice` items the loss is the Brier score of the softmax over
  the valid option labels. One factor differs; `tests/test_training.py` asserts the two configs
  are otherwise equal.

Why Brier for V1c [IMPL]: restricted log loss would differ from V1's full-vocabulary
cross-entropy only by ignoring non-label tokens (which hold 4% of the mass), so it would not
test anything. Brier is a different proper scoring rule with bounded penalties: it does not
keep pushing confidence up on items already answered correctly, which is the mechanism by
which SFT usually ends up overconfident. With supervised labels this is the exact-gradient
version of "reward = proper score of the reported distribution"; RLCD's RL machinery is only
needed when labels are not available, so this is the closest analogue we can test, not a
reproduction. The base model is already calibrated (dev ECE 0.032), so V1c is judged on
whether it *keeps* calibration while matching V1's accuracy.

A side effect to watch: Brier gradients are smaller than cross-entropy gradients at the same
learning rate, so V1c may learn the choice format more slowly. If V1c is less accurate, that
confound has to be ruled out (e.g. a V1c run at a higher learning rate) before blaming the
loss.

**Adapter and optimiser.** LoRA rank 16, alpha 32, dropout 0.05 on the attention, Gated
DeltaNet and FFN projections of all 32 layers. AdamW, lr 1e-4, 20 warmup steps, cosine to
1e-5, 8 sequences per optimizer step, one epoch = 300 steps, gradient clipping at 1.0,
gradient checkpointing, bf16 base weights with fp32 adapter weights. One seed first; the plan
calls for three per trained variant.

**Evaluation.** Adapters are evaluated unmerged (bit-identical to the training forward pass)
through the same harness: dev suite (1,000 per readout) for `choice` and `generate`, the
400-example dev prefix for `generate_cot`, as for V0.

**Robustness.** Training saves a resume point every 25 optimizer steps; a crashed run started
again with the same config continues from it and reproduces the uninterrupted run (tested).
A different config refuses to resume.

**Probe on the real model** (4 optimizer steps, 32 sequences, commit fd8e0ef plus the target
fix pending; two other jobs computing on the card throughout):

- 30,474,240 trainable parameters (0.72% of 4.21B); adapter file 116 MB.
- Peak VRAM 9.06 GB (8.0 GB of it is the frozen model). No spill into shared memory.
- 19 s per optimizer step once warm (about 2.4 s per sequence); the first step took 295 s.
  A 300-step run is therefore about 1.7 h including validation, under this contention.
- Direct-answer loss was about 3 per token, which led to the target fix above: most of it was
  the lone `\n` token. Four steps say nothing about learning.

## 2026-10-02/03: first queue run failed; fixed

The M3 queue (launched 09:01) never finished `train_v1`. Attempt 1 reached step 50 (validation
choice accuracy 0.875 at that point) and died writing its second resume point:
`PermissionError: [WinError 5]` on deleting the old `last/` folder. The checkpoints lived under
OneDrive, which held a handle on the folder while it synced 366 MB of resume state. Attempts
2-5 found no usable resume state (the old folder was already emptied), restarted from step 0,
and died the same way at step 25. Nothing was trained to completion; the queue stopped at
09:59 after five attempts. My own environment note said to keep large files out of OneDrive;
the default checkpoint path ignored it.

Fixes: `paths.checkpoints` is now `~/Documents/Codex/jev-qwen/checkpoints` (outside OneDrive;
variant `adapter:` paths are relative to it), and resume state is one file written to a temp
name and swapped in with `os.replace`, with retries on transient locks. No directory is ever
deleted. Tests updated; the stale folder was removed.

Decisions taken 2026-10-03 (user: "follow your recommendations"): H1's efficiency bound is
paired latency (EXPERIMENT.md section 8); V2b keeps V1's full adapter and differs only by the
active loop (EXPERIMENT.md section 6); queues keep the machine awake while on mains power.

**Tested on the tiny model** (`tests/test_training.py`, 10 tests): data disjointness and
balance, target encoding, Brier values, LR schedule, loss falls for both losses, exact resume,
training through the loop driver (for V2b), and the harness loading the adapter with outputs
identical to the trained model.

## 2026-10-03: V1 results (adapter `v1_lora_s0`, commit c51596d)

Training: 300 optimizer steps in 68.5 min, final validation choice accuracy 0.906 (the
resume save at step 25 now works). Evaluation through the harness:

| Readout | V0 dev | V1 dev | Change |
|---------|--------|--------|--------|
| `choice` (1,000) | 0.713 | **0.880** | +16.7 |
| `generate` (1,000) | 0.667 | 0.792 | +12.5 |
| `generate_cot` (400 prefix) | 0.875 | 0.922 | +4.7 |

**H2 holds.** On the trained difficulties 1-3 (600 items) `choice` goes from 0.773 to 0.925
(+15.2 points; exact McNemar p = 2e-17), above the 10-point bound. The direct-answer readout gains
+9.8, just under it. The gain also transfers to the never-trained difficulties 4-5 (+19.0 for
`choice`), so the adapter is not only memorising the trained depths.

**H1, accuracy half: holds narrowly.** On the same 400 items V1 `choice` scores 0.877 against
`generate_cot` 0.922, a 4.5-point gap (bound: 5). The gap itself is significant (18 items only
`choice` gets right, 36 only `generate_cot`; p = 0.02) and concentrated in sequential-state tasks
(`var_trace` 0.78 vs 0.94, `arith_chain` 0.88 vs 1.00); on `graph_hops` `choice` is ahead
(0.64 vs 0.56). **Efficiency half: pending** the registered paired measurement
(`scripts/bench_h1_latency.py`, queue xb3). The unpaired medians from separate runs are 0.42 s vs
5.28 s (12.6x), which is not the registered statistic.

Calibration: V1 `choice` ECE 0.069 against V0's 0.032. Standard SFT roughly doubled the
calibration error while removing V0's over-refusal on `auth_gate` (21% to 0%). This is the
effect V1c is meant to address.
