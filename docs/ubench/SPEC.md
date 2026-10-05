# SPEC-U1: utility benchmark for a multi-label-space Jev-style adapter

Registered 2026-10-05, before any U1 training or evaluation. A follow-up to xbench (SPEC v1,
`docs/xbench/SPEC.md`), whose records and rules it reuses. Changes after registration are dated
addenda at the end.

## 1. Question and scope

xbench v1 found that post-training the 4B model on one generic decision suite (V1) costs
accuracy on suites it was not trained on, partly by installing priors over answer labels. It
also found that a loop trained on one algorithmic task adds hops, while a loop trained on the
mixed suite adds nothing. U1 asks three things:

1. Does one adapter trained on the **training splits of the utility suites themselves** (their
   label spaces and prompt formats) remove the transfer cost without losing home-suite accuracy?
2. Does it match the single-task specialist on S1?
3. Does a trained loop still add hops when the training mix is not a single task?

Scope: utility suites only (S1, S2, S3, S4, S7). The control suites S5 and S6 are deferred by
the user's decision of 2026-10-05: a good blue-team gate needs specialised synthetic data first.

## 2. Arms

New (three training seeds each: 0, 1, 2):

| Arm | Training | Evaluated |
|---|---|---|
| J-multi | V1's recipe (LoRA r16, CE loss, lr 1e-4, batch 8, one epoch) on the U1 mix, `configs/train/multi_lora.yaml` | S1, S2, S4, S7, S7p (all seeds); S3 (seed 0, zero-shot) |
| J-multi-loop | J-multi with layers 12-15 run twice in training, `configs/train/multi_loop.yaml`; evaluated at 1, 2, 3 passes (r1-r3, tail lens) | S1, S2, S4, S7, S7p (all seeds); S3 not applicable (C1: the loop driver has no KV cache) |

Comparators are reused from xbench v1 records: J-V0, J-V1 (seeds 0-2), J-V1c, J-V1-rmp (0-2),
J-V2b-rmp (0-2), J-cot-V0 and J-cot-V1, the RMP decoders, scripts, routers, TRMs, ControlTRM-LDT
and Bonsai-8B. The paired tests below also need two new comparator runs: J-V1 seeds 1 and 2 on
S2 and S7p.

## 3. Training data (as built and checked before registration)

| Part | Rows | Source | Identity check |
|---|---|---|---|
| S4 | 2,400 | jev-qwen train split, three formats | identical to V1's rows (same ids and formats) |
| S1 | 2,400 | RMP train region, 5 families x 480 | rows_sha256 55b96105... = J-V1-rmp's |
| S2 | 234 x 3 | Hermes-Skills C-sig and near-miss `train` split, minus the 8 few-shot rows | rows_sha256 1f237e0a... |
| S7 | 54 x 4 | hermes-lite registered `train_rows`, minus the 3 few-shot rows; shuffled shortlist, a fresh shuffle per copy; negatives trained to ABSTAIN | rows_sha256 f49e45f0... |

Totals: 5,718 rows, 714 optimizer steps. J-multi's data is a strict superset of V1's and of
J-V1-rmp's, so comparisons with them isolate what adding the other suites' label spaces does.
Validation (monitored only, no model selection): dev val 96, RMP val 100, S2 `val_seen` 50.

Never trained on: everything on SPEC v1's never-train list (section 9); every S3 split (S3 stays
zero-shot); S7 `held_cases`; the S2 evaluation splits.

## 4. Evaluation

S1 uses the registered items. S2 uses the holdouts. S3 uses its 24 eval tasks, zero-shot with
chunked prefill. S4 uses the dev suite (1,000 items, `choice`). For text arms S7 is measured
on **S7p** (shuffled shortlist, addendum A2 of SPEC v1), because the registered order puts
every gold at option A. Registered S7 is also run, descriptively. Records go to
`results/xbench/<suite>/<arm>[-s<seed>][-r<n>]` in the v1 schema.

## 5. Tests (fixed now)

"Pooled" means: discordant (seed, item) pairs over seeds 0-2, compared with a one-sided exact
binomial test (p = 1/2), each seed paired with the same seed of the comparator.

| ID | Claim | Test |
|---|---|---|
| T1 | Transfer to S1: J-multi > J-V1 on S1 core4 depths 1-8 | pooled one-sided |
| T2 | Transfer to S7: J-multi > J-V1 on S7p | pooled one-sided |
| T3 | S2: J-multi > J-V0 | pooled one-sided (J-V0 has no seeds; each J-multi seed is paired with it) |
| H | Home suite non-inferior: J-multi vs J-V1 on S4 | mean over seeds of the paired accuracy difference; item bootstrap (10,000 resamples, seed 0); non-inferior if the 95% lower bound > -0.02 |
| S | Specialist parity: J-multi vs J-V1-rmp on S1 core4 depths 1-8 | same bootstrap; non-inferior if the 95% lower bound > -0.02 |
| L | Loop in a mixture: J-multi-loop-r2 > J-multi on S1 core4 depths 1-8 | pooled one-sided |

T1-T3 are one family, Holm-adjusted at alpha 0.05. H and S are non-inferiority claims. L is
tested alone at alpha 0.05.

Descriptive, not tested:
- S1 depths 9-12 for the U1 arms, against LOOP-T-ds and the reasoning readouts.
- Depth-indexed readiness of J-multi-loop on pointer chasing, by seed, with the v1 item bootstrap.
- S3 utility of J-multi with and without the LDT veto, against J-V1.
- Calibration (ECE, NLL) on every suite.

**Reliability (v1 rule, section 6).** Applied to a pool of the v1 registered arms plus the U1
seed-0 arms. On S7 the pool uses S7p records for text arms and S7 records for the
order-invariant arms (lexical router, TRM router). The question is whether any Jev-style arm
becomes reliable on any utility suite, in the all-arm or the neural pool.

## 6. Order and stop rules

Queue `configs/queues/u1.yaml`:
1. The J-V1 seed 1 and 2 comparator runs.
2. Seed 0: J-multi train and evaluate, then J-multi-loop train and evaluate.
3. Seeds 1 and 2, in the same order.

The seed-0 results may be reported as interim and labelled so. Tests T1-T3, H, S and L are
decided only on all three seeds. If a run fails, it is retried by the queue; a seed that cannot
be completed is reported as missing, not replaced.

## 7. Unchanged from SPEC v1

Record schema, claim labels, statistics (Wilson, exact McNemar, Holm, percentile bootstrap) and
hygiene: other repos read-only, one GPU job at a time, artifacts outside OneDrive, no paid API
calls.

## Addenda

(none)
