# SPEC-U6: training on real proposals, and the cost of the Campsite training

Registered 2026-10-08, before any U6 arm was run. Your choice that day ("2 and 3", from the U5
summary's next steps):
- **Part T.** Train on real model proposals instead of synthetic perturbations, and check whether
  the Jev's repair gain transfers.
- **Part F.** Measure what J-u4's Campsite training cost on the suites J-multi was built for.

Changes after registration are dated addenda.

## 1. Part T: training on real proposals

**Held out.** The SPEC-U5 proposal set (474 proposals by J-V0 and Bonsai-8B on 237 puzzles: 117
std, 60 8x8, 60 10x10) stays the test set. Nothing below trains on it.

**Training proposals.**
- **Proposers and prompt:** the same as SPEC-U5 (J-V0 and Bonsai-8B, greedy, three solved U4 train
  shots).
- **Puzzles:**

| Split | Puzzles | Proposals |
|---|---|---|
| realtrain | the U4 train pool (398 std) plus 100 8x8 and 100 10x10 newly planted with hermes-lite's routines (seed 20261014) | 2 x 598 = 1,196 |
| realval | the U4 val pool (79 std) plus 20 8x8 and 20 10x10 (seed 20261015) | 2 x 119 = 238 |

- **Disjointness.** Every pool is disjoint by puzzle hash from the U5 proposal puzzles, the U4 and
  U5 test sets, and the other split. This was checked before registration.

**Labels.** As in SPEC-U5:
- **Decision label:** the menu-ceiling order (commit, c_repair, dual_repair, reject), computed with
  the U5 projection rules: dual_repair capped at 60 s on 8x8 and skipped on 10x10.
- **Repair target:** the CSP gold grid.

**Arms.**

| Arm | What it is | Instances |
|---|---|---|
| J-real | each seed's J-u4 adapter (`u4_lora_s{s}`), trained further on the realtrain proposals: decide + repair rows in the zero-shot format, V1 optimiser recipe, 1 epoch, 2,392 rows = 299 steps; validation on a fixed 100-item realval subset | seeds 0-2 |
| Decision-TRM-real | the SPEC-U4 Decision-TRM recipe trained on the realtrain proposals, steps chosen on realval | seeds 0-2 |
| for comparison | the SPEC-U5 records: J-u4, c_repair, dual_repair, identity, CSP re-solve, the flow policies, the menu ceiling and the Decision-TRM (synthetic-trained) | |

**Evaluation.** J-real decides and writes repairs on the U5 proposals, and on the U4 test set to see
whether the synthetic skill survives (descriptive).

**Registered tests** (U5 proposals; pooled discordant (instance, item) pairs; exact binomial; Holm
over T1-T3 at alpha 0.05):

| ID | Claim | Measure | Test |
|---|---|---|---|
| T1 | J-real written repair vs c_repair | repair success | two-sided |
| T2 | J-real written repair > J-u4 written repair | repair success | one-sided |
| T3 | J-real decider vs Decision-TRM-real | decide success | two-sided |

**Descriptive:**
- success by band;
- the deciders' unsafe and reject rates (does training on real failures teach the Jev to reject?);
- J-real on the U4 test set.

## 2. Part F: what the Campsite training cost

J-u4 seeds 0-2 run on J-multi's suites, with the same commands as SPEC-U1:
- S1, S2, S4, S7 and S7p (`scripts/xbench.py j`);
- U2 (`scripts/xbench.py u2 j`).

Each is paired with the J-multi record of the same seed.

**Tests.**
- **F1:** S1 core4, depths 1-8.
- **F2:** S2.
- **F3:** S4.
- **F4:** S7p.
- **F5:** U2 end-to-end success with the Jev routing and gating (pipeline C).

**Decision rule.** J-u4 is non-inferior to J-multi on a suite iff the lower end of the two-sided 99%
item-bootstrap interval of the mean paired accuracy difference (J-u4 minus J-multi, averaged over
seeds) is above -0.02.
- The margin and bootstrap are SPEC-U1's.
- The level is Bonferroni over the five suites (95% would be SPEC-U1's single-test level).

## 3. Limitations, stated now

- **Real failures are not near misses.** On 8x8 and 10x10 no menu action fixes any U5 proposal.
  Training there teaches the Jev to solve or to reject, not to repair.
- **J-real starts from J-u4.** It mixes synthetic and real training; Part T measures the added real
  data, not real data alone.
- **Part F covers J-u4 only.** J-real's own cost on the suites is not measured.

## 4. Order

Queue `configs/queues/u6.yaml`. It waits for an idle GPU.
1. Part F evaluations, J-u4 seeds 0-2.
2. Proposals on realtrain/realval: J-V0, then Bonsai-8B.
3. CPU step: projections and Decision-TRM-real.
4. For each seed 0-2: train J-real, then evaluate J-real.

The report is `scripts/u6_report.py`, written before any run; it produces `reports/ubench/u6.md`.

## Addenda

(none)
