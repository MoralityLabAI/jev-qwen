# SPEC-U7: a compact looped reasoner on Campsite, and one Jev for everything

Registered 2026-11-07, before any U7 arm was trained. Your request that day: "we have more RAM and
free GPU make a plan to finish this thread maybe we can learn something about compact reasoning
looped transformers". Changes after registration are dated addenda.

U7 closes the jev-qwen thread with two parts:
- **Part L** asks what a compact looped transformer, trained from scratch, does on the task where
  the 4B Jev was finally tested best: verifier-scored Campsite repair. It tests whether looping
  buys test-time compute and size extrapolation.
- **Part M** trains one Jev on everything, to see whether U6's specialisation costs go away.

## 0. Looked at before registration

These are design probes and checks only; no part-L arm was trained and no Jev was run.
- **Puzzle supply.** The generator yields 1,500 distinct standard puzzles in 4 s; 91% of 400 have a
  single solution.
- **The large set (L7)** builds in 97 s.
  - The hermes-lite tree-first planter did not finish 26 12x12 puzzles in 15 min.
  - The tents-first planter below does.
  - The CSP solver needs at most 0.15 s at 14x14 (3 puzzles).
- **CPU unit tests of the model code** (`tests/test_u7.py`): buckets, encoding, shapes, parameter
  matching, loss, and a 4-example fit.

## 1. Part L: compact looped transformer (`src/jevq/xbench/u7_loop.py`)

**Task.** Repair a Campsite candidate: output tent or no tent for every non-tree cell. Inputs per
cell:
- whether the puzzle has a tree there;
- the candidate's symbol (T, X, C, or none for a missing/ragged cell);
- the row and column tent targets.

"Solve" is the same task from the zero-tent copy.

**Model.**
- Each cell is a token: d = 128, 4 heads, FFN 512, pre-LN.
- Attention adds a learned per-head bias for the relation between two cells (self, one of the 8
  neighbours, same row, same column, other). There are no absolute positions, so one set of weights
  runs on any grid size.
- The readout is one shared linear head.

**Arms** (same width, data, order, steps and per-block readout):

| Arm | Structure | Parameters |
|---|---|---|
| LOOP | one 2-layer block, weight-tied, iterated T times with the input re-added each iteration; supervised after every iteration; T_train = 16 | 402,018 |
| DEEP | 16 untied 2-layer blocks, supervised after every block (compute-matched to LOOP at T = 16) | 6,351,618 |
| FLAT | one 2-layer block applied once (parameter-matched to LOOP) | 402,018 |

Seeds 0-2 for each arm.

**Loss.** Cross-entropy on non-tree cells, taking the minimum over up to 4 solutions of the puzzle
(puzzles with several solutions are not penalised for choosing another), averaged over supervised
steps.

**Optimiser.** AdamW lr 1e-3, warmup 500, cosine to 10%, weight decay 0.01, clip 1.0, batch 128,
20,000 steps, bf16 autocast.

**Training data** (built once by the CPU step):
- 8,000 fresh standard puzzles from hermes-lite's generator (seed 20261107, 1,600 of them 6x6). Any
  puzzle whose hash is in a U-series test pool is dropped: U4 test, U5 test, U5 proposal puzzles, U5
  large.
- Each puzzle gives nine candidates: the eight SPEC-U4 types and the zero-tent copy.
- Also the standard-size SPEC-U6 realtrain proposals, upsampled x10.
- **Validation:** the U4 val pool items and the standard-size realval proposals (reported, not used
  for selection).
- **Only grids up to 6x6 are trained on.**

**Evaluation sets** (official verifier on the decoded grid):

| Set | Items | Content |
|---|---|---|
| u4test | 480 | SPEC-U4 synthetic, standard sizes |
| u5test | 480 | SPEC-U5 synthetic, standard sizes |
| proposals | 474 | SPEC-U5 real J-V0/Bonsai proposals: 117 std, 60 8x8, 60 10x10 puzzles, two proposers |
| l7 | 480 | new large synthetic set: the eight U4 types x 15 items at each of 8x8, 10x10, 12x12, 14x14 |

L7 details:
- 8x8 and 10x10 use hermes-lite's planter (seed 20261108).
- 12x12 and 14x14 use a tents-first planter: non-touching tents at random, then one tree on a free
  side of each, 20% density, each puzzle checked with the verifier. The planted grid is the gold.
- Hashes of held-out pools are excluded.
- On l7: identity, c_repair (capped at 60 s per call from 12x12), dual_repair (capped at 60 s at
  8x8, skipped above) and CSP re-solve.

**Evaluation iterations.** LOOP at T in {1, 2, 4, 8, 16, 32, 64, 128}; DEEP with its 16 blocks;
FLAT once.

**Registered tests** (pooled discordant (instance, item) pairs, same seed paired; exact binomial;
Holm over L1-L4):

| ID | Claim | Set | Test |
|---|---|---|---|
| L1 | LOOP (T 16) vs J-u4 (written repair, SPEC-U5) | u4test | two-sided |
| L2 | LOOP (T 16) vs J-real (written repair, SPEC-U6) | proposals | two-sided |
| L3 | LOOP (T 64) > DEEP | l7 | one-sided |
| L4 | LOOP (T 64) > LOOP (T 16): test-time iterations help past the trained depth | l7 | one-sided |

**Descriptive:**
- every arm by size and type;
- LOOP's accuracy against T at each size;
- the first iteration at which the decoded grid passes;
- parameters and training time.

## 2. Part M: one Jev for everything (J-mix)

**Training.** J-mix is trained from the base model with J-multi's recipe on the union of every
training source so far (`configs/train/mix_lora.yaml`):
- V1's decision questions;
- the RMP rows;
- S2 (x3) and S7 (x4);
- the U4 decide/repair items (SPEC-U5);
- the realtrain decide/repair proposals (SPEC-U6).

That gives 14,128 rows, 1,766 steps, seeds 0-2.

**Evaluation.**
- J-multi's suites: S1, S2, S4, S7, S7p and U2.
- The U4 test set and the U5 proposals: decide and written repair, zero-shot format.

**Registered tests:**
- **M1-M5:** J-mix is non-inferior to J-multi on S1 (core4, depths 1-8), S2, S4, S7p and U2
  (pipeline C). The rule is SPEC-U6 part F's: two-sided 99% item-bootstrap interval of the mean
  paired difference above -0.02, Bonferroni over the five.
- **M6:** J-mix vs J-real, written repair on the proposals (two-sided).
- **M7:** J-mix vs J-u4, written repair on the U4 test set (two-sided).
- Holm over M6-M7.

## 3. Limitations, stated now

- **Synthetic training mix.** LOOP/DEEP/FLAT see far more synthetic data than J-u4 did (about 72k
  items against 3,009). L1 compares a model trained for the task from scratch with a 4B model
  adapted on less data.
- **Generators differ by size.** L7 above 10x10 comes from a different planter than everything
  else. A size effect there can partly be a generator effect; 8x8 and 10x10 share hermes-lite's.
- **The solver stays cheap.** Re-solving remains trivially cheap at every size; part L is about
  what the looped model learns, not about need.
- **Part M competes for the GPU and RAM.** Part M is the heavy part (about 3.5 h of training per
  seed). It runs after part L and may be delayed by other sessions' jobs.

## 4. Order and resources

**Shared machine.** The GPU is shared and leased through vramorama: one job of mine at a time,
chained only after the previous one ends.
1. **CPU step,** detached: training data, the L7 set and its comparators (`xbench.py u7 cpu`).
2. **GPU job 1** (`vramorama run --vram 3G`): queue `configs/queues/u7l.yaml`. Nine training runs,
   resumable from checkpoints every 1,000 steps, then their evaluations.
3. **GPU job 2** (`vramorama run --vram 11G`), after job 1: queue `configs/queues/u7m.yaml`. J-mix
   seeds 0-2, each trained and then evaluated.

**Report.** `scripts/u7_report.py`, written before any run, produces `reports/ubench/u7.md`.

## Addenda

**U7-A1 (2026-11-07, order only; no U7 arm had run).** SPEC-U8 (small Jevs) was registered the same
day and runs between part L and part M. Part M's design is unchanged.
