# SPEC-U5: reject paths, the Jev trained on the U4 items, and real proposals on larger puzzles

Registered 2026-10-07, before any U5 arm was run. On 2026-10-07 you chose ("1, 2, 3") all three next
steps from the U4 summary. Changes after registration are dated addenda.

1. **Part R.** Add a reject path to the infused skill's Campsite flow policy.
2. **Part J.** Train the Jev on the same U4 train items the Decision-TRM saw, so the U4 comparison
   is like-for-like.
3. **Part P.** Test on real model-proposed candidates, and on puzzles large enough that re-solving
   is not free.

U5 reuses SPEC-U4's verifier, generator, repair modules, prompts and Decision-TRM recipe
(`src/jevq/xbench/u4_campsite.py`); the new code is in `src/jevq/xbench/u5_campsite.py`.

## 0. Design probes run before registration

Run on throwaway seeds, with no U5 item and no arm.

**Cost by puzzle size** (hermes-lite planting and CSP routines, one tent dropped from the solution):

| Size | CSP solve | c_repair | dual_repair |
|---|---|---|---|
| 8x8 | <= 22 nodes, ~0 s | <= 0.05 s | up to 38 s |
| 10x10 | <= 294 nodes, < 5 ms | ~0.5 s | > 60 s |
| 12x12 | <= 399 nodes, ~10 ms | 15-78 s | > 60 s |
| 14x14 / 16x16 | not reached: planting alone > 60 s (6 of 6 attempts) | | |

**Consequences for Part P.**
- With these tools, re-solving never becomes the expensive step. The projections become impractical
  first, and the generator stops before the solver slows.
- Part P therefore uses 8x8 and 10x10 as the large band, and reports the costs.
- dual_repair runs with a 60 s cap per call at 8x8 and is not run at 10x10. A timeout or skip counts
  as "no projection": the original candidate stands.

## 1. Item sets

| Set | What | Items |
|---|---|---|
| u4test | the SPEC-U4 test set, unchanged (`02e802f76cdaabf2`) | 480 |
| u5test | fresh synthetic set, SPEC-U4 construction (8 types x 60); generator seed 20261011, 110 puzzles (22 of them 6x6), 107 after dropping overlaps; sha256 prefix `05e8f23e50a6feb8` | 480 |
| proposals | one greedy proposal per (proposer, puzzle) over the proposal puzzles (below), each scored as it stands; the proposal is the candidate, an unparseable one is an empty grid | 2 x 237 |

**Proposal puzzles:**
- **std:** 117 puzzles from hermes-lite's generator (seed 20261012, 120 generated, 24 of them 6x6,
  3 dropped as overlaps);
- **large:** 60 8x8 and 60 10x10 puzzles from hermes-lite's own planting and CSP routines (seed
  20261013, 20% trees, as its generator).

**Proposers:**
- **J-V0**, the generic Qwen3.5-4B-Base, greedy, through the jev-qwen generation runner;
- **Bonsai-8B**, greedy raw completion on the pinned llama-server.

Both get the same prompt: the Campsite rules, three solved U4 train puzzles, and the puzzle. The
token budget is 2 x cells + 16.

All pools are disjoint by puzzle hash, checked against every SPEC-U4 pool. No set is used for
training except the U4 train and val pools.

## 2. Part R: reject paths

**Policies.** Both Hermes-Skills flow policies (`c_repair_if_c_fail`, `dual_repair_if_any_sig_fail`),
each in three modes:
- **plain:** as in SPEC-U4.
- **noop-reject:** reject when the grid about to be committed is the unchanged candidate and the
  verifier has already failed it. This covers a commit of a failing candidate, a projection that
  returns nothing (or times out or is skipped), and a projection that changes nothing. It needs no
  extra verifier call.
- **verify-reject:** commit only a grid that passes the verifier, at the cost of one verifier call
  after the repair.

**Properties by construction.** Neither mode can lower success; verify-reject has zero unsafe
commits; noop-reject has no more unsafe commits than plain. These are checked by tests, not
statistics. The empirical question is how many unsafe commits noop-reject leaves, on synthetic and
on real candidates.

**Also reported:**
- the Decision-TRM, retrained with the SPEC-U4 recipe on U4 train (seeds 0-2) and applied unchanged;
- the menu ceiling;
- CSP re-solve, with nodes and seconds.

**Deliverable.** A patch for Hermes-Skills' flow-policy sweep, adding the noop-reject and
verify-reject variants. It goes under `patches/hermes-skills/`; I do not apply it (that repo is
read-only for me).

## 3. Part J: J-u4, the Jev trained on the U4 train items

**Training** (`configs/train/u4_lora.yaml`):
- **Start:** each seed's J-multi adapter (seed s from `multi_lora_s{s}`). It is trained further, with
  the J-multi LoRA shape and V1 optimiser recipe: lr 1e-4, warmup 20, 8 sequences per step, 1 epoch.
- **Data:** the 3,009 U4 train items, each used twice:
  - **decide:** choice over commit / c_repair / dual_repair / reject, labelled with the Decision-TRM's
    labels (menu-ceiling order);
  - **repair:** target = the gold grid.
- **Format:** both use the zero-shot prompt (the U4 instruction and the item, no shots). That gives
  6,018 rows and 752 steps.
- **Validation:** a fixed 100-item subset of the U4 val pool.

**Evaluation.** The zero-shot format on u4test (decide + repair), u5test (decide) and proposals
(decide + repair).

## 4. Part P: real proposals

On the proposal items:
- **Repairers:** identity, c_repair, dual_repair (capped or skipped as above), J-u4 writing the grid,
  and CSP re-solve.
- **Deciders:** the six Part R policies, the Decision-TRM, J-u4, and the menu ceiling.
- **Reported by size band and proposer:**
  - raw proposal quality (pass rate as written, share parsed as a full grid);
  - costs (seconds per projection, timeouts and skips, CSP nodes and seconds).

## 5. Registered tests

Pooled discordant (instance, item) pairs; exact binomial; a single-instance arm is paired with every
instance of the other; same-seed pairing when both have three. Holm over J1-J3 and P1-P2 at alpha
0.05.

| ID | Claim | Set | Measure | Test |
|---|---|---|---|---|
| J1 | J-u4 decider vs Decision-TRM | u4test | decide success | two-sided |
| J2 | J-u4 written repair vs dual_repair | u4test | repair success | two-sided |
| J3 | J-u4 decider > J-multi decider (SPEC-U4 few-shot run) | u4test | decide success | one-sided |
| P1 | J-u4 decider vs Decision-TRM | proposals | decide success | two-sided |
| P2 | J-u4 written repair vs c_repair | proposals | repair success | two-sided |

P2 uses c_repair because it is the projection that runs at every size.

## 6. Limitations, stated now

- J-u4 starts from J-multi and is trained only on U4 items. Forgetting of the other suites is not
  measured here.
- The proposers are small models. Their proposals on 8x8 and 10x10 may be far from any solution, so
  "repair" there is close to "solve".
- The Decision-TRM's features include the puzzle size scaled by 6, so 8x8 and 10x10 are outside its
  training range.
- The 60 s cap and the 10x10 skip for dual_repair are cost limits I set. A deployed skill might
  choose others.

## 7. Order

Queue `configs/queues/u5.yaml`. It waits for an idle GPU.
1. J-V0 proposals, then Bonsai-8B proposals.
2. CPU step: projections, reject paths, Decision-TRM, CSP.
3. For each seed 0-2: train J-u4, then evaluate J-u4.

The report is `scripts/u5_report.py`, written before any run; it produces `reports/ubench/u5.md`.

## Addenda

(none)
