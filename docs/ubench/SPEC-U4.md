# SPEC-U4: Campsite repair head-to-head, scored by the official verifier

Registered 2026-10-07, before any U4 arm was run. Your choice that day ("1", after U3): a real repair
test, where repaired artifacts are scored by a verifier, instead of the 88 post-hoc rudder rows.
Changes after registration are dated addenda.

## 1. What is compared

The infused intellect3-logic skill repairs Campsite candidates with deterministic code. U4 runs that
code on fresh broken candidates and compares it with the Jev in two roles:

- **Part D, decision.** Given the puzzle, a candidate grid and the verifier's report on it, choose
  `commit` (as is), `c_repair`, `dual_repair` or `reject`. The final committed grid is scored.
- **Part G, generation.** Write the repaired grid directly. Its output is scored like a repair module's.

**The repair modules** come from Hermes-Skills `research/scripts/run_intellect3_camp_gate_micro_env.py`
at `HEAD`. They are executed unchanged from the committed source, read with `git show`.
- `c_repair` is `c_only_projection`: a minimum-edit projection of tents onto the row and column tent
  counts, keeping the candidate's trees.
- `dual_repair` is `dual_signature_projection`: it also projects trees onto the tree counts.
- Both take an `expected` grid and read only its shape and its T and C row and column counts. I pass
  the CSP solution, and every item asserts that these counts equal the puzzle's visible tree layout
  and constraints. So no hidden information enters.
- When a projection returns nothing (e.g. wrong shape), the original candidate stands, as in
  Hermes-Skills' flow policies (`c_repair or original`).

**The verifier and puzzles** are hermes-lite's `agent.intellect3_logic`:
- the official Campsite semantics (`verify_candidate`): syntax, shape, trees unchanged, row counts,
  column counts, no touching tents, and a perfect tent-tree matching;
- its generator `generate_unseen_tasks`: shapes 3x5 to 5x5 with 20% trees, plus 6x6 as a size shift;
- its CSP solver, which provides the gold grid and the re-solve ceiling.

## 2. Items (`src/jevq/xbench/u4_campsite.py`)

**Puzzle pools.** Three pools, made disjoint by puzzle hash. The generator's puzzle space is small:
the train seed reproduces 2 test puzzles and the val seed 1, so train and val drop every puzzle that
already occurs in an earlier pool.

| Pool | Puzzles generated | Of which 6x6 | Generator seed | Puzzles kept | Items |
|---|---|---|---|---|---|
| test | 90 | 18 | 20261008 | 90 | 480 |
| train | 400 | 80 | 20261009 | 398 | 3,009 |
| val | 80 | 16 | 20261010 | 79 | 602 |

- **Fingerprints.** Test items JSON sha256 prefix `02e802f76cdaabf2`. The Hermes-Skills repair source
  at HEAD has sha256 prefix `d6c87060e3b163e6`.
- **The test candidates' failed checks** (verifier only):
  - correct, drop_tent, tree_mutation and bad_shape each have a single pattern;
  - extra_tent, move_in_row, move_any and swap_rect mix touching and matching failures;
  - swap_rect never fails a count.

**Candidates.** The gold grid G is the CSP solution. Eight candidate types are built from G with a
seeded RNG:

| Type | Construction | Verifier |
|---|---|---|
| correct | G | passes |
| drop_tent | one tent -> X | fails |
| extra_tent | one tent added next to a tree | fails |
| move_in_row | one tent moved to another cell of its row | fails |
| move_any | one tent moved to a cell next to a tree, in another row and column | fails |
| swap_rect | two tents at (r1,c1),(r2,c2) moved to (r1,c2),(r2,c1); counts unchanged | fails |
| tree_mutation | one tree -> X | fails |
| bad_shape | last cell of one row removed | fails |

- If a construction passes the verifier (another valid solution) or is impossible, the puzzle is
  skipped for that type.
- **Test set:** for each type, the first 60 eligible test puzzles in a seeded order, giving 480 items.
- **Train and val:** every eligible (puzzle, type) pair.
- **Checks done before registration, on items only.** Counts per type; every defective candidate fails
  the verifier and every correct one passes; the projection inputs equal the visible puzzle. No
  repair module, decider or generator was run.

## 3. Part D: deciders

**Actions and outcomes.**
- `commit` commits the candidate.
- `c_repair` / `dual_repair` apply the module and commit its output (or the original when it
  returns nothing).
- `reject` commits nothing.
- **Success:** the committed grid passes the verifier. **Unsafe:** the committed grid fails. A reject
  is neither.

**Arms.**

| Arm | Information | Instances |
|---|---|---|
| J-V0, generic Qwen3.5-4B-Base | prompt below; 4 train shots, one per best action | 1 |
| J-multi, the Jev | same | seeds 0-2 |
| J-multi-loop, 2 passes | same | seeds 0-2 |
| Decision-TRM | features below | seeds 0-2 |
| always-commit | rule | 1 |
| c_repair_if_c_fail | Hermes-Skills flow policy: c_repair if the C signature fails, else commit | 1 |
| dual_repair_if_any_sig_fail | Hermes-Skills flow policy: dual_repair if the T or C signature fails, else commit | 1 |
| menu ceiling | uses post-repair verification: the best action per item | 1 |
| CSP re-solve | discards the candidate and solves the puzzle | 1 |

- **The menu ceiling's order** is commit if the candidate passes, else c_repair if its output passes,
  else dual_repair if its output passes, else reject. The same ordering labels the Decision-TRM's
  training rows and picks the shots.
- **The Jev prompt** gives the rules, the action descriptions, and the objective: end with a
  committed grid that passes every rule, or reject if no action can. It then shows the puzzle grid,
  the row and column tent constraints, the candidate grid, and the verifier's failed checks with its
  row and column tent counts.
- **The Jev's options** are lettered in a per-item shuffled order; the readout is `choice`.
- **The Decision-TRM** is hermes-lite's `TinyRecursivePolicy` structure: input projection, a tanh
  recurrence over 4 steps, hidden size 64, and a 4-way action head.
  - Features: the 7 verifier gates, official pass, shape, the share of rows and of columns whose
    tent count is off, the tent and tree count differences, Hermes-Skills' T and C signature
    passes, and the puzzle size.
  - It is trained with cross-entropy on the train items; full-batch steps from {50, 200, 500, 1000}
    are chosen on val accuracy.

## 4. Part G: repairers

- **Score:** an item succeeds iff the repaired grid passes the verifier. All 480 items are scored,
  including the 60 correct ones, so harm counts.
- **Arms:**
  - identity (no repair), c_repair, dual_repair, and CSP re-solve (ceiling);
  - J-V0 and J-multi seeds 0-2, generating the grid greedily. Looped generation is not supported by
    the generation runner, so the looped Jev has no Part G arm.
- **Generation prompt:** the rules, three train shots (drop_tent, tree_mutation and move_any, each
  answered with its gold grid), and the item, with the same fields as Part D. It asks for the
  repaired grid only, one space-separated row per line.
- **Decoding:** up to 120 new tokens, stopping after the grid's row count. An unparseable or
  wrong-shape output fails.
- **Descriptive:** the edit distance to the candidate and to the gold grid.

## 5. Registered tests

Pooled discordant (instance, item) pairs; exact binomial; a single-instance arm is paired with every
instance of the other; Holm over D1, D2 and G1 at alpha 0.05.

| ID | Claim | Measure | Test |
|---|---|---|---|
| D1 | J-multi decider vs Decision-TRM | Part D success, 480 items | two-sided |
| D2 | J-multi decider > J-V0 decider | Part D success | one-sided |
| G1 | J-multi generated repair vs dual_repair | Part G success, 480 items | two-sided |

**Descriptive:**
- every arm by candidate type;
- unsafe and reject rates;
- the fixed flow policies and both ceilings;
- the looped Jev decider;
- the choice label mass;
- generation length and latency.

## 6. Limitations, stated now

- **The candidates are synthetic perturbations** of a valid solution, not model proposals. Their mix
  is fixed by design, not drawn from real failures.
- **The puzzles are small** (up to 6x6), as in hermes-lite's generator. Projections and the CSP
  solver are cheap here.
- **Part D deciders do not see post-repair verification.** In a deployed pipeline the verifier would
  gate the commit, which is why the menu ceiling is reported.
- **No arm was trained on Campsite grids**, except the Decision-TRM on U4 train items. The Jev's
  training has no Campsite grids.

## 7. Order

- **CPU step:** modules, rules, ceilings, Decision-TRM seeds 0-2.
- **Queue `configs/queues/u4.yaml`:** J-V0, J-multi seeds 0-2 and J-multi-loop seeds 0-2, Part D
  and (non-looped) Part G in one step per arm. It waits for an idle GPU.
- **Report:** `scripts/u4_report.py`, written before any run, produces `reports/ubench/u4.md`.

## Addenda

(none)
