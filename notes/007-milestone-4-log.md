# Milestone 4 log: V2b, trained recurrence (2026-10-03)

V2b is V1's recipe with the loop active during training: same LoRA config (r16 on every
attention, DeltaNet and FFN projection), same 2,400 training questions in three formats, same
300 steps and seed. The one difference is that layers 12-15 run twice in every training forward
(EXPERIMENT.md, decision recorded 2026-10-03). Queue xb1, steps `train_v2b` and `j_v2b`.

## Training

| | V1 | V1c | V2b |
|---|---|---|---|
| mean loss, steps 1-50 | 0.233 | 0.171 | 0.255 |
| mean loss, steps 251-300 | 0.113 | 0.060 | 0.094 |
| median seconds per step | 13.1 | 13.3 | 15.7 |

Wall time 89 min (5,340 s). The loop costs about 20% per training step.

## Evaluation

One run at three passes per item. The tail lens after pass r is exactly the output of a run
stopped at r (instrument.py, validated in M2), so the records J-V2b-r1, -r2 and -r3 are exact
one-, two- and three-pass outputs of the same adapter. r2 is the trained depth; r1 is the
`n_iters: 1` ablation that H4's mechanism criterion asks for. The three records share one
latency measurement (the three-pass run); M2 measured 12.2% paired latency per extra pass.

### S4 (dev suite, 1,000 items, `choice`)

| | J-V1 | J-V2b-r1 | J-V2b-r2 | J-V2b-r3 |
|---|---|---|---|---|
| all | 0.880 | 0.847 | 0.875 | 0.873 |
| multi-step classes (750) | 0.840 | 0.800 | 0.833 | 0.831 |
| difficulty 4-5 (400) | 0.812 | 0.748 | 0.810 | 0.802 |
| multi-step, difficulty 4-5 (300) | 0.750 | 0.673 | 0.747 | 0.737 |
| multi-step, difficulty 1-3 (450) | 0.900 | 0.884 | 0.891 | 0.893 |
| ECE | 0.069 | 0.099 | 0.080 | 0.082 |

Paired exact McNemar against J-V1 (discordant counts V2b-only / V1-only):
r1 24/57 (p = 3e-4), r2 36/41 (p = 0.65), r3 41/48 (p = 0.53). On multi-step difficulty 4-5:
r2 23/24 (p = 1.0). Across passes on multi-step difficulty 4-5: r2 vs r1 31/9 (p = 7e-4),
r3 vs r2 10/13 (p = 0.68).

By task: the loop helps `relation_hops` (V1 0.94, r2 0.98) and `var_trace` (0.71, 0.73), and
hurts `graph_hops` (0.68, 0.62) and `arith_chain` (0.86, 0.83).

"Multi-step classes" is not defined in EXPERIMENT.md. Read here as the four classes whose
difficulty knob is a number of sequential steps: arithmetic, logic, planning, multihop. This
reading was chosen after the outcome. The verdict below is the same for all classes and for
difficulty 4-5 alone.

### H4: falsified

- *Criterion 1:* "false if V2b at its best `n_iters` is within 3 points of V1 at matched
  budget on the multi-step classes." Best is r2 at 0.833 against 0.840: within 3 points, and
  below.
- *Criterion 2:* "or if its accuracy does not increase with `n_iters` at difficulty 4-5." It
  increases from one pass to the trained two (0.673 to 0.747), then not to three (0.737). The
  rise from r1 is the adapter returning to the depth it was trained at, not extra depth helping:
  r2 only recovers V1's level.
- *Mechanism:* the extra passes are not idle. Option-distribution KL from pass 1 to 2 is 0.099
  (mean) and the argmax changes on 61 of 1,000 items; from 2 to 3 it is 0.049 with 39 changes.
  The recurrence computes something, but at its trained depth the result equals V1 without it.

### S1 and the other suites (xbench)

- **S1:** more passes are worse: r1 0.342, r2 0.326, r3 0.304 (J-V1 0.289). The passes change
  many answers (r1 to r2: 392 of 1,275), so on S1 the recurrence moves answers without
  improving them.
- **S2:** r1 1.000, r2 0.946 (2 of 18 true rejects committed), r3 0.959 (J-V1 0.946).
- **S5 monitor:** attack success r1 0.20, r2 0.10, r3 0.00 with 2 of 23 legitimate actions
  blocked (J-V1 0.10, no false blocks).
- **S6 scripted ladder, S4 deny targets:** r2 and r3 flip none of 54 targets; J-V1 flips 5. The
  5 are the same targets, so the paired exact test is 5 vs 0, p = 0.0625. r1 flips 3.

### H7 (control), interim

- Not falsified on the dev suite: no recurrent read raises the unsafe-allow rate (0 of 54 deny
  items for V1 and every V2b pass count).
- Stronger form: untestable on dev. V1's rate with override notes is already 0 (31 deny items at
  difficulty 3 and up), so it cannot fall.
- The injection ladder points the same way as the stronger form: 5 flips for V1, 0 at the
  trained depth. That is one seed and not significant (p = 0.0625); the replicates (xb4,
  addendum A1) are the test.

### Main claim (EXPERIMENT.md section 8)

V2b beats V1 on none of the three registered axes:
- accuracy at equal compute: r1 is worse;
- compute at equal accuracy: r2 matches V1 at about 12% more latency;
- difficulty extrapolation: r2 at difficulty 4-5 is 0.810 against 0.812.

V3 (latent) and V4 (halting) have not been built, so the main claim is open only through them.

## J-V2b-rmp: the same comparison on one algorithmic task (xbench S1, 2026-10-04)

J-V1-rmp and J-V2b-rmp are the V1 and V2b recipes (300 steps, 2,400 rows) on RMP train-region
rows (five pointer-chase, reachability and sudoku families), evaluated on S1 at one, two and
three passes as above.

| S1 | J-V1-rmp | J-V2b-rmp-r1 | J-V2b-rmp-r2 | J-V2b-rmp-r3 |
|---|---|---|---|---|
| core4, depths 1-8 (675) | 0.649 | 0.599 | 0.724 | 0.686 |
| core4, depths 9-12 (300) | 0.433 | 0.400 | 0.430 | 0.457 |
| pointer chase, depth 4 / 5 / 6 | 0.72 / 0.48 / 0.44 | 0.40 / 0.24 / 0.08 | 0.92 / 0.92 / 0.76 | 0.84 / 0.64 / 0.72 |

- **Trained depth vs J-V1-rmp, depths 1-8:** r2 against J-V1-rmp is 103 vs 52 discordant items,
  p = 5e-5.
- **Across passes:** r2 against r1 is 130 vs 45 (p = 9e-11); r3 against r2 is 37 vs 63
  (p = 0.012).
- **Depths 9-12:** no difference.
- **Readiness on pointer chase:** depth-indexed by the registered rule (rho = 0.894 over depths
  1, 2, 4 and 5); depth 3 never reaches 0.9. Not depth-indexed on the other families.

So the conclusion that trained recurrence does not help is specific to the mixed dev suite (H4,
which is registered on dev and stays falsified). On a single task with a consistent sequential
step, the same recipe makes the second pass do real work. One seed; no replicate is queued for
the RMP arms.
