# SPEC-U1: final summary (2026-10-07)

One Jev-style adapter (J-multi, plus a looped twin) was trained on the training splits of the
utility suites, each in its own label space. It was compared with the xbench v1 arms. The
registration is `docs/ubench/SPEC.md` (cac47df, before any U1 run). Full tables with intervals
are in `u1.md`; the run log, incidents and interim results are in `notes/009-u1-log.md`.

## Completion

| Item | State |
|---|---|
| u1.yaml steps | 14/14 done, every one with keep-awake. One step (`j_multi_s1`) was killed by a restart and rerun; `train_multi_loop_s0` was lost to a GPU-driver crash and rerun. |
| Registered tests | decided on three seeds |
| Report | `reports/ubench/u1.md`: Wilson intervals, bootstrap intervals, readiness bootstrap |
| Paper | U1 section with the generated table `paper/generated/u1_table.tex` |
| Published | github.com/MoralityLabAI/jev-qwen and huggingface.co/AlephFunk/jev-qwen (all U1 adapters, aggregate records, paper) |

## Answers

**Registered tests**

| Test | Result | Verdict |
|---|---|---|
| T1 transfer to S1 (vs J-V1) | 755 vs 185, Holm p = 4e-82 | supported |
| T2 transfer to S7p (vs J-V1) | 26 vs 22, p = 0.33 | **not supported** |
| T3 S2 (vs J-V0) | 9 vs 0, Holm p = 0.004 | supported |
| H home suite S4 (vs J-V1) | -0.2 points [-1.3, +0.9] | non-inferior |
| S specialist parity on S1 (vs J-V1-rmp) | -2.3 points [-4.3, -0.3] | **not shown non-inferior** |
| L loop in a mixture, S1 | 300 vs 180, p = 2e-8 | supported |

**1. Does training on the target label spaces remove the transfer cost?** Mostly.
- S1 depths 1-8 rise from 0.351 (V1) to 0.632 over three seeds.
- S2 reaches 1.000 in every seed.
- S4 is unchanged (0.867 vs 0.868).
- Calibration improves: S4 ECE 0.055-0.059 against 0.069-0.078.
- Shuffled S7 routing does not improve (0.880 vs 0.870). More than half the held S7 positives (54 of 105) target contracts absent from the 57 registered training rows, so the training split cannot teach those routes.

**2. Does it match the single-task specialist?** Almost: 2.3 points below J-V1-rmp on S1, with an interval that excludes the registered -2 point margin. A multi-task adapter at the same rank gives up a little specialist accuracy.

**3. Does a trained loop still add hops in a mixed training set?** Yes, on average: J-multi-loop at two passes scores 0.691 against 0.632 on S1 depths 1-8 (L). The gain is uneven: +9 and +8 points at seeds 0 and 2, about 0 at seed 1. The loop does not make the readout depth-indexed in any seed, does not help past depth 8 (0.393 vs 0.424), and costs nothing on average on S4 (0.864 vs 0.867).

**Against the smallest reliable executors (v1 rule).** No Jev-style arm, U1 or v1, is reliable on any utility suite. The best arms are unchanged:

| Suite | Best arm | Its score | Best Jev-style arm |
|---|---|---|---|
| S1 depths 1-8 | 460k-parameter LOOP-T-ds | 0.999 | J-multi-loop 0.691 |
| S1 depths 9-12 | J-cot-V0 (emitted reasoning) | 0.717 | J-multi 0.424 |
| S7 routing | 6k-parameter TRM router | 0.992 | J-multi-loop 0.898 (shuffled) |
| S3 | ControlTRM-LDT | 0.687 utility | J-multi-LDT 0.630 |

Where the Jev clearly wins is against a generic small Qwen used directly: S1 0.632 vs 0.406, S4 0.867 vs 0.713, S2 1.000 vs 0.959.

## Non-replications and failures

- **L is uneven across seeds.** Seed 1 shows no loop gain (0.649 vs 0.644). The pooled test passes, but the effect size depends on the seed.
- **The seed-0 S4 cost of the loop (-2.6 points) did not replicate:** -0.3 points on average.
- **J-V2b-rmp's v1 readiness result** (depth-indexed at one seed) does not carry over: J-multi-loop is depth-indexed in 0 of 3 seeds.
- **Incidents** (details in notes/009):
  - UI1: a GPU-driver crash (bugcheck 0x116 VIDEO_TDR_FAILURE) during `train_multi_loop_s0`;
  - UI2: the Hugging Face upload failed on a flaky network (Xet errors, DNS) until it was batched and run without Xet;
  - UI3: a restart from the Start menu killed `j_multi_s1`; the GPU was then shared with another session's RMP jobs and a game, so U1 now waits for an idle GPU.

  No result was lost; every interrupted step was rerun in full.

## Decisions needed from you

1. **S7 routing.** The Jev cannot learn routes absent from its training rows, and the TRM router (0.992) already generalises to them. Either accept the TRM router as the executor for routing, or build a larger S7 training split that covers more contracts. That would need new registered rows from hermes-lite, which is your call.
2. **End-to-end test (proposed U2).** You asked whether a small Jev beats a TRM-infused skill selected by a generic small Qwen. Measured component by component, the Jev selects slightly better than a generic small Qwen and ties the TRM skill gate on S2, but the TRM router beats both at selection. A registered end-to-end run on the hermes-lite control mesh would settle it. The arms would be: generic Qwen picks then TRM skill; Jev picks then TRM skill; TRM router picks then TRM skill; Jev alone.
3. **Control (S5/S6).** Still deferred until there is specialised blue-team synthetic data, as you decided.
4. **GPU stability.** One VIDEO_TDR_FAILURE bugcheck occurred during looped training. Driver and TDR settings are yours to change; I left them alone.
5. **Paper.** The U1 section is in `paper/main.pdf`. Authorship, venue and whether U1 stays in this paper or becomes its own short note are your call.
