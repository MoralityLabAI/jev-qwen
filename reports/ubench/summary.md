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

## Your decisions (2026-10-07)

1. S7 routing: the TRM router stays the routing executor for now.
2. End-to-end test: approved. It will be pre-registered as SPEC-U2 before any run.
3. Control (S5/S6): deferred.
4. GPU: steady power and internet now; driver and TDR settings unchanged.
5. Paper author: Patrick Dugan, Red Team Gladiatorics.

## U2: end-to-end pipelines (SPEC-U2, decided 2026-10-07)

This is the test of your question: does a small Jev beat a TRM-infused skill picked by a
generic small Qwen? It runs on a synthetic joint set of 600 items: requests for the Campsite logic
skill, each paired with a commit/veto state, plus confusable and negative requests. Details are in
`notes/010-u2-log.md` and `reports/ubench/u2.md`.

**Registered tests**
- **T1:** the Jev picking and deciding (0.762) does not beat the generic Qwen picking with the TRM
  gate deciding (0.773). Holm p = 0.92: **not supported**.
- **T3:** the Jev pipeline beats the generic Qwen doing both steps (0.762 vs 0.740, Holm p = 0.009).
- **T2:** the TRM-router pipeline is far worse here (0.517). It has no abstain path, and it routes
  the new request phrasing poorly.

**Descriptive**
- **Gates:** the TRM gate is perfect on the commit/veto rule. The Jev gate is about 95% right, and
  every miss is a zero-delta, non-exact repair, which the rule rejects and the Jev commits.
- **Routing:** the Jev is the better router on confusable and negative requests.
- **Best combinations:**
  - looped Jev doing both steps, 0.793;
  - Jev routing with the TRM gate, 0.787, with the lowest unsafe rate among the strong pipelines
    (1.3%).
- **Post hoc:** looped Jev vs generic Qwen + TRM gate, two-sided p = 0.036, exploratory.

**Reading.** On this test, put the Jev (or looped Jev) at the routing step and keep the TRM gate
for commit/veto. A Jev alone is not better than the generic-Qwen + TRM-skill pipeline, because its
gate misses the zero-delta boundary that the TRM learned.

**Possible next steps (yours to choose)**
1. Train the Jev's gate format with more zero-delta rows from the S2 training split (the data
   already exists), or simply keep the TRM gate.
2. Give the TRM router an abstain path before relying on it for negatives.
3. Validate U2 on real request traffic if any exists. The requests here are templated.

## U3: repair rudder and the TRM router's abstain path (SPEC-U3, decided 2026-10-07)

You asked to compare the TRM repair modules of the infused skills. No trained repair TRM exists:
the skills repair with deterministic code (for example `c_repair` / `dual_repair`). The learned or
prompted piece is the rudder, which picks the repair action and whether to commit it. You chose to
replicate Hermes-Skills' 88-row repair-rudder benchmark. Details are in `notes/011-u3-log.md` and
`reports/ubench/u3.md`.

**Part R, repair rudder (54 holdout rows for the tests)**
- **R1 not supported.** The Jev does not beat the generic small Qwen with leak-free retrieval
  (0.580 vs 0.630 joint).
- **R2 no difference shown.** The repair TRM, trained here for the first time as Hermes-Skills had
  planned, scores 0.630 (Holm p = 0.35).
- **R3 identical.** With the published retrieval, the Jev equals the published Qwen3.5-27B (both
  0.667, the ceiling).
- **The published retrieval leaks the answer.** It matches on the eval row's own labels, and 98.6%
  of its examples share the row's commit action. With it, a 4B base model, the Jev, 9B and 27B all
  reach the same 0.795 on 88 rows. The scale "lift" in the published table is retrieval.
- **Without the leak,** the generic Qwen, the repair TRM and a lookup table tie (34/54). The Jev is
  the only arm that vetoes all four bad repairs whose outcome is not visible beforehand. It also
  rejects some good ones, so it has the lowest unsafe rate but lower joint accuracy.

**Part A, abstain path**
- **A1 not supported.** Fitting the router's unused abstain head on the frozen trunk changes nothing:
  it never fires.
- **The router is most confident where it should refuse.** On U2 negatives it picks the named,
  forbidden skill with mean top probability 0.55, against 0.20 on real requests. So a confidence
  threshold cannot fix it either.
- **A2.** The Jev router with the TRM gate stays far ahead (0.787 vs 0.517).

**Possible next steps (yours to choose)**
1. A real repair test needs repaired artifacts scored by a verifier, for example the Campsite
   head-to-head offered earlier, rather than the 88 post-hoc rows.
2. For the TRM router, retrain with the abstain loss or an explicit negation feature (post hoc).
   Alternatively keep the Jev as the router; it already handles negatives (0.915).

## U4: Campsite repair head-to-head (SPEC-U4, decided 2026-10-07)

**Setup.** 480 fresh broken Campsite candidates on hermes-lite puzzles, 8 defect types x 60, scored by
the official verifier. The repair modules are Hermes-Skills' intellect3-logic projections, run from
the committed source. Details are in `notes/012-u4-log.md` and `reports/ubench/u4.md`.

**Registered tests**
- **D1, deciding:** choose commit, c_repair, dual_repair or reject. A 9.7k-parameter recursive TRM
  policy trained on synthetic items reaches 0.727 (ceiling 0.733) with 1.9% unsafe. The Jev reaches
  0.569 with 18.7% unsafe. The TRM is far better.
- **D2:** the Jev does not beat the generic small Qwen as a decider (0.569 vs 0.590). Not supported.
- **G1, repairing:** dual_repair fixes 0.731 of candidates. A Jev writing the repaired grid fixes
  0.184, mostly by copying the candidate. The projection is far better.

**Descriptive**
- The looped Jev is the best Jev decider (0.643) but still unsafe on 22% of items.
- The skill's own flow policies never reject, so every unfixable candidate becomes a broken commit
  (27% unsafe for the dual policy).
- Re-solving with the CSP solver fixes everything on puzzles this small.

**Reading.** In the infused intellect3-logic skill, the deterministic repair code plus a tiny trained
decision policy beat the 4B Jev in both roles. The main safety gap is the skill's missing reject
path, not its repair code. Caveat: the TRM was trained on the same defect types; the Jev saw four
examples and no Campsite training.

**Possible next steps (yours to choose)**
1. Add the Decision-TRM's reject path to the skill's flow policy. On this set it removes almost all
   unsafe commits at no cost in success.
2. Train the Jev on the U4 train items, the same data the TRM saw, to make D1 a like-for-like test.
3. Test on real model-proposed candidates instead of synthetic perturbations, and on puzzles large
   enough that re-solving is not free.
