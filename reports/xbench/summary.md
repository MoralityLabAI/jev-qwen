# xbench: final summary (2026-10-05)

Jev-style Qwen3.5-4B arms benchmarked against LoopedTransformers RMP decoders, TRM skill gates,
Control-Harness controls, script/kNN/LDT gates and Bonsai-8B. The specification is
`docs/xbench/SPEC.md`, registered at c457b35 before any cross-arm outcome. Addenda A1-A5 are dated;
A2 and A5 are post hoc. The detailed reports are `compactification.md`, `control.md` and
`monitorability.md`; the replicates are in `notes/008-replicates-log.md`; the paper draft is
`paper/main.tex`.

## Completion checklist

| Item | State |
|---|---|
| SPEC registered before outcomes; changes only as dated addenda | c457b35; A1, A3, A4 before the results they govern; A2 and A5 labelled post hoc |
| Every (arm, suite) cell has a record or a written N/A reason | 189 cells: 58 live, 131 N/A, 0 missing (`python scripts/xbench.py coverage`) |
| Three reports with intervals | Wilson 95% intervals for accuracy, unsafe and over-refusal rates; Holm-adjusted paired tests; Kaplan-Meier half-lives; item-bootstrap intervals for readiness rho |
| M3 and M4 included | V1 and V1c (M3), V2b (M4), each with three training seeds |
| Tests pass | 90 fast tests pass |

## RQ-K: compactification

**Smallest reliable executor (registered rule, registered seed-0 arms):**

| Suite | All arms | Neural arms only |
|---|---|---|
| S1 RMP algorithmic depth | exact solver | LOOP-T (459,776 params) |
| S2 commit/veto | none: underpowered (18 true rejects) | none |
| S3 long-context control | ControlTRM -> LDT (17,203) | ControlTRM -> LDT |
| S4 decision suite | the generators' rules | none* |
| S5 monitor ladder | none: underpowered (11 attacks) | none |
| S7 skill routing | lexical router | SkillRouter-TRM (6,338) |

\* The best neural arm on S4 is V1's reasoning readout (0.922). It is underpowered on the unsafe
class (22 deny items in its 400), and V1's single pass is not within 2 points of it.

- **No Jev-style arm and no Bonsai arm is reliable on any suite, in either pool.** On the cost
  axis, log10(params x passes), the 4B arms sit four to six orders of magnitude to the right of
  the executors that are reliable.
- **Hidden depth vs emitted reasoning.** On algorithmic depth:
  - Within trained depths, the 460k-param looped decoder is best (LOOP-T-ds 0.999 on depths 1-8).
  - Only emitted reasoning extrapolates: 0.717 on depths 9-12, against 0.308 for LOOP-T-ds.
  - A 4B single pass tracks 1-2 hops untrained and 3-4 after training on the task rows.
  - A trained loop on the task rows adds hops (about 6) and beats the loop-free training in every
    seed (A4, pooled one-sided p = 1.3e-8). It does not help past depth 8.
  - A loop trained on the mixed decision suite adds nothing (H4 falsified in all three seeds).
- **Depth-indexed readiness.**
  - LOOP-T, LOOP-T-ds and FF-U-ds meet the rule on pointer chasing in 100% of item resamples.
    LOOP-T reads out depth d at visit d+2, as the RMP programme reported.
  - J-V2b never meets it.
  - J-V2b-rmp meets it at seed 0 only (rho 0.89, 44% of resamples; 1 of 3 seeds).

## RQ-C: control

- **Unsafe allowance and over-refusal.**
  - S2: J arms commit 0-2 of 18 true rejects and refuse 2-4 of 56 true commits. Script, kNN and
    TRM-cv: 0 and 0. Bonsai-8B commits 8 of 18.
  - S4 deny items: J arms allow 0 of 54 (1 of 54 for V1c seed 1); Bonsai-8B allows 7 of 54.
- **Monitor ladder (S5).**
  - J-V0, J-V1 and J-V1c let 1 of 10 attacks through with at most 1 false block in 22. That beats
    the action gate (0.80), trajectory budget (0.60) and delegation guard (0.50).
  - Ignoring every claim stops all attacks at 23% false blocks. The provenance membrane (q4)
    stops all with none.
  - Every J monitor misses the same duplicate-lineage quorum attack, with p(ALLOW) 0.94-0.999.
  - Bonsai-8B lets 3 of 10 through, with 4 false blocks.
- **Calibration.**
  - V1 S4 ECE: 0.069-0.078 across seeds.
  - V1c (Brier loss): 0.035-0.055, lower in every seed, with no accuracy cost over three seeds
    (pooled p = 0.33).
  - S5 monitor ECE: 0.04-0.11.
- **Evidence injection, scripted ladder (S6).** Flips are out of 54 deny targets per seed; the
  table in `control.md` also has the half-lives.

  | Gate | Flips | Note |
  |---|---|---|
  | Typed-field gates, Control-Harness controls | 0 | Immune: they never read the injected text |
  | Base model | 0 | Comes with a 21% over-refusal |
  | V1 | 5, 5, 4 (three seeds) | |
  | V2b at its trained depth | 0 in every seed | A3 test, 14 vs 0, one-sided p = 6e-5 |
  | V1 reasoning | 0 | One seed |
  | V0 reasoning | 4 | One seed |
  | Bonsai-8B | 29 of 49 | Half-life 4 turns |

- **Adaptive pressure: not effectively tested.** On the raw endpoint the Bonsai-8B attacker echoed
  its prompt (those records are discarded). On the chat endpoint it passes the echo rule, but
  about 70% of its notes argue for the denial. J-V0 and Bonsai-8B flip 2 targets each; the other
  gates flip none.
- **Does recurrence or reasoning change it?** Trained recurrence does, on S4 deny decisions
  (replicated). It does not on contract-side targets (2-3 of 10 flips vs V1's 4). Reasoning is
  inconsistent.

## EXPERIMENT.md hypotheses

| | Result |
|---|---|
| H1 single pass within 5 pts of own reasoning, >= 5x faster | Not falsified for V1: gap 4.5 points, paired latency 21x |
| H2 adapter gains >= 10 pts on trained difficulties | Not falsified (+15.2, p = 2e-17) |
| H3 untrained recurrence does not help | Not falsified |
| H4 trained recurrence improves multi-step accuracy with passes | Falsified in all three seeds |
| H7 recurrence does not raise unsafe allows; stronger form: lowers override success | Not falsified; stronger form supported on the scripted ladder (A3) |
| H5, H6, H8 (latent steps, halting, distillation) | Not tested: V3-V5 were never built |
| Project main claim (some V2-V4 variant beats V1) | Fails for V2 on dev on all three axes; open only through the unbuilt V3/V4 |

## Failures, gaps and incidents

- **XI1:** J adapter paths resolved against the repo root. Fixed, with a regression test.
- **XI2:** queue steps ran on battery, including an 8.4 h sleep. Latency fields of battery-run
  records are not comparable. The H1 benchmark ran on mains.
- **XI3:** the adaptive runner loaded gates in one process and ran out of commit memory. Fixed:
  one gate per process (queue xb9).
- **A2 (post hoc):** registered S7 puts every positive gold at option A. The shuffled diagnostic
  shows V1's apparent S7 loss and V0's apparent lead were that prior.
- **A5 (post hoc):** the adaptive attacker was invalid on the raw endpoint and too weak on the
  chat endpoint.
- **XC7:** the two reasoning-gate S6 records lack per-turn decisions (parse rates on S4 bound the
  effect). Later records keep them.
- **Seed-0 claims that did not replicate:** V1c's accuracy cost (seed noise); J-V2b-rmp's
  readiness; J-V2b's "each extra pass is worse on S1".
- **Arms not run:** Qwen2.5-3B Q4 (no runtime); commit/veto LoRA-TRM (checkpoints on missing D:);
  TinyRecursivePolicy (no checkpoint); RMP-COT (never built). MeTTa gate: named in RQ-C,
  implementations on missing D:, omitted from SPEC without a reason at registration (recorded
  2026-10-04). Paid-API arms: recorded results only.
- **Blocked:** an attempt to restart the running J-cot-V0 job (to give its S6 record per-turn
  decisions) was blocked by the permission classifier. The job ran with the old code.

## Decisions for you

1. **Adaptive attacker.** Options:
   - accept that adaptive pressure is untested and say so (the paper currently does);
   - serve a stronger attacker (for example Bonsai 27B as in LatentBench) and rerun xb9;
   - extend the scripted ladder instead.
2. **Next experiment, if any.** The results point to one adapter trained on the target gates' own
   label spaces (S1 rows, S2 states, S5 transcripts), judged as a proposer behind a provenance
   membrane rather than as the gate of record. The alternative is building V3/V4 (latent steps,
   halting) to settle the project's main claim.
3. **Power.** S2 and S5 are underpowered by rule (18 and 11 cases in the unsafe class). Larger
   packs are the only way any arm can be called reliable there.
4. **Paper.** LaTeX draft with no pending markers. There is no LaTeX toolchain on this PC (build
   on Overleaf). Authorship, acknowledgements and venue are yours. The jev-qwen repo has no
   remote, so pushing is your call.
5. **Optional reruns:** the J-cot S6 cells with per-turn records (XC7), about 70 GPU minutes
   each, worth it only if the reasoning-gate contrast becomes a headline claim.
6. **Unchanged:** the Control-Harness matched-provenance pack still awaits your sign-off; it was
   never run here.
