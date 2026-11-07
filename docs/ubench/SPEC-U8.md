# SPEC-U8: small Jevs, scale and calibrated-decision RL

Registered 2026-11-07, before any U8 arm was trained or any small base was run. Your request that
day: "I want to learn how they JEV-ify that and if that can be replicated at a lower scale yet".
Changes after registration are dated addenda.

U8 asks two questions on the benchmark the 4B Jevs were scored on:
- **Scale.** Does J-multi's recipe still work when the base is a fifth or half the size, in the same
  model family?
- **Feedback.** Does training on bandit feedback alone (the only reading of TypeSafe's RLCD that has
  been published) keep the calibration that full-label Brier training gives?

## 0. What is known, read 2026-11-07

**TypeSafe.**
- The method behind Jev is still unpublished: no paper, weights or architecture.
- Its public statements are unchanged from `notes/001-jev-public-facts.md`: typed decisions in one
  query, a "parallel sampler", and a training method named RLCD.

**Open small-scale replications since then.**
- **eve-rlcd** (github.com/anthony-maio/eve-rlcd, MIT).
  - Qwen3-0.6B-Base. The state is prefilled once and question suffixes are batched on the shared
    KV cache; it reads option-letter logits.
  - Training: a 100-step SFT warmup, then 500 bandit steps. The policy samples an option a ~ p and
    sees only c = [a is right]. Reward r = c - p_a, REINFORCE with a leave-one-out baseline over 4
    samples.
  - Results, in distribution: RLCD accuracy 0.808, ECE 0.023. The RLVR control (r = c) reaches
    0.778 and ECE 0.213. SFT on 32k labels reaches 0.817 and ECE 0.014.
- **Laya** (convaiinnovations/laya-typed-decisions, Apache-2.0).
  - A ModernBERT-large encoder (421M), non-autoregressive.
  - Trained with proper-scoring-rule REINFORCE plus soft CE to teacher distributions.
  - On its own four synthetic workflows: 0.766 accuracy against Jev's published 0.727, but ECE
    0.213 against 0.144.
- **Narrow Decision Models** (arXiv 2610.25553). A bidirectional encoder with a block attention mask
  over a state -> question -> option prefix tree. This gives option-order invariance and independent
  questions by construction.

**This project so far.**
- At 4B, Brier training (J-V1c) improved calibration and the gain replicated.
- J-multi (CE, one adapter on the utility suites' training splits) is the 4B reference here.

**Gap U8 fills.** Every replication above is a narrow specialist evaluated in distribution, and none
was scored against a 4B model of the same family on held-out suites.

## 1. Arms

| Arm | Base | Training | Seeds |
|---|---|---|---|
| J-V0-0.8B | Qwen3.5-0.8B-Base | none (4-shot, as J-V0) | - |
| J-V0-2B | Qwen3.5-2B-Base | none | - |
| J-multi-0.8B | 0.8B | J-multi's recipe unchanged (`configs/train/u8_multi08.yaml`) | 0-2 |
| J-multi-2B | 2B | J-multi's recipe unchanged (`u8_multi2b.yaml`) | 0-2 |
| J-brier-0.8B | 0.8B | J-multi's rows, every row as a choice item, Brier with the gold label (`u8_brier08.yaml`) | 0-2 |
| J-rlcd-0.8B | 0.8B | the same rows, bandit feedback, r = c - p_a (`u8_rlcd08.yaml`) | 0-2 |
| J-rlvr-0.8B | 0.8B | the same rows, bandit feedback, r = c (`u8_rlvr08.yaml`) | 0-2 |

**Bases.**
- Qwen3.5-0.8B-Base at Hub commit dc7cdfe2 and Qwen3.5-2B-Base at b1485b2f
  (`arms.BASES`).
- Both have the 4B base's architecture: Qwen3_5, hybrid Gated DeltaNet, full attention every 4th
  layer, the same tokenizer. So J-multi's LoRA targets apply unchanged.

**Training recipe for every trained arm.**
- J-multi's: LoRA r16 / alpha 32 on the same modules, lr 1e-4, batch 8, one epoch, 714 optimizer
  steps.
- The R arms differ from each other only in `loss.choice`.

**Data.**
- J-multi's 5,718 rows (S4 2,400; S1 2,400; S2 234 x 3; S7 54 x 4).
- For the R arms, the S4 rows are rendered in choice format instead of round-robin choice /
  generate / generate_cot. The example ids are identical (`tests/test_u8.py`).

**4B references** (records already exist, not rerun): J-V0, J-V1c, J-multi seeds 0-2.

## 2. Bandit losses (`training.bandit_loss`)

**Feedback.** For each choice item, k = 4 options are sampled from the model's distribution p over
the valid labels. Only c_j = [a_j is the gold option] is used for each sample.

**Loss.**

    loss = -2 * mean_j (r_j - b_j) * log p(a_j),
    b_j = mean of the other three r

- rlcd: r = c - p_a.
- rlvr: r = c.

**Expected gradients** (Monte Carlo check in `tests/test_u8.py`):
- For rlcd, the expected gradient equals the Brier score's gradient exactly. J-rlcd-0.8B and
  J-brier-0.8B optimise the same objective in expectation and differ only in what is revealed.
- For rlvr, it is -2 times the gradient of p(gold): an accuracy objective with no calibration term.

This is eve-rlcd's open reading of RLCD. TypeSafe's own method is unpublished.

## 3. Evaluation

**Suites.**
- S1, S2, S4, S7 and S7p through `xbench.py j`, on the same items as J-multi's records.
- Single forward pass, softmax over the item's option labels.

**Pooled set P.** S1 core4 depths 1-8, S2, S4 and S7p. Each (suite, item) counts once per seed.

**Calibration.** ECE with 10 equal-width bins on the top-option probability, over P with seeds
pooled.

## 4. Registered tests

**Non-inferiority rule.** Mean over seeds of the paired accuracy difference on P. Non-inferior if
the lower end of the two-sided 95% item-bootstrap interval (10,000 resamples, seed 0; one resample
of P's items serves every seed) is above -0.02.

**Superiority rule.** Pooled discordant (seed, item) pairs, one-sided exact binomial. An unseeded
comparator is paired with every seed.

| ID | Claim | Rule |
|---|---|---|
| Z1 | J-multi-2B is non-inferior to J-multi (4B) on P | non-inferiority |
| Z2 | J-multi-0.8B is non-inferior to J-multi (4B) on P | non-inferiority |
| Z3 | J-multi-0.8B > J-V0-0.8B on P | superiority |
| R1 | J-rlcd-0.8B > J-V0-0.8B on P: bandit feedback alone teaches | superiority |
| R2 | J-rlcd-0.8B is non-inferior to J-brier-0.8B on P accuracy | non-inferiority |
| R3 | ECE(J-rlvr-0.8B) > ECE(J-rlcd-0.8B) on P | item bootstrap of the ECE difference; holds if its 5th percentile > 0 |
| R4 | ECE(J-rlcd-0.8B) - ECE(J-brier-0.8B) < 0.02 on P | the same bootstrap; holds if its 95th percentile < 0.02 |

- Holm at alpha 0.05 over Z3 and R1.
- Z1, Z2, R2 and R4 are interval claims at the stated levels; R3 is one-sided at 5%.

**Descriptive:**
- every arm on every suite (accuracy with Wilson interval, ECE, Brier);
- S1 held-out depths 9 and above;
- the scale curve 0.8B / 2B / 4B;
- training time, peak VRAM, median latency;
- training-log accuracy and validation curves for the R arms.

## 5. Limitations, stated now

- **Not TypeSafe's method.** RLCD here is eve-rlcd's open reading.
- **The R arms start from the base.** They get one epoch, with no supervised warmup; eve-rlcd warmed
  up on labels first. Bandit feedback carries less information per row than a label, so R2 is a
  demanding comparison at equal data.
- **One learning rate.** lr 1e-4 was chosen for supervised LoRA training. If RLVR collapses at it,
  R3 still holds or fails, but the collapse is partly a learning-rate effect.
- **Same suites in training and test.** Arms train on the suites' training splits and are tested on
  their test items. Out of distribution are only S1's depths 9 and above (descriptive) and the S7p
  shuffle.
- **No RL at 2B or 4B.** The RL comparison is at 0.8B only.
- **No latency test of shared-prefix question branches** (eve-rlcd's batched `ask`). Records keep
  per-item latency as before.

## 6. Order and resources

- **Downloads.** The two small bases are downloaded from huggingface.co/Qwen: Qwen3.5-0.8B-Base,
  1.77 GB, and Qwen3.5-2B-Base, 4.57 GB. The queue is not launched before you approve that.
  - The local `Documents\Codex\qwen35_08b_base` is not used: its weights' sha256 (04b1c301...)
    differs from the Hub file (c2b1e5a1...).
- **GPU.** After SPEC-U7 part L, before part M: one vramorama job (`--vram 8G`) running
  `configs/queues/u8.yaml`.
  - About 10 GPU-hours estimated from J-multi's 4B timings (training 76-104 min, suites 10-15 min
    per seed).
- **Report.** `scripts/u8_report.py`, written before any run, produces `reports/ubench/u8.md`.

## Addenda

(none)
