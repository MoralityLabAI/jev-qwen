# Replicates log (addenda A1, A3, A4; 2026-10-04)

Seed replicates sit outside the registered R1/R2 pools (SPEC addendum A1). Seeds change training
order and LoRA initialisation. Evaluation items and few-shot prompts stay at seed 0
(`training.build_items` uses eval_seed). Seed 0 is the registered record.

## V1 and V1c (queue xb4)

| | seed 0 | seed 1 | seed 2 | mean (SD) |
|---|---|---|---|---|
| S4 accuracy, V1 | 0.880 | 0.861 | 0.864 | 0.868 (0.010) |
| S4 accuracy, V1c | 0.861 | 0.849 | 0.880 | 0.863 (0.016) |
| S4 ECE, V1 / V1c | 0.069 / 0.055 | 0.075 / 0.035 | 0.078 / 0.052 | |
| S4 NLL, V1 / V1c | 0.382 / 0.334 | 0.380 / 0.334 | 0.367 / 0.339 | |
| S4 unsafe allows, V1 / V1c | 0/54 / 0/54 | 0/54 / 1/54 | 0/54 / 0/54 | |
| S1 accuracy, V1 | 0.289 | 0.310 | 0.335 | 0.311 (0.023) |
| S1 accuracy, V1c | 0.351 | 0.354 | 0.369 | 0.358 (0.010) |

- **S4 accuracy, V1c against V1:** pooled discordant pairs over the three seeds are 95 vs 110,
  exact p = 0.33. The seed-0 result (V1c 1.9 points lower, p = 0.034) does not replicate.
- **Calibration:** V1c has the lower ECE and the lower NLL in every seed.
- **Transfer to S1:** V1c is higher in every seed (pooled 364 vs 186, p = 3e-14).
- **Seed-to-seed spread of V1 alone:** 0.880 vs 0.861, paired p = 0.076.

## V2b (H4 replicate, queue xb4)

| | seed 0 | seed 1 |
|---|---|---|
| S4 multi-step classes, V1 / V2b-r2 | 0.840 / 0.833 (p = 0.65) | 0.815 / 0.827 (p = 0.36) |
| S4 multi-step d4-5, V2b r1 / r2 / r3 | 0.673 / 0.747 / 0.737 | 0.690 / 0.710 / 0.690 |
| S1, V1 / V2b r1 / r2 / r3 | 0.289 / 0.342 / 0.326 / 0.304 | 0.310 / 0.334 / 0.359 / 0.340 |

- **H4:** stays falsified at seed 1. V2b is within 3 points of V1 on the multi-step classes, and
  accuracy does not rise with passes at difficulty 4-5.
- **S1:** the seed-0 pattern "each extra pass is worse" does not replicate (seed 1 peaks at the
  trained depth). V2b is above V1 on S1 in both seeds.
- Seed 2: pending.
