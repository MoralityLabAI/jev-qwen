---
base_model: Qwen/Qwen3.5-4B-Base
library_name: peft
license: apache-2.0
tags:
  - lora
  - typed-decisions
  - looped-transformer
  - ai-control
  - benchmark
---

# jev-qwen: Jev-style typed-decision adapters for Qwen3.5-4B-Base

LoRA adapters from the *Small Jev* experiments (Patrick Dugan, Red Team Gladiatorics). TypeSafe's Jev is described as a
"System One" model: it returns a typed, calibrated decision over a pre-enumerated output space from
a single query. These adapters give an open 4B base model a single-pass, option-scored readout
of that kind. They are benchmarked against compact looped decoders, TRM skill gates,
script/LDT/Control-Harness gates and a 1-bit 8B LLM.

**This is not Jev.** Jev's weights and architecture are not public; nothing here measures Jev.

- Code, configs, pre-registration, reports: https://github.com/MoralityLabAI/jev-qwen
- Paper draft: `paper.pdf` in this repo (also `paper/` on GitHub)
- Aggregate run records: `results/xbench/<suite>/<arm>/record.json`. Item-level records are not
  published, because they contain held-out evaluation items of other projects.

## Adapters

All adapters are LoRA r16 / alpha 32 on q/k/v/o, the DeltaNet projections (in_proj_qkv, in_proj_z,
out_proj) and the FFN (gate/up/down) of all 32 layers: 30.5M trainable parameters, 0.72% of the
model. Each was trained for 300 steps. `_s0`, `_s1`, `_s2` are training seeds; seed 0 is the
pre-registered arm.

| Folder | Arm | Training |
|---|---|---|
| `adapters/v1_lora_s{0,1,2}` | J-V1 | 2,400 synthetic decision questions (8 families, difficulties 1-3) in three formats: option scoring, short answer, one-line reasoning |
| `adapters/v1c_lora_s{0,1,2}` | J-V1c | as V1, with a Brier loss on the option distribution |
| `adapters/v1c_lr3_s0` | J-V1c-lr3 | V1c at 3x the learning rate (a control) |
| `adapters/v2b_lora_s{0,1,2}` | J-V2b | as V1, with layers 12-15 run twice in every training forward |
| `adapters/v1_rmp_s{0,1,2}` | J-V1-rmp | V1 recipe on 2,400 RMP train-region rows (pointer chasing, gating, reachability, sudoku4) |
| `adapters/v2b_rmp_s{0,1,2}` | J-V2b-rmp | V2b recipe on the same RMP rows |
| `adapters/multi_lora_s{0,1,2}` | J-multi (SPEC-U1) | V1's recipe on the training splits of the utility suites in their own label spaces: V1's 2,400 decision questions, J-V1-rmp's 2,400 RMP rows, the S2 commit/veto train split (x3), hermes-lite's registered S7 train rows (x4, shuffled shortlists); 714 steps |
| `adapters/multi_loop_s{0,1,2}` | J-multi-loop (SPEC-U1) | J-multi with layers 12-15 run twice in training (use the loop at inference) |

The V2b adapters expect the loop at inference: layers 12-15 run twice. The repo's external
layer-schedule driver does this (`src/jevq/looped.py`, `LoopSpec(12, 16, 2)`). With the stock
forward they behave like a one-pass exit, which is less accurate.

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B-Base", torch_dtype="bfloat16")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base")
model = PeftModel.from_pretrained(base, "AlephFunk/jev-qwen", subfolder="adapters/v1c_lora_s0")
```

The `choice` readout renders the options as single-token labels (" A", " B", ...) in a few-shot
prompt and takes a softmax over those label logits at the answer position: one forward pass,
no emitted tokens. See `src/jevq/xbench/jrunner.py`.

## Headline results (from the paper)

- **Single pass vs reasoning.** V1 scores 0.880 on the 1,000-item decision suite (base model
  0.713). It is within 4.5 points of its own one-line reasoning readout, at 1/21 of the paired
  latency.
- **Calibration.** V1c has a lower ECE than V1 in all three seeds (0.035-0.055 vs 0.069-0.078),
  with no accuracy cost.
- **Recurrence.** Training with the loop on the mixed suite does not beat V1 (falsified in three
  seeds). On RMP rows the loop adds hops within the trained depths in every seed. Neither
  extrapolates past depth 8; only emitted reasoning does.
- **Control.** Under a scripted evidence-injection ladder, V2b at its trained depth flips 0 of 162
  deny decisions over three seeds; V1 flips 14. As monitors on the Control-Harness dev pack, the
  single-pass arms stop 9 of 10 scripted attacks. All of them miss the duplicate-lineage quorum
  attack with full confidence.
- **Reliability.** No Jev-style adapter meets the pre-registered reliability rule on any suite.
  Where the decision state is typed, scripts and 6k-17k-parameter TRMs are smaller and already
  reliable.

## Follow-up U1: training on the target label spaces (three seeds, pre-registered)

| Arm (mean over seeds) | S1 depths 1-8 | S2 | S4 | S7, shuffled |
|---|---|---|---|---|
| V1 | 0.351 | 0.968 | 0.868 | 0.870 |
| J-multi | 0.632 | 1.000 | 0.867 | 0.880 |
| J-multi-loop, 2 passes | 0.691 | 1.000 | 0.864 | 0.898 |

- **Transfer:** J-multi removes most of V1's transfer cost on S1 (Holm p = 4e-82) with no loss on S4 (non-inferior).
- **Calibration:** better than V1 on S4 in every seed.
- **Routing:** it does not route better on shuffled S7.
- **Specialist:** it stays about 2 points below the single-task J-V1-rmp on S1.
- **Loop:** adds S1 accuracy within trained depths (p = 2e-8), unevenly across seeds.
- **Reliability:** no Jev-style adapter meets the pre-registered rule on any utility suite. See `reports/ubench/` on GitHub.

## Limitations

- The tasks are synthetic and templated.
- Two control suites are underpowered (18 and 11 cases in the unsafe class).
- The adaptive attacker was too weak to test adaptive pressure.
- Three seeds per arm. Details and every deviation are in `notes/xbench-log.md` on GitHub.
