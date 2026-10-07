# SPEC-U3: repair-rudder replication, and an abstain path for the TRM router

Registered 2026-10-07, before any U3 arm was run. Your request that day: "2. and mostly I want to
compare the TRM Repair modules from the infused skills in performance". When I reported that no
trained repair TRM exists, you chose the repair-rudder replication. Changes after registration are
dated addenda.

## 0. What was looked at before registration

- **Repair rudder.**
  - The published per-arm and per-split summaries of the Hermes-Skills repair-rudder runs.
  - The label structure of the 88 evaluation rows: counts of repair action x commit action x failure
    label.
  - The published runner, `research/scripts/run_3b_repair_training_rudder_benchmark.py`.
- **U2.** The results of every U2 pipeline, including B (TRM router + TRM gate), which fails all 90
  negatives.
- **Not run.** No Jev, TRM, kNN or lookup arm has been run on the rudder rows. The abstain head has
  not been trained.

## 1. What exists, and what this compares

**Repair in Hermes-Skills.**
- The TRM-infused skills repair with deterministic code. Examples are intellect3-logic's `c_repair`
  and `dual_repair` (minimum-edit projections onto the tent row and column counts) and the canonical
  re-writers of the structured skills.
- The learned or prompted part is the **repair rudder**: from the pre-repair state, it chooses which
  repair action to apply and whether to commit the result.
- Hermes-Skills benchmarked rudders on the near-miss repair curriculum, on the 88 non-train rows of
  `research/generated/near_miss_repair_curriculum/splits`:

| split | rows | content |
|---|---|---|
| val_seen | 34 | intellect3-logic |
| holdout_seen | 36 | intellect3-logic |
| holdout_unseen_family | 18 | six other skill families, failure labels absent from train |

- **Rudders tested so far:**
  - Qwen2.5-3B-Instruct, Qwen3.5-9B and Qwen3.5-27B (Q4 GGUF);
  - MeTTa rule gates.
- **The trained repair TRM** (`candidate_trainer_specs.json`, spec `repair_verifier_logic_c_signature`)
  was planned but never trained. The runner says so: "It does not claim trained repair-TRM weights
  exist yet."

U3-R replicates that benchmark with the Jev, the generic small Qwen and the missing repair TRM added,
and imports the published rows for paired comparison.

**Properties of the benchmark, stated now:**
- **Leaky retrieval.** The published `repair_training_rudder` arm retrieves its four examples with
  the eval row's own `bucket`, `action` and `target_action` in the match. Its examples therefore tend
  to share the answer. I keep that protocol, but only as a labelled replication arm, and add a
  leak-free retrieval.
- **Unattainable unseen-family actions.** The allowed repair actions are the six seen in train. The
  18 unseen-family rows need actions outside that list, so only the MeTTa rules (hand-written per
  family) can be right on them. Any arm that chooses from the allowed list is capped at 70/88 joint
  accuracy, 36/54 on the holdout rows.
- **Outcome access.** `metta_static_gate_rudder` and `metta_validator_gate` read the row's `bucket`,
  which is the repair outcome. They are reported as published ceilings, not as pre-repair rudders.
- **Lookup structure.** In-family repair actions follow from `failure_label` and `after_arm`. Commit
  actions follow from `failure_label`, except on `c_signature_fail` rows, where the outcome decides
  and the pre-repair state may not show it.

## 2. Part R: repair rudder

**Rows and scoring.**
- The 88 rows above, scored as published:
  - repair action correct;
  - commit action (`commit` or `reject_or_abstain`) correct;
  - joint (both correct).
- Added: **unsafe**, a commit on a row whose target is `reject_or_abstain`.
- **Registered tests use the 54 holdout rows.** val_seen is reported separately, because J-multi used
  it as validation data and the repair TRM selects its epochs on it.

**Protocols** (prompt modes, all with the published state fields: env_family, trm_role, before_arm,
after_arm, before_reward, failure_label, candidate_excerpt, route_gate, validate_gate):

| Protocol | Repair action | Examples | Published counterpart |
|---|---|---|---|
| raw | chosen by the arm | none | `raw_3b_rudder` |
| retrieval | chosen by the arm | 4 train rows, retrieved on pre-repair fields only | (new, leak-free) |
| retrieval-published | chosen by the arm | 4 train rows, published retrieval (leaky) | `repair_training_rudder` |
| action-space | fixed by the MeTTa rule | none | `metta_action_space_rudder` |
| static-gate | MeTTa rule; the commit action from the MeTTa static rule where it fires (it reads `bucket`), else the arm's action-space answer | none | `metta_static_gate_rudder` |

**Jev-family readout.**
- The Jev arms use the `choice` readout in two steps:
  1. the repair action, as a lettered list of the six allowed actions in a per-row shuffled order;
  2. commit or reject, with the arm's own chosen repair shown (in action-space, the MeTTa repair is
     shown).
- The labels are the S2 ones (`commit` / `reject`). This splits the published single JSON answer
  into two choices with the same content.

**Arms.**

| Arm | What it is | Instances |
|---|---|---|
| J-V0 | generic small Qwen (Qwen3.5-4B-Base) | 1 |
| J-multi | the Jev (SPEC-U1 adapter) | seeds 0-2 |
| J-multi-loop | the looped Jev, 2 passes | seeds 0-2 |
| Repair-TRM | the planned repair TRM | seeds 0-2 |
| kNN | k = 5 on the same features | 1 |
| lookup | the majority repair and commit action per (failure_label, after_arm repair keyword) cell in train, falling back to the global majority | 1 |
| 3B / 9B / 27B | the published rows, imported unchanged | 1 each |
| MeTTa rules | recomputed here; they must equal the published rows | 1 |

- **Repair-TRM details.**
  - It is the TinyTRM structure of TRM-cv (input projection, four recursive residual steps, layer
    norm, hidden size 2048) with two heads: repair action (6) and commit action (2).
  - It is trained on the 156 train rows.
  - Its features are one-hots of env_family, trm_role, before_arm and failure_label over the train
    vocabulary plus an unknown slot, the repair keyword in after_arm, and before_reward.
  - Epochs are chosen from {8, 25, 50, 100} by val_seen joint accuracy.
- **The 3B, 9B and 27B rows** are imported from Hermes-Skills `HEAD` (`git show`, read-only). The
  local 3B rows are equal on disk.

**Registered tests** (joint accuracy on the 54 holdout rows; pooled discordant (instance, row) pairs;
exact binomial; a single-instance arm is paired with every instance of the other; Holm over R1-R3 at
alpha 0.05):

| ID | Claim | Test |
|---|---|---|
| R1 | J-multi (retrieval) > J-V0 (retrieval) | one-sided |
| R2 | J-multi (retrieval) vs Repair-TRM | two-sided |
| R3 | J-multi (retrieval-published) vs published Qwen3.5-27B `repair_training_rudder` (same protocol) | two-sided |

Descriptive, for every arm and protocol:
- the three accuracies and unsafe rate on all 88 rows and per split;
- the structural ceilings;
- whether the published-retrieval port reproduces the published `retrieved_case_ids`.

## 3. Part A: an abstain path for the TRM router

**The gap.**
- hermes-lite's `TinyRecursiveSkillRouter` (6,338 parameters) already has an `abstain_head`.
- Training never fits it: the loss covers the score head only.
- `trm_typed` ignores it.
- This is why pipeline B chose the forbidden skill on every U2 negative.

**SkillRouter-TRM+abstain** (new):
- **Model.** The registered checkpoint (`trm_router.pt`), unchanged, with only `abstain_head` fitted.
- **Training rows.**
  - hermes-lite's registered S7 `train_rows.jsonl` (34 positives, label 0; 23 negatives, label 1),
    which J-multi also trained on;
  - each contract's `route.positive_examples` (label 0), the router's own training rows.
- **Fit.** Each row is scored over the same lexical top-5 shortlist the router sees at evaluation.
  The fit is full-batch logistic loss for 500 AdamW steps (lr 1e-2), so the result is deterministic.
- **Decision.** Abstain iff the head's probability is > 0.5; otherwise the TRM router's choice stands.
- **Exclusions.** No U2 item and no S7 held case is used.
- **Disclosed in advance.** Four of the six U2 negative templates use the S7 training negatives'
  phrasing ("Do not activate X; the request explicitly asks for ..."), so part of U2 is near the
  training distribution. That holds for J-multi too.

**SkillRouter-TRM+threshold** (descriptive): abstain iff the router's top probability is < tau, with
tau chosen to maximise abstain accuracy on the same training rows.

**Pipelines** (U2 items, scored by `u2_joint.end_to_end`):
- **B-abstain:** SkillRouter-TRM+abstain picks, TRM-cv decides.
- **B-threshold:** SkillRouter-TRM+threshold picks, TRM-cv decides.

**Registered tests** (end-to-end success on the 600 U2 items; pooled as in SPEC-U2; Holm over A1-A2):

| ID | Claim | Test |
|---|---|---|
| A1 | B-abstain > B | one-sided |
| A2 | B-abstain vs H (Jev picks, TRM gate decides) | two-sided |

Descriptive:
- route accuracy by kind (joint, confusable, negative);
- unsafe rate;
- the 128 registered S7 held cases for both new routers: pass rate on positives and negatives.

## 4. Limitations, stated now

- The rudder rows are post-hoc projections from earlier runs (evidence class `post_hoc_projection`),
  and holdout_seen has 36 rows. Differences of a few rows are within noise.
- The repair choice is close to a lookup, so the benchmark tests label-space mapping more than
  repair skill. Unseen-family repair actions are unattainable for every arm except the MeTTa rules.
- J-multi was trained on the S2 near-miss train rows. Those rows hold the same cases' commit labels,
  with post-repair information. Repair-TRM, kNN and lookup train on the same rows; the published LLM
  rudders saw them only as retrieved examples.
- The abstain head is trained on 57 + 75 short rows. Its negatives follow one template.

## 5. Order

- **Code.** Repair rudder: `src/jevq/xbench/u3_rudder.py`. Abstain path: `src/jevq/xbench/u3_abstain.py`.
- **Report.** `scripts/u3_report.py`, written before any run, produces `reports/ubench/u3.md`.
- **CPU steps** (Repair-TRM, kNN, lookup, MeTTa rules, receipt import, abstain head, threshold) run
  first.
- **GPU queue.** `configs/queues/u3.yaml` runs J-V0, J-multi seeds 0-2 and J-multi-loop seeds 0-2.
  It waits for an idle GPU.

## Addenda

(none)
