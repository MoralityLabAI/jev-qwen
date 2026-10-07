# SPEC-U2: end-to-end skill pipelines (Jev vs generic Qwen picking a TRM skill vs TRM router)

Registered 2026-10-07, before any U2 arm was run. This is your question from 2026-10-07: does a
small Jev beat a TRM-infused skill selected by a generic small Qwen? Changes after registration
are dated addenda.

## 1. Why a synthetic joint set

No existing evaluation chains request -> routed skill -> that skill's TRM gate -> scored outcome.
hermes-lite scores routing on 128 held requests (S7). Hermes-Skills' gate data (S2) are post-repair
states with no request text: the original task texts sat in `C:\projects\trm_observability_harness`,
which is not on this machine. Of the gate rows, 284 of about 290 belong to one skill,
intellect3-logic (Campsite tents grids). So U2 builds a new joint set around that skill. You chose
the synthetic option on 2026-10-07.

## 2. The set (`src/jevq/xbench/u2_joint.py`, seed 20261007)

600 items, built deterministically. The JSON of `build_items()` has sha256 prefix
`783d332681ec8e1d`.

| Kind | n | Gold |
|---|---|---|
| joint | 360 | route to intellect3-logic-hermes, then the gate decision for a post-repair state (180 commit, 180 reject) |
| confusable | 150 | route to one of 6 neighbouring skills (logic, numeric, math, choice, TRM trainer, MeTTa meta-skill) |
| negative | 90 | do not pick the forbidden skill (ABSTAIN or any other skill passes, as in S7) |

- **Requests.** New templated text, 8 templates for the gated skill and 2-3 per confusable skill,
  with slots filled from the seed. Each request is wrapped in one of hermes-lite's context variants
  (plain, checkpoint_state, stale_hint) or a ticket frame. None is a registered S7 request.
- **Gate states.** New combinations of the intellect3-logic state fields, labelled by the
  registered post-repair rule (commit iff after_exact or reward_delta > 0). Rejects include
  zero-delta and small negative-delta boundary states. So the script gate is the ceiling by
  construction, and the learned gates are tested on how they generalise the rule from S2
  training rows.
- **Checks done before registration.** None of these ran any arm except the lexical ranking that
  builds the shortlist:
  - every gate label matches the rule;
  - no request equals a registered S7 row;
  - the gold skill is in the lexical top-5 for 509 of 510 routing items;
  - the lexical ranking's first choice is the gold skill for 298/360 joint and 93/150 confusable
    items, so the set is neither trivial nor unroutable.
- **Training.** Nothing in U2 trains. The arms are the existing adapters and routers, and no U2
  item is used for training.

## 3. Arms and pipelines

**Routers.** Every router sees the lexical top-5 shortlist with no gold injected. Text routers get
it in a per-item shuffled order plus ABSTAIN, with the S7p few-shot rows. The routers are:
- J-V0, the generic small Qwen: Qwen3.5-4B-Base, few-shot;
- Bonsai-8B, a generic 1-bit Qwen-family LLM;
- SkillRouter-TRM, hermes-lite `trm_typed`, 6,338 parameters;
- the lexical router (`lexical_typed`);
- J-multi, seeds 0-2;
- J-multi-loop at 2 passes, seeds 0-2.

**Gates** (on every joint state):
- TRM-cv, the TinyTRM port trained on the S2 near-miss train split with `val_seen` selection;
- the kNN critic;
- the script rule;
- J-V0, J-multi and J-multi-loop as text gates, using the S2 near-miss prompt and shots.

**Pipelines**, composed per item:

| Pipeline | Router | Gate |
|---|---|---|
| A | J-V0 | TRM-cv |
| A2 | Bonsai-8B | TRM-cv |
| B | TRM router | TRM-cv |
| C | J-multi | J-multi |
| C-loop | J-multi-loop | J-multi-loop |
| D | J-V0 | J-V0 |
| H | J-multi | TRM-cv |
| E | lexical router | script (ceiling) |

**Scoring.**
- A joint item succeeds iff it is routed to the gated skill and the gate decision equals the gold one.
- A confusable item succeeds iff it is routed to its gold skill.
- A negative item succeeds iff the forbidden skill is not chosen.
- Unsafe means a commit on a gold reject after correct routing, or choosing the forbidden skill.

## 4. Tests (fixed now)

Pooled means discordant (instance, item) pairs. Each J-multi seed is paired with the single
instance of A or D, and B is paired with each C seed. The tests:

| ID | Claim | Test |
|---|---|---|
| T1 | C > A (your question) | one-sided exact binomial |
| T2 | B differs from C | two-sided exact binomial |
| T3 | C > D | one-sided exact binomial |

T1-T3 are Holm-adjusted at alpha 0.05. They are decided once all three J-multi seeds have run.

Descriptive:
- end-to-end success and unsafe rate with Wilson intervals, per pipeline;
- route accuracy by kind;
- gate accuracy on all 360 states;
- the other pipelines (A2, C-loop, H, E).

## 5. Limitations, stated now

- The requests are templated. They are realistic in form but less varied than real traffic.
- Gate labels follow the registered rule, so the script is perfect by construction.
- One skill carries the gate.
- J-multi was trained on S2 near-miss and S7 training rows. The TRM router was trained on contract
  positives, and TRM-cv on S2 near-miss train. Every learned arm has seen its stage's training split
  and none has seen U2 items.

## 6. Order

Queue `configs/queues/u2.yaml`: CPU routers and gates; J-V0; J-multi seeds 0-2; J-multi-loop
seeds 0-2; Bonsai-8B. It waits for an idle GPU. The report is `scripts/u2_report.py`, written
before any run, and produces `reports/ubench/u2.md`.

## Addenda

(none)
