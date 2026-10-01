# 001 - What is publicly known about Jev (collected 2026-10-01)

Jev is a hosted, proprietary model from TypeSafe AI, announced 2026-09-15. No weights, no
technical report. Everything below comes from web pages read on 2026-10-01; third-party pages
were read through an automated summariser, so wording is paraphrased unless quoted.

## Tier A - stated by TypeSafe (primary source)

Source: https://typesafe.ai/blog/introducing-system-one-models-and-jev

- A "System One Model": "unstructured state in, typed probabilistic decisions out".
- "a new model architecture, parallel sampler for maximum efficiency"; it "generates all
  outputs in a single query".
- Training method: "Reinforcement Learning for Calibrated Decisions (RLCD)", aiming at
  "answers with epistemically honest probabilities on System One tasks".
- Outputs and their structure are defined in advance; the model "never makes type errors";
  it "gives up string generation". Cardinality up to 255.
- "End-to-end response time is 70ms-500ms"; "40x-200x faster" than LLMs at similar quality on
  System One tasks. Vendor-measured; TypeSafe notes its workflow evals were written by its own
  team and that the reference answers were an average of two frontier LLMs.
- Price: $0.042 per million input tokens; output not metered.

**Not stated by TypeSafe:** parameter count, backbone or base model, whether any recurrent,
looped or latent computation is used, number of forward passes.

## Tier B - third-party observation with a described method

- **Behavioural reverse engineering** (https://archerhume.com/posts/jevs-architecture-unmasked/):
  shared state encoded once with isolated question branches (token accounting is additive; one
  question cannot read another's text); server time nearly flat up to roughly 100 questions;
  options must be present before the decision; no token-by-token decoding; no sign of diffusion,
  iterative refinement, recurrence or latent loops. Calibration on a 1,200-item MMLU sample:
  ECE 0.031. Backbone guess (causal decoder, sparse MoE, ~10B active parameters from latency) is
  the author's inference and is marked low/medium confidence by the author.
- **Security testing** (https://blog.checkpoint.com/ai-security/jev-is-not-a-language-model-but-it-breaks-like-one-prompt-injection-against-a-typed-decision-model/):
  evidence-style injections (documents that report warning signs as already resolved) flipped
  Jev's verdict in 59% of runs; marking documents untrusted or adding anti-injection
  instructions barely helped; a comparison LLM dropped from 67% to 19% attack success when
  reasoning was enabled. Their conclusion: typed output constrains what a model can say, not
  what it can be convinced of.

## Tier C - commentary; not relied on

- https://www.datacamp.com/blog/system-one-models-jev (overview; repeats vendor numbers)
- https://www.explainx.ai/blog/how-does-jev-work-rlcd-system-one-model-explained-2026
- https://arxiviq.substack.com/p/jev-and-the-emergence-of-system-one (presents MoE routing,
  REINFORCE with a group-mean baseline, proper scoring rules, and a list of arXiv ids as fact;
  none of it was traced to a primary source here)

These disagree with each other on basics. Example: one says the tokenizer matches no public
tokenizer, another that it tracks OpenAI's o200k, another speculates Qwen lineage. Treat every
architecture detail beyond Tier A as unknown.

## What this means for the experiment

| Jev property | Evidence | In this project |
|--------------|----------|-----------------|
| Typed decision from a single pass | Tier A | `choice` readout on every variant |
| Calibrated probabilities | Tier A | ECE / Brier / NLL on every `choice` run; proposed V1c |
| Shared state, parallel independent questions | Tier B | few-shot block is a shared prefix; parallel branches are future work |
| Recurrent / latent computation | **none; Tier B reports its absence** | V2-V4 are our hypotheses, not a reproduction |
| Susceptible to evidence-style injection; reasoning helps | Tier B | control tasks with override notes; H7 |

Follow-up worth doing before M3: read the arXiv papers that reference Jev (e.g. the RLCD and
Jev-as-a-judge preprints named in the Tier C commentary) and promote anything verifiable.
