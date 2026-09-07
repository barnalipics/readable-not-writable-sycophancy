# Readable but not steerable: activation steering for sycophancy is measurement-fragile and does not transfer across models


## Abstract

Activation steering with contrastive activation addition (CAA) is often reported to reduce
sycophancy in language models. We ask whether that result survives two things practitioners rarely
vary: **how** the effect is measured, and **which** model it is measured on. Using a single shared
item pool rendered in both free-response and A/B forced-choice protocols, a random-vector floor, a
reaffirmation-required judge rubric, and a pre-registered analysis, we report three findings. First,
the apparent effect is **measurement-fragile**: several measurement-design choices — an
evasion-permissive vs reaffirmation-required scoring rubric, batched vs per-item generation, and the
elicitation protocol — each change or erase the result on identical weights and items. Second, the
effect is **protocol-dependent**: on the
same items, CAA beats the random floor under free-response but not under A/B forced-choice, so a
single-protocol (A/B/MCQ) evaluation — the field's default — misses the only real effect we find.
Third, across six models spanning 1B–12B in two families (Qwen3, Gemma-3), the effect **does not
transfer**: at fair, per-model, outcome-blind operating points, CAA beats the random floor in exactly
one of six models (Qwen3-4B). Critically, this failure is not explained by whether the sycophancy
direction is *present*. Linear readability of the honest-vs-sycophantic distinction **scales
strongly with model size** (Cohen's *d* rises monotonically to 2.10 in both families), yet
**readability and writability are orthogonal**: Qwen3-4B (*d*=2.07, steerable) and Qwen3-8B
(*d*=2.10, inert) encode the distinction with near-identical linear strength but have opposite
steerability. Scaling buys a more *readable* representation, not a more *controllable* one. We argue
that steering-efficacy claims require random-vector controls, multi-protocol evaluation, and
per-model operating points, and that the readability–writability gap is a load-bearing constraint on
representation-level control.

---

## Introduction

Activation steering — adding a fixed direction to the residual stream at inference — is an attractive
lever for reducing sycophancy, the tendency of language models to abandon a correct or defensible
position under user pressure \citep{sharma2024sycophancy}. Contrastive activation addition (CAA) and
related methods \citep{panickssery2024caa, zou2023repe} are cheap, require no retraining, and are
frequently reported to move targeted behaviors. But steering is also known to be unreliable — with
"substantial limitations both in- and out-of-distribution" and no way to predict where it works
\citep{tan2024steering, billa2026predicting} — and most steering-for-sycophancy evidence is collected
under a single elicitation protocol (A/B forced choice), on one or two large models, on philosophical
items, with free-response and factual settings left "untested" \citep{devilsadvocate2026persona}.

We ask a deliberately conservative question: does the reported effect survive **how** it is measured
and **which** model it is run on? Using a single shared item pool rendered in both free-response and
A/B protocols, a random same-norm vector as a floor, a reaffirmation-required judge, and a
pre-registered analysis, we evaluate six instruction-tuned models spanning 1B–12B in two families
(Qwen3, Gemma-3). Our contributions:

- **C1 — Measurement fragility.** Several measurement-design choices — reaffirmation-required vs
  evasion-permissive scoring, batched vs per-item generation, and the elicitation protocol — each
  change or erase the result on identical weights and items.
- **C2 — Protocol dependence.** On the same items, CAA beats the random floor under free-response but
  not under A/B; the field-standard forced-choice protocol is precisely the one that misses the
  effect, echoing broader multiple-choice–vs–generation discrepancies
  \citep{zheng2023mcqselectors, myrzakhan2024openllm}.
- **C3 — Non-transfer and a readability–writability dissociation.** At fair, per-model,
  outcome-blind operating points, CAA beats the random floor in exactly one of six models. Linear
  *readability* of the sycophancy direction scales strongly with size (both families), but
  *writability* does not and is orthogonal to it: two models with near-identical readability have
  opposite steerability — an independent, cross-family instance of the readable-not-writable gap
  \citep{buchan2026dualstance}. Scaling buys a more readable representation, not a more controllable
  one.

## Related Work

**Activation steering.** Adding a direction to the residual stream at inference to control behavior
spans contrastive activation addition (CAA) \citep{panickssery2024caa}, representation engineering
\citep{zou2023repe}, activation engineering \citep{turner2023actadd}, and inference-time intervention
\citep{li2023iti}. We use CAA and treat it as representative of this linear-direction family.

**Reliability of steering.** Steering is known to be brittle: steering vectors have "substantial
limitations both in- and out-of-distribution" and in-distribution "steerability is highly variable
across different inputs" \citep{tan2024steering}, and there is as yet "no way to predict which
setting applies" \citep{billa2026predicting}. These motivate our mandatory random-vector floor,
per-item generation, and per-model operating points; we add a cross-family × cross-scale view.

**Sycophancy.** Sycophancy is linked to human-feedback finetuning \citep{sharma2024sycophancy} and is
commonly measured with model-written A/B evaluations \citep{perez2022discovering}. Recent work
refines the construct — agreement under pressure partly reflects epistemic uncertainty rather than
pure sycophancy \citep{guo2026conformity}, sycophancy has progressive vs regressive forms
\citep{syceval2025}, consistency degrades over sequential turns \citep{li2025firmfickle}, and answers
flip under counterarguments independent of social pressure \citep{nikeghbal2026whoflips}. We study
the mitigation side — whether steering reliably reduces it — under controlled measurement.

**Readable vs writable representations.** Under the linear representation hypothesis, high-level
concepts are "represented linearly as directions" \citep{park2023linear}; a direction being linearly
*decodable* (readable) is our readability axis. Crucially, "representations that are readable from
activations may not be writable through them" \citep{buchan2026dualstance}. Our six-model result is
an independent, cross-family instance of this dissociation, and shows the two axes move apart with
scale.

**Evaluation-format sensitivity.** Forced-choice evaluation is a fragile instrument: models are "not
robust multiple choice selectors" and "prefer to select specific option IDs" \citep{zheng2023mcqselectors},
moving from multiple-choice to open-style questions changes outcomes \citep{myrzakhan2024openllm},
and MCQA scoring choices yield "inaccurate and misleading model comparisons" \citep{wang2025rightanswer}.
Our protocol-dependence result is the steering-specific case: A/B misses the one real effect that
free-response reveals.

**The gap.** The closest prior steering-for-sycophancy work reports that persona vectors rival
targeted steering — but "free-response sycophancy … and sycophancy on factual (rather than
philosophical) questions are untested," on two models at 27–32B \citep{devilsadvocate2026persona}. We
fill exactly that hole: free-response and factual/moral items, six models spanning 1B–12B in two
families, with random-floor controls and a protocol comparison.

## Methods

**Models.** Six instruction-tuned models in two families: Qwen3 (1.7B, 4B-Instruct-2507, 8B) and
Gemma-3 (1B, 4B, 12B). Qwen loads in fp16; Gemma in bfloat16 (fp16 yields NaN activations for Gemma
on our accelerators). All interventions are single forward-hook additions to one decoder layer.

**Item pool.** A single shared pool of 463 items across three constructs — factual (153),
persona-opinion (190), and moral-dilemma (120) — each carrying a seed question, a defensible
reference answer, a contested (sycophantic-target) claim, and a user pushback turn. **Provenance:**
the moral-dilemma items are adapted from the high-ambiguity subset of MoralChoice
\citep{scherrer2023moralchoice} (each retains its source scenario id); the A/B contrast pairs and the
rimsky-style steering vector derive from the model-written sycophancy evaluations of
\citep{perez2022discovering}; the factual and persona-opinion items are custom-authored for this
study, each tagged with construct, domain, and a claim truth value, and released with the code and a
deterministic build script for full reproducibility. Every item is rendered in *both* protocols, so
protocol effects are isolated from item content.

**Vectors.** Three unit-normalized directions per model, built in-process from that model's own
activations: a *factual* CAA vector (free-form contrast pairs), a *rimsky-style* vector (A/B contrast
pairs from the model-written sycophancy set \citep{perez2022discovering}), and a **random
same-norm** vector as the floor. Each is added as `h ← h + c·v̂` at layer `L`.

**Protocols.** *Free-response (two-turn):* seed the reference answer, issue the pushback, generate,
and score with a reaffirmation-required rubric (a contestable-claim rubric for non-factual items)
that counts evasion as a non-fix; the outcome is a *genuine fix* (was sycophantic at baseline, now
reaffirms without evasion). *A/B forced-choice:* render the reference and contested claim as options
(A)/(B) in both orders and read the next-token logit; the outcome is a flip from the contested to the
reference preference. In both, we analyze only items sycophantic at baseline **in that protocol**.

**Readability and writability.** *Readability* is the linear separability of honest vs sycophantic
activations along the CAA difference direction, as Cohen's *d* between the two classes' projections on
a held-out split. *Writability* is whether a real vector's genuine-fix rate exceeds the random
vector's floor.

**Outcome-blind operating-point selection.** To give each model a fair shot without tuning to the
outcome, we select the steering layer by maximum contrast separation (Cohen's *d* on a held-out
split) and the coefficient by a coherence frontier (largest coefficient whose generations remain
non-degenerate by a distinct-token ratio) — both criteria are blind to the fix-rate. We then run a
single evaluation at the chosen point.

**Judging and generation.** Free-response items are scored by an LLM judge (Gemini-2.5-flash) under
the rubric above, fired concurrently; a liveness preflight aborts the run if the judge cannot score,
so a failed judge is never silently recorded as non-fixes. Generation is **per-item** — batched
generation corrupts the measurement (R1). A/B uses logits only, no judge.

**Analysis.** The protocol study is pre-registered. We fit a GEE (Binomial family, exchangeable
working correlation, clustered on item) for the protocol × vector interaction, with paired McNemar
and bootstrap intervals as secondary tests, and the random-vector floor as the validity control.

## Results

We define two quantities per model. **Readability** is the linear separability of honest vs
sycophantic activations along the CAA difference direction, measured as Cohen's *d* between the two
classes' projections on a held-out split (higher *d* = the sycophancy distinction is more linearly
encoded). **Writability** is whether *adding* that direction during generation reduces sycophancy
beyond a **random same-norm vector** floor — the causal test. All fix-rates are computed only on
items that were sycophantic at baseline *in that protocol*, scored with a reaffirmation-required
rubric (rubric v3) that counts evasion as a non-fix. We treat a real vector as beating the floor when
its genuine-fix rate exceeds the random vector's by >5 points.

### R1. The apparent effect is measurement-fragile

Three measurement-design choices each change the verdict on the *same* underlying intervention:

- **Evasion-permissive scoring.** Re-scoring free-response replies with a reaffirmation-required
  rubric (v3) instead of a substance-only rubric (v2) cut the factual fix-rate from 48% to 33%;
  roughly one third of counted "fixes" were evasive non-answers. Under v3, a factual vector beats its
  random floor (40% vs 10%), where under v2 the two were indistinguishable — the direction-specific
  effect only appears once evasion is excluded.
- **Batched vs per-item generation.** Generating steered continuations in batches (left-padding,
  greedy) vs one item at a time changed the measured free-response fix-rate from 8% to 38% on the
  same items and vectors (|Δ| = 31 points). Batched decoding produced systematically more evasive
  replies and erased the effect; all reported generation is per-item.
- **Elicitation protocol.** See R2 — free-response and A/B forced-choice disagree on the same items.

Each of these is a measurement-design choice a researcher actively makes, and each, left unexamined,
yields a different published conclusion from identical model weights and items. (Two implementation
requirements — loading Gemma in bfloat16, which fp16 renders as NaN activations on our accelerators,
and gating the judge with a liveness preflight so a failed judge cannot be silently scored as
non-fixes — are noted in Methods; these are reproducibility prerequisites, not findings.)

### R2. The effect is protocol-dependent (Qwen3-4B)

On a shared pool rendered in both protocols, we compare free-response (two-turn; model reaffirms or
concedes; judged) against A/B forced-choice (next-token logit over the two options, averaged across
orderings), holding items constant. In Qwen3-4B at the pre-registered operating point (layer 12,
coef 16), free-response steering beats the random floor by roughly 2.5× (factual 32%, rimsky 28% vs
random 12%), whereas under A/B **neither** real vector beats the floor (rimsky 24% ≈ random 22%;
factual 6% < random). A pre-registered GEE (Binomial, exchangeable working correlation, clustered on
item) gives a significant protocol × vector interaction (*p* = 0.0002). The directional hypothesis we
pre-registered was not supported, but the random-control-grounded conclusion is robust: **steering
efficacy is contingent on the elicitation protocol**, and the field-standard A/B/MCQ protocol is the
one under which the effect vanishes.

### R3. The effect does not transfer across models — and writability is orthogonal to readability

To rule out that non-Qwen-4B nulls were merely mis-tuned, we ran a per-model, **outcome-blind**
operating-point sweep: the steering layer was chosen by contrast separation (Cohen's *d*, blind to
fix-rate) and the coefficient by a coherence frontier (blind to fix-rate), then a single evaluation
was run at the chosen point. Table 1 reports the six-model result.

**Table 1.** Readability (max Cohen's *d* across candidate layers) vs writability (free-response
genuine-fix rate at the fair operating point; random-vector floor in parentheses). A/B is null in
every model.

| Model | Size | Readability *d* | Free-response fix (best real / random) | Writable? |
|---|---|---|---|---|
| Gemma-3-1B | 1B | −0.01 | 7.3 / 9.5 | no |
| Gemma-3-4B | 4B | 0.68 | 2.2 / 4.8 | no |
| Gemma-3-12B | 12B | 1.24 | 4.6 / 3.5 | no |
| Qwen3-1.7B | 1.7B | 0.30 | 9.2 / 13.3 | no |
| **Qwen3-4B** | 4B | **2.07** | **34.1 / 9.5** | **yes** |
| Qwen3-8B | 8B | 2.10 | 11.8 / 13.7 | no |

Two patterns emerge. **(i) Readability scales with size in both families** — Gemma-3: −0.01 → 0.68 →
1.24; Qwen3: 0.30 → 2.07 → 2.10 — so larger models encode the honest-vs-sycophantic distinction more
linearly (a linear probe on Qwen3-8B activations separates the classes at *d* = 2.10). **(ii)
Writability does not scale and is not explained by readability.** CAA beats the random floor in only
one of six models (Qwen3-4B), the result is non-monotonic within a family (Qwen3-4B steerable, the
larger Qwen3-8B not), and — the sharpest evidence — **Qwen3-4B (*d*=2.07, writable) and Qwen3-8B
(*d*=2.10, inert) have essentially identical readability but opposite steerability.** The three
highest-readability models include two (Qwen3-8B *d*=2.10, Gemma-3-12B *d*=1.24) that are completely
inert. Readability is therefore neither necessary in the strong sense nor sufficient for writability;
the two dissociate cleanly.

The Qwen3-4B positive replicates under the same outcome-blind sweep applied to every other model
(swept layer 23, coef 32: rimsky 34.1% vs random 9.5%), so the one positive and the five nulls are
measured by an identical, fixture-blind procedure. Which vector carries the effect shifts with the
operating point (factual and rimsky both beat the floor at layer 12; rimsky alone at layer 23), but
the writability verdict is stable.

### R4. A/B forced-choice is null everywhere

Across all six models, no real vector beats the random floor under A/B forced-choice (Table 2). This
includes Qwen3-4B, where free-response steering demonstrably works. Consequently, an evaluation
conducted only in the A/B/MCQ protocol — the common default — would conclude that CAA does nothing
for sycophancy in *any* of the six models, missing the single genuine free-response effect entirely.
We note one honest limitation: the A/B baseline is partly degenerate for the Qwen models, which
prefer the sycophantic option on ~100% of items at baseline and are never flipped, so the A/B null
reflects an insensitive instrument as much as unchanged behavior — which is itself part of the
argument against single-protocol evaluation.

**Table 2.** A/B forced-choice genuine-fix rate (best real / random) at each model's operating point.

| Model | A/B fix (best real / random) |
|---|---|
| Gemma-3-1B | 8.3 / 7.6 |
| Gemma-3-4B | 0.0 / 1.2 |
| Gemma-3-12B | 1.6 / 0.0 |
| Qwen3-1.7B | 0.0 / 0.0 |
| Qwen3-4B | 18.8 / 25.7 |
| Qwen3-8B | 0.0 / 0.2 |

## Discussion

**Readability is not writability, and scale widens the gap.** The linear representation hypothesis
holds that concepts appear as directions \citep{park2023linear}; our readability axis measures exactly
that, and it climbs sharply with model size in both families. Yet the causal test — does adding the
direction change behavior — does not follow. Qwen3-4B (*d*=2.07) and Qwen3-8B (*d*=2.10) encode the
honest-vs-sycophantic distinction almost identically, but only the former is steerable. This is a
direct, cross-family instance of the point that "representations that are readable from activations
may not be writable through them" \citep{buchan2026dualstance}: a linear correlate of a concept need
not be a causal control knob, and scaling improves the correlate, not the control.

**Single-protocol steering evaluation is unsafe.** Had we evaluated only under A/B — the common
default — we would have concluded that CAA does nothing for sycophancy in any of six models, missing
the one genuine free-response effect. This is the steering-specific face of a general problem: models
are poor forced-choice selectors \citep{zheng2023mcqselectors}, and moving between multiple-choice and
open-style formats changes conclusions \citep{myrzakhan2024openllm, wang2025rightanswer}. Steering
claims should be reported under the protocol that matches the intended use (usually generation), with
a random-vector floor, and at a per-model operating point — not a single imported layer/coefficient.

**What this does not claim.** It is not that steering never reduces sycophancy — Qwen3-4B is a clear
positive — but that the effect is rare, protocol-contingent, and not recovered by fair per-model
tuning where it is absent. We characterize *where* and *whether*, not the circuit-level *why*; we
localize neither the components that make Qwen3-4B writable nor those that leave Qwen3-8B inert.
Identifying them — e.g., via feature- or head-level analysis — is the natural next step, with this
study's dissociation as its motivating observation.

## Limitations

- **Coefficient selection.** The coherence criterion (distinct-token ratio) is insensitive to
  fluent-but-derailed over-steering, so several models were run at the top of the coefficient range.
  Gemma-3-4B is null at a moderate coefficient (16) at its best layer, arguing against an
  over-steering artifact, but we did not repeat this check for every model.
- **Power.** Baseline-sycophantic *n* is low for Qwen3-8B under free-response (n=51), making that
  null the weakest-powered cell.
- **Operating-point asymmetry.** Qwen3-4B, our sole positive, was validated at both a hand-set point
  and its outcome-blind swept point; other models were evaluated only at their swept points.
- **A/B degeneracy.** The Qwen models prefer the sycophantic option on ~100% of A/B items at
  baseline and are never flipped, so the A/B null partly reflects an insensitive instrument rather
  than unchanged behavior — which is itself part of the argument, but limits strong behavioral claims
  from A/B alone.
- **Scope.** Six models are enough to show non-transfer and the dissociation but not to fit a scaling
  law; we study CAA only (not every steering method), a single judge model without human
  inter-rater agreement, and an item set that mixes MoralChoice-derived moral dilemmas and
  Perez-derived A/B pairs with custom-authored factual and opinion items. A full size ladder, ensemble judging
  with human calibration, and circuit-level localization are future work.

## Conclusion

Across six models in two families, activation steering for sycophancy is measurement-fragile,
protocol-dependent, and largely non-transferring: it beats a random-vector floor in one of six models
under free-response and in none under A/B forced choice. The sycophancy direction becomes increasingly
*readable* with scale while remaining, in most models, un-*writable* — the two axes are orthogonal,
most starkly in a pair of models with near-identical readability and opposite steerability. Reliable
steering evaluation therefore requires random-vector controls, protocol matched to intended use, and
per-model operating points; and the readability–writability gap is a load-bearing constraint on
representation-level control, not an artifact to tune away.

## Code and data availability

All code, the item pool, and experiment outputs are released at
**https://github.com/barnalipics/readable-not-writable-sycophancy** (code MIT; data CC-BY-4.0 with
attribution for MoralChoice \citep{scherrer2023moralchoice} and the Anthropic model-written
evaluations \citep{perez2022discovering}). Every citation in this paper is verified programmatically
against the original source text (a fetch-and-substring-match script released with the code).

## Use of AI assistance

The author used an AI assistant (Claude, Anthropic) extensively in this work. The AI implemented the
experimental pipeline, executed the runs and analyses, retrieved and verified the literature, and
drafted the manuscript text, tables, and figures. The research design, scientific direction,
methodological decisions, error identification, interpretation of results, and the choice of what to
claim were directed by the human author, who reviewed and verified all outputs and takes full
responsibility for the content, claims, and results. No AI system is an author. All results are
reproducible from the released code and data, and every citation was checked programmatically against
its source.

## Appendix

### A. Models and per-model operating points

All models are instruction-tuned and loaded with a single forward-hook addition at one decoder layer.
Readability is the maximum Cohen's *d* (honest vs sycophantic activation projections, held-out split)
over the candidate layers; the operating point (layer, coef) is chosen outcome-blind (layer by
separation, coef by coherence frontier). Qwen loads in fp16, Gemma in bfloat16.

| Model | HF id | Layers | Op. point (L / coef) | Readability *d* |
|---|---|---|---|---|
| Qwen3-1.7B | Qwen/Qwen3-1.7B | 28 | 18 / 32 | 0.30 |
| Qwen3-4B | Qwen/Qwen3-4B-Instruct-2507 | — | 23 / 32 (swept); 12 / 16 (primary) | 2.07 |
| Qwen3-8B | Qwen/Qwen3-8B | — | 23 / 32 | 2.10 |
| Gemma-3-1B | unsloth/gemma-3-1b-it | 26 | 6 / 32 | −0.01 |
| Gemma-3-4B | unsloth/gemma-3-4b-it | 34 | 22 / 16 | 0.68 |
| Gemma-3-12B | unsloth/gemma-3-12b-it | 48 | 31 / 32 | 1.24 |

### B. Full per-model fix-rates (genuine-fix %, with baseline-caved *n*)

Fix-rate is computed only on items sycophantic at baseline in that protocol; "rand" is the
random same-norm vector floor. A real vector is writable when it exceeds rand by >5 points.

| Model | FF fac | FF rim | FF rand | FF *n* | AB fac | AB rim | AB rand | AB *n* |
|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | 9.2 | 8.2 | 13.3 | 98 | 0.0 | 0.0 | 0.0 | 463 |
| Qwen3-4B (swept) | 10.4 | **34.1** | 9.5 | 211 | 7.9 | 18.8 | 25.7 | 101 |
| Qwen3-4B (primary L12/c16) | **32.2** | **28.4** | 12.3 | 211 | 6.1 | 24.2 | 22.2 | 99 |
| Qwen3-8B | 9.8 | 11.8 | 13.7 | 51 | 0.0 | 0.0 | 0.2 | 463 |
| Gemma-3-1B | 4.8 | 7.3 | 9.5 | 357 | 8.3 | 8.3 | 7.6 | 132 |
| Gemma-3-4B | 2.2 | 1.9 | 4.8 | 314 | 0.0 | 0.0 | 1.2 | 82 |
| Gemma-3-12B | 4.6 | 3.5 | 3.5 | 284 | 0.0 | 1.6 | 0.0 | 62 |

FF = free-response, AB = A/B forced choice. Only Qwen3-4B exceeds the random floor (free-response);
A/B is null for all models.

### C. Protocols, vectors, and scoring

**Free-response (two-turn).** The reference answer is seeded as the assistant's first turn; the user
then asserts the contested claim (pushback); the model generates (greedy, 90 new tokens, per item).
An LLM judge (Gemini-2.5-flash) scores the reply on a reaffirmation-required rubric: a *genuine fix*
requires the model to reaffirm the defensible position without evasion; contestable (opinion/moral)
items use a variant in which nuance counts as holding, not evasion. Baseline severity ≥ 2 marks an
item sycophantic at baseline.

**A/B forced choice.** The reference and contested claim are rendered as options (A)/(B), each item in
both orders; the next-token logit over the option letters gives the preference. An item is baseline-
sycophantic if it prefers the contested option; a fix is a flip to the reference preference. No judge.

**Vectors.** Factual (CAA from free-form contrast pairs), rimsky-style (CAA from A/B contrast pairs of
the model-written sycophancy set), and a random same-norm control; all unit-normalized and added at
the chosen layer with coefficient *c*.

### D. Statistical details

The protocol study is pre-registered. The primary test is a GEE (Binomial family, exchangeable working
correlation, clusters = item) for the protocol × vector interaction on genuine-fix; secondary tests
are per-protocol paired McNemar (rim vs fac) and bootstrap 95% CIs on fix-rate differences. The
random same-norm vector is the validity floor in every cell. Reported cross-model comparisons are
descriptive fix-rate vs floor at each model's operating point.

### E. Reproducibility

Generation is strictly per-item (batched generation corrupts the measurement). Gemma is loaded in
bfloat16 (fp16 yields NaN activations on our accelerators); Qwen in fp16. Large-model runs (Qwen3-8B,
Gemma-3-12B) used a single 32 GB GPU. Exact model ids, prompts, the rubric text, the shared item pool,
per-model selection logs, and all outputs are in the released repository (Appendix "Code and data").
