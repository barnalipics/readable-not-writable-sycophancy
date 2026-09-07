# Readable but not steerable

**Activation steering for LLM sycophancy is measurement-fragile and doesn't transfer across models; readability ≠ writability.**

Code and data for the paper *"Readable but not steerable: activation steering for sycophancy is
measurement-fragile and does not transfer across models."*
Barna Lipics (Eötvös Loránd University). Preprint: _arXiv link TBA_.

---

## TL;DR

We test whether contrastive activation addition (CAA) really reduces sycophancy, varying **how** it is
measured and **which** model it runs on. Across six instruction-tuned models (Qwen3 and Gemma-3, 1B–12B),
with a random-vector floor, a reaffirmation-required judge, and a pre-registered analysis:

1. **Measurement-fragile** — the scoring rubric, batched-vs-per-item generation, and elicitation protocol each change or erase the result on identical weights and items.
2. **Protocol-dependent** — CAA beats the random floor under free-response but **not** under A/B forced choice; single-protocol (A/B/MCQ) evaluation misses the only real effect.
3. **Doesn't transfer + a readability–writability dissociation** — at fair, per-model, outcome-blind operating points, CAA beats the random floor in **1 of 6** models. Linear *readability* of the sycophancy direction scales with size; *writability* does not and is orthogonal to it.

| Model | Size | Readability *d* | Free-response writable? |
|---|---|---|---|
| Gemma-3-1B | 1B | ~0 | no |
| Gemma-3-4B | 4B | 0.68 | no |
| Gemma-3-12B | 12B | 1.24 | no |
| Qwen3-1.7B | 1.7B | 0.30 | no |
| **Qwen3-4B** | 4B | **2.07** | **yes** |
| Qwen3-8B | 8B | 2.10 | no |

Qwen3-4B (*d*=2.07, steerable) and Qwen3-8B (*d*=2.10, inert) encode the distinction almost identically
yet steer oppositely — scaling buys a more *readable* representation, not a more *controllable* one.

## Repository layout

```
scripts/     pipeline (build pool, run models, sweep operating points, analysis) + paper/citation tooling
src/         steering (caa_ab.py) and judge (judge.py)
data/        shared item pool + per-construct eval sets + CAA contrast pairs  (see DATA_LICENSE.md)
results/     experiment outputs (multimodel_flip.jsonl, sweep_*.jsonl, analysis json)
writeup/     paper (paper_draft.md, paper.tex, paper.pdf) + verified citations
```

## Setup

```bash
pip install -e .          # or: uv sync
cp .env.example .env      # add your GEMINI_API_KEY (needed for free-response judging)
```

## Reproduce

```bash
# 1. (re)build the shared item pool
python scripts/35_build_shared_pool.py

# 2. run a model through both protocols (fixed operating point)
MODEL="Qwen/Qwen3-4B-Instruct-2507" python scripts/38_multimodel_run.py

# 3. per-model outcome-blind operating-point sweep (layer=separation, coef=coherence) + eval
MODEL="Qwen/Qwen3-4B-Instruct-2507" JUDGE_WORKERS=16 python scripts/41_sweep.py

# 4. analysis
python scripts/39_multimodel_analysis.py
```
Gemma models load in bfloat16 (fp16 yields NaN activations); Qwen in fp16. Generation is per-item.

## Verifiable citations

Every citation in the paper is checked programmatically against the real source text — no hand-typed
quote survives unverified:

```bash
python scripts/fetch_source.py 2312.06681 2606.11205 ...   # fetch real arXiv full text
python scripts/verify_citations.py                          # assert every quote + title is present
```
`writeup/citations.json` holds `{key, arxiv_id, title_expect, verbatim quote, our_claim}`; the verifier
also fails if the draft cites any unverified key. (Fetched source texts are a local cache and are not
redistributed — see `.gitignore`.)

## Data provenance & license

- **Code** (`scripts/`, `src/`): MIT — see `LICENSE`.
- **Data** (`data/`, `results/`): CC-BY-4.0. The moral items are adapted from **MoralChoice**
  (Scherrer et al. 2023) and the A/B/steering-vector data from **Anthropic model-written evaluations**
  (Perez et al. 2022), both CC-BY-4.0; factual and opinion items are custom-authored. Full attribution
  and list of modifications in **`DATA_LICENSE.md`**.

## Citing

```bibtex
@misc{lipics2026readable,
  title  = {Readable but not steerable: activation steering for sycophancy is measurement-fragile and does not transfer across models},
  author = {Lipics, Barna},
  year   = {2026},
  note   = {arXiv preprint (link TBA)}
}
```
