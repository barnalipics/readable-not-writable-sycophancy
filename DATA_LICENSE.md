# Data license & attribution

All **data** in this repository (`data/`, `results/`) is released under
**[Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/)**.
(Code under `scripts/` and `src/` is MIT — see `LICENSE`.)

If you use the data, please cite this repository / the accompanying paper **and** the
third-party sources below.

## Original (our) material — CC-BY-4.0
Authored by us for this study, free to reuse with attribution:

- `data/raw/eval_L2A.jsonl`, `data/raw/eval_L2A_boost.jsonl` — custom **factual/math** sycophancy items.
- `data/raw/eval_en.jsonl`, `data/raw/eval_opinion_boost.jsonl`, `data/raw/eval_opinion_boost2.jsonl` — custom **persona-opinion** items.
- `data/raw/caa_contrast.jsonl`, `data/raw/caa_contrast_opinion.jsonl` — custom CAA contrast pairs.
- `results/*` — our experimental outputs.

## Third-party material (redistributed under CC-BY-4.0, with modifications)

### MoralChoice — `data/raw/eval_moralchoice_opinion.jsonl`
Adapted from **MoralChoice** (high-ambiguity subset).
- Source: Scherrer, Shi, Feder, Blei. *Evaluating the Moral Beliefs Encoded in LLMs* (2023), arXiv:2307.14324.
- Original: https://github.com/ninodimontalcino/moralchoice · https://huggingface.co/datasets/ninoscherrer/moralchoice
- License: CC-BY-4.0.
- **Changes made:** high-ambiguity scenarios reformatted into two-turn (seed + user-pushback) and A/B items; added defensible reference answers and a contestable-claim target; original scenario ids retained (`source_id`).

### Anthropic model-written evaluations (sycophancy) — `data/external/rimsky_caa_sycophancy_generate.json`
Derived from Anthropic's **model-written evaluations**.
- Source: Perez et al. *Discovering Language Model Behaviors with Model-Written Evaluations* (2022), arXiv:2212.09251.
- Original: https://github.com/anthropics/evals
- License: CC-BY-4.0.
- **Changes made:** sycophancy A/B pairs selected/reformatted for CAA steering-vector construction and held-out A/B evaluation.

CC-BY-4.0 permits redistribution and adaptation provided the creators are credited, changes are
indicated, and a link to the license is given — as done above.
