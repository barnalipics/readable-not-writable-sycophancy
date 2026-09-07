"""Serious CAA vector construction (Rimsky et al. 2024) with BATCHED activation reads.

A/B format: each item has a question with (A)/(B) options, an answer_matching_behavior (the
sycophantic choice) and answer_not_matching_behavior (the honest choice). We append each choice
as the assistant turn, read the residual at the last (answer) token, and take
    v = mean(honest) - mean(sycophantic)
so that adding v steers toward honesty. Batched with left-padding so the last token aligns.

Also builds a vector from free-form contrast pairs (our factual set) for the two-vector comparison.
"""
from __future__ import annotations

import json

import torch


def get_decoder_layers(model):
    """Locate the decoder block list across architectures (Llama/Qwen: model.model.layers;
    multimodal e.g. Gemma-3: model.model.language_model.layers; GPT-style: transformer.h)."""
    for f in (lambda m: m.model.layers,
              lambda m: m.model.language_model.layers,
              lambda m: m.language_model.model.layers,
              lambda m: m.transformer.h):
        try:
            layers = f(model)
            if layers is not None and len(layers) > 0:
                return layers
        except AttributeError:
            continue
    raise RuntimeError("cannot locate decoder layers for this architecture")


@torch.no_grad()
def _read_last_resid_batch(model, tok, prompts, layer, device, batch_size=8):
    """Return [n, d] residual at layer `layer` (output), last token, for each prompt. Left-padded.
    Uses a forward hook on the target layer (lighter than output_hidden_states, which returns all
    layers and OOMs/swaps on 16GB with long prompts)."""
    old_side = tok.padding_side
    old_trunc = tok.truncation_side
    tok.padding_side = "left"
    tok.truncation_side = "left"   # keep the END (answer token we read); trim leading context only
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cap = {}

    def hook(_m, _i, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out)[:, -1, :].float().cpu()

    handle = get_decoder_layers(model)[layer].register_forward_hook(hook)
    outs = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    try:
        for bi, i in enumerate(range(0, len(prompts), batch_size)):
            batch = prompts[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=640).to(device)
            model(**enc)
            outs.append(cap["h"])  # left-pad => -1 is the real last token
            if bi % 10 == 0:
                print(f"    read batch {bi + 1}/{n_batches}", flush=True)
    finally:
        handle.remove()
        tok.padding_side = old_side
        tok.truncation_side = old_trunc
    return torch.cat(outs, 0)


def _ab_prompt(tok, question, answer):
    msgs = [{"role": "user", "content": question},
            {"role": "assistant", "content": answer}]
    # no generation prompt: we want the activation AT the answer token
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


def build_vector_ab(model, tok, items, layer, device, limit=None, batch_size=16):
    """items: list of {question, answer_matching_behavior, answer_not_matching_behavior}."""
    if limit:
        items = items[:limit]
    honest_prompts, syco_prompts = [], []
    for it in items:
        honest_prompts.append(_ab_prompt(tok, it["question"], it["answer_not_matching_behavior"]))
        syco_prompts.append(_ab_prompt(tok, it["question"], it["answer_matching_behavior"]))
    h = _read_last_resid_batch(model, tok, honest_prompts, layer, device, batch_size)
    s = _read_last_resid_batch(model, tok, syco_prompts, layer, device, batch_size)
    diffs = h - s
    return diffs.mean(0), {"n": len(items), "norm": float(diffs.mean(0).norm()),
                           "per_pair_norm_mean": float(diffs.norm(dim=1).mean())}


def build_vector_freeform(model, tok, pairs, layer, device, batch_size=16):
    """pairs: list of {context, honest, sycophantic} (our factual set). Read last-token resid of
    context+continuation."""
    honest_prompts = [p["context"] + p["honest"] for p in pairs]
    syco_prompts = [p["context"] + p["sycophantic"] for p in pairs]
    h = _read_last_resid_batch(model, tok, honest_prompts, layer, device, batch_size)
    s = _read_last_resid_batch(model, tok, syco_prompts, layer, device, batch_size)
    diffs = h - s
    return diffs.mean(0), {"n": len(pairs), "norm": float(diffs.mean(0).norm())}


def load_rimsky(path):
    return json.load(open(path))
