"""MULTI-MODEL analysis: does the protocol-contingency (steering beats random in free-form, not in
A/B) generalize across families x sizes? Reads results/multimodel_flip.jsonl (+ the single-model
protocol_flip.jsonl if present, tagged as its model). CPU only.

Per model: descriptive fix-rates + the two key checks (free-form real>random? A/B real~random?).
Pooled: GEE protocol x vector interaction with MODEL controlled. Writes multimodel_analysis.json.
"""
from __future__ import annotations

import glob
import json
import os

ROOT = os.environ.get("ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

rows = []
for fp in glob.glob(f"{ROOT}/results/multimodel_flip*.jsonl"):
    if "analysis" in fp:
        continue
    rows += [json.loads(l) for l in open(fp) if l.strip()]
# fold in the original single-model run (tag it)
sp = f"{ROOT}/results/protocol_flip.jsonl"
if os.path.exists(sp):
    for l in open(sp):
        if not l.strip():
            continue
        r = json.loads(l)
        r.setdefault("model", "Qwen/Qwen3-4B-Instruct-2507")
        if r.get("coef") == 16:
            rows.append(r)
print(f"loaded {len(rows)} rows; models: {sorted(set(r['model'] for r in rows))}")


def fixrate(rr, model, proto, vec):
    v = [r for r in rr if r["model"] == model and r["protocol"] == proto and r["vector"] == vec
         and r["baseline_syco"] == 1 and r["reduced"] is not None and r.get("coef", 16) == 16]
    k = sum(r["reduced"] for r in v)
    return k, len(v)


models = sorted(set(r["model"] for r in rows))
out = {"per_model": {}}
print("\n=== PER-MODEL (fix-rate on caved; does free-form beat random & A/B not?) ===")
for mdl in models:
    d = {}
    for proto in ("freeform", "ab"):
        cell = {}
        for vec in ("fac", "rim", "rand"):
            k, n = fixrate(rows, mdl, proto, vec)
            cell[vec] = {"pct": round(100*k/n, 1) if n else None, "n": n}
        d[proto] = cell
    ff, ab = d["freeform"], d["ab"]
    def beats(c, proto):  # real vector beats random by >5pts
        r = proto["rand"]["pct"]
        return (proto[c]["pct"] is not None and r is not None and proto[c]["pct"] - r > 5)
    ff_works = beats("fac", ff) or beats("rim", ff)
    ab_fails = not (beats("fac", ab) or beats("rim", ab))
    d["pattern_holds"] = bool(ff_works and ab_fails)
    out["per_model"][mdl] = d
    short = mdl.split("/")[-1]
    print(f"\n  {short}:")
    print(f"    freeform: fac {ff['fac']['pct']} rim {ff['rim']['pct']} rand {ff['rand']['pct']} (n~{ff['rand']['n']})")
    print(f"    ab      : fac {ab['fac']['pct']} rim {ab['rim']['pct']} rand {ab['rand']['pct']} (n~{ab['rand']['n']})")
    print(f"    -> free-form steering works: {ff_works};  A/B fails control: {ab_fails};  PATTERN HOLDS: {d['pattern_holds']}")

n_hold = sum(1 for mdl in models if out["per_model"][mdl]["pattern_holds"])
print(f"\n  PATTERN HOLDS in {n_hold}/{len(models)} models")
out["pattern_holds_count"] = f"{n_hold}/{len(models)}"

# ---- pooled GEE with model controlled ----
print("\n=== POOLED: protocol x vector interaction, MODEL controlled (GEE) ===")
try:
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    d = [r for r in rows if r["vector"] in ("fac", "rim") and r["baseline_syco"] == 1
         and r["reduced"] is not None and r.get("coef", 16) == 16]
    df = pd.DataFrame(d)
    df["proto_ff"] = (df["protocol"] == "freeform").astype(int)
    df["vec_rim"] = (df["vector"] == "rim").astype(int)
    df["uid"] = df["model"] + "|" + df["item_id"]
    mod = smf.gee("reduced ~ proto_ff * vec_rim + C(construct) + C(model)", groups="uid",
                  data=df, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
    res = mod.fit()
    ic = "proto_ff:vec_rim"
    out["pooled_interaction"] = {"coef": round(float(res.params[ic]), 3), "p": round(float(res.pvalues[ic]), 5), "n_obs": len(df), "n_models": len(models)}
    print(res.summary())
    print(f"\n  POOLED interaction proto_ff:vec_rim coef={float(res.params[ic]):+.3f} p={float(res.pvalues[ic]):.5f}")
except Exception as e:
    out["pooled_error"] = str(e)
    print(f"  GEE failed: {e}")

json.dump(out, open(f"{ROOT}/results/multimodel_analysis.json", "w"), indent=2)
print("\nwrote results/multimodel_analysis.json")
