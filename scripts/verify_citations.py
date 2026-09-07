"""DETERMINISTIC citation verifier. For every record in writeup/citations.json, load the REAL source
text (writeup/sources/<arxiv_id>.txt, fetched by fetch_source.py) and assert that BOTH the expected
title and the verbatim quote occur in it, under whitespace/quote normalization only. No LLM, no
network, no fuzzy matching. A citation is valid ONLY if it PASSES here; anything else is treated as
hallucination and must not be cited.

Exit code 0 iff all records pass. Run: python scripts/verify_citations.py
"""
from __future__ import annotations
import json
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(_ROOT, "writeup", "sources")
CITES = os.path.join(_ROOT, "writeup", "citations.json")

_CURLY_APOS = ["’", "‘", "ʼ", "`"]
_CURLY_QUOT = ["“", "”"]
_DASHES = ["–", "—", "−"]


def norm(s: str) -> str:
    s = s.lower()
    for a in _CURLY_APOS:
        s = s.replace(a, "'")
    for a in _CURLY_QUOT:
        s = s.replace(a, '"')
    for a in _DASHES:
        s = s.replace(a, "-")
    return re.sub(r"\s+", " ", s).strip()


def main() -> int:
    cites = json.load(open(CITES))
    fails = 0
    print(f"verifying {len(cites)} citation records against {SRC_DIR}\n")
    for c in cites:
        key, aid = c["key"], c["arxiv_id"]
        path = os.path.join(SRC_DIR, f"{aid}.txt")
        if not os.path.exists(path):
            print(f"[FAIL] {key:28s} ({aid})  no source file -> fetch it first")
            fails += 1
            continue
        src = norm(open(path).read())
        t_ok = norm(c["title_expect"]) in src
        q_ok = norm(c["quote"]) in src
        ok = t_ok and q_ok
        fails += 0 if ok else 1
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {key:28s} ({aid})  title:{'y' if t_ok else 'N'} quote:{'y' if q_ok else 'N'}")
        if not ok:
            if not t_ok:
                print(f"        title not found: {c['title_expect'][:70]!r}")
            if not q_ok:
                print(f"        QUOTE not found: {c['quote'][:80]!r}")
    print(f"\n{'='*50}\n{len(cites)-fails}/{len(cites)} records passed; {fails} failed")
    if fails:
        print("REJECT the failed citations (treat as hallucination) or fix the quote to verbatim source text.")

    # also guard the draft: every \citep{key} in the paper must be a verified key
    draft = os.path.join(_ROOT, "writeup", "paper_draft.md")
    if os.path.exists(draft):
        valid = {c["key"] for c in cites}
        used = set()
        for grp in re.findall(r"\\citep\{([^}]+)\}", open(draft).read()):
            used |= {k.strip() for k in grp.split(",")}
        stray = used - valid
        print(f"\ndraft cites {len(used)} keys; stray (unverified) keys: {sorted(stray) if stray else 'NONE'}")
        if stray:
            fails += 1

    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
