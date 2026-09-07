"""Fetch the REAL full text of an arXiv paper to writeup/sources/<id>.txt for deterministic
citation verification. Tries ar5iv (full HTML) first, falls back to the arXiv abstract page.
Strips tags to plain text + normalizes whitespace. No LLM in the loop — pure download + regex.

Usage: python scripts/fetch_source.py 2312.06681 [2310.13548 ...]
"""
from __future__ import annotations
import html
import os
import re
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_ROOT, "writeup", "sources")
os.makedirs(OUT_DIR, exist_ok=True)


def strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style)\b.*?>.*?</\1>", " ", raw)
    txt = re.sub(r"(?s)<[^>]+>", " ", raw)
    txt = html.unescape(txt)
    return re.sub(r"\s+", " ", txt).strip()


def fetch(aid: str) -> None:
    urls = [
        f"https://ar5iv.org/abs/{aid}",
        f"https://ar5iv.labs.arxiv.org/html/{aid}",
        f"https://arxiv.org/abs/{aid}",
    ]
    for u in urls:
        try:
            raw = subprocess.run(["curl", "-sL", "--max-time", "45", "-A", "Mozilla/5.0 (citation-verify)", u],
                                 capture_output=True, timeout=60).stdout.decode("utf-8", "ignore")
            txt = strip_html(raw)
            if len(txt) > 1500:
                path = os.path.join(OUT_DIR, f"{aid}.txt")
                with open(path, "w") as f:
                    f.write(txt)
                kind = "FULL(ar5iv)" if "ar5iv" in u else "ABSTRACT-ONLY(arxiv)"
                print(f"[OK] {aid}: {len(txt):>7d} chars  {kind}  <- {u}")
                print(f"      head: {txt[:220]}")
                return
        except Exception as e:
            print(f"      (try failed {u}: {type(e).__name__})")
    print(f"[FAIL] {aid}: no source retrieved")


if __name__ == "__main__":
    for aid in sys.argv[1:]:
        fetch(aid)
