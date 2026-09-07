"""Render writeup/paper.pdf from paper_draft.md without a TeX engine:
pandoc -> standalone HTML (citations resolved from references_verified.bib via --citeproc) ->
headless Chrome --print-to-pdf (falls back to macOS cupsfilter). Deterministic, re-runnable.
"""
from __future__ import annotations
import os
import re
import subprocess
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
md = open(os.path.join(_ROOT, "writeup", "paper_draft.md")).read()

title = re.search(r"^#\s+(.+)", md, re.M).group(1).strip()
body = md[re.search(r"##\s+Abstract", md).start():]
body = re.sub(r"\\citep\{([^}]+)\}",
              lambda m: "[" + "; ".join("@" + k.strip() for k in m.group(1).split(",")) + "]",
              body)

YAML = (
    "---\n"
    f'title: "{title}"\n'
    "author:\n"
    "  - \"Barna Lipics — Eötvös Loránd University — barna.lipics@student.elte.hu\"\n"
    "  - \"Ágnes Buvár — Eötvös Loránd University — buvar.agnes@ppk.elte.hu\"\n"
    "bibliography: references_verified.bib\n"
    "link-citations: true\n"
    "---\n\n"
)
CSS = """
@page { margin: 2cm; }
body { font-family: Georgia,'Times New Roman',serif; font-size:10.5pt; line-height:1.42; text-align:justify; }
header#title-block-header .title { font-size:15pt; }
h1 { font-size:14pt; } h2 { font-size:12pt; margin-top:1.1em; border-bottom:1px solid #ccc; padding-bottom:2px; }
table { border-collapse:collapse; font-size:9pt; } th,td { border:1px solid #999; padding:2px 6px; }
code { font-family:monospace; font-size:9pt; } #refs { font-size:9pt; }
"""
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
html = os.path.join(_ROOT, "writeup", "paper.html")
pdf = os.path.join(_ROOT, "writeup", "paper.pdf")

with tempfile.TemporaryDirectory() as d:
    mdp = os.path.join(d, "p.md"); open(mdp, "w").write(YAML + body)
    cssp = os.path.join(d, "s.css"); open(cssp, "w").write(CSS)
    subprocess.run(["pandoc", mdp, "--citeproc", "--bibliography",
                    os.path.join(_ROOT, "references_verified.bib"),
                    "-s", "--embed-resources", "-c", cssp, "-o", html], check=True, timeout=120)

if os.path.exists(CHROME):
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", f"file://{html}"], capture_output=True, timeout=120)
if not os.path.exists(pdf) or os.path.getsize(pdf) < 5000:
    subprocess.run(f"cupsfilter {html} > {pdf}", shell=True, timeout=120)

data = open(pdf, "rb").read()
pages = data.count(b"/Type /Page") - data.count(b"/Type /Pages")
print(f"wrote {pdf} ({len(data)//1024} KB, ~{pages} pages) and {html}")
