"""Assemble writeup/paper.tex from writeup/paper_draft.md (Markdown) + references_verified.bib.
Splits the H1 title and the Abstract into a proper LaTeX preamble; pandoc-converts the body
(Introduction onward), keeping \\citep and tables; wires up natbib + the verified bib. Re-runnable.
"""
from __future__ import annotations
import os
import re
import subprocess
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
md = open(os.path.join(_ROOT, "writeup", "paper_draft.md")).read()

title = re.search(r"^#\s+(.+)", md, re.M).group(1).strip()
abstract = re.search(r"##\s+Abstract\s*(.+?)\n##\s", md, re.S).group(1).strip()
body = md[re.search(r"##\s+Introduction", md).start():]


def pandoc(src: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(src)
        p = f.name
    out = subprocess.run(["pandoc", p, "-f", "markdown", "-t", "latex", "--wrap=preserve"],
                         capture_output=True, timeout=60).stdout.decode("utf-8")
    os.unlink(p)
    return out


abstract_tex = pandoc(abstract)
body_tex = pandoc(body)

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs,longtable,array}
\usepackage{graphicx}
\usepackage[numbers,round]{natbib}
\usepackage[hidelinks]{hyperref}
\title{%s}
\author{
  Barna Lipics\\ E\"otv\"os Lor\'and University\\ \texttt{barna.lipics@student.elte.hu}
  \and
  \'Agnes Buv\'ar\\ E\"otv\"os Lor\'and University\\ \texttt{buvar.agnes@ppk.elte.hu}
}
\date{\today}
\begin{document}
\maketitle
\begin{abstract}
%s
\end{abstract}
""" % (title, abstract_tex)

TAIL = r"""
\bibliographystyle{plainnat}
\bibliography{references_verified}
\end{document}
"""

out = os.path.join(_ROOT, "writeup", "paper.tex")
with open(out, "w") as f:
    f.write(PREAMBLE + "\n" + body_tex + TAIL)
# bib next to the tex for compilation convenience
import shutil
shutil.copy(os.path.join(_ROOT, "references_verified.bib"), os.path.join(_ROOT, "writeup", "references_verified.bib"))
print(f"wrote {out}  ({len(open(out).read())} chars)")
print(f"title: {title}")
print(f"citep keys in tex: {len(set(re.findall(r'\\citep\{([^}]+)\}', open(out).read())))}")
