#!/usr/bin/env python3
"""Update the FAIR score numbers written in README/HTML/generator files from fair-report/fair-report.json.

Run after regenerating the FAIR report so badges and texts agree with the report.
Only well-defined patterns are replaced (badge URLs, 'FAIR Score: x/100', 'Overall Score: x/100',
'FAIR Compliance Score: x/100', 'assessment (x/100)', '(Currently: x)', 'Current Score: x/100', per-principle '(x/100)' headings
and the FAIR Score metric card). Data files are never touched.

Usage: python scripts/update_fair_score_strings.py [--dry-run]
"""
import json
import os
import re
import sys

dry = "--dry-run" in sys.argv
rep = json.load(open("fair-report/fair-report.json", encoding="utf-8"))
fmt = lambda x: ("%g" % round(x, 2))  # noqa: E731
total = fmt(rep["scores"]["total"])
pr = {k.capitalize(): fmt(v["percentage"]) for k, v in rep["scores"]["principles"].items()}

rules = [
    (r"FAIR_Score-[\d.]+%2F100", f"FAIR_Score-{total}%2F100"),
    (r"(?<!Target )FAIR Score: [\d.]+/100", f"FAIR Score: {total}/100"),
    (r"Overall Score: [\d.]+/100", f"Overall Score: {total}/100"),
    (r"FAIR Compliance Score: [\d.]+/100", f"FAIR Compliance Score: {total}/100"),
    (r"FAIR assessment \([\d.]+/100\)", f"FAIR assessment ({total}/100)"),
    (r"\(Currently: [\d.]+\)", f"(Currently: {total})"),
    (r"Current Score: [\d.]+/100", f"Current Score: {total}/100"),
    (r'(<div class="metric-value">)[\d.]+%(</div>\s*<div class="metric-label">FAIR Score)', rf"\g<1>{total}%\g<2>"),
]
for name, val in pr.items():
    rules.append((rf"({name}) \([\d.]+/100\)", rf"\g<1> ({val}/100)"))

changed = 0
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in (".git", "data", "node_modules") and not d.startswith("fair_reports")]
    for fn in files:
        if not fn.endswith((".md", ".html", ".py")):
            continue
        p = os.path.join(root, fn)
        if fn == "update_fair_score_strings.py":
            continue
        try:
            s = open(p, encoding="utf-8").read()
        except UnicodeDecodeError:
            continue
        n = s
        for pat, rep_ in rules:
            n = re.sub(pat, rep_, n)
        if n != s:
            changed += 1
            print(("would update " if dry else "updated ") + p)
            if not dry:
                open(p, "w", encoding="utf-8").write(n)
print(f"{changed} file(s) {'would change' if dry else 'changed'}; total score {total}/100")
