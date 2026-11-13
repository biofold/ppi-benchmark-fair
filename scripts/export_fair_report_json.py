#!/usr/bin/env python3
"""Export the machine-readable FAIR assessment of this repository.

Reads the raw JSON report written by scripts/test_fair_gitrepo.py
(fair_reports/fair_comparison_<timestamp>.json) and writes a compact,
documented JSON file (default: fair-report/fair-report.json) that follows
fair-report/fair-report.schema.json.

Usage:
    python scripts/test_fair_gitrepo.py repos.txt        # repos.txt: one repository URL
    python scripts/export_fair_report_json.py fair_reports/fair_comparison_*.json \
        --repo-url https://github.com/<owner>/<repo> --commit <sha> \
        --doi 10.5281/zenodo.<concept-id> -o fair-report/fair-report.json

Only the Python standard library is required.
"""
import argparse
import json
import sys
from datetime import datetime, timezone

PRINCIPLES = ["findable", "accessible", "interoperable", "reusable"]


def build(raw, repo_url, commit=None, release=None, doi=None, license_id=None):
    scores = [s for s in raw.get("scores", []) if s.get("repository", "").rstrip("/") == repo_url.rstrip("/")]
    if not scores:
        sys.exit(f"Repository {repo_url} not found in the raw report")
    s = scores[0]
    improvements = [i for i in raw.get("improvements", []) if i.get("repository", "").rstrip("/") == repo_url.rstrip("/")]
    principles = {}
    for p in PRINCIPLES:
        d = s["detailed_scores"][p]
        principles[p] = {
            "score": d["score"],
            "max_score": d["max_score"],
            "percentage": round(d["percentage"], 2),
            "missing_criteria": d["missing"],
        }
    out = {
        "schema_version": "1.0",
        "report_type": "FAIR repository assessment",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "assessed_at": raw.get("timestamp"),
        "repository": {
            "url": repo_url,
            "commit": commit,
            "release_tag": release,
            "doi": doi,
            "license": license_id,
        },
        "tool": {
            "name": "test_fair_gitrepo.py",
            "description": "Rule-based FAIR checklist run on a clone of the repository; "
                           "each principle is scored 0-100 from weighted criteria and "
                           "the total is the mean of the four principle scores.",
            "path": "scripts/test_fair_gitrepo.py",
        },
        "scores": {
            "total": round(s["total"], 2),
            "scale_max": 100,
            "principles": principles,
        },
        "metadata_summary": {
            "metadata_files_found": s.get("metadata_files_count"),
            **{k: v for k, v in s.get("metadata_summary", {}).items()},
        },
        "improvements": [
            {
                "principle": i["principle"],
                "missing_criterion": i["missing"],
                "action": i["action"],
                "priority": i["priority"],
                "potential_points": i["potential_points"],
            }
            for i in improvements
        ],
        "limitations": [
            "The score is a checklist of repository-level FAIR indicators, not a certification of FAIR compliance.",
            "Some criteria (e.g. repository topics, wiki, issue templates) are inferred from files in the clone and cannot see GitHub settings.",
            "The assessed commit precedes the commit that adds this file.",
        ],
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_report", help="fair_comparison_<timestamp>.json from test_fair_gitrepo.py")
    ap.add_argument("--repo-url", required=True)
    ap.add_argument("--commit")
    ap.add_argument("--release")
    ap.add_argument("--doi")
    ap.add_argument("--license", dest="license_id")
    ap.add_argument("-o", "--output", default="fair-report/fair-report.json")
    a = ap.parse_args()
    with open(a.raw_report, encoding="utf-8") as f:
        raw = json.load(f)
    out = build(raw, a.repo_url, a.commit, a.release, a.doi, a.license_id)
    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("Wrote %s: total %s/100" % (a.output, out["scores"]["total"]))


if __name__ == "__main__":
    main()
