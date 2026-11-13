#!/usr/bin/env python3
"""Check the status of the PDB entries of the benchmark in the RCSB PDB.

Reads the PDB identifiers from the curated benchmark file, asks the RCSB holdings service for the
current status of every entry and reports entries that are no longer current (obsolete/withdrawn),
with the replacing entry when RCSB provides one. Entries already handled in
data/replacement_structures/replacements.tsv are marked as handled.

Usage:
    python scripts/check_pdb_status.py [--csv data/benchmark_annotated_updated_30042023.csv]
        [--replacements data/replacement_structures/replacements.tsv]
        [--output fair-report/pdb_status_report.json]
Only the Python standard library is needed. The script never modifies benchmark data.
"""
import argparse
import csv
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

API = "https://data.rcsb.org/rest/v1/holdings/status?ids="


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data/benchmark_annotated_updated_30042023.csv")
    ap.add_argument("--replacements", default="data/replacement_structures/replacements.tsv")
    ap.add_argument("--output", default="fair-report/pdb_status_report.json")
    a = ap.parse_args()

    ids = set()
    with open(a.csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ids.add(r["ID"][:4].upper())
    ids = sorted(ids)

    handled = {}
    if os.path.exists(a.replacements):
        with open(a.replacements, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                handled[r["obsolete_pdb_id"].upper()] = r["replacement_pdb_id"].upper()

    status = {}
    for i in range(0, len(ids), 100):
        batch = ids[i:i + 100]
        req = urllib.request.Request(API + ",".join(batch), headers={"User-Agent": "ppi-benchmark-fair-status-check"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            for item in json.load(resp):
                h = item.get("rcsb_repository_holdings_combined", {})
                status[item["rcsb_id"].upper()] = (h.get("status"), h.get("id_code_replaced_by_latest"))
    missing = [i for i in ids if i not in status]

    not_current = []
    for i in ids:
        st, rep = status.get(i, (None, None))
        if st and st != "CURRENT":
            not_current.append({
                "pdb_id": i, "status": st, "replaced_by": rep,
                "handled_in_package": i in handled,
                "replacement_in_package": handled.get(i),
            })
    unhandled = [e for e in not_current if not e["handled_in_package"]]
    report = {
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "RCSB PDB holdings status service (data.rcsb.org)",
        "entries_checked": len(ids),
        "entries_not_found": missing,
        "not_current": not_current,
        "not_current_and_unhandled": len(unhandled),
    }
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print(f"{len(ids)} entries checked; {len(not_current)} not current; {len(unhandled)} not yet handled; {len(missing)} not found")
    for e in not_current:
        print(f"  {e['pdb_id']}: {e['status']} -> {e['replaced_by']} ({'handled' if e['handled_in_package'] else 'NEW'})")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(f"### PDB status check\n{len(ids)} entries checked, {len(not_current)} not current, "
                    f"**{len(unhandled)} not yet handled**.\n\n")
            for e in not_current:
                f.write(f"- {e['pdb_id']}: {e['status']} -> {e['replaced_by']} ({'handled' if e['handled_in_package'] else 'NEW'})\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
