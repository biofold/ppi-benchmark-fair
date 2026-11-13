#!/usr/bin/env python3
"""Validate the metadata files of this FAIR package (run by .github/workflows/validate.yml).

Checks performed (a check is skipped when its files are absent):
  1. .zenodo.json: required fields, creators, licence, communities.
  2. CITATION.cff: valid YAML with the required fields; title and version agree with .zenodo.json.
  3. fair-report/fair-report.json against fair-report/fair-report.schema.json.
  4. SCO: scrna-schema/sco-benchmark-experiment-1.json against scrna-schema/schema.json.
  5. PPI: bioschema/fair_metadata_package.json against bioschema/schema.json, and every
     bioschema/interface_protein_pairs/interface_*.json file is JSON-LD with @context and @type.

Usage: python scripts/validate_metadata.py   (exit code 1 if any check fails)
Requires: jsonschema, pyyaml
"""
import glob
import json
import os
import sys

import yaml
from jsonschema import validators

try:  # jsonschema >= 4
    from jsonschema import Draft202012Validator as DefaultValidator
except ImportError:  # older jsonschema (e.g. the 3.2 pinned by some tools)
    from jsonschema import Draft7Validator as DefaultValidator

errors = []


def fail(msg):
    errors.append(msg)
    print("FAIL:", msg)


def ok(msg):
    print("ok:  ", msg)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate(data, schema, label):
    cls = validators.validator_for(schema, default=DefaultValidator)
    errs = sorted(cls(schema).iter_errors(data), key=lambda e: list(e.path))
    if errs:
        for e in errs[:10]:
            fail(f"{label}: {e.message} at {'/'.join(str(p) for p in e.path) or '<root>'}")
    else:
        ok(label)


zen = cff = None
if os.path.exists(".zenodo.json"):
    zen = load_json(".zenodo.json")
    for key in ("title", "creators", "license", "upload_type", "version", "description"):
        if not zen.get(key):
            fail(f".zenodo.json: missing '{key}'")
    for c in zen.get("creators", []):
        if not c.get("name"):
            fail(".zenodo.json: creator without name")
    if not zen.get("communities"):
        fail(".zenodo.json: no communities")
    ok(".zenodo.json structure checked")

if os.path.exists("CITATION.cff"):
    with open("CITATION.cff", encoding="utf-8") as f:
        cff = yaml.safe_load(f)
    for key in ("cff-version", "title", "authors", "version", "license", "doi"):
        if not cff.get(key):
            fail(f"CITATION.cff: missing '{key}'")
    ok("CITATION.cff structure checked (full schema check: cffconvert --validate)")
    if zen:
        if cff.get("title") != zen.get("title"):
            fail("title differs between CITATION.cff and .zenodo.json")
        if str(cff.get("version")) != str(zen.get("version")):
            fail(f"version differs: CITATION.cff {cff.get('version')} vs .zenodo.json {zen.get('version')}")

rep = "fair-report/fair-report.json"
sch = "fair-report/fair-report.schema.json"
if os.path.exists(rep) and os.path.exists(sch):
    validate(load_json(rep), load_json(sch), "fair-report.json against its JSON Schema")
    r = load_json(rep)
    if cff and r["repository"].get("release_tag") and str(cff["version"]) not in r["repository"]["release_tag"]:
        fail(f"fair-report.json release_tag {r['repository']['release_tag']} does not match CITATION.cff version {cff['version']}")

if os.path.exists("scrna-schema/schema.json"):
    validate(load_json("scrna-schema/sco-benchmark-experiment-1.json"), load_json("scrna-schema/schema.json"),
             "SCO metadata record against Single Cell Schema")

if os.path.exists("bioschema/schema.json") and os.path.exists("bioschema/fair_metadata_package.json"):
    validate(load_json("bioschema/fair_metadata_package.json"), load_json("bioschema/schema.json"),
             "bioschema/fair_metadata_package.json against bioschema/schema.json")
files = sorted(glob.glob("bioschema/interface_protein_pairs/interface_*.json"))
if files:
    bad = 0
    for p in files:
        try:
            d = load_json(p)
            if not (isinstance(d, dict) and "@context" in d and "@type" in d):
                bad += 1
                fail(f"{p}: not JSON-LD (missing @context/@type)")
        except Exception as e:  # noqa: BLE001
            bad += 1
            fail(f"{p}: {e}")
    if not bad:
        ok(f"{len(files)} interface JSON-LD files parse and have @context and @type")

print()
if errors:
    print(f"{len(errors)} problem(s) found")
    sys.exit(1)
print("All metadata checks passed")
