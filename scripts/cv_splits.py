#!/usr/bin/env python3
"""Homology-aware cross-validation splits of the PPI benchmark: load, describe and add them to the JSON-LD.

The split files in data/cv_splits/ (one CSV per strategy with the columns entry_id, label, group_key, fold) are the
splits published with the OpenEBench implementation of this benchmark (Zenodo concept DOI 10.5281/zenodo.24323529,
https://github.com/elixir-unibo/oeb_3dbioinfo_ppi_benchmark). They are NOT taken from Schweke et al. 2023, which
does not describe how its own folds were built; see data/cv_splits/README.md.

Used by scripts/ppi_benchmark_fair.py (option --cv-splits) and, to bring existing JSON-LD files up to date without
a full regeneration, from the command line:

    python scripts/cv_splits.py patch [--splits data/cv_splits] [--bioschema bioschema]

The patch is idempotent: properties of an earlier run are removed before the current ones are added.
"""
import argparse
import csv
import glob
import hashlib
import json
import os

# (key, property suffix, file name, short method description)
STRATEGIES = [
    ("1", "Strategy1", "cv_fold_strategy1_cath_superfamily_gene.csv", "CATH superfamily / UniProt gene grouping"),
    ("2", "Strategy2", "cv_fold_strategy2_mmseqs2_strict.csv", "MMseqs2 sequence clustering (25% identity, 50% coverage)"),
    ("2b", "Strategy2b", "cv_fold_strategy2b_blastclust_reference.csv", "BLASTClust reference clustering (-S 25 -L 0.5 -b F)"),
    ("3", "Strategy3", "cv_fold_strategy3_combined_union.csv", "union of Strategy 1 and Strategy 2b (recommended)"),
]
PROP_PREFIXES = ("CVGroup_", "CVFold_")
RAW_BASE = "https://raw.githubusercontent.com/biofold/ppi-benchmark-fair/master/data/cv_splits/"


def load_cv_splits(directory):
    """Return {suffix: {...}} with the assignments of every strategy, or {} when the directory is absent."""
    splits = {}
    for key, suffix, fname, desc in STRATEGIES:
        path = os.path.join(directory, fname)
        if not os.path.exists(path):
            continue
        assign, labels = {}, {}
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                iid = row["entry_id"].strip().upper()
                assign[iid] = (row["group_key"], int(row["fold"]))
                labels[iid] = int(row["label"])
        with open(path, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        groups = {g for g, _ in assign.values()}
        folds = sorted({f for _, f in assign.values()})
        splits[suffix] = {"key": key, "file": fname, "desc": desc, "assign": assign, "labels": labels,
                          "n_entries": len(assign), "n_groups": len(groups), "folds": folds, "sha256": sha}
    return splits


def _label(info):
    return f"Strategy {info['key']} ({info['desc']})"


def split_properties(interface_id, splits):
    """PropertyValue items (group, fold) of one interface for every strategy that assigns it."""
    out = []
    iid = str(interface_id).strip().upper()
    for key, suffix, fname, desc in STRATEGIES:
        info = splits.get(suffix)
        if not info or iid not in info["assign"]:
            continue
        group, fold = info["assign"][iid]
        out.append({"@type": "PropertyValue", "name": "CVGroup_" + suffix, "value": group,
                    "description": f"Homology group of the cross-validation splitting {_label(info)}: "
                                   f"interfaces of one group are always placed in the same fold"})
        out.append({"@type": "PropertyValue", "name": "CVFold_" + suffix, "value": fold,
                    "description": f"Fold (0-{max(info['folds'])}) of the {len(info['folds'])}-fold cross-validation "
                                   f"of Strategy {info['key']}"})
    return out


def apply_to_additional_properties(props, interface_id, splits):
    """Remove earlier CV properties and append the current ones (idempotent). Returns the new list."""
    kept = [p for p in props if not str(p.get("name", "")).startswith(PROP_PREFIXES)]
    return kept + split_properties(interface_id, splits)


def _interface_id_of(item):
    for p in item.get("additionalProperty", []):
        if p.get("name") == "InterfaceID":
            return p.get("value")
    ident = item.get("identifier")  # the per-interface files carry the id only as identifier
    return ident if isinstance(ident, str) else None


def distribution_entries(splits, base_url=RAW_BASE):
    out = []
    for key, suffix, fname, desc in STRATEGIES:
        info = splits.get(suffix)
        if not info:
            continue
        out.append({"@type": "cr:FileObject", "@id": "file_cv_" + suffix.lower(),
                    "name": f"Cross-validation split, Strategy {info['key']} (CSV)",
                    "description": f"Homology groups and 5-fold assignment of the {info['n_entries']} interfaces: {info['desc']}",
                    "contentUrl": base_url + fname, "encodingFormat": "text/csv", "sha256": info["sha256"]})
    return out


def recordsets(splits):
    out = []
    for key, suffix, fname, desc in STRATEGIES:
        info = splits.get(suffix)
        if not info:
            continue
        fid = "file_cv_" + suffix.lower()

        def field(name, text, dtype):
            return {"@type": "cr:Field", "name": name, "description": text, "dataType": dtype,
                    "source": {"fileObject": fid, "extract": {"column": name}}}
        out.append({"@type": "cr:RecordSet", "@id": "recordset_cv_" + suffix.lower(), "name": "cv_split_" + suffix.lower(),
                    "description": f"Cross-validation split, Strategy {info['key']}: {info['desc']}",
                    "field": [field("entry_id", "Interface identifier (lower case, e.g. 1a17_6)", "https://schema.org/Text"),
                              field("label", "Numeric label (1=physiological, 0=non-physiological)", "https://schema.org/Integer"),
                              field("group_key", "Homology group; all its interfaces are in the same fold", "https://schema.org/Text"),
                              field("fold", "Fold of the 5-fold cross-validation (0-4)", "https://schema.org/Integer")],
                    "key": ["entry_id"]})
    return out


def statistics(splits):
    st = {}
    for key, suffix, fname, desc in STRATEGIES:
        info = splits.get(suffix)
        if not info:
            continue
        sizes = {}
        for _, f in info["assign"].values():
            sizes[f] = sizes.get(f, 0) + 1
        st["strategy_" + key] = {"file": "data/cv_splits/" + fname, "method": info["desc"], "entries": info["n_entries"],
                                 "groups": info["n_groups"], "folds": len(info["folds"]),
                                 "entries_per_fold": [sizes[f] for f in sorted(sizes)], "sha256": info["sha256"]}
    return st


def manifest_entry(splits):
    return {"directory": "data/cv_splits", "strategies": statistics(splits),
            "properties_added": ["CVGroup_Strategy<N>: homology group", "CVFold_Strategy<N>: fold (0-4)"],
            "note": "Interfaces flagged 'to delete' in the curated annotation have no split assignment."}


def patch_dataset(dataset, splits):
    """Patch a dataset JSON-LD dict in place: interface items, distribution, record sets."""
    for item in dataset.get("hasPart", []):
        iid = _interface_id_of(item)
        if iid is not None:
            item["additionalProperty"] = apply_to_additional_properties(item.get("additionalProperty", []), iid, splits)
    ids = {d.get("@id") for d in distribution_entries(splits)}
    dataset["distribution"] = [d for d in dataset.get("distribution", []) if d.get("@id") not in ids] + distribution_entries(splits)
    rids = {r.get("@id") for r in recordsets(splits)}
    dataset["recordSet"] = [r for r in dataset.get("recordSet", []) if r.get("@id") not in rids] + recordsets(splits)
    return dataset


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _dump(obj, path, indent=2):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False)


def patch_files(splits_dir, bioschema_dir):
    splits = load_cv_splits(splits_dir)
    if not splits:
        raise SystemExit(f"no split files found in {splits_dir}")
    p = os.path.join(bioschema_dir, "dataset_with_interfaces.json")
    ds = patch_dataset(_load(p), splits)
    _dump(ds, p)
    n = 0
    for path in sorted(glob.glob(os.path.join(bioschema_dir, "interface_protein_pairs", "interface_*.json"))):
        d = _load(path)
        iid = _interface_id_of(d)
        if iid is None:
            continue
        d["additionalProperty"] = apply_to_additional_properties(d.get("additionalProperty", []), iid, splits)
        _dump(d, path)
        n += 1
    p = os.path.join(bioschema_dir, "fair_metadata_package.json")
    if os.path.exists(p):
        pkg = _load(p)
        patch_dataset(pkg["bioschemas_markup"]["dataset"], splits)
        pkg["cv_split_statistics"] = statistics(splits)
        _dump(pkg, p)
    p = os.path.join(bioschema_dir, "manifest.json")
    if os.path.exists(p):
        man = _load(p)
        man["cv_splits"] = manifest_entry(splits)
        _dump(man, p)
    return splits, n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["patch"])
    ap.add_argument("--splits", default="data/cv_splits")
    ap.add_argument("--bioschema", default="bioschema")
    a = ap.parse_args()
    s, n = patch_files(a.splits, a.bioschema)
    print(f"patched dataset_with_interfaces.json, fair_metadata_package.json, manifest.json and {n} interface files "
          f"with {len(s)} strategies")
