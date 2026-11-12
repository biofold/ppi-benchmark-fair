# Replacement structures for obsolete benchmark entries

**These files are not part of the benchmark.** The ELIXIR 3D-BioInfo PPI benchmark (1,677 interfaces, 1,645 PDB entries) is unchanged.
This folder only holds the current coordinate files of the PDB entries that replaced three benchmark entries that the wwPDB has since made obsolete.
The benchmark's own structure files for these three entries are still in `data/benchmark_pdb_format/` and `data/benchmark_mmcif_format/`.

| Obsolete entry in the benchmark | Benchmark interface | Replacement entry | Obsolete (removed) on | File |
| :-- | :-- | :-- | :-- | :-- |
| 4NK0 | 4NK0_2 | [6YQD](https://www.rcsb.org/structure/6YQD) | 2020-04-29 | `6YQD.cif.gz` |
| 4ZAI | 4ZAI_4 | [7JNY](https://www.rcsb.org/structure/7JNY) | 2020-10-07 | `7JNY.cif.gz` |
| 5A0J | 5A0J_1 | [6OH6](https://www.rcsb.org/structure/6OH6) | 2019-04-17 | `6OH6.cif.gz` |

Checksums, sizes and source URLs are in `replacements.tsv`.

## Provenance
- Source: RCSB PDB file download service (`https://files.rcsb.org/download/<ID>.cif.gz`), files stored exactly as served (mmCIF, gzip).
- Retrieved: 2026-11-11.
- Removal dates and replacement IDs come from the RCSB removed-entry record (`https://data.rcsb.org/rest/v1/holdings/removed/<ID>`); the replacement IDs agree with those recorded in the benchmark metadata.
- Licence: PDB archive data are available under CC0 1.0 (wwPDB). The CC BY-NC 4.0 licence of this repository applies to the FAIR metadata package, not to these coordinates.

## Why they are kept apart from the benchmark
- The benchmark labels (physiological / non-physiological) were assigned to the original entries and interfaces (Schweke et al. 2023, doi:10.1002/pmic.202200323). A replacement entry is a new deposition with its own refinement (resolution, number of deposited chains), so the labelled interface cannot be assumed to be identical.
  RCSB reports 2 deposited chains for 6YQD and 1 each for 7JNY and 6OH6.
- Substituting them would change the published benchmark. They are provided for users who want to compare the obsolete benchmark structures with the current ones or to re-derive the interfaces.

## How they are referenced
The metadata snapshots `bioschema/pdb_metadata/6YQD.json`, `7JNY.json` and `6OH6.json` describe these entries, and the `ReplacedBy` property of the three obsolete benchmark items points to them.
