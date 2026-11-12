# Label curation: published benchmark and the annotation file used in this package

The ELIXIR 3D-BioInfo benchmark of homodimer interfaces was published with **836 physiological and 841 non-physiological** interfaces
(Schweke et al. 2023, [doi:10.1002/pmic.202200323](https://doi.org/10.1002/pmic.202200323)).
After publication the benchmark authors curated a number of labels and released an updated annotation file.
**This package uses the updated file, which has 844 physiological and 833 non-physiological interfaces.** This page describes how the two differ and where the information comes from.

## The two annotation files

| | Published file | Updated file (used here) |
| :-- | :-- | :-- |
| File | `benchmark_annotated.csv` (copy in this folder: `benchmark_annotated_published.csv`) | `data/benchmark_annotated_updated_30042023.csv` |
| Source | [vibbits/Elixir-3DBioInfo-Benchmark-Protein-Interfaces](https://github.com/vibbits/Elixir-3DBioInfo-Benchmark-Protein-Interfaces), commit `ff464f1` (22 Feb 2023) | same repository, commits `d4f218e` (2 May 2023) and `1abc4b6` (3 May 2023) |
| Rows | 1,677 | 1,677 |
| Physiological | **836** | **844** |
| Non-physiological | **841** | **833** |
| Extra column | none | `comments` (curation flags) |

The two files have the same 1,677 interfaces (1,645 distinct PDB entries). They differ in the label of 16 interfaces and in the `comments` column.

## What changed

- **16 labels were changed**: 12 interfaces from non-physiological to physiological (`physio after curation`) and 4 from physiological to non-physiological (`non-physio after curation`). Net effect: +8 physiological, -8 non-physiological.
  All 16 are QSalign-derived (empty `InterfaceID`). This agrees with the paper: 141 QSalign non-physiological dimers - 12 + 4 = 133 in the updated file; the 700 ProtCID-derived non-physiological interfaces are unchanged.
- **26 interfaces were flagged but keep their label**: 20 marked `ambiguous` and 6 marked `to delete`. The six `to delete` rows are **not removed** from the file. All 1,677 interfaces are therefore kept in this package, and the flags are exposed so that users can filter them.
- Unflagged interfaces: 1,635.

How the curated set compares, for users who want to apply a filter:

| Set | Interfaces | Physiological | Non-physiological |
| :-- | --: | --: | --: |
| Published labels | 1,677 | 836 | 841 |
| Updated labels (this package) | 1,677 | 844 | 833 |
| Updated labels without the 6 `to delete` rows | 1,671 | 840 | 831 |
| Updated labels without the 26 flagged rows | 1,651 | 822 | 829 |

## Where the labels appear in the metadata

Each interface item in `bioschema/dataset_with_interfaces.json` and in `bioschema/interface_protein_pairs/` has:

| Property | Meaning |
| :-- | :-- |
| `physio` / `label` | label of the updated file (used by this package) |
| `PublishedPhysio` | label in the published benchmark (`benchmark_annotated.csv`) |
| `Comments` | curation flag of the updated file: `physio after curation`, `non-physio after curation`, `ambiguous`, `to delete` (absent for unflagged interfaces) |

Users who need results comparable with the paper should use `PublishedPhysio`.
`PublishedPhysio` is written by `scripts/ppi_benchmark_fair.py` when it is given `--published-csv` (see the README).

## How the curation was done

The following is reconstructed from the benchmark repository: its `ROC_analysis/misclassified_entries_allmethods-03-18-2023.xlsx` (18 March 2023), the commit history and the `comments` column. We did not find a separate written protocol, so the criteria below are inferred from those records and were not described by the curators in a document.

1. **Selection.** For every interface the spreadsheet records whether the consensus score, the Random Forest classifier and AlphaFold classified it correctly. 168 interfaces were classified wrongly by at least two of the three approaches (141 by two, 27 by all three).
2. **Manual review.** A second sheet (`ExcludingNA`) lists 150 of them with a manual `Conclusion` (70 physiological, 52 non-physiological, 22 "not sure", 6 "recommend deletion") and the supporting evidence: ProtCID or ProtCAD crystal-form clustering (in how many crystal forms and PDB entries the interface is found), chain and domain clusters with sequence identity, PDBBA/PISABA assemblies, and notes from the literature (gel filtration, SEC/MALS).
3. **Outcome.** In 18 of the 122 decisive conclusions the published label disagreed with the conclusion. 16 of them were relabelled (the table below). The other two, 5wme_4 and 6np6_5, were left non-physiological although the conclusion was physiological (for 6np6_5 the note says "maybe"). Twenty of the 22 "not sure" interfaces are flagged `ambiguous` (1juh_9 and 6r80_2 are not flagged), and the six "recommend deletion" ones are flagged `to delete`. The deletion recommendations are dated June 2020 in the notes.
4. **Timeline.** Commit `d4f218e` (2 May 2023, "new benchmark file with QIfang analysis") changed 11 labels, of which 10 are in the final file (5hs8_1 was reverted). Commit `1abc4b6` (3 May 2023, "update benchmark file with latest analysis") added six more (3e6d_1, 3lf6_1, 3ljb_2, 4ckn_2, 4o1r_2, 4ofd_1) and added the `comments` column.

### The 16 relabelled interfaces

"Wrong approaches" is the number of the three approaches (consensus score, Random Forest, AlphaFold) that classified the interface wrongly. The evidence column is quoted from the spreadsheet. The full table is in `curated_label_changes_16.csv`; the 26 flagged interfaces with their evidence are in `flagged_entries_26.csv`.

| Interface | Change | Wrong approaches | Evidence recorded in the spreadsheet |
| :-- | :-- | --: | :-- |
| 1fp3_2 | non-physio. -> physio. | 3 | ProtCID=Physio same as 2gz6 interface, 3/12 CFs |
| 2rgv_1 | non-physio. -> physio. | 2 | ProtCID=Physio (22/24 CFs) PDBBA 27/28 entries; PISABA 27/28 entries; same as 4ray |
| 2v54_1 | non-physio. -> physio. | 2 | ProtCID=Physio: 31/37 CFs, 100/123 entries |
| 3e6d_1 | non-physio. -> physio. | 2 | chain cluster: 69/70 CFs |
| 3lf6_1 | non-physio. -> physio. | 2 | chain cluster 6/12 CFs, seq id = 18 |
| 3ljb_2 | non-physio. -> physio. | 2 | no chain cluster. Domain cluster: 12/22 CFs. |
| 4ckn_2 | non-physio. -> physio. | 2 | chain cluster 4/6 CFs, 4/6 entries, seq id = 19. domain cluster 7/9 CFs, 7/9 entries, seq id = 11 |
| 4ldz_1 | non-physio. -> physio. | 3 | ProtCAD=physio; 13/22 CFs, some bound to DNA |
| 4ray_1 | non-physio. -> physio. | 2 | ProtCID=Physio (22/24 CFs; same Pfam/interface as 2rgv) |
| 4zms_1 | non-physio. -> physio. | 3 | ProtCID=Physio (13/22 CFs) same Pfam cluster as 4ldz |
| 5ckl_1 | non-physio. -> physio. | 3 | ProtCAD=physio; 4/14 CFs (4/5 of these UNPs), including 3 CFs of this UNP; other is 62% seqid |
| 5svg_1 | non-physio. -> physio. | 2 | ProtCID=Physio (11/55 CFs) Paper says dimer by SEC/MALS |
| 2gz6_1 | physio. -> non-physio. | 3 | ProtCID=NonPhysio(other interface preferred; 3/12 CFs) Assigned by authors; gel  filtration (not shown in paper) claims dimer. |
| 4nsv_1 | physio. -> non-physio. | 3 | ProtCID=nonphysio;no cluster out of 36 CFs |
| 4o1r_2 | physio. -> non-physio. | 2 | no clusters. 20 CFs of this Pfam architecture, the largest cluster has 2 CFs. But no clusters of this PDB |
| 4ofd_1 | physio. -> non-physio. | 2 | no clusters of this interface. Other interface (4ofd_5) preferred in domain cluster 58/199 CFs. |

## Points to be aware of

- **Selection effect.** Interfaces were reviewed because the classifiers got them wrong. Relabelling them to the reviewer's conclusion makes those classifiers look better on the updated labels. Performance figures that are to be compared with the paper should use the published labels (`PublishedPhysio`).
- **The benchmark README is outdated.** The sentence in the source repository's `benchmark/README.md` ("2 entries were reassigned from physiological to non-physiological, and 11 from non-physiological to physiological. 6 additional entries were deleted") was last edited on 2 May 2023, before the second commit. On 2 May the file had 9 non-physiological to physiological and 2 physiological to non-physiological changes; the current file has 12 and 4, and the six entries are flagged but not deleted. The figures on this page are computed from the files, not from that README.
- **Published analyses use the published labels.** The Random Forest output files of the benchmark repository (1,648 of the 1,677 interfaces) agree with the published file for all entries they contain; 16 of them disagree with the updated file.

## Citation

Please cite the original work:
Schweke H, Xu Q, Tauriello G, et al. Discriminating physiological from non-physiological interfaces in structures of protein complexes: A community-wide study. *Proteomics* 2023 Sep;23(17):e2200323. doi:[10.1002/pmic.202200323](https://doi.org/10.1002/pmic.202200323). PMID 37365936; PMCID PMC10937251. Epub 27 June 2023.
The curation described above was carried out by the benchmark authors in the source repository; this package documents it and does not add to it.
