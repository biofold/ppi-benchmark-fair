# Homology-aware cross-validation splits

Five-fold cross-validation assignments of the PPI benchmark under four ways of defining homology. In each file all
interfaces of a group (`group_key`) have the same `fold`, so that homologous interfaces never end up in both the
training and the held-out part of a cross-validation.

**Origin.** These splits were produced for the OpenEBench implementation of this benchmark
([Zenodo concept DOI 10.5281/zenodo.24323529](https://doi.org/10.5281/zenodo.24323529),
[repository](https://github.com/elixir-unibo/oeb_3dbioinfo_ppi_benchmark),
[interactive page](https://elixir.unibo.it/tools/oeb_3dbioinfo_ppi_benchmark/)). The files here are byte-identical copies of
`datasets/metrics_reference/cv_fold_strategy*.csv` of that package; the copy in the OpenEBench package is the master. They are
**not** the folds of Schweke *et al.* (2023), whose 5-fold analysis selects the best score and threshold of each group (Methods 2.3 and Figure S1 of their supplement) and is not reproduced here. Strategy 2b is the BLASTClust clustering of this benchmark computed by the authors of this package (published here as `data/ppi_benchmark_seqs.clust_25_05`); Strategies 1 and 3 are grouping rules defined for this package, and Strategy 2 is a clustering computed for this package with MMseqs2. The full description,
including the reasons for each strategy, is in [`CV_SPLITTING_STRATEGIES.md`](CV_SPLITTING_STRATEGIES.md) (copied from that package).

## Files

Each CSV has the columns `entry_id` (interface identifier in lower case, for example `1a17_6`), `label` (1 physiological,
0 non-physiological, curated annotation of 30 April 2023), `group_key` and `fold` (0-4).

| Strategy | File | Homology defined by | Groups | Entries per fold | SHA-256 |
| :---- | :---- | :---- | ----: | :---- | :---- |
| 1 | `cv_fold_strategy1_cath_superfamily_gene.csv` | CATH superfamily, UniProt gene as fallback | 853 | 334, 334, 335, 334, 334 | `b4b6698f40c6c6be02c3e8a05aa093d9afd34bebe58cf857f0770013a7f4cb36` |
| 2 | `cv_fold_strategy2_mmseqs2_strict.csv` | MMseqs2, 25% identity, 50% coverage | 1,076 | 335, 334, 334, 335, 333 | `1444c1571a0e48e74e56c8aab71b3449638bc954b36517eceebb05748fb273a0` |
| 2b | `cv_fold_strategy2b_blastclust_reference.csv` | BLASTClust `-S 25 -L 0.5 -b F` (`data/ppi_benchmark_seqs.clust_25_05`) | 1,045 | 335, 333, 333, 336, 334 | `41c2c64eb75178808c5c55fddb0df28a53ccd40eef357a1850ade79019bcecab` |
| 3 | `cv_fold_strategy3_combined_union.csv` | Union of Strategy 1 and 2b (connected components) | 688 | 321, 319, 392, 319, 320 | `31d70491956842985072adb86446486d2586c10bcec2c479b48d5b0600f923ba` |

Strategy 3 is the most conservative (largest group: 132 interfaces) and the recommended default; its folds are less even
because of the large groups.

## Coverage

Each file covers 1,671 of the 1,677 interfaces of the benchmark. The six interfaces flagged "to delete" in the curated
annotation (`2X2A_1`, `3P0C_3`, `3UCR_1`, `4O1W_1`, `4WOV_3`, `5Z70_2`) are excluded and have no assignment. The labels in the files
agree with the `label` of this package for all 1,671 interfaces. The groups of Strategy 2b are exactly the BLASTClust clusters
(`ClusterID`) of this package restricted to the 1,671 interfaces: 1,050 clusters become 1,045 once the six interfaces are removed.

## In the metadata

The same information is attached to every interface in the JSON-LD (`bioschema/dataset_with_interfaces.json` and
`bioschema/interface_protein_pairs/interface_<ID>.json`) as the properties `CVGroup_Strategy1/2/2b/3` and
`CVFold_Strategy1/2/2b/3`, and described in the Croissant `distribution` (file objects with checksums) and `recordSet`.
They are added by `scripts/ppi_benchmark_fair.py --cv-splits data/cv_splits` or, to update existing JSON-LD files,
`python scripts/cv_splits.py patch`.

## Use in the machine-learning script

`scripts/ppi_ml_croissant.py` trains and evaluates its models with the BLASTClust clusters by default (`--cv-strategy blastclust`, the setting of the published ML report, unchanged). `--cv-strategy 1|2|2b|3` uses the groups and the folds of the chosen strategy instead; the six interfaces without an assignment are left out. The properties `CVGroup_*` and `CVFold_*` describe the split and are never used as features. Cross-validated results on this package (5 folds, seed 42, accuracy / ROC-AUC), for comparison:

| Setting | Random Forest | SVM | Logistic Regression |
| :---- | :---- | :---- | :---- |
| `blastclust` (default, GroupKFold) | 0.773 / 0.859 | 0.761 / 0.852 | 0.761 / 0.829 |
| Strategy 1 | 0.784 / 0.864 | 0.776 / 0.859 | 0.768 / 0.839 |
| Strategy 2 | 0.764 / 0.851 | 0.767 / 0.852 | 0.759 / 0.830 |
| Strategy 2b | 0.777 / 0.857 | 0.758 / 0.850 | 0.763 / 0.827 |
| Strategy 3 | 0.780 / 0.858 | 0.764 / 0.847 | 0.761 / 0.826 |

The models use six interface features plus `ClusterSize`, which is a property of the BLASTClust clustering and not of the interface; it is kept as in the published report.

## Example

```python
import pandas as pd
df = pd.read_csv("data/cv_splits/cv_fold_strategy3_combined_union.csv")
for k in range(5):
    train, test = df[df.fold != k], df[df.fold == k]   # no group is on both sides
```

Folds are used as they are provided. A threshold chosen on the held-out fold, or a model trained on interfaces of the held-out
groups, gives optimistic results; the OpenEBench page evaluates a method by learning the threshold on the four training folds only.
