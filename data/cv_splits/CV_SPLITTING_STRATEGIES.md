> Copy of `CV_SPLITTING_STRATEGIES.md` of the OpenEBench package (https://github.com/elixir-unibo/oeb_3dbioinfo_ppi_benchmark). Strategy 2b uses the BLASTClust output of this repository, `data/ppi_benchmark_seqs.clust_25_05`. See `README.md` in this folder for the files, checksums and coverage.

# Multiple cross-validation splitting strategies for the PPI benchmark

**These splits are defined for this package; they are not the folds of Schweke et al.** (Strategies 1 and 3 are grouping rules defined for this package, Strategy 2 is a clustering computed for this package with MMseqs2, and Strategy 2b is the BLASTClust clustering of this benchmark, published in `data/ppi_benchmark_seqs.clust_25_05`.) The manuscript describes a 5-fold cross-validated ROC analysis used to select the best score of each group and its threshold (the point of the ROC most distant from the diagonal; Methods 2.3 and Figure S1 of the supplement). The folds of that analysis are not reproduced here. The splits below keep homologous interfaces together in one fold, and are a proposal of this package.

## Why a plain random 5-fold split is unsafe here

1000 of 1671 benchmark entries (60%) share a CATH superfamily with at least one other entry.
A random split would routinely place near-identical or homologous interfaces in both the
training and held-out folds of the same run, inflating any cross-validated performance estimate.

## Strategy 1 -- CATH superfamily / UniProt gene grouping (original proposal)

Group key: CATH superfamily (primary), UniProt gene id (fallback when superfamily is missing),
singleton (fallback when both are missing). 853 groups, largest group 42 entries. Cheap --
reuses annotation columns already present in `benchmark_annotated.csv`, no external tool
required. Weakness: CATH/gene annotation is missing for 149-247 entries (depending on column),
and two entries can share no CATH superfamily yet still be sequence-homologous (e.g. a
superfamily not yet assigned, or homology below CATH's classification resolution).

## Strategy 2 -- Sequence-identity clustering with MMseqs2 (this package's own run)

Real pairwise sequence identity, not an annotation proxy. One representative chain sequence
was extracted per benchmark entry directly from the PDB structure files in
`benchmark_pdb_format/` (BioPython `PPBuilder`, 1677/1677 succeeded, saved as
`benchmark_sequences.fasta`), then clustered with:

```
mmseqs easy-cluster benchmark_sequences.fasta clusterRes tmp \
    --min-seq-id 0.25 -c 0.5 --cov-mode 1 --cluster-mode 1
```

Parameters chosen to approximate the BLASTClust clustering of this benchmark (`blastclust -i sequences.fasta -o
clusters.txt -S 25 -L 0.5 -b F`, Strategy 2b below; 25%% identity, 50%% coverage of the shorter sequence,
asymmetric/one-sided coverage requirement) as closely as MMseqs2's model allows:
`--min-seq-id 0.25` <-> `-S 25`; `-c 0.5 --cov-mode 1` <-> `-L 0.5 -b F` (one-sided coverage);
`--cluster-mode 1` (connected-component/single-linkage-like) <-> blastclust's transitive
single-linkage clustering (MMseqs2's *default* cluster mode is greedy set-cover, which is NOT
equivalent to blastclust and would under-merge transitively-linked chains).

Result: 1076 clusters, largest 25 entries. Validated against the BLASTClust reference below:
154/200 sampled entries had an EXACT partner-set match between this run and the reference
(remaining disagreement is expected from BLAST-vs-k-mer prefiltering and implementation
differences between blastclust and MMseqs2 -- the two tools are not identical algorithms even
with matched nominal parameters).

## Strategy 2b -- BLASTClust reference clustering (computed by the authors of this package)

The clustering of this benchmark computed by the authors of this package with the NCBI BLASTClust program:
`blastclust -i sequences.fasta -o clusters.txt -S 25 -L 0.5 -b F`, published at
https://biofold.github.io/ppi-benchmark-fair and
https://raw.githubusercontent.com/biofold/ppi-benchmark-fair/refs/heads/master/data/ppi_benchmark_seqs.clust_25_05 .
**100%% of the file's 1677 entries matched this package's own entry_id convention exactly**
(case-insensitive) -- confirming this is the clustering for the identical benchmark, not a
different/related dataset. 1045 clusters, largest 27 entries; the 6 entries dropped during this
package's curation ("to delete" comments) are present in the reference file but correctly
absent from the fold assignment here. **This is the most defensible sequence-identity split
available and should be preferred over Strategy 2 wherever reproducibility of a specific split
across groups/papers matters.**

## Strategy 3 -- Combined union of Strategy 1 + Strategy 2b (RECOMMENDED)

Two entries are forced into the same fold if EITHER signal links them: same CATH
superfamily/gene (Strategy 1) OR same blastclust sequence cluster (Strategy 2b), computed as
connected components of the union graph. This is the most conservative option against leakage
(a structural match some sequence-clustering misses due to low sequence identity at a conserved
fold, or vice versa, is still caught), at the cost of coarser groups (688 components, largest
132 entries) and therefore somewhat less balanced folds (49.1-53.8%% per class fraction vs.
49.4-52.8%% for Strategy 1 alone). Recommended default for any future reproduction work on this
benchmark unless a specific single-signal split is required for comparison with another study.

## Comparison summary

| strategy | n groups | largest group | folds leak? | fold size range | physio fraction range |
|---|---|---|---|---|---|
| 1: CATH superfamily/gene | 853 | 42 | no | 334-335 | 0.494-0.528 |
| 2: MMseqs2 (own run) | 1076 | 25 | no | 333-335 | 0.500-0.506 |
| 2b: blastclust (reference) | 1045 | 26 | no | 333-336 | 0.495-0.519 |
| 3: Combined union | 688 | 132 | no | 319-392 | 0.491-0.538 |

All four give exactly zero groups spanning more than one fold (verified). Files:
`datasets/metrics_reference/cv_fold_strategy{1,2,2b,3}_*.csv`, plus the underlying
`benchmark_sequences.fasta`, `blastclust_reference_S25_L05_bF.txt`, and
`mmseqs2_strict_cluster.tsv`.

## What to do next if you want an exact strategy choice baked into the pipeline

`workflow/crossval_metrics.py` and `workflow/select_best_score_cv.py` currently read
`cv_fold_assignment.csv` (Strategy 1, unchanged for backward compatibility). Point `--
goldstandard_dir` at a directory containing one of the new strategy files renamed to
`cv_fold_assignment.csv` to use a different strategy, or pass `--cv_file` explicitly if you
extend those scripts to accept it.
