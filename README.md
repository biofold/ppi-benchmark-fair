# **ELIXIR - Protein-Protein Interface Benchmark - FAIR Metadata**

**[→ View the FAIR Data Portal](https://biofold.github.io/ppi-benchmark-fair)**

> **Original work.** The benchmark described here was published by Schweke H, Xu Q, Tauriello G, *et al.* (2023) *Discriminating physiological from non-physiological interfaces in structures of protein complexes: A community-wide study.* Proteomics 2023 Sep;23(17):e2200323, [doi:10.1002/pmic.202200323](https://doi.org/10.1002/pmic.202200323) (PMID [37365936](https://pubmed.ncbi.nlm.nih.gov/37365936/), PMCID [PMC10937251](https://pmc.ncbi.nlm.nih.gov/articles/PMC10937251/)). The data come from the community repository [vibbits/Elixir-3DBioInfo-Benchmark-Protein-Interfaces](https://github.com/vibbits/Elixir-3DBioInfo-Benchmark-Protein-Interfaces). This repository adds FAIR metadata; please cite the paper when you use the benchmark (see [Citation](#citation)).


[![FAIR Score: 87.5/100](https://img.shields.io/badge/FAIR_Score-87.5%2F100-brightgreen)](https://github.com/biofold/ppi-benchmark-fair)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
![DOI: assigned at the first Zenodo release](https://img.shields.io/badge/DOI-assigned%20at%20first%20release-lightgrey.svg)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Schema.org+Bioschemas](https://img.shields.io/badge/Metadata-Schema.org%2BBioschemas-blue)](https://schema.org/)
[![MLCommons Croissant](https://img.shields.io/badge/ML-Croissant_1.0-yellow)](https://mlcommons.org/croissant/)

## **Overview**

This repository provides FAIR-compliant metadata for the _ELIXIR 3D-BioInfo Benchmark for Protein–Protein Interfaces_, a dataset comprising 1,677 homodimer interfaces from 1,645 protein crystal structures, 844 physiological and 833 non-physiological in the annotation file used here for machine learning and structural bioinformatics applications.

**Relation to the published benchmark.** The paper reports 836 physiological and 841 non-physiological interfaces. This package uses the later, curated annotation file of the benchmark authors (`benchmark_annotated_updated_30042023.csv`), in which 16 labels were revised: 844 physiological and 833 non-physiological. Every interface carries both labels (`physio` for the curated file, `PublishedPhysio` for the published one) and its curation flag (`Comments`). See [`data/curation/README.md`](data/curation/README.md) for what was changed, the evidence recorded for each change and the points to be aware of.

The `scripts/ppi_benchmark_fair.py` script generates machine-readable JSON-LD metadata compliant with multiple standards:

| Standard | Purpose | Compliance |
| :---- | :---- | :---- |
| [Schema.org](https://schema.org/) | Dataset discovery | ✅ Full |
| Bioschemas | Life-science extensions | ✅ Profile 1.0-RELEASE |
| MLCommons Croissant | ML dataset interoperability | ✅ Version 1.0 |
| EDAM Ontology | Bioinformatics operations | ✅ Integrated |
| Gene Ontology | Molecular functions | ✅ GO:0005515 |

## **Quick Start**

`bash`

*`# Clone repository`*  
`git clone https://github.com/biofold/ppi-benchmark-fair.git`  
`cd ppi-benchmark-fair`

*`# Install requirements`*  
`pip install -r requirements.txt`

*`# Generate basic metadata`*  
`python scripts/ppi_benchmark_fair.py --output bioschema`

*`# Generate enriched metadata (recommended)`*  
`python scripts/ppi_benchmark_fair.py \`  
  `--fetch-pdb-metadata \`  
  `--check-pdb-label \`  
  `--cluster data/ppi_benchmark_seqs.clust_25_05 \`  
  `--published-csv data/curation/benchmark_annotated_published.csv \`  
  `--output bioschema \`  
  `--log-level INFO`

## **Dataset Statistics**

| Metric | Value | Details |
| :---- | :---- | :---- |
| Total Interfaces | 1,677 | Balanced binary classification |
| Physiological (TRUE) | 844 | 50.3% of dataset (836 in the published benchmark) |
| Non-physiological (FALSE) | 833 | 49.7% of dataset (841 in the published benchmark) |
| Curation flags | 16 relabelled, 20 ambiguous, 6 to delete | all 1,677 interfaces kept; see `data/curation/` |
| Unique PDB Structures | 1,645 | 1,650 distinct strings in the `ID` column of the CSV, which mixes bare PDB codes with `<code>_<N>` ones; 1,673 is the number of mmCIF files |
| Sequence Clusters | 1,050 | BLASTClust, 25% sequence identity, 50% length coverage |
| Resolution Range | 0.90–3.94 Å | median 2.20 Å (1,642 entries; the 3 obsolete entries are not served by RCSB) |
| Experimental Method | X-ray diffraction | 1,642 of 1,642 entries served by RCSB (checked 2026-11-11) |

## **FAIR Compliance Score: 87.5/100**

### **Findable (85/100) ✅**

* ✅ Rich Metadata: 1,681 metadata files with structured annotations  
* ✅ Persistent Identifier: DOI minted by Zenodo at the first release (pending)  
* ✅ Keywords: EDAM ontology terms integrated  
* ⚠️  Possible Improvement: Repository topics, GitHub releases, documentation wiki

### **Accessible (80/100) ✅**

* ✅ Open Access: CC BY-NC 4.0 license  
* ✅ Standard Protocols: HTTPS access to all data  
* ✅ Multiple Formats: CSV, JSON-LD, PDB, mmCIF  
* ⚠️  Possible Improvement: Access rights specification in metadata

### **Interoperable (90/100) ✅**

* ✅ [Schema.org/Bioschemas](https://schema.org/Bioschemas): Full compliance  
* ✅ MLCommons Croissant: ML dataset standard  
* ✅ Domain Vocabularies: EDAM, GO, Pfam, SCOP  
* ⚠️  Possible Improvement: Data schema documentation, format information

### **Reusable (95/100) ✅**

* ✅ Provenance: Full processing pipeline documented  
* ✅ Contact Information: ELIXIR 3D-BioInfo Community  
* ✅ Examples: Usage scripts provided  
* ✅ Info: Explicit LICENSE file, CITATION.cff, issue templates

## **Metadata Architecture**

### **Generated File Structure**

`text`

`bioschema/`  
`├── dataset_with_interfaces.json          # Complete dataset (all 1,677 interfaces, about 29 MB)`  
`├── interface_protein_pairs/              # Individual interface files`  
`│   ├── interface_1A17_6.json             # ProtCID format`  
`│   ├── interface_1A3A_1.json             # QSalign format`  
`│   └── ... (1,677 files total)`  
`├── pdb_metadata/                         # In-repo snapshot of the PDB metadata, one file per PDB entry`  
`│   ├── 1A17.json                         # link target of PDBStructureMetadata (RCSB record, retrieved 2026-01-03)`  
`│   └── ... (1,645 benchmark entries + 3 replacements of obsolete entries)`  
`├── fair_metadata_package.json            # FAIR assessment package`  
`├── embedded_markup.html                  # Dataset-level HTML with JSON-LD for web discovery`  
`├── manifest.json                         # File inventory`  
`└── pdb_metadata_cache.json               # Cached PDB metadata of the original run`

`data/`  
`├── benchmark_pdb_format/, benchmark_mmcif_format/   # structure files of the benchmark interfaces`  
`├── benchmark_annotated_updated_30042023.csv         # annotation file used here (844 / 833)`  
`├── curation/                                        # published labels, the 16 label changes and the 26 flags`  
`└── replacement_structures/                          # current mmCIF files of the entries that replaced 3 obsolete ones (not part of the benchmark)`

Each interface item links to its PDB metadata in two ways: the in-repo snapshot (`value` of `PDBStructureMetadata`) and the live RCSB record (`url`). The three large files (`dataset_with_interfaces.json`, `fair_metadata_package.json`, `embedded_markup.html`) used to be stored with Git LFS. They are now ordinary files, so a plain `git clone` or HTTPS download returns the data.

### **Data Schema**

The metadata follows a hierarchical structure:

`json`

`{`  
  `"@context": ["https://schema.org/", {"cr": "https://mlcommons.org/croissant/1.0"}],`  
  `"@type": ["Dataset", "cr:Dataset"],`  
  `"dct:conformsTo": "https://bioschemas.org/profiles/Dataset/1.0-RELEASE",`  
  `"hasPart": [ /* 1,677 interface items */ ],`  
  `"distribution": [`  
    `{`  
      `"@type": "DataDownload",`  
      `"encodingFormat": "text/csv",`  
      `"contentUrl": "https://.../benchmark_annotated_updated_30042023.csv"`  
    `}`  
  `]`

`}`

### **Interface ID Formats**

* ProtCID: `{PDB_ID}_{integer}` (e.g., `1A17_6`)  
* QSalign: `{PDB_ID}_{integer}` (e.g., `1A3A_1`)  
* Cleaned: Original InterfaceID with standardized formatting

## **Usage Examples**

### **Basic Python Usage**

`python`

`import json`  
`import pandas as pd`  
`from pathlib import Path`

*`# Load dataset metadata`*  
`with open('bioschema/dataset_with_interfaces.json', 'r') as f:`  
    `dataset = json.load(f)`

`print(f"Dataset: {dataset['name']}")`  
`print(f"Total interfaces: {dataset['numberOfItems']}")  # 1677`  
`print(f"License: {dataset['license']}")`

*`# Load specific interface`*  
`interface_file = 'bioschema/interface_protein_pairs/interface_1A17_6.json'`  
`with open(interface_file, 'r') as f:`  
    `interface = json.load(f)`

*`# Extract features for ML`*  
`features = {}`  
`for prop in interface['additionalProperty']:`  
    `if prop['name'] == 'physio':`  
        `features['label'] = 1 if prop['value'] else 0`  
    `elif prop['name'] == 'Buried Surface Area (BSA)':`  
        `features['bsa'] = prop['value']`  
    `elif prop['name'] == 'ClusterID':`  
        `features['cluster'] = prop['value']  # For stratified splits`

`print(f"Features: {features}")`

### **Command Line Options**

`bash`

*`# Full metadata generation with all enrichments`*  
`python scripts/ppi_benchmark_fair.py \`  
  `--csv-url "https://raw.githubusercontent.com/.../benchmark.csv" \`  
  `--mmcif-url "https://raw.githubusercontent.com/.../mmcif/" \`  
  `--pdb-url "https://raw.githubusercontent.com/.../pdb/" \`  
  `--fetch-pdb-metadata \           # Fetch PDB metadata from RCSB API`  
  `--check-pdb-label \              # Validate chains in PDB files`  
  `--check-cif-label \              # Validate chains in mmCIF files`  
  `--cluster "data/clusters.txt" \  # Add sequence cluster information`  
  `--published-csv "data/curation/benchmark_annotated_published.csv" \  # Add the published label (PublishedPhysio)`  
  `--output "metadata_output" \     # Custom output directory`  
  `--log-level INFO \              # Logging detail: DEBUG, INFO, WARNING, ERROR, CRITICAL`  
  `--separator "," \               # CSV separator`  
  `--auto-detect                    # Try other separators if the given one fails`

Two further options control the PDB metadata: by default each interface links to the snapshot `<output>/pdb_metadata/<PDB_ID>.json` (published at `--pdb-snapshot-url`) and to the live RCSB record; `--inline-pdb-metadata` instead embeds the full RCSB record in every interface item, which gives files of more than 100 MB.

## **Sequence Clustering Information**

Interfaces are clustered with BLASTClust at 25% sequence identity and 50% length coverage:

`bash`

*`# Command used for clustering`*  
`blastclust -i sequences.fasta -o clusters.txt -S 25 -L 0.5 -b F`

*`# Interpretation:`*  
*`# -S 25: similarity threshold; values of 3 or more are read by blastclust as a percentage of identical residues, i.e. 25% identity`*  
*`# -L 0.5: minimum length coverage 50%`*  
*`# -b F: coverage is required on only one of the two sequences of a pair`*

*`# Result: 1,050 clusters across 1,677 interfaces (763 singletons, largest cluster 27)`*

Cluster Properties in Metadata:

* `ClusterID`: First InterfaceID in each BLASTClust line  
* `ClusterSize`: Number of interfaces in the cluster  
* `ClusterMembers`: List of other interfaces in same cluster  
* `ClusterMethod`: "BLASTClust sequence clustering"  
* `ClusterMethodOptions`: "-S 25 \-L 0.5 \-b F"

## **PDB Metadata Enrichment**

When `--fetch-pdb-metadata` is enabled, the script fetches comprehensive metadata from RCSB PDB API:

`python`

*`# Example of enriched PDB metadata`*  
`{`  
  `"resolution": 2.1,`  
  `"experimental_method": "X-RAY DIFFRACTION",`  
  `"source_organism": ["Homo sapiens"],`  
  `"sequences": {`  
    `"A": "MSEK...",  # ALL chain sequences (not just representative)`  
    `"B": "MSEK..."`  
  `},`  
  `"is_homomer": True,`  
  `"homomer_type": "dimer",`  
  `"chain_count": 2,`  
  `"unique_sequences": 1`

`}`

## **For Different User Groups**

### **🤖 Machine Learning Researchers**

* Stratified Splits: Cluster-based partitioning for non-redundant training/validation  
* Feature-Rich: 50+ structural and sequence features per interface  
* ML-Ready: Croissant 1.0 compatible for automatic pipeline integration  
* Reproducible: Versioned features with full provenance

### **🔬 Structural Biologists**

* Structurally Validated: Chain IDs verified against biological assemblies  
* Biologically Annotated: GO terms, organism information, experimental details  
* FAIR Compliant: Persistent identifiers, provenance tracking  
* Multi-Format: PDB and mmCIF representations available

### **📊 Data Stewards & Curators**

* Schema-Compliant: Bioschemas and [Schema.org](https://schema.org/) adherence  
* Provenance-Aware: Clear lineage from raw data to enriched metadata  
* Extensible: `additionalProperty` for custom annotations  
* Web-Discoverable: JSON-LD embedded in HTML for search engines

### **💻 Bioinformaticians**

* API-Ready: Structured JSON for programmatic access  
* Standards-Based: EDAM, GO, UniProt cross-references  
* Pipeline-Friendly: Modular, incremental metadata generation  
* Version-Controlled: Git-tracked with clear update history

## **Technical Implementation**

### **Design Principles**

1. Modular Enrichment: Optional steps (PDB metadata, assembly validation, clustering)  
2. Provenance Preservation: All modifications timestamped and logged  
3. Schema Safety: Extensions use `additionalProperty` without breaking validation  
4. FAIR-by-Design: Built with Findable, Accessible, Interoperable, Reusable principles

### **Processing Pipeline**

`text`

`CSV Data → Parse → ProteinInterface Objects → Optional Enrichment:`  
  `├── Assembly Chain Validation (--check-pdb-label/--check-cif-label)`  
  `├── PDB Metadata Fetching (--fetch-pdb-metadata)`  
  `└── Cluster Assignment (--cluster file)`  
    `↓`

`Generate Metadata → Validate → Output Files`

## **FAIR Improvements Roadmap**

Based on the FAIR assessment (87.5/100), here are planned improvements:

### **Low Priority (Potential \+5 points)**

* Create `.github/ISSUE_TEMPLATE/` for structured issue reporting  

### **Medium Priority (Potential \+40 points)**

* Add `schema.json` or `data_dictionary.md` describing data structure  
* Specify access rights in metadata (open/restricted/closed)  
* Include format information (MIME types, extensions) in metadata

### **Low Priority (Potential \+15 points)**

* Add repository topics on GitHub settings  
* Create GitHub releases with version tags  
* Enable and populate documentation wiki

Target FAIR Score: 100/100 (Currently: 87.5)

## **License and Attribution**

### **License**

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)  
This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/). The full text is in the file [`LICENSE`](LICENSE). The PDB entries and metadata that are mirrored from the Protein Data Bank (`bioschema/pdb_metadata/`, `data/replacement_structures/`) remain available under the CC0 1.0 terms of the wwPDB archive.

### **Citation**

`bibtex`

`@article{schweke2023discriminating,`
   `title={Discriminating physiological from non-physiological interfaces in structures of protein complexes: A community-wide study},`
   `author={Schweke, Hugo and Xu, Qifang and Tauriello, Gerardo and Pantolini, Lorenzo and Schwede, Torsten and Cazals, Frédéric and Lhéritier, Alix and Fernandez-Recio, Juan and Rodríguez-Lumbreras, Luis Angel and Schueler-Furman, Ora and Varga, Julia K. and Jiménez-García, Brian and Réau, Manon F. and Bonvin, Alexandre M. J. J. and Savojardo, Castrense and Martelli, Pier-Luigi and Casadio, Rita and Tubiana, Jérôme and Wolfson, Haim J. and Oliva, Romina and Barradas-Bautista, Didier and Ricciardelli, Tiziana and Cavallo, Luigi and Venclovas, Česlovas and Olechnovič, Kliment and Guerois, Raphael and Andreani, Jessica and Martin, Juliette and Wang, Xiao and Terashi, Genki and Sarkar, Daipayan and Christoffer, Charles and Aderinwale, Tunde and Verburgt, Jacob and Kihara, Daisuke and Marchand, Anthony and Correia, Bruno E. and Duan, Rui and Qiu, Liming and Xu, Xianjin and Zhang, Shuang and Zou, Xiaoqin and Dey, Sucharita and Dunbrack, Roland L. and Levy, Emmanuel D. and Wodak, Shoshana J.},`
   `journal={Proteomics},`
   `volume={23},`
   `number={17},`
   `pages={e2200323},`
   `year={2023},`
   `month={sep},`
   `publisher={Wiley},`
   `doi={10.1002/pmic.202200323},`
   `pmid={37365936},`
   `pmcid={PMC10937251}`
`}`

`@dataset{capriotti2026ppifair,`
   `title={ELIXIR - Protein-Protein Interface Benchmark - FAIR Metadata},`
   `author={Capriotti, Emidio and Martelli, Pier Luigi and Savojardo, Castrense},`
   `year={2026},`
   `version={1.0.0},`
   `publisher={Zenodo},`
   `url={https://github.com/biofold/ppi-benchmark-fair},`
   `note={DOI assigned by Zenodo at the first release; cite the concept DOI once it is available}`
`}`

Project report (Zenodo, all versions): Capriotti E, Martelli PL, Savojardo C. *FAIRification of the ELIXIR 3D-Bioinfo Protein-Protein Interface Benchmark*. Zenodo, 2026. [doi:10.5281/zenodo.24144349](https://doi.org/10.5281/zenodo.24144349). The first version is [doi:10.5281/zenodo.24144350](https://doi.org/10.5281/zenodo.24144350).

Please cite **both** the original paper and this metadata package. The author list of the paper is the one of the published article (46 authors, Crossref and PubMed).


### **Dataset Reference Publication**

"Discriminating physiological from non-physiological interfaces in structures of protein complexes: A community-wide study"  
📄 [https://doi.org/10.1002/pmic.202200323](https://doi.org/10.1002/pmic.202200323)

### **Contact**

* ELIXIR 3D-BioInfo Community: [https://elixir-europe.org/platforms/3d-bioinfo](https://elixir-europe.org/platforms/3d-bioinfo)  
* Repository Issues: [GitHub Issues](https://github.com/biofold/ppi-benchmark-fair/issues)  
* Dataset Source: [Original Benchmark](https://github.com/vibbits/Elixir-3DBioInfo-Benchmark-Protein-Interfaces)

## **Related Resources**

| Resource | Description | Link |
| :---- | :---- | :---- |
| ELIXIR 3D-BioInfo | Community platform | [https://elixir-europe.org/platforms/3d-bioinfo](https://elixir-europe.org/platforms/3d-bioinfo) |
| Source Code | This repository | [https://github.com/biofold/ppi-benchmark-fair](https://github.com/biofold/ppi-benchmark-fair) |
| Dataset Repository | Original benchmark data | [https://github.com/vibbits/Elixir-3DBioInfo-Benchmark-Protein-Interfaces](https://github.com/vibbits/Elixir-3DBioInfo-Benchmark-Protein-Interfaces) |
| Bioschemas | Life-science metadata standards | [https://bioschemas.org](https://bioschemas.org/) |
| MLCommons Croissant | ML dataset standard | [https://mlcommons.org/croissant](https://mlcommons.org/croissant) |
| RCSB PDB API | Protein Data Bank API | [https://data.rcsb.org](https://data.rcsb.org/) |

### **Contributing**

We welcome contributions to improve FAIR compliance\! See our [GitHub Issues](https://github.com/biofold/ppi-benchmark-fair/issues) for current improvement tasks.

### **Support**

For questions or support, please:

1. Check the FAIR Improvements Roadmap section  
2. Review existing [GitHub Issues](https://github.com/biofold/ppi-benchmark-fair/issues)  
3. Create a new issue with detailed description

---

*This project is part of the ELIXIR 3D-BioInfo Community and Tools Platform efforts to promote FAIR data practices in structural bioinformatics.*  
