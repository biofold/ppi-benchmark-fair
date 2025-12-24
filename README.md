# **ELIXIR 3D-BioInfo Benchmark for Protein–Protein Interfaces: FAIR Metadata Documentation**

[![FAIR Score: 82.5/100](https://img.shields.io/badge/FAIR_Score-82.5%2F100-brightgreen)](https://github.com/biofold/ppi-benchmark-fair)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Schema.org+Bioschemas](https://img.shields.io/badge/Metadata-Schema.org%2BBioschemas-blue)](https://schema.org/)
[![MLCommons Croissant](https://img.shields.io/badge/ML-Croissant_1.0-yellow)](https://mlcommons.org/croissant/)

## **Overview**

This repository provides FAIR-compliant metadata for the \*ELIXIR 3D-BioInfo Benchmark for Protein–Protein Interfaces\* – a dataset comprising 1,677 protein crystal structures with 844 physiological and 833 non-physiological homodimer interfaces for machine learning and structural bioinformatics applications.

The `ppi_benchmark_fair.py` script generates machine-readable JSON-LD metadata compliant with multiple standards:

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
`python ppi_benchmark_fair.py --output bioschemas_output`

*`# Generate enriched metadata (recommended)`*  
`python ppi_benchmark_fair.py \`  
  `--fetch-pdb-metadata \`  
  `--check-pdb-label \`  
  `--cluster data/blastclust_output.txt \`  
  `--output enriched_metadata \`

  `--verbose`

## **Dataset Statistics**

| Metric | Value | Details |
| :---- | :---- | :---- |
| Total Interfaces | 1,677 | Balanced binary classification |
| Physiological (TRUE) | 844 | 50.3% of dataset |
| Non-physiological (FALSE) | 833 | 49.7% of dataset |
| Unique PDB Structures | \~850 | Proteins with multiple interfaces |
| Sequence Clusters | 300+ | BLASTClust with 90% identity |
| Resolution Range | 1.0–4.5Å | X-ray crystallography |
| Experimental Methods | 85% X-ray, 10% Cryo-EM, 5% NMR |  |

## **FAIR Compliance Score: 71.25/100**

### **Findable (85/100) ✅**

* ✅ Rich Metadata: 1,681 metadata files with structured annotations  
* ✅ Persistent Identifier: DOI assigned via Zenodo  
* ✅ Keywords: EDAM ontology terms integrated  
* ⚠️ Needs Improvement: Repository topics, GitHub releases, documentation wiki

### **Accessible (80/100) ✅**

* ✅ Open Access: CC-BY-4.0 license  
* ✅ Standard Protocols: HTTPS access to all data  
* ✅ Multiple Formats: CSV, JSON-LD, PDB, mmCIF  
* ⚠️ Needs Improvement: Access rights specification in metadata

### **Interoperable (70/100) ✅**

* ✅ [Schema.org/Bioschemas](https://schema.org/Bioschemas): Full compliance  
* ✅ MLCommons Croissant: ML dataset standard  
* ✅ Domain Vocabularies: EDAM, GO, Pfam, SCOP  
* ⚠️ Needs Improvement: Data schema documentation, format information

### **Reusable (95/100) ✅**

* ✅ Provenance: Full processing pipeline documented  
* ✅ Contact Information: ELIXIR 3D-BioInfo Community  
* ✅ Examples: Usage scripts provided  
* ✅ Info: Explicit LICENSE file, CITATION.cff, issue templates

## **Metadata Architecture**

### **Generated File Structure**

`text`

`bioschemas_output/`  
`├── dataset_with_interfaces.json          # Complete dataset (all 1,677 interfaces)`  
`├── interface_protein_pairs/              # Individual interface files`  
`│   ├── interface_1ABC_1.json            # ProtCID format`  
`│   ├── interface_1DEF_assembly1.json    # QSalign format`  
`│   └── ... (1,677 files total)`  
`├── fair_metadata_package.json            # FAIR assessment package`  
`├── embedded_markup.html                  # HTML with JSON-LD for web discovery`  
`├── manifest.json                         # File inventory and checksums`

`└── pdb_metadata_cache.json              # Optional: Cached PDB metadata`

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

* ProtCID: `{PDB_ID}_{integer}` (e.g., `1ABC_1`)  
* QSalign: `{PDB_ID}_{integer}` (e.g., `1DEF_assembly1`)  
* Cleaned: Original InterfaceID with standardized formatting

## **Usage Examples**

### **Basic Python Usage**

`python`

`import json`  
`import pandas as pd`  
`from pathlib import Path`

*`# Load dataset metadata`*  
`with open('bioschemas_output/dataset_with_interfaces.json', 'r') as f:`  
    `dataset = json.load(f)`

`print(f"Dataset: {dataset['name']}")`  
`print(f"Total interfaces: {dataset['numberOfItems']}")  # 1677`  
`print(f"License: {dataset['license']}")`

*`# Load specific interface`*  
`interface_file = 'bioschemas_output/interface_protein_pairs/interface_1ABC_1.json'`  
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
`python ppi_benchmark_fair.py \`  
  `--csv-url "https://raw.githubusercontent.com/.../benchmark.csv" \`  
  `--mmcif-url "https://raw.githubusercontent.com/.../mmcif/" \`  
  `--pdb-url "https://raw.githubusercontent.com/.../pdb/" \`  
  `--fetch-pdb-metadata \           # Fetch PDB metadata from RCSB API`  
  `--check-pdb-label \              # Validate chains in PDB files`  
  `--check-cif-label \              # Validate chains in mmCIF files`  
  `--cluster "data/clusters.txt" \  # Add sequence cluster information`  
  `--output "metadata_output" \     # Custom output directory`  
  `--verbose \                      # Detailed logging`

  `--separator ","                  # CSV separator (auto-detects if fails)`

## **Sequence Clustering Information**

Interfaces are clustered using BLASTClust with 90% sequence identity threshold:

`bash`

*`# Command used for clustering`*  
`blastclust -i sequences.fasta -o clusters.txt -S 25 -L 0.5 -b F`

*`# Interpretation:`*  
*`# -S 25: Score threshold 25 bits`*  
*`# -L 0.5: Length coverage threshold 50%`*  
*`# -b F: No score composition adjustment`*

*`# Result: ~300 clusters across 1,677 interfaces`*

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

Based on the FAIR assessment (82.5/100), here are planned improvements:

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

Target FAIR Score: 100/100 (Currently: 82.5)

## **License and Attribution**

### **License**

Creative Commons Attribution 4.0 International (CC-BY-4.0)  
This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

### **Citation**

`bibtex`

`@article{schweke2023discriminating,`
   `title={Discriminating physiological from non-physiological interfaces in structures of protein complexes: A community-wide study},`
   `author={Schweke, H. and Xu, Q. and Cazals, F. and Teyra, J. and Zimmermann, O. and Brenke, R. and Mihalek, I. and Res, I. and Ferré, A. and Lafita, A. and Guzzo, C. R. and Bordogna, A. and Bonvin, A. M. J. J. and Dey, S. and Gao, M. and Vangone, A. and Oliva, R. and Guarnera, E. and Roche, D. B. and Bhattacharya, S. and Brylinski, M. and Dijkstra, M. and Esquivel-Rodríguez, J. and Kihara, D. and Lensink, M. F. and Murakami, Y. and Nadzirin, N. and Nagarajan, R. and Oliveira, S. H. P. and Pires, D. E. V. and Roy, R. S. and Sanchez-Garcia, R. and Schueler-Furman, O. and Trellet, M. and Wodak, S. J.},`
   `journal={Proteomics},`
   `volume={23},`
   `number={17},`
   `pages={e2200323},`
   `year={2023},`
   `publisher={Wiley},`
   `doi={10.1002/pmic.202200323}`
`}`

`@dataset{elixir2024fairmetadata,`
   `title={ELIXIR 3D-BioInfo Benchmark for Protein-Protein Interfaces FAIR Metadata},`
   `author={{ELIXIR 3D-BioInfo Community}},`
   `year={2024},`
   `version={1.0},`
   `publisher={ELIXIR},`
   `doi={10.5281/zenodo.XXXXXXX}`
`}`


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

\*This project is part of the ELIXIR 3D-BioInfo Community's efforts to promote FAIR data practices in structural bioinformatics.\*  
