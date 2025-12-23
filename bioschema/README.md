::: titlepage
**ELIXIR 3D-BioInfo Benchmark for\
Protein--Protein Interfaces:\
FAIR Metadata Documentation**

Technical Documentation & Schema Specification

![image](elixir-logo.pdf){width="40%"}

Version 1.0

2025-12-23
:::

# Overview {#sec:overview}

This document provides comprehensive documentation for the
**Bioschemas-based FAIR metadata** of the *ELIXIR 3D-BioInfo Benchmark
for Protein--Protein Interfaces*. The dataset comprises **1,677 curated
protein-protein interfaces** designed for machine learning and
structural bioinformatics applications.

The `ppi_benchmark_fair.py` script generates **machine-readable JSON-LD
metadata** compliant with Schema.org and Bioschemas standards, enabling:

- **FAIR data publication** (Zenodo, WorkflowHub)

- **Google Dataset Search indexing**

- **ML dataset registries** (Croissant-compatible)

- **Semantic web interoperability**

- **Reproducible structural bioinformatics**

# Metadata Architecture {#sec:architecture}

## Core Standards and Vocabularies

::: {#tab:standards}
  ----------------------------------------------------------------
  **Standard**      **Purpose**              **Profile**
  ----------------- ------------------------ ---------------------
  Schema.org        Generic dataset modeling Dataset,
                                             DataCatalogItem

  Bioschemas        Life-science extensions  Protein,
                                             MolecularEntity

  MLCommons         ML dataset               MLDataset
  Croissant         interoperability         

  EDAM Ontology     Bioinformatics           Topic, Operation
                    operations               

  Gene Ontology     Molecular functions      GO:0005515 (protein
  (GO)                                       binding)
  ----------------------------------------------------------------

  : Standards and vocabularies used in the metadata
:::

All metadata is encoded as **JSON-LD** with proper `@context`
definitions.

## JSON Schema Structure {#subsec:json-schema}

The generated metadata follows a hierarchical structure with three main
components.

### Top-level Dataset (`dataset_with_interfaces.json`)

``` {#lst:dataset style="json" caption="Top-level dataset structure" label="lst:dataset"}
{
  "@context": [
    "https://schema.org/",
    "https://bioschemas.org/",
    {"@vocab": "https://bioschemas.org/"}
  ],
  "@type": "Dataset",
  "@id": "https://example.org/dataset/ppi-benchmark",
  "name": "ELIXIR 3D-BioInfo Benchmark for Protein–Protein Interfaces",
  "description": "A benchmark dataset of 1,677 protein-protein interfaces...",
  "hasPart": [
    "interface_1.jsonld",
    "interface_2.jsonld",
    ...
  ],
  "keywords": [
    "protein-protein interaction",
    "structural bioinformatics",
    "machine learning",
    "3D structures"
  ],
  "license": "https://creativecommons.org/licenses/by/4.0/",
  "version": "1.0.0",
  "datePublished": "2024-01-15",
  "creator": [
    {
      "@type": "Organization",
      "name": "ELIXIR 3D-BioInfo Community"
    }
  ],
  "distribution": [
    {
      "@type": "DataDownload",
      "encodingFormat": "application/json+ld",
      "contentUrl": "https://example.org/files/dataset_with_interfaces.json"
    }
  ]
}
```

### Individual Interface (`interface_*.json`)

Each of the 1,677 interfaces in the `interface_protein_pair/` directory
is represented as a `DataCatalogItem`:

``` {#lst:interface style="json" caption="Individual interface structure" label="lst:interface"}
{
  "@type": "DataCatalogItem",
  "@id": "https://example.org/interface/1A2B_C_D",
  "identifier": "1A2B_C_D",
  "name": "Interface between chains C and D in PDB 1A2B",
  "description": "Protein-protein interface with annotated features...",
  "additionalProperty": [
    {
      "@type": "PropertyValue",
      "name": "LabelChain1",
      "value": "C"
    },
    {
      "@type": "PropertyValue",
      "name": "LabelChain2",
      "value": "D"
    },
    {
      "@type": "PropertyValue",
      "name": "ClusterID",
      "value": "cluster_42"
    },
    {
      "@type": "PropertyValue",
      "name": "features",
      "value": "{\"residue_count\": 45, \"area\": 1200.5, ...}"
    },
    {
      "@type": "PropertyValue",
      "name": "labels",
      "value": "{\"binding_affinity\": \"high\", \"interface_type\": \"obligate\"}"
    }
  ],
  "mainEntity": {
    "@type": "Protein",
    "@id": "https://www.rcsb.org/structure/1A2B",
    "identifier": "PDB:1A2B",
    "name": "Example protein complex",
    "taxonomicRange": {
      "@id": "https://www.ncbi.nlm.nih.gov/taxonomy/9606",
      "name": "Homo sapiens"
    },
    "hasMolecularFunction": {
      "@id": "http://purl.obolibrary.org/obo/GO_0005515",
      "name": "protein binding"
    },
    "hasRepresentation": [
      {
        "@type": "MolecularEntity",
        "encodingFormat": "chemical/x-pdb",
        "contentUrl": "https://files.rcsb.org/download/1A2B.pdb"
      },
      {
        "@type": "MolecularEntity",
        "encodingFormat": "chemical/x-mmCIF",
        "contentUrl": "https://files.rcsb.org/download/1A2B.cif"
      }
    ]
  },
  "isPartOf": {
    "@id": "https://example.org/dataset/ppi-benchmark",
    "@type": "Dataset"
  }
}
```

### Supporting Files

- `embedded_markup.html`: HTML file with embedded JSON-LD for web
  discovery

- `fair_metadata_package.json`: Complete metadata package for FAIR
  assessment

- `manifest.json`: Inventory of all generated files with checksums

- `interface_protein_pair/`: Directory containing all 1,677 individual
  interface files

# Pipeline Workflow {#sec:workflow}

## Four-Stage Metadata Generation (`ppi_benchmark_fair.py`)

<figure id="fig:pipeline">

<figcaption>Metadata generation pipeline stages</figcaption>
</figure>

## ML Evaluation (`ppi_ml_croissant.py`)

- **Input**: Fully enriched JSON-LD metadata

- **Purpose**: Validates ML usability via Croissant standard

- **Output**: Performance metrics and ML-ready data splits

- **Compliance**: MLCommons Croissant 1.0 compatible

# File Inventory {#sec:files}

::: {#tab:files}
  ------------------------------------------------------------------------------------------------
  **File**                            **Description**                 **Size/Count**
  ----------------------------------- ------------------------------- ----------------------------
  **File**                            **Description**                 **Size/Count**

                                                                      

  `dataset_with_interfaces.json`      Complete dataset metadata       $\sim$`<!-- -->`{=html}15 MB

  `interface_protein_pair/*.json`     Individual interface files      1,677 files

  `embedded_markup.html`              Web-optimized JSON-LD embedding $\sim$`<!-- -->`{=html}5 MB

  `fair_metadata_package.json`        FAIR assessment package         $\sim$`<!-- -->`{=html}20 MB

  `manifest.json`                     File inventory and checksums    $\sim$`<!-- -->`{=html}50 KB

  `base_metadata.json`                Initial, reproducible metadata  $\sim$`<!-- -->`{=html}8 MB

  `metadata_with_label_chains.json`   Chain-validated metadata        $\sim$`<!-- -->`{=html}10 MB

  `metadata_full_enriched.json`       PDB-enriched metadata           $\sim$`<!-- -->`{=html}15 MB

  `index.html`                        Human-readable dataset portal   $\sim$`<!-- -->`{=html}2 MB

  **Total dataset size**              Including structural file       $\sim$`<!-- -->`{=html}500
                                      references                      MB
  ------------------------------------------------------------------------------------------------

  : Complete file inventory of the dataset
:::

# User-Specific Features {#sec:features}

## For Machine Learning Researchers

- **Stratified splits**: Cluster-based partitioning for non-redundant
  training/validation

- **Feature-rich**: 50+ structural and sequence features per interface

- **ML-ready**: Croissant-compatible format for automatic pipeline
  integration

- **Reproducible**: Versioned features and labels with provenance

## For Structural Biologists

- **Structurally validated**: Chain identifiers verified against
  biological assemblies

- **Biologically annotated**: GO terms, organism information,
  experimental details

- **FAIR compliant**: Persistent identifiers, provenance tracking

- **Multi-format**: PDB and mmCIF representations available

## For Data Stewards & Curators

- **Schema-compliant**: Bioschemas and Schema.org adherence

- **Provenance-aware**: Clear lineage from raw data to enriched metadata

- **Extensible**: `additionalProperty` for custom annotations

- **Web-discoverable**: JSON-LD embedded in HTML for search engines

## For Bioinformaticians

- **API-ready**: Structured JSON for programmatic access

- **Standards-based**: EDAM, GO, UniProt cross-references

- **Pipeline-friendly**: Modular, incremental metadata generation

- **Version-controlled**: Git-tracked with clear update history

# Technical Implementation {#sec:implementation}

## Design Principles

1.  **Incremental Enrichment**: Each stage adds metadata without
    overwriting previous data

2.  **Provenance Preservation**: All modifications are traceable and
    timestamped

3.  **Schema Safety**: Extensions use `additionalProperty` without
    breaking validation

4.  **FAIR-by-Design**: Built with Findable, Accessible, Interoperable,
    Reusable principles

## File Naming Convention

    interface_{PDB_ID}_{Chain1}_{Chain2}.json
    Example: interface_1A2B_C_D.json

## Context Definitions

All JSON-LD files include standard context definitions:

``` {style="json"}
"@context": [
  "https://schema.org/",
  "https://bioschemas.org/",
  {"@vocab": "https://bioschemas.org/"}
]
```

# Usage Examples {#sec:usage}

## Accessing a Specific Interface

``` {#lst:python-access .python language="Python" caption="Python example for accessing interfaces" label="lst:python-access"}
import json
import os

# Load individual interface
interface_path = os.path.join('interface_protein_pair', 
                             'interface_1A2B_C_D.json')
with open(interface_path, 'r') as f:
    interface = json.load(f)
    
print(f"Interface ID: {interface['identifier']}")
print(f"Chains: {interface['additionalProperty'][0]['value']}-"
      f"{interface['additionalProperty'][1]['value']}")
print(f"Cluster: {interface['additionalProperty'][2]['value']}")

# Extract features
features = json.loads(interface['additionalProperty'][3]['value'])
print(f"Residue count: {features.get('residue_count', 'N/A')}")
print(f"Interface area: {features.get('area', 'N/A')} Å²")
```

## Loading Full Dataset Metadata

``` {#lst:python-dataset .python language="Python" caption="Loading complete dataset" label="lst:python-dataset"}
import json

# Load complete dataset
with open('dataset_with_interfaces.json', 'r') as f:
    dataset = json.load(f)
    
print(f"Dataset: {dataset['name']}")
print(f"Description: {dataset['description'][:100]}...")
print(f"Total interfaces: {len(dataset['hasPart'])}")
print(f"License: {dataset['license']}")

# Access first 5 interfaces
for interface_ref in dataset['hasPart'][:5]:
    print(f" - {interface_ref}")
```

# Dataset Statistics {#sec:statistics}

::: {#tab:stats}
  **Metric**                                             **Value**
  ----------------------------------- ----------------------------
  Total Interfaces                                           1,677
  Unique PDB Entries                    $\sim$`<!-- -->`{=html}850
  Organisms Covered                                   150+ species
  Experimental Methods (X-ray)                                 85%
  Experimental Methods (Cryo-EM)                               10%
  Experimental Methods (NMR)                                    5%
  Resolution Range                                    1.0Å -- 4.5Å
  Sequence Clusters (90% identity)                            300+
  Metadata Properties per Interface                           100+

  : Summary statistics of the PPI benchmark dataset
:::

# License and Attribution {#sec:license}

## License

::: center
**Creative Commons Attribution 4.0 International (CC-BY-4.0)**

<https://creativecommons.org/licenses/by/4.0/>
:::

## Citation

    @dataset{elixir_ppi_benchmark_2024,
      title = {ELIXIR 3D-BioInfo Benchmark for Protein-Protein Interfaces},
      author = {{ELIXIR 3D-BioInfo Community}},
      year = {2024},
      version = {1.0},
      publisher = {ELIXIR},
      doi = {10.XXXX/YYYY},
      url = {https://example.org/dataset/ppi-benchmark}
    }

## Acknowledgments

- ELIXIR 3D-BioInfo Community for dataset curation

- RCSB PDB for providing structural data and API access

- Bioschemas community for metadata standards development

- All contributors and maintainers of the benchmark

# Related Resources {#sec:resources}

- **ELIXIR 3D-BioInfo**:
  <https://elixir-europe.org/platforms/3d-bioinfo>

- **Bioschemas Protein Profile**:
  <https://bioschemas.org/profiles/Protein>

- **MLCommons Croissant**: <https://mlcommons.org/croissant/>

- **RCSB PDB API**: <https://data.rcsb.org/index.html>

- **Schema.org**: <https://schema.org/>

- **EDAM Ontology**: <http://edamontology.org/>

- **Gene Ontology**: <http://geneontology.org/>

**Note**: This document describes version 1.0 of the FAIR metadata for
the ELIXIR 3D-BioInfo Protein-Protein Interface Benchmark. Last updated:
2025-12-23.
