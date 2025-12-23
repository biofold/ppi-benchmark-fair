# ELIXIR 3D-BioInfo Benchmark for Protein–Protein Interfaces: FAIR Metadata Documentation

**Version 1.0** · **Technical Documentation & Schema Specification**

---

## Overview

This document provides comprehensive documentation for the **Bioschemas-based FAIR metadata** of the *ELIXIR 3D-BioInfo Benchmark for Protein–Protein Interfaces*. The dataset comprises **1,677 curated protein-protein interfaces** designed for machine learning and structural bioinformatics applications.

The `ppi_benchmark_fair.py` script generates **machine-readable JSON-LD metadata** compliant with Schema.org and Bioschemas standards, enabling:

- **FAIR data publication** (Zenodo, WorkflowHub)
- **Google Dataset Search indexing**
- **ML dataset registries** (Croissant-compatible)
- **Semantic web interoperability**
- **Reproducible structural bioinformatics**

---

## Metadata Architecture

### Core Standards and Vocabularies

| Standard | Purpose | Profile |
|----------|---------|---------|
| **Schema.org** | Generic dataset modeling | Dataset, DataCatalogItem |
| **Bioschemas** | Life-science extensions | Protein, MolecularEntity |
| **MLCommons Croissant** | ML dataset interoperability | MLDataset |
| **EDAM Ontology** | Bioinformatics operations | Topic, Operation |
| **Gene Ontology (GO)** | Molecular functions | GO:0005515 (protein binding) |

All metadata is encoded as **JSON-LD** with proper `@context` definitions.

### JSON Schema Structure

The generated metadata follows a hierarchical structure with three main components.

#### Top-level Dataset (`dataset_with_interfaces.json`)

```json
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
    "..."
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


#### Supporting Files

* `embedded_markup.html`: HTML file with embedded JSON-LD for web discovery
* `fair_metadata_package.json`: Complete metadata package for FAIR assessment
* `manifest.json`: Inventory of all generated files with checksums
* `interface_protein_pair/`: Directory containing all 1,677 individual interface files

---

## Pipeline Workflow

### Four-Stage Metadata Generation (`ppi_benchmark_fair.py`)

**Metadata Generation Pipeline Stages:**

1. **Base Metadata Generation**
   * *Input*: Raw CSV annotations from GitHub repository
   * *Output*: Initial JSON-LD with interface identifiers
   * *Files*: `base_metadata.json`, initial `interface_*.json`
   * *Reproducibility*: Deterministic, offline-capable

2. **Structural Chain Validation**
   * *Input*: JSON-LD from Stage 1 + PDB/mmCIF files
   * *Process*: Validates chain identifiers against structural files
   * *Adds*: `LabelChain1`, `LabelChain2`, provenance data

3. **PDB Metadata Enrichment**
   * *Input*: Validated JSON-LD from Stage 2
   * *Process*: Queries RCSB PDB API
   * *Adds*: Experimental details, sequences, citations

4. **Sequence Cluster Annotation**
   * *Input*: Enriched JSON-LD from Stage 3 + BlastClust output
   * *Process*: Maps sequence clusters to interfaces
   * *Adds*: `ClusterID` for dataset splitting

### ML Evaluation (`ppi_ml_croissant.py`)

* *Input*: Fully enriched JSON-LD metadata
* *Purpose*: Validates ML usability via Croissant standard
* *Output*: Performance metrics and ML-ready data splits
* *Compliance*: MLCommons Croissant 1.0 compatible

---

## File Inventory

| File | Description | Size/Count |
|------|-------------|------------|
| `dataset_with_interfaces.json` | Complete dataset metadata | ~15 MB |
| `interface_protein_pair/*.json` | Individual interface files | 1,677 files |
| `embedded_markup.html` | Web-optimized JSON-LD embedding | ~5 MB |
| `fair_metadata_package.json` | FAIR assessment package | ~20 MB |
| `manifest.json` | File inventory and checksums | ~50 KB |
| `base_metadata.json` | Initial, reproducible metadata | ~8 MB |
| `metadata_with_label_chains.json` | Chain-validated metadata | ~10 MB |
| `metadata_full_enriched.json` | PDB-enriched metadata | ~15 MB |
| `index.html` | Human-readable dataset portal | ~2 MB |
| **Total dataset size** | Including structural file references | **~500 MB** |

---

## User-Specific Features

### For Machine Learning Researchers

* **Stratified splits**: Cluster-based partitioning for non-redundant training/validation
* **Feature-rich**: 50+ structural and sequence features per interface
* **ML-ready**: Croissant-compatible format for automatic pipeline integration
* **Reproducible**: Versioned features and labels with provenance

### For Structural Biologists

* **Structurally validated**: Chain identifiers verified against biological assemblies
* **Biologically annotated**: GO terms, organism information, experimental details
* **FAIR compliant**: Persistent identifiers, provenance tracking
* **Multi-format**: PDB and mmCIF representations available

### For Data Stewards & Curators

* **Schema-compliant**: Bioschemas and Schema.org adherence
* **Provenance-aware**: Clear lineage from raw data to enriched metadata
* **Extensible**: `additionalProperty` for custom annotations
* **Web-discoverable**: JSON-LD embedded in HTML for search engines

### For Bioinformaticians

* **API-ready**: Structured JSON for programmatic access
* **Standards-based**: EDAM, GO, UniProt cross-references
* **Pipeline-friendly**: Modular, incremental metadata generation
* **Version-controlled**: Git-tracked with clear update history

---

## Technical Implementation

### Design Principles

1. **Incremental Enrichment**: Each stage adds metadata without overwriting previous data
2. **Provenance Preservation**: All modifications are traceable and timestamped
3. **Schema Safety**: Extensions use `additionalProperty` without breaking validation
4. **FAIR-by-Design**: Built with Findable, Accessible, Interoperable, Reusable principles

### File Naming Convention


---

## Usage Examples

### Accessing a Specific Interface

```python
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
