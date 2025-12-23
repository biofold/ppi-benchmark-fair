% Title page
\begin{titlepage}
    \centering
    \vspace*{2cm}
    {\Huge\bfseries ELIXIR 3D-BioInfo Benchmark for\\Protein–Protein Interfaces:\\FAIR Metadata Documentation\par}
    \vspace{1.5cm}
    {\Large Technical Documentation \& Schema Specification\par}
    \vspace{2cm}
    \includegraphics[width=0.4\textwidth]{elixir-logo.pdf} % Replace with actual logo
    \vfill
    {\large Version 1.0\par}
    {\large \today\par}
\end{titlepage}

\tableofcontents
\newpage

###  Overview

\label{sec:overview}

This document provides comprehensive documentation for the **Bioschemas-based FAIR metadata** of the *ELIXIR 3D-BioInfo Benchmark for Protein–Protein Interfaces*. The dataset comprises **1,677 curated protein-protein interfaces** designed for machine learning and structural bioinformatics applications.

The `ppi\_benchmark\_fair.py` script generates **machine-readable JSON-LD metadata** compliant with Schema.org and Bioschemas standards, enabling:



*  **FAIR data publication** (Zenodo, WorkflowHub)
*  **Google Dataset Search indexing**
*  **ML dataset registries** (Croissant-compatible)
*  **Semantic web interoperability**
*  **Reproducible structural bioinformatics**


###  Metadata Architecture

\label{sec:architecture}

####  Core Standards and Vocabularies


\begin{table}[h]
    \centering
    \caption{Standards and vocabularies used in the metadata}
    \label{tab:standards}
    \begin{tabular}{@{}>{\raggedright}p{0.25\linewidth}>{\raggedright}p{0.35\linewidth}>{\raggedright}p{0.3\linewidth}@{}}
        \toprule
        **Standard** & **Purpose** & **Profile** \\\\
        \midrule
        Schema.org & Generic dataset modeling & Dataset, DataCatalogItem \\\\
        Bioschemas & Life-science extensions & Protein, MolecularEntity \\\\
        MLCommons Croissant & ML dataset interoperability & MLDataset \\\\
        EDAM Ontology & Bioinformatics operations & Topic, Operation \\\\
        Gene Ontology (GO) & Molecular functions & GO:0005515 (protein binding) \\\\
        \bottomrule
    \end{tabular}
\end{table}

All metadata is encoded as **JSON-LD** with proper `@context` definitions.

####  JSON Schema Structure

\label{subsec:json-schema}

The generated metadata follows a hierarchical structure with three main components.

\subsubsection{Top-level Dataset (`dataset\_with\_interfaces.json`)}



    [style=json, caption={Top-level dataset structure}, label=lst:dataset]
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


\subsubsection{Individual Interface (`interface\_*.json`)}

Each of the 1,677 interfaces in the `interface\_protein\_pair/` directory is represented as a `DataCatalogItem`:



    [style=json, caption={Individual interface structure}, label=lst:interface]
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


\subsubsection{Supporting Files}



*  `embedded\_markup.html`: HTML file with embedded JSON-LD for web discovery
*  `fair\_metadata\_package.json`: Complete metadata package for FAIR assessment
*  `manifest.json`: Inventory of all generated files with checksums
*  `interface\_protein\_pair/`: Directory containing all 1,677 individual interface files


###  Pipeline Workflow

\label{sec:workflow}

####  Four-Stage Metadata Generation (\texttt{ppi\_benchmark\_fair.py
)}

\begin{figure}[h]
    \centering
    \fbox{\parbox{0.9\textwidth}{
        

1.  **Base Metadata Generation**
\begin{itemize}
1.  \textit{Input}: Raw CSV annotations from GitHub repository
1.  \textit{Output}: Initial JSON-LD with interface identifiers
1.  \textit{Files}: `base\_metadata.json`, initial `interface\_*.json`
1.  \textit{Reproducibility}: Deterministic, offline-capable
\end{itemize}

1.  **Structural Chain Validation**
\begin{itemize}
1.  \textit{Input}: JSON-LD from Stage 1 + PDB/mmCIF files
1.  \textit{Process}: Validates chain identifiers against structural files
1.  \textit{Adds}: `LabelChain1`, `LabelChain2`, provenance data
\end{itemize}

1.  **PDB Metadata Enrichment**
\begin{itemize}
1.  \textit{Input}: Validated JSON-LD from Stage 2
1.  \textit{Process}: Queries RCSB PDB API
1.  \textit{Adds}: Experimental details, sequences, citations
\end{itemize}

1.  **Sequence Cluster Annotation**
\begin{itemize}
1.  \textit{Input}: Enriched JSON-LD from Stage 3 + BlastClust output
1.  \textit{Process}: Maps sequence clusters to interfaces
1.  \textit{Adds}: `ClusterID` for dataset splitting
\end{itemize}

    }}
    \caption{Metadata generation pipeline stages}
    \label{fig:pipeline}
\end{figure}

####  ML Evaluation (\texttt{ppi\_ml\_croissant.py
)}



*  **Input**: Fully enriched JSON-LD metadata
*  **Purpose**: Validates ML usability via Croissant standard
*  **Output**: Performance metrics and ML-ready data splits
*  **Compliance**: MLCommons Croissant 1.0 compatible


###  File Inventory

\label{sec:files}

\begin{longtable}{@{}>{\raggedright}p{0.35\linewidth}>{\raggedright}p{0.45\linewidth}>{\raggedright}p{0.15\linewidth}@{}}
    \caption{Complete file inventory of the dataset}
    \label{tab:files} \\\\
    \toprule
    **File** & **Description** & **Size/Count** \\\\
    \midrule
    \endfirsthead
    
    \toprule
    **File** & **Description** & **Size/Count** \\\\
    \midrule
    \endhead
    
    \midrule
    \multicolumn{3}{r}{\textit{Continued on next page}} \\\\
    \endfoot
    
    \bottomrule
    \endlastfoot
    
    `dataset\_with\_interfaces.json` & Complete dataset metadata & $\sim$15 MB \\\\
    `interface\_protein\_pair/*.json` & Individual interface files & 1,677 files \\\\
    `embedded\_markup.html` & Web-optimized JSON-LD embedding & $\sim$5 MB \\\\
    `fair\_metadata\_package.json` & FAIR assessment package & $\sim$20 MB \\\\
    `manifest.json` & File inventory and checksums & $\sim$50 KB \\\\
    `base\_metadata.json` & Initial, reproducible metadata & $\sim$8 MB \\\\
    `metadata\_with\_label\_chains.json` & Chain-validated metadata & $\sim$10 MB \\\\
    `metadata\_full\_enriched.json` & PDB-enriched metadata & $\sim$15 MB \\\\
    `index.html` & Human-readable dataset portal & $\sim$2 MB \\\\
    \hline
    **Total dataset size** & Including structural file references & $\sim$500 MB \\\\
\end{longtable}

###  User-Specific Features

\label{sec:features}

####  For Machine Learning Researchers




*  **Stratified splits**: Cluster-based partitioning for non-redundant training/validation
*  **Feature-rich**: 50+ structural and sequence features per interface
*  **ML-ready**: Croissant-compatible format for automatic pipeline integration
*  **Reproducible**: Versioned features and labels with provenance


####  For Structural Biologists




*  **Structurally validated**: Chain identifiers verified against biological assemblies
*  **Biologically annotated**: GO terms, organism information, experimental details
*  **FAIR compliant**: Persistent identifiers, provenance tracking
*  **Multi-format**: PDB and mmCIF representations available


####  For Data Stewards \& Curators




*  **Schema-compliant**: Bioschemas and Schema.org adherence
*  **Provenance-aware**: Clear lineage from raw data to enriched metadata
*  **Extensible**: `additionalProperty` for custom annotations
*  **Web-discoverable**: JSON-LD embedded in HTML for search engines


####  For Bioinformaticians




*  **API-ready**: Structured JSON for programmatic access
*  **Standards-based**: EDAM, GO, UniProt cross-references
*  **Pipeline-friendly**: Modular, incremental metadata generation
*  **Version-controlled**: Git-tracked with clear update history


###  Technical Implementation

\label{sec:implementation}

####  Design Principles




1.  **Incremental Enrichment**: Each stage adds metadata without overwriting previous data
1.  **Provenance Preservation**: All modifications are traceable and timestamped
1.  **Schema Safety**: Extensions use `additionalProperty` without breaking validation
1.  **FAIR-by-Design**: Built with Findable, Accessible, Interoperable, Reusable principles


####  File Naming Convention


\begin{verbatim}
interface_{PDB_ID}_{Chain1}_{Chain2}.json
Example: interface_1A2B_C_D.json
\end{verbatim}

####  Context Definitions


All JSON-LD files include standard context definitions:



    [style=json]
    "@context": [
    "https://schema.org/",
    "https://bioschemas.org/",
    {"@vocab": "https://bioschemas.org/"}
    ]


###  Usage Examples

\label{sec:usage}

####  Accessing a Specific Interface




    [language=Python, caption={Python example for accessing interfaces}, label=lst:python-access]
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


####  Loading Full Dataset Metadata




    [language=Python, caption={Loading complete dataset}, label=lst:python-dataset]
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


###  Dataset Statistics

\label{sec:statistics}

\begin{table}[h]
    \centering
    \caption{Summary statistics of the PPI benchmark dataset}
    \label{tab:stats}
    \begin{tabular}{@{}lr@{}}
        \toprule
        **Metric** & **Value** \\\\
        \midrule
        Total Interfaces & 1,677 \\\\
        Unique PDB Entries & $\sim$850 \\\\
        Organisms Covered & 150+ species \\\\
        Experimental Methods (X-ray) & 85% \\\\
        Experimental Methods (Cryo-EM) & 10% \\\\
        Experimental Methods (NMR) & 5% \\\\
        Resolution Range & 1.0Å -- 4.5Å \\\\
        Sequence Clusters (90% identity) & 300+ \\\\
        Metadata Properties per Interface & 100+ \\\\
        \bottomrule
    \end{tabular}
\end{table}

###  License and Attribution

\label{sec:license}

####  License


\begin{center}
    **Creative Commons Attribution 4.0 International (CC-BY-4.0)**
    
    \vspace{0.5em}
    \url{https://creativecommons.org/licenses/by/4.0/}
\end{center}

####  Citation


\begin{verbatim}
@dataset{elixir_ppi_benchmark_2024,
  title = {ELIXIR 3D-BioInfo Benchmark for Protein-Protein Interfaces},
  author = {{ELIXIR 3D-BioInfo Community}},
  year = {2024},
  version = {1.0},
  publisher = {ELIXIR},
  doi = {10.XXXX/YYYY},
  url = {https://example.org/dataset/ppi-benchmark}
}
\end{verbatim}

####  Acknowledgments




*  ELIXIR 3D-BioInfo Community for dataset curation
*  RCSB PDB for providing structural data and API access
*  Bioschemas community for metadata standards development
*  All contributors and maintainers of the benchmark


###  Related Resources

\label{sec:resources}



*  **ELIXIR 3D-BioInfo**: \url{https://elixir-europe.org/platforms/3d-bioinfo}
*  **Bioschemas Protein Profile**: \url{https://bioschemas.org/profiles/Protein}
*  **MLCommons Croissant**: \url{https://mlcommons.org/croissant/}
*  **RCSB PDB API**: \url{https://data.rcsb.org/index.html}
*  **Schema.org**: \url{https://schema.org/}
*  **EDAM Ontology**: \url{http://edamontology.org/}
*  **Gene Ontology**: \url{http://geneontology.org/}


\vspace{1em}
\noindent**Note**: This document describes version 1.0 of the FAIR metadata for the ELIXIR 3D-BioInfo Protein-Protein Interface Benchmark. Last updated: \today.