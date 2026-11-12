# **Data Dictionary: Protein-Protein Interface Dataset** 

## **Overview**

A benchmark dataset containing 1,677 homodimer interfaces from 1,645 protein crystal structures, 844 physiological and 833 non-physiological in the annotation file used here (836 and 841 in the published benchmark), for evaluating protein-protein interface scoring functions. Designed for machine learning applications in structural bioinformatics.

## **Dataset Metadata**

| Property | Value |
| :---- | :---- |
| Name | Protein-Protein Interaction Interface Benchmark Dataset |
| Identifier | [https://doi.org/10.5281/zenodo.24149636](https://doi.org/10.5281/zenodo.24149636) |
| URL | [https://github.com/vibbits/Elixir-3DBioInfo-Benchmark-Protein-Interfaces](https://github.com/vibbits/Elixir-3DBioInfo-Benchmark-Protein-Interfaces) |
| License | CC BY-NC 4.0 ([https://creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/)) |
| Version | 1.0 |
| Date Published | 2023-04-30 |
| Date Modified | 2025-12-26 |
| Size | 1677 entries |
| Schema Conformance | ML Commons Croissant 1.0 & BioSchemas Dataset Profile 1.0-RELEASE |

## **Root Dataset Properties**

| Field | Type | Required | Description | Example/Format |
| :---- | :---- | :---- | :---- | :---- |
| `@context` | Array | Yes | JSON-LD context URLs | `["https://schema.org/", {"cr": "https://mlcommons.org/croissant/1.0"}]` |
| `@type` | Array | Yes | Dataset types | `["Dataset", "cr:Dataset"]` |
| `@id` | String (URI) | Yes | Dataset identifier URL | `"https://github.com/vibbits/Elixir-3DBioInfo-Benchmark-Protein-Interfaces"` |
| `cr:conformsTo` | String | Yes | ML Commons Croissant spec | `"https://mlcommons.org/croissant/1.0"` |
| `dct:conformsTo` | String | Yes | BioSchemas profile | `"https://bioschemas.org/profiles/Dataset/1.0-RELEASE"` |
| `name` | String | Yes | Dataset name | `"Protein-Protein Interaction Interface Benchmark Dataset"` |
| `description` | String | Yes | Comprehensive description | Text description |
| `identifier` | String (URI) | Yes | DOI identifier | `"https://doi.org/10.5281/zenodo.24149636"` |
| `url` | String (URI) | Yes | GitHub repository URL | `"https://github.com/vibbits/Elixir-3DBioInfo-Benchmark-Protein-Interfaces"` |
| `license` | String (URI) | Yes | License URL | `"https://creativecommons.org/licenses/by-nc/4.0/"` |
| `keywords` | Array\[DefinedTerm\] | Yes | EDAM ontology keywords | Array of DefinedTerm objects |
| `creator` | Array\[Organization\] | Yes | Creator organizations | Array of Organization objects |
| `datePublished` | String (Date) | Yes | Publication date | `"2023-04-30"` (YYYY-MM-DD) |
| `publisher` | Organization | Yes | Publisher organization | Organization object |
| `version` | String | Yes | Dataset version | `"1.0"` (Major.Minor) |
| `citation` | ScholarlyArticle | Yes | Citation reference | ScholarlyArticle object |
| `variableMeasured` | Array\[PropertyValue\] | Yes | Measured variables | Array of PropertyValue objects |
| `measurementTechnique` | Array\[String\] | Yes | Experimental techniques | `["X-ray crystallography", "..."]` |
| `dateCreated` | String (Date) | Yes | Creation date | `"2023-04-30"` |
| `dateModified` | String (Date) | Yes | Last modification date | `"2025-12-26"` |
| `maintainer` | Organization | Yes | Maintaining organization | Organization object |
| `size` | String | Yes | Dataset size description | `"1677 entries"` |
| `hasPart` | Array\[Interface\] | Yes | Interface entries | Array of Interface objects |

## **Interface Entry Structure (hasPart items)**

| Field | Type | Required | Description | Pattern/Example |
| :---- | :---- | :---- | :---- | :---- |
| `@type` | String | Yes | Data catalog item type | `"DataCatalogItem"` |
| `name` | String | Yes | Interface name | `^Interface [A-Z0-9]{4}_\d+$` |
| `description` | String | Yes | Interface description | Text description |
| `identifier` | String | Yes | Interface identifier | `^[A-Z0-9]{4}_\d+$` |
| `url` | String (URI) | Yes | RCSB PDB URL | `^https://www\.rcsb\.org/structure/[A-Z0-9]{4}$` |
| `additionalProperty` | Array\[PropertyValue\] | Yes | Interface properties | Array (min 20 items) |
| `mainEntity` | Protein | Yes | Protein entity | Protein object |

## **Interface Additional Properties \- Tabular View**

### **Core Interface Properties**

| Property Name | Type | Required | Description | Allowed Values | Example |
| :---- | :---- | :---- | :---- | :---- | :---- |
| `InterfaceID` | String | Yes | Unique interface identifier | Pattern: `[A-Z0-9]{4}_\d+` | `"1A17_6"` |
| `InterfaceSource` | String | Yes | Source of interface ID | `"ProtCID"`, `"QSalign"` | `"ProtCID"` |
| `physio` | Boolean | Yes | Physiological classification in the curated annotation file used by this package (844 physiological, 833 non-physiological) | `true`, `false` | `false` |
| `PublishedPhysio` | Boolean | Yes | Classification in the published benchmark (Schweke et al. 2023; 836 physiological, 841 non-physiological). Differs from `physio` for 16 interfaces | `true`, `false` | `false` |
| `Comments` | String | No | Curation flag of the updated annotation file: `physio after curation`, `non-physio after curation`, `ambiguous` or `to delete`. Absent for the 1,635 unflagged interfaces. See `data/curation/README.md` | String | `"physio after curation"` |
| `label` | Integer | Yes | Numeric label | `0`, `1` | `0` |
| `AuthChain1` | String | Yes | First chain in interface | Single letter or `"nan"` | `"A"` |
| `AuthChain2` | String | Yes | Second chain in interface | Single letter or `"nan"` | `"A"` |
| `SymmetryOp1` | String | Yes | Symmetry operation chain 1 | String or `"nan"` | `"1_555"` |
| `SymmetryOp2` | String | Yes | Symmetry operation chain 2 | String or `"nan"` | `"15_545"` |
| `LabelChain1` | String | Yes | First chain in assembly | Single letter | `"A"` |
| `LabelChain2` | String | Yes | Second chain in assembly | Single letter | `"B"` |

### **Interface Metrics**

| Property Name | Type | Required | Unit | Description | Validation | Example |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| `Buried Surface Area (BSA)` | Number | Yes | Å² | Total buried surface area | \> 0 | `4153.16` |
| `Atomic Contacts` | Integer | Yes | count | Number of atomic contacts | ≥ 0 | `138` |
| `BSA Polar` | Number | Yes | Å² | Polar component of BSA | \> 0 | `1651.43` |
| `BSA Apolar` | Number | Yes | Å² | Apolar component of BSA | \> 0 | `2501.74` |
| `Fraction Polar` | Number | Yes | ratio | Fraction of polar contacts | 0-1 | `0.3976` |
| `Fraction Apolar` | Number | Yes | ratio | Fraction of apolar contacts | 0-1 | `0.6024` |

### **Biological Information**

| Property Name | Type | Required | Description | Format | Example |
| :---- | :---- | :---- | :---- | :---- | :---- |
| `Gene` | String | Yes | Gene name | String | `"PPP5C"` |
| `Superfamily` | String | Yes | Protein superfamily | SCOP format | `"1.25.40.10"` |
| `Pfam` | String | No | Pfam domain | PFxxxxx | `"PF00359"` |

### **Sequence Clustering**

| Property Name | Type | Required | Description | Format | Example |
| :---- | :---- | :---- | :---- | :---- | :---- |
| `ClusterID` | String | Yes | Sequence cluster ID | String | `"1A17_6"` |
| `ClusterSize` | Integer | Yes | Number in cluster | \> 0 | `1` |
| `ClusterMembers` | Array\[String\] | Yes | Other interfaces in cluster | Array of IDs | `["3O77_2", ...]` |
| `ClusterMethod` | String | Yes | Clustering method | String | `"BLASTClust sequence clustering"` |
| `ClusterMethodOptions` | String | Yes | Clustering parameters | String | `"-S 25 -L 0.5 -b F"` |

### **PDB Metadata Properties**

| Property Name | Type | Required | Description | Validation | Example |
| :---- | :---- | :---- | :---- | :---- | :---- |
| `PDBStructureMetadata` | String (URL) | Yes | Link to the in-repo snapshot of the RCSB metadata of the PDB entry (`bioschema/pdb_metadata/<PDB_ID>.json`, see "PDB Metadata Snapshot Files"). The property `url` is the live RCSB record, `valueReference` the parsed summary, and `additionalProperty` the interface-specific context (`interface_chains`, `interface_sequence_analysis`, `enrichment_timestamp`) | URL | `"https://biofold.github.io/ppi-benchmark-fair/bioschema/pdb_metadata/1A17.json"` |
| `PDBMetadataStatus` | String | Yes | Metadata status | String | `"Complete metadata available"` |
| `Resolution` | Number | Yes | X-ray resolution | \> 0 (Å) | `2.45` |
| `ExperimentalMethod` | String | Yes | Experimental method (all benchmark entries served by RCSB are X-ray diffraction) | String | `"X-RAY DIFFRACTION"` |
| `SourceOrganism` | String | Yes | Source organism | String | `"Homo sapiens"` |
| `HomomericStructure` | Boolean | Yes | Is homomeric | Boolean | `true` |
| `HomomerType` | String | Yes | Type of homomer | String | `"monomer"` |
| `ChainCount` | Integer | Yes | Number of chains | \> 0 | `1` |
| `UniqueSequenceCount` | Integer | Yes | Unique sequences | \> 0 | `1` |
| `Chain_*_Sequence` | String | Yes\* | Amino acid sequence | Pattern: `^[A-Z]+$` | `"RDEPPADGALK..."` |
| `PrimaryCitation` | String | Yes | Publication title | String | Title text |
| `PublicationDOI` | String | Yes | Publication DOI | DOI format | `"10.1093/emboj/17.5.1192"` |
| `Entity_*` | String | Yes\* | Entity information | String | `"Chains: A, Length: 166"` |
| `NCBI_Taxonomy_ID` | String | Yes | Taxonomy ID | Pattern: `^\d+$` | `"9606"` |
| `AssociatedInterface` | String | Yes | Interface ID | Pattern: `[A-Z0-9]{4}_\d+` | `"1A17_6"` |
| `InterfaceChains` | String | Yes | Chains in interface | String | `"A-A"`, `"nan-nan"` |
| `InterfaceClassification` | String | Yes | Interface class | String | `"Non-physiological"` |
| `LabelInterfaceChains` | String | Yes | Assembly chains | String | `"A-B"` |
| `PDB_ID` | String | Yes | PDB identifier | Pattern: `^[A-Z0-9]{4}$` | `"1A17"` |

*Note: Chain\_Sequence and Entity*\* are required for each chain/entity present in the structure.

## **Protein Entity Structure (mainEntity)**

| Field | Type | Required | Description | Pattern/Example |
| :---- | :---- | :---- | :---- | :---- |
| `@context` | String | Yes | [Schema.org](https://schema.org/) context | `"https://schema.org"` |
| `@type` | String | Yes | Protein type | `"Protein"` |
| `@id` | String | Yes | Protein identifier | `^#protein_[A-Z0-9]{4}$` |
| `dct:conformsTo` | String | Yes | BioSchemas profile | `"https://bioschemas.org/profiles/Protein/0.11-RELEASE"` |
| `identifier` | String | Yes | PDB ID | `^[A-Z0-9]{4}$` |
| `name` | String | Yes | Protein name | `^Protein [A-Z0-9]{4}$` |
| `description` | String | Yes | Protein description | Text description |
| `url` | String (URI) | Yes | RCSB PDB URL | `^https://www\.rcsb\.org/structure/[A-Z0-9]{4}$` |
| `taxonomicRange` | TaxonomicRange | Yes | Organism taxonomy | TaxonomicRange object |
| `alternateName` | Array\[String\] | Yes | Alternative names | Min 2 items, includes `PDB {ID}` |
| `hasRepresentation` | Array\[ProteinRep\] | Yes | Structure files | Min 3 items |
| `additionalProperty` | Array\[PropertyValue\] | Yes | Protein properties | Similar to interface properties |
| `hasMolecularFunction` | DefinedTerm | Yes | Molecular function | DefinedTerm object |
| `bioChemInteraction` | Protein | Yes | Interaction info | Protein object |

## **TaxonomicRange Object Structure**

| Field | Type | Required | Description | Example |
| :---- | :---- | :---- | :---- | :---- |
| `@type` | String | No | DefinedTerm type | `"DefinedTerm"` |
| `name` | String | No | Organism name | `"Homo sapiens"` |
| `inDefinedTermSet` | String | No | Taxonomy set | `"https://www.ncbi.nlm.nih.gov/taxonomy"` |
| `description` | String | No | Description | `"Organism taxonomy information from PDB metadata"` |
| `termCode` | String | No | Taxonomy ID | `"9606"` |
| `url` | String (URI) | No | Taxonomy browser | `"https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=9606"` |
| `identifier` | String | No | Identifier | `"taxonomy:9606"` |

## **ProteinRepresentation Object Structure**

| Field | Type | Required | Description | Allowed Values |
| :---- | :---- | :---- | :---- | :---- |
| `@type` | String | Yes | PropertyValue type | `"PropertyValue"` |
| `name` | String | Yes | Representation type | `"PDB Structure"`, `"mmCIF Structure"`, `"PDB ID"` |
| `value` | String (URI) | Yes | URL or identifier | URI for files, ID for PDB ID |
| `description` | String | Yes | Description | Text description |

## **PDB Metadata Snapshot Files**

`bioschema/pdb_metadata/<PDB_ID>.json` holds the entry-level PDB metadata of each benchmark entry (1,645 files) and of the three entries that replaced obsolete benchmark entries (6YQD, 7JNY, 6OH6; marked with `snapshot.role`). The records were retrieved from the RCSB PDB Data API (`https://data.rcsb.org/rest/v1`) by `scripts/ppi_benchmark_fair.py`. PDB archive data are available under CC0 1.0.

| Field | Content |
| :---- | :---- |
| `pdb_id` | PDB identifier |
| `snapshot` | Provenance: `retrieved` (date), `source`, `live_record` (RCSB URL), `license`, and, where relevant, `note`, `amendments`, `role` and `replaces` |
| `structure_metadata` | Entry summary parsed by the generator: resolution, dates, experimental method, source organism, primary citation, chain and entity information, sequences, homomer type |
| `entity_metadata` | Polymer-entity records as returned by RCSB, keyed by entity ID (empty for obsolete entries, which RCSB no longer serves) |
| `rcsb_removed_holding` | Only for obsolete entries: the RCSB removal record (replacement ID, deposition, release and removal dates, title), stored as returned |

Most records were retrieved on 2026-01-03. Records for 1P35, 3G0K, 4NQQ (first retrieval incomplete), the three obsolete entries (removal record added) and the three replacements carry the retrieval date 2026-11-11 in `snapshot.retrieved`. `experimental_method` was filled on 2026-11-11 from the RCSB `exptl.method` field for all entries (the first run did not capture it); this is recorded in `snapshot.amendments`.
The interface-specific part of the original record (`interface_chains`, `interface_sequence_analysis`, `enrichment_timestamp`) stays in the interface item, as `additionalProperty` of `PDBStructureMetadata`.

## **Data File Structure**

### **Available File Formats**

| Format | Location Pattern | Example | Compression |
| :---- | :---- | :---- | :---- |
| PDB Format | `https://raw.githubusercontent.com/biofold/ppi-benchmark-fair/master/data/benchmark_pdb_format/{interface_id}.pdb.gz` | `1a17_6.pdb.gz`, `1a3a_1.pdb.gz` | gzip |
| mmCIF Format | ProtCID interfaces: `https://raw.githubusercontent.com/biofold/ppi-benchmark-fair/master/data/benchmark_mmcif_format/{pdb_id}_{n}.cif.gz`; QSalign interfaces: `.../benchmark_mmcif_format/{pdb_id}.cif.gz` | `1a17_6.cif.gz` (ProtCID), `1a3a.cif.gz` (QSalign) | gzip |

### **File Naming Convention**

* PDB ID: lowercase (e.g., `1a17` not `1A17`)  
* PDB files: one per interface, named after the interface ID (`{pdb_id}_{n}.pdb.gz`, 1,677 files)  
* mmCIF files: ProtCID-derived interfaces are named `{pdb_id}_{n}.cif.gz`; QSalign-derived interfaces share one file per PDB entry, `{pdb_id}.cif.gz` (1,673 files for 1,677 interfaces). The file name is therefore not always built from the interface ID.  
* Compression: all files are gzipped (`.gz` extension)
* `data/replacement_structures/` holds the current mmCIF files of the entries that replaced three obsolete benchmark entries; they are not part of the benchmark.

## **Data Validation Rules**

### **Required Fields Validation**

Interface entries must have:

1. Unique InterfaceID  
2. Valid PDB ID pattern (`[A-Z0-9]{4}_\d+`)  
3. `physio` boolean value  
4. `label` integer (0 or 1\)  
5. BSA and atomic contacts values  
6. Valid taxonomicRange with NCBI taxonomy ID

Protein entities must have:

1. Valid PDB ID pattern (`[A-Z0-9]{4}`)  
2. At least 3 structure representations  
3. Valid taxonomicRange object  
4. Sequence information for all chains  
5. Correct alternate name pattern

### **Data Type Validation**

| Property | Type Constraint | Validation Rule |
| :---- | :---- | :---- |
| `physio` | Boolean | Must be `true` or `false` |
| `label` | Integer | Must be `0` or `1` |
| `BSA`, `BSA Polar`, `BSA Apolar` | Number | Must be positive |
| `Atomic Contacts` | Integer | Must be non-negative |
| `Resolution` | Number | Must be positive |
| `ChainCount`, `UniqueSequenceCount`, `ClusterSize` | Integer | Must be positive |
| `Fraction Polar`, `Fraction Apolar` | Number | Must be 0-1 |
| `NCBI_Taxonomy_ID` | String | Must be numeric digits |
| Chain sequences | String | Must match `^[A-Z]+$` pattern |

### **Consistency Rules**

1. Interface Classification: `physio: true` must match `label: 1`, and `physio: false` must match `label: 0`  
2. Chain Sequences: Must match sequence length reported in metadata  
3. Taxonomy IDs: Must be valid NCBI taxonomy IDs and consistent across references  
4. URLs: Must be accessible and follow expected patterns  
5. Cross-References: Interface IDs must match between InterfaceID and name patterns  
6. PDB IDs: Must be consistent across all references to the same structure  
7. File Naming: Must follow lowercase PDB ID convention

## **Usage Examples**

### **Accessing Interface Data**

`python`

*`# Example patterns`*  
`interface_id = entry["identifier"]  # e.g., "1A17_6"`  
`classification = entry["physio"]    # Boolean: true/false`  
`bsa_value = entry["Buried Surface Area (BSA)"]  # Number: 4153.16`

`contacts = entry["Atomic Contacts"]  # Integer: 138`

### **Accessing Protein Data**

`python`

*`# Example patterns`*  
`protein_name = protein["name"]  # e.g., "Protein 1A17"`  
`organism = protein["SourceOrganism"]  # e.g., "Homo sapiens"`  
`taxonomy_id = protein["NCBI_Taxonomy_ID"]  # e.g., "9606"`

`chain_a_seq = protein["Chain_A_Sequence"]  # Amino acid sequence`

### **Downloading Structure Files**

`python`

*`# URL construction patterns`*  
`pdb_id = "1A17"`  
`interface_num = "6"`

*`# PDB file URL`*  
`pdb_url = f"https://raw.githubusercontent.com/biofold/ppi-benchmark-fair/main/data/benchmark_pdb_format/{pdb_id.lower()}_{interface_num}.pdb.gz"`

*`# mmCIF file URL`*

`mmcif_url = f"https://raw.githubusercontent.com/biofold/ppi-benchmark-fair/main/data/benchmark_mmcif_format/{pdb_id.lower()}_{interface_num}.cif.gz"`

## **License and Attribution**

License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

When using this dataset, please cite:

1. Dataset DOI: [https://doi.org/10.5281/zenodo.24149636](https://doi.org/10.5281/zenodo.24149636)  
2. Original Publication: "Discriminating physiological from non-physiological interfaces in structures of protein complexes: A community-wide study" (DOI: 10.1002/pmic.202200323)  
3. ELIXIR 3D-BioInfo Community

## **Contact and Support**

| Contact Point | Details |
| :---- | :---- |
| Maintainer | ELIXIR 3D-BioInfo Community |
| URL | [https://elixir-europe.org/platforms/3d-bioinfo](https://elixir-europe.org/platforms/3d-bioinfo) |
| GitHub Repository | [https://github.com/vibbits/Elixir-3DBioInfo-Benchmark-Protein-Interfaces](https://github.com/vibbits/Elixir-3DBioInfo-Benchmark-Protein-Interfaces) |

## **Version History**

| Version | Date | Description |
| :---- | :---- | :---- |
| 1.0.0 | first Zenodo archive | 1,677 interfaces; light JSON-LD files with an in-repo PDB metadata snapshot; documentation of the label curation |
| Latest | 2025-12-26 | Updated metadata enrichment |

This data dictionary provides comprehensive documentation for all fields, validation rules, and usage patterns in the Protein-Protein Interaction Interface Benchmark Dataset schema. All properties follow JSON Schema validation rules and are compatible with both BioSchemas and ML Commons Croissant standards.  
