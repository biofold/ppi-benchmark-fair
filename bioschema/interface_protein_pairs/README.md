# **Data Dictionary: Protein-Protein Interface JSON Schema**

## **Overview**

This data dictionary describes the structure and semantics of protein-protein interaction interface data following [schema.org](https://schema.org/) and Bioschemas standards. The data represents structural and functional information about protein interfaces from the Protein Data Bank (PDB).

---

## **Root Level Fields**

### **@context**

* Type: String (constant)  
* Description: JSON-LD context specifying the vocabulary  
* Value: `"https://schema.org"`  
* Required: Yes

### **@type**

* Type: String (constant)  
* Description: [Schema.org](https://schema.org/) type of the object  
* Value: `"DataCatalogItem"`  
* Required: Yes

### **name**

* Type: String  
* Description: Name of the protein-protein interface  
* Format: `"Interface {PDB_ID}_{InterfaceNumber}"`  
* Example: `"Interface 1A17_6"`  
* Required: Yes

### **description**

* Type: String  
* Description: Human-readable description of the interface  
* Format: `"Protein-protein interaction interface between chains {Chain1} and {Chain2}"`  
* Example: `"Protein-protein interaction interface between chains A and A"`  
* Required: No

### **identifier**

* Type: String  
* Description: Unique identifier for the interface  
* Format: `"{PDB_ID}_{InterfaceNumber}"`  
* Example: `"1A17_6"`  
* Required: Yes

### **mainEntity**

* Type: Object (Protein)  
* Description: The primary protein entity involved in the interface  
* Required: Yes  
* See: Protein Object

### **additionalProperty**

* Type: Array of PropertyValue objects  
* Description: Additional interface-specific properties and metrics  
* Required: Yes  
* See: Interface Properties

---

## **Protein Object**

Describes the protein structure and metadata.

### **Core Protein Properties**

| Field | Type | Description | Required |
| :---- | :---- | :---- | :---- |
| `@type` | String | Type of entity | Yes |
| `@id` | String | Internal identifier | Yes |
| `dct:conformsTo` | String | Bioschemas profile URL | Yes |
| `identifier` | String | PDB ID | Yes |
| `name` | String | Protein name | Yes |
| `description` | String | Detailed description | No |
| `url` | String | RCSB PDB URL | No |
| `alternateName` | String\[\] | Alternative names/IDs | No |

### **Taxonomic Information**

* `taxonomicRange`: Object containing organism information  
  * `name`: Organism name (e.g., "Homo sapiens")  
  * `termCode`: NCBI taxonomy ID (e.g., "9606")  
  * `url`: Taxonomy browser URL

### **Structural Representations**

* `hasRepresentation`: Array of file references  
  * Type: PropertyValue  
  * Includes: PDB files, mmCIF files, identifiers

### **Protein Properties**

* `hasMolecularFunction`: Gene Ontology term for molecular function  
* `bioChemInteraction`: Description of interaction partner

---

## **PropertyValue Object**

Generic container for key-value pairs with metadata.

| Field | Type | Description | Examples |
| :---- | :---- | :---- | :---- |
| `@type` | String | Always "PropertyValue" | "PropertyValue" |
| `name` | String | Property name | "Resolution", "Gene", "BSA" |
| `value` | Mixed | Property value | 2.45, "PPP5C", 4153.16 |
| `description` | String | Human-readable description | "X-ray crystallography resolution" |
| `unitCode` | String | Measurement unit | "Å", "Å²" |
| `valueReference` | Object | Reference to structured data | Parsed metadata object |

---

## **Protein Properties (additionalProperty in Protein Object)**

### **Structural Metadata**

| Property Name | Type | Description | Example |
| :---- | :---- | :---- | :---- |
| `PDB_ID` | String | Protein Data Bank identifier | "1A17" |
| `AssociatedInterface` | String | Interface identifier | "1A17\_6" |
| `InterfaceSource` | String | Source database | "ProtCID" |
| `InterfaceChains` | String | Chain identifiers | "A-A" |
| `InterfaceClassification` | String | Interface type | "Non-physiological" |
| `LabelInterfaceChains` | String | Assembly chain labels | "A-B" |
| `Resolution` | Number | Structure resolution in Ångströms | 2.45 |
| `HomomericStructure` | Boolean | Whether structure is homomeric | true |
| `HomomerType` | String | Type of homomeric assembly | "monomer" |
| `ChainCount` | Integer | Number of chains | 1 |
| `UniqueSequenceCount` | Integer | Unique amino acid sequences | 1 |

### **Biological Information**

| Property Name | Type | Description | Example |
| :---- | :---- | :---- | :---- |
| `Gene` | String | Gene name | "PPP5C" |
| `Superfamily` | String | Protein superfamily | "1.25.40.10" |
| `SourceOrganism` | String | Organism name | "Homo sapiens" |
| `NCBI_Taxonomy_ID` | String | Taxonomy ID | "9606" |

### **Sequence Information**

| Property Name | Type | Description | Example |
| :---- | :---- | :---- | :---- |
| `Chain_A_Sequence` | String | Amino acid sequence | "RDEPPADGALKRAEEL..." |
| `sequenceLength` | Integer | Number of residues | 166 |
| `entityId` | String | Entity identifier | "1" |

### **Publication Information**

| Property Name | Type | Description | Example |
| :---- | :---- | :---- | :---- |
| `PrimaryCitation` | String | Publication title | "The structure of the tetratricopeptide..." |
| `PublicationDOI` | String | DOI of publication | "10.1093/emboj/17.5.1192" |

### **Clustering Information**

| Property Name | Type | Description | Example |
| :---- | :---- | :---- | :---- |
| `ClusterID` | String | Sequence cluster ID | "1A17\_6" |
| `ClusterSize` | Integer | Number in cluster | 1 |
| `ClusterMembers` | Array | Other interfaces in cluster | \[\] |
| `ClusterMethod` | String | Clustering method | "BLASTClust sequence clustering" |

### **PDB Metadata**

* `PDBStructureMetadata`: Large JSON string containing parsed RCSB metadata  
* `PDBMetadataStatus`: Status of metadata retrieval

---

## **Interface Properties (additionalProperty at Root Level)**

### **Interface Identification**

| Property Name | Type | Description | Example |
| :---- | :---- | :---- | :---- |
| `InterfaceSource` | String | Source database | "ProtCID" |
| `physio` | Boolean | Physiological interface? | false |
| `label` | Integer | Binary label (1=physio, 0=non-physio) | 0 |
| `PDB_ID` | String | PDB identifier | "1A17" |

### **Chain Information**

| Property Name | Type | Description | Example |
| :---- | :---- | :---- | :---- |
| `AuthChain1` | String | First chain in original structure | "A" |
| `AuthChain2` | String | Second chain in original structure | "A" |
| `LabelChain1` | String | First chain in assembly | "A" |
| `LabelChain2` | String | Second chain in assembly | "B" |
| `SymmetryOp1` | String | Symmetry operation for chain 1 | "1\_555" |
| `SymmetryOp2` | String | Symmetry operation for chain 2 | "15\_545" |

### **Interface Biophysical Properties**

| Property Name | Type | Unit | Description | Example |
| :---- | :---- | :---- | :---- | :---- |
| `Buried Surface Area (BSA)` | Number | Å² | Total buried surface area | 4153.16 |
| `Atomic Contacts` | Integer | count | Number of atomic contacts | 138 |
| `BSA Polar` | Number | Å² | Polar component of BSA | 1651.43 |
| `BSA Apolar` | Number | Å² | Apolar component of BSA | 2501.74 |
| `Fraction Polar` | Number | ratio | Fraction of polar contacts | 0.3976 |
| `Fraction Apolar` | Number | ratio | Fraction of apolar contacts | 0.6024 |

### **Biological Context**

| Property Name | Type | Description | Example |
| :---- | :---- | :---- | :---- |
| `Gene` | String | Gene name | "PPP5C" |
| `Superfamily` | String | Protein superfamily | "1.25.40.10" |

---

## **PDBStructureMetadata Structure**

The `PDBStructureMetadata` property contains a JSON string with detailed structural metadata:

### **Basic Structure Info**

`json`

`{`

  `"resolution": 2.45,`

  `"experimentalMethod": "X-RAY DIFFRACTION",`

  `"depositionDate": "1997-12-23T00:00:00+0000",`

  `"releaseDate": "1998-04-29T00:00:00+0000",`

  `"sourceOrganism": ["Homo sapiens"],`

  `"spaceGroup": "P 21 21 21",`

  `"rFactor": 0.187,`

  `"entityCount": 3,`

  `"chainCount": 1,`

  `"uniqueSequences": 1,`

  `"isHomomer": true,`

  `"homomerType": "monomer"`

`}`

### **Sequence Information**

* `sequences`: Dictionary mapping chain IDs to amino acid sequences  
* `chain_info`: Metadata for each chain (organism, length, type)  
* `entity_info`: Detailed entity information including UniProt mappings

### **Annotations**

* Gene Ontology (GO): Molecular function, cellular component, biological process  
* Pfam/InterPro: Protein domains and families  
* Structural Features: Hydropathy, disordered regions, binding sites

### **Clustering**

* Sequence identity clusters at various thresholds (30%, 50%, 70%, 90%, 95%, 100%)  
* Aligned regions between cluster members

### **Ligand/Cofactor Information**

* Binding compounds with SMILES strings, assay values, and mechanisms  
* Targets from ChEMBL, DrugBank, Pharos

---

## **Data Types and Formats**

### **Strings**

* PDB IDs: 4-character codes (e.g., "1A17")  
* Chain IDs: Single letters (e.g., "A", "B")  
* Gene Names: Standard nomenclature (e.g., "PPP5C")  
* DOIs: Valid DOI format (e.g., "10.1093/emboj/17.5.1192")  
* URLs: Valid HTTP/HTTPS URLs

### **Numbers**

* Resolution: Positive floating-point (Ångströms)  
* BSA: Positive floating-point (Å²)  
* Fractions: 0.0 to 1.0  
* Counts: Positive integers

### **Booleans**

* `true`/`false` for binary properties  
* Used for: `physio`, `HomomericStructure`

### **Arrays**

* Used for: `alternateName`, `additionalProperty`, `ClusterMembers`  
* May be empty arrays `[]`

---

## **Controlled Vocabularies**

### **Interface Classification**

* `"Physiological"` \- Biologically relevant interface  
* `"Non-physiological"` \- Crystal packing artifact

### **Homomer Types**

* `"monomer"` \- Single chain  
* `"dimer"` \- Two identical chains  
* `"trimer"` \- Three identical chains  
* `"tetramer"` \- Four identical chains  
* `"polymer"` \- Multiple identical chains

### **Experimental Methods**

* `"X-RAY DIFFRACTION"`  
* `"ELECTRON MICROSCOPY"`  
* `"SOLUTION NMR"`  
* `"ELECTRON CRYSTALLOGRAPHY"`

---

## **Data Relationships**

`text`

`Root (DataCatalogItem)`

`├── mainEntity (Protein)`

`│   ├── taxonomicRange (Organism)`

`│   ├── hasRepresentation (Structure Files)`

`│   ├── hasMolecularFunction (GO Term)`

`│   ├── bioChemInteraction (Partner Protein)`

`│   └── additionalProperty (Protein Metadata)`

`└── additionalProperty (Interface Properties)`

    `├── Chain identifiers`

    `├── Biophysical metrics`

    `├── Biological context`

    `└── Clustering info`

---

## **Quality Indicators**

### **Completeness**

* Complete metadata: `PDBMetadataStatus: "Complete metadata available"`  
* Partial metadata: May have null/missing values for some fields

### **Validation**

* All PDB IDs should exist in RCSB PDB  
* Chain identifiers should match structure  
* Sequences should be valid amino acid strings  
* URLs should be accessible

### **Consistency**

* `InterfaceChains` should match `AuthChain1`/`AuthChain2`  
* `physio` and `label` should be consistent (true=1, false=0)  
* `HomomericStructure` should match `homomerType`

---

## **Usage Notes**

1. Interface IDs: `{PDB_ID}_{InterfaceNumber}` format is standard  
2. Chain Labels: Assembly chains may differ from auth chains due to symmetry  
3. BSA Calculations: Values computed using specific software (e.g., NACCESS)  
4. Clustering: Based on sequence similarity using BLASTClust  
5. Timestamps: `enrichment_timestamp` indicates when metadata was last fetched

This data dictionary provides comprehensive documentation for developers, data scientists, and bioinformaticians working with protein-protein interface data.

