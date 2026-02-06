# ELIXIR 3D-BioInfo FAIR Benchmark — Schema Summary (visual README)

This README gives a compact, human-friendly summary of the JSON Schema used for the ELIXIR 3D-BioInfo PPI benchmark FAIR package. It's intended for quick visualization (tree view), validation tips, and an immediately usable minimal JSON example that satisfies the schema's required fields.

---

## At-a-glance (top-level structure)

- Root (object)
  - Required:
    - `fair_metadata` (object)
      - `findability` (object)
        - `persistent_identifier` (uri)
        - `rich_metadata` (boolean)
        - `searchable_registry` (array[string])
        - `versioning` (object) — `current_version`, `versioning_system`, `repository` (uri)
      - `accessibility` (object) — `protocol`, `authentication`, `long_term_availability`
      - `interoperability` (object)
        - `data_formats` (array[string])
        - `vocabularies` (array[string])
        - `standards_compliance` (array[string])
      - `reusability` (object)
        - `license` (string)
        - `provenance` (object) — `creation_date` (date), `creators` (array[string]), `methods`, `reference_publication` (uri), `data_sources`
        - `documentation` (object) — `readme` (uri), `methods_paper` (uri), `ml_ready` (boolean), `task_type`, `input_modality`, `target_column`, `features` (array[string])
    - `bioschemas_markup` (object)
      - Required:
        - `dataset` (object) — a Bioschemas `Dataset` JSON-LD object (very strict in the schema)
          - Many required dataset fields, notably:
            - `@context` (array of strings and/or objects)
            - `@type` (array[string])
            - `@id` (uri)
            - `name`, `description`, `identifier`, `url`, `license`, `keywords` (array of DefinedTerm objects)
            - `creator` (array of Person/Organization objects)
            - `datePublished` (date)
            - `publisher` (object)
            - `version`, `citation` (object)
            - `variableMeasured` (array)
            - `measurementTechnique` (array)
            - `dateCreated` (date), `dateModified` (date)
            - `maintainer`, `size`, `hasPart` (array of dataset parts)
  - Optional (but present in schema):
    - `cluster_statistics`, `dataset_statistics`, `interface_id_handling`, `pdb_metadata_statistics` (all objects with domain-specific fields)
- `additionalProperties: false` at the root — the schema is strict about top-level keys.

---

## Visual tree (short)

- fair_metadata
  - findability
    - persistent_identifier (uri)
    - rich_metadata (bool)
    - searchable_registry [string]
    - versioning { current_version, versioning_system, repository (uri) }
  - accessibility { protocol, authentication, long_term_availability }
  - interoperability { data_formats[], vocabularies[], standards_compliance[] }
  - reusability
    - license
    - provenance { creation_date(date), creators[], methods[], reference_publication(uri), data_sources[] }
    - documentation { readme(uri), methods_paper(uri), ml_ready(bool), task_type, input_modality, target_column, features[] }

- bioschemas_markup
  - dataset (Bioschemas Dataset object — many required properties, see below)
    - @context, @type, @id, name, description, identifier, url, license, keywords[], creator[], datePublished, publisher{}, version, citation{}, variableMeasured[], measurementTechnique[], dateCreated, dateModified, maintainer{}, size, hasPart[]

- cluster_statistics, dataset_statistics, interface_id_handling, pdb_metadata_statistics (optional summary objects)

---

## Important notes & recommendations

- The schema is strict (additionalProperties: false at root). The JSON document you validate must include `fair_metadata` and `bioschemas_markup` at top level.
- The `bioschemas_markup.dataset` object in the schema expects many Bioschemas and Croissant properties. If your generator emits extra Croissant fields (e.g., `distribution`, `recordSet`, `cr:*`) and your validator expects a stricter Bioschemas-only dataset, you will either need to:
  - Validate the *full* FAIR package (top-level object with `fair_metadata` + `bioschemas_markup`), or
  - Prune the dataset object to match the strict Bioschemas dataset shape, or
  - Relax the schema to accept Croissant-specific keys.
- Dates in the schema use `format: "date"` (YYYY-MM-DD). If you have timestamps, consider converting to date or allowing `date-time`.
- Several Bioschemas fields may accept a single object or an array in practice (e.g., `creator`, `@type`); consider allowing `oneOf: object | array` if you want more tolerant validation.
- Reuse: The schema contains repeated object shapes (creator, additionalProperty, publisher, etc.). Consider moving them into `$defs` and using `$ref` for maintainability.

---

## Minimal valid JSON skeleton (fills only required fields)

Below is a compact example that fills required fields so you can visualize structure. It is intentionally minimal — replace placeholder values with real ones.

```json
{
  "fair_metadata": {
    "findability": {
      "persistent_identifier": "https://doi.org/10.5281/zenodo.XXXXXXX",
      "rich_metadata": true,
      "searchable_registry": ["Google Dataset Search"],
      "versioning": {
        "current_version": "1.0",
        "versioning_system": "Git",
        "repository": "https://github.com/OWNER/REPO"
      }
    },
    "accessibility": {
      "protocol": "HTTPS",
      "authentication": "None required",
      "long_term_availability": "GitHub + Zenodo"
    },
    "interoperability": {
      "data_formats": ["CSV", "PDB", "mmCIF", "JSON-LD"],
      "vocabularies": ["Schema.org", "Bioschemas"],
      "standards_compliance": ["Bioschemas Dataset Profile 1.0-RELEASE"]
    },
    "reusability": {
      "license": "CC-BY-4.0",
      "provenance": {
        "creation_date": "2023-04-30",
        "creators": ["ELIXIR 3D-BioInfo Community"],
        "methods": ["QSalign", "ProtCID"],
        "reference_publication": "https://pubmed.ncbi.nlm.nih.gov/37365936/",
        "data_sources": ["PDB"]
      },
      "documentation": {
        "readme": "https://github.com/OWNER/REPO/blob/main/README.md",
        "methods_paper": "https://doi.org/10.1000/example",
        "ml_ready": true,
        "task_type": "Binary classification",
        "input_modality": "Protein structures + tabular features",
        "target_column": "physio",
        "features": ["ID", "InterfaceID", "bsa", "contacts"]
      }
    }
  },
  "bioschemas_markup": {
    "dataset": {
      "@context": ["https://schema.org/"],
      "@type": ["Dataset"],
      "@id": "https://doi.org/10.5281/zenodo.XXXXXXX",
      "name": "Protein-Protein Interaction Interface Benchmark Dataset",
      "description": "Minimal dataset description",
      "identifier": "https://doi.org/10.5281/zenodo.XXXXXXX",
      "url": "https://github.com/OWNER/REPO",
      "license": "https://creativecommons.org/licenses/by/4.0/",
      "keywords": [
        { "@type": "DefinedTerm", "name": "protein-protein interaction", "inDefinedTermSet": "http://example.org/terms" }
      ],
      "creator": [
        { "@type": "Organization", "name": "ELIXIR 3D-BioInfo Community", "url": "https://elixir-europe.org" }
      ],
      "datePublished": "2023-04-30",
      "publisher": { "@type": "Organization", "name": "ELIXIR Europe", "url": "https://elixir-europe.org" },
      "version": "1.0",
      "citation": { "@type": "ScholarlyArticle", "name": "Title", "url": "https://doi.org/10.1000/example", "sameAs": "https://doi.org/10.1000/example" },
      "variableMeasured": [
        { "@type": "PropertyValue", "name": "physio", "description": "Binary label" }
      ],
      "measurementTechnique": ["X-ray crystallography"],
      "dateCreated": "2023-04-30",
      "dateModified": "2023-04-30",
      "maintainer": { "@type": "Organization", "name": "ELIXIR 3D-BioInfo Community", "url": "https://elixir-europe.org" },
      "size": "1 entry",
      "hasPart": [
        {
          "@type": "Dataset",
          "name": "Interface 1",
          "description": "Example interface",
          "identifier": "1ABC_1",
          "url": "https://www.rcsb.org/structure/1ABC",
          "additionalProperty": [
            { "@type": "PropertyValue", "name": "InterfaceID", "value": "1ABC_1" }
          ],
          "mainEntity": {
            "@context": "https://schema.org",
            "@type": "MolecularEntity",
            "@id": "https://www.rcsb.org/structure/1ABC",
            "dct:conformsTo": "https://bioschemas.org/",
            "identifier": "1ABC",
            "name": "PDB 1ABC",
            "description": "Example PDB entry",
            "url": "https://www.rcsb.org/structure/1ABC",
            "taxonomicRange": { "@type": "DefinedTerm", "name": "Homo sapiens", "inDefinedTermSet": "https://www.ncbi.nlm.nih.gov/taxonomy" },
            "alternateName": ["Chain A-Chain B"],
            "hasRepresentation": [
              { "@type": "PropertyValue", "name": "PDB Structure", "value": "https://example.org/1ABC.pdb" }
            ],
            "additionalProperty": [
              { "@type": "PropertyValue", "name": "resolution", "value": 2.1 }
            ],
            "hasMolecularFunction": { "@type": "DefinedTerm", "name": "binding", "inDefinedTermSet": "http://purl.obolibrary.org/obo/GO_0005515" },
            "bioChemInteraction": { "@type": "Interaction", "name": "protein-protein interaction", "description": "Physical binding" }
          }
        }
      ]
    }
  }
}
```

---

## How to validate locally

Use the Python `jsonschema` package (Draft-07):

```python
import json
from jsonschema import Draft7Validator

schema = json.load(open("schema.json", "r", encoding="utf-8"))
data = json.load(open("dataset_with_interfaces.json", "r", encoding="utf-8"))

validator = Draft7Validator(schema)
errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
for e in errors:
    print(e.message)
    print("path:", list(e.path))
```

Common validation failures:
- Missing `fair_metadata` or `bioschemas_markup` at the top level.
- `additionalProperties` errors when dataset contains keys the schema forbids (e.g., if the schema expects a strict Bioschemas dataset but you included Croissant fields like `distribution` or `recordSet`).
- Missing `hasPart` or other required dataset fields.

---

## Quick visualization tools

- JSON Schema viewers (web):
  - https://json-schema.github.io/json-schema-viewer/
  - https://www.jsonschemavalidator.net/
- Convert schema to Markdown / HTML:
  - `json-schema-to-markdown` (npm) — produces human-readable docs
  - `docson` (for JSON-LD) or `spectacle` family tools
- For automated docs, consider generating Markdown from schema `$defs` using `json-schema-to-markdown` or `genson` helpers.

---

## Final recommendations

- If your generator emits Croissant/Croissant-specific fields (e.g., `cr:...`, `distribution`, `recordSet`) and you want validation against this schema:
  - Validate the full FAIR package (top-level object). The provided script writes the full package to `dataset_with_interfaces.json`.
  - Or update the schema to allow Croissant properties inside `bioschemas_markup.dataset` (recommended if you want to include Croissant metadata).
- Consider moving repeated object definitions into `$defs` in the schema for readability and easier maintenance.
- If you want, I can:
  - Produce a one-page HTML rendering of this schema (auto-generated),
  - Produce a pruned dataset output that matches a strict Bioschemas-only schema,
  - Or generate a fully expanded `README` with field-by-field descriptions extracted from the schema.

---

License: CC0 — use and adapt as needed.
Contact: ecapriotti (for questions about this schema summary)
