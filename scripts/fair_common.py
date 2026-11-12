"""
Shared definitions for the PPI benchmark FAIR metadata generator.

Single source of truth for license, creators, the citation of the original work
(Schweke et al. 2023) and the handling of the PDB metadata snapshot.
"""
import copy
import json
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------- identifiers
FAIR_LICENSE_URL = "https://creativecommons.org/licenses/by-nc/4.0/"
FAIR_LICENSE_NAME = "CC BY-NC 4.0"
FAIR_LICENSE_SPDX = "CC-BY-NC-4.0"
FAIR_PORTAL_URL = "https://biofold.github.io/ppi-benchmark-fair"
FAIR_REPO_URL = "https://github.com/biofold/ppi-benchmark-fair"
FAIR_RAW_BASE = "https://raw.githubusercontent.com/biofold/ppi-benchmark-fair/master/"
UPSTREAM_REPO_URL = "https://github.com/vibbits/Elixir-3DBioInfo-Benchmark-Protein-Interfaces"
DEFAULT_PDB_SNAPSHOT_URL = FAIR_PORTAL_URL + "/bioschema/pdb_metadata/"

# ------------------------------------------------------------------- creators
_AFFILIATION = {"@type": "Organization", "name": "University of Bologna"}
FAIR_CREATORS: List[Dict[str, Any]] = [
    {"@type": "Person", "@id": "https://orcid.org/0000-0002-2323-0963", "name": "Emidio Capriotti",
     "givenName": "Emidio", "familyName": "Capriotti", "url": "https://orcid.org/0000-0002-2323-0963",
     "affiliation": _AFFILIATION},
    {"@type": "Person", "@id": "https://orcid.org/0000-0002-0274-5669", "name": "Pier Luigi Martelli",
     "givenName": "Pier Luigi", "familyName": "Martelli", "url": "https://orcid.org/0000-0002-0274-5669",
     "affiliation": _AFFILIATION},
    {"@type": "Person", "@id": "https://orcid.org/0000-0002-7359-0633", "name": "Castrense Savojardo",
     "givenName": "Castrense", "familyName": "Savojardo", "url": "https://orcid.org/0000-0002-7359-0633",
     "affiliation": _AFFILIATION},
]

# ------------------------------------------------------ original work (Schweke)
# Author list of the published article (version of record), taken from the Crossref record of
# doi:10.1002/pmic.202200323 (46 authors) and cross-checked against PubMed (PMID 37365936).
# The PMC author manuscript lists only the first 37 of these on its first page; the other 9
# (Terashi, Sarkar, Christoffer, Aderinwale, Verburgt, Duan, Qiu, X. Xu, Zhang) appear in the published byline.
SCHWEKE_DOI = "10.1002/pmic.202200323"
SCHWEKE_TITLE = ("Discriminating physiological from non-physiological interfaces in structures of "
                 "protein complexes: A community-wide study")
SCHWEKE_AUTHORS = json.loads(r'''[
 {
  "given": "Hugo",
  "family": "Schweke"
 },
 {
  "given": "Qifang",
  "family": "Xu"
 },
 {
  "given": "Gerardo",
  "family": "Tauriello"
 },
 {
  "given": "Lorenzo",
  "family": "Pantolini"
 },
 {
  "given": "Torsten",
  "family": "Schwede"
 },
 {
  "given": "Frédéric",
  "family": "Cazals"
 },
 {
  "given": "Alix",
  "family": "Lhéritier"
 },
 {
  "given": "Juan",
  "family": "Fernandez-Recio"
 },
 {
  "given": "Luis Angel",
  "family": "Rodríguez-Lumbreras"
 },
 {
  "given": "Ora",
  "family": "Schueler-Furman"
 },
 {
  "given": "Julia K.",
  "family": "Varga"
 },
 {
  "given": "Brian",
  "family": "Jiménez-García"
 },
 {
  "given": "Manon F.",
  "family": "Réau"
 },
 {
  "given": "Alexandre M. J. J.",
  "family": "Bonvin"
 },
 {
  "given": "Castrense",
  "family": "Savojardo"
 },
 {
  "given": "Pier-Luigi",
  "family": "Martelli"
 },
 {
  "given": "Rita",
  "family": "Casadio"
 },
 {
  "given": "Jérôme",
  "family": "Tubiana"
 },
 {
  "given": "Haim J.",
  "family": "Wolfson"
 },
 {
  "given": "Romina",
  "family": "Oliva"
 },
 {
  "given": "Didier",
  "family": "Barradas-Bautista"
 },
 {
  "given": "Tiziana",
  "family": "Ricciardelli"
 },
 {
  "given": "Luigi",
  "family": "Cavallo"
 },
 {
  "given": "Česlovas",
  "family": "Venclovas"
 },
 {
  "given": "Kliment",
  "family": "Olechnovič"
 },
 {
  "given": "Raphael",
  "family": "Guerois"
 },
 {
  "given": "Jessica",
  "family": "Andreani"
 },
 {
  "given": "Juliette",
  "family": "Martin"
 },
 {
  "given": "Xiao",
  "family": "Wang"
 },
 {
  "given": "Genki",
  "family": "Terashi"
 },
 {
  "given": "Daipayan",
  "family": "Sarkar"
 },
 {
  "given": "Charles",
  "family": "Christoffer"
 },
 {
  "given": "Tunde",
  "family": "Aderinwale"
 },
 {
  "given": "Jacob",
  "family": "Verburgt"
 },
 {
  "given": "Daisuke",
  "family": "Kihara"
 },
 {
  "given": "Anthony",
  "family": "Marchand"
 },
 {
  "given": "Bruno E.",
  "family": "Correia"
 },
 {
  "given": "Rui",
  "family": "Duan"
 },
 {
  "given": "Liming",
  "family": "Qiu"
 },
 {
  "given": "Xianjin",
  "family": "Xu"
 },
 {
  "given": "Shuang",
  "family": "Zhang"
 },
 {
  "given": "Xiaoqin",
  "family": "Zou"
 },
 {
  "given": "Sucharita",
  "family": "Dey"
 },
 {
  "given": "Roland L.",
  "family": "Dunbrack"
 },
 {
  "given": "Emmanuel D.",
  "family": "Levy"
 },
 {
  "given": "Shoshana J.",
  "family": "Wodak"
 }
]''')
SCHWEKE_CITATION: Dict[str, Any] = {
    "@type": "ScholarlyArticle",
    "@id": "https://doi.org/" + SCHWEKE_DOI,
    "name": SCHWEKE_TITLE,
    "author": [{"@type": "Person", "givenName": a["given"], "familyName": a["family"],
                "name": a["given"] + " " + a["family"]} for a in SCHWEKE_AUTHORS],
    "datePublished": "2023-06-27",  # online (Epub) date; issue: September 2023
    "isPartOf": {"@type": "PublicationIssue", "issueNumber": "17",
                 "isPartOf": {"@type": "PublicationVolume", "volumeNumber": "23",
                              "isPartOf": {"@type": "Periodical", "name": "Proteomics"}}},
    "pagination": "e2200323",
    "identifier": ["https://doi.org/" + SCHWEKE_DOI,
                   {"@type": "PropertyValue", "propertyID": "PMID", "value": "37365936"},
                   {"@type": "PropertyValue", "propertyID": "PMCID", "value": "PMC10937251"}],
    "url": "https://pubmed.ncbi.nlm.nih.gov/37365936/",
    "sameAs": "https://doi.org/" + SCHWEKE_DOI,
}
FAIR_IS_BASED_ON: List[Dict[str, Any]] = [
    {"@type": "ScholarlyArticle", "@id": "https://doi.org/" + SCHWEKE_DOI, "name": SCHWEKE_TITLE,
     "url": "https://doi.org/" + SCHWEKE_DOI},
    {"@type": "Dataset", "@id": UPSTREAM_REPO_URL,
     "name": "ELIXIR 3D-BioInfo Benchmark for Protein-Protein Interfaces (original benchmark repository)",
     "url": UPSTREAM_REPO_URL},
]
DERIVATION_NOTE = (
    " This package is derived from the benchmark of Schweke et al. (2023, Proteomics 23(17):e2200323, "
    "doi:10.1002/pmic.202200323). It uses the post-publication curated annotation file (30 April 2023), "
    "in which 16 interface labels were revised relative to the published split of 836 physiological and "
    "841 non-physiological interfaces."
)


def apply_provenance(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Set license, creators, citation of the original work and identifiers on a Dataset block."""
    dataset["license"] = FAIR_LICENSE_URL
    dataset["creator"] = copy.deepcopy(FAIR_CREATORS)
    dataset["citation"] = copy.deepcopy(SCHWEKE_CITATION)
    dataset["isBasedOn"] = copy.deepcopy(FAIR_IS_BASED_ON)
    dataset["@id"] = FAIR_REPO_URL
    dataset["url"] = FAIR_PORTAL_URL
    if DERIVATION_NOTE.strip() not in dataset.get("description", ""):
        dataset["description"] = dataset.get("description", "") + DERIVATION_NOTE
    return dataset


# ------------------------------------------------------- PDB metadata snapshot
# Keys of the generated PDB record that depend on the interface, not on the PDB entry.
INTERFACE_CONTEXT_KEYS = ("interface_chains", "interface_sequence_analysis", "enrichment_timestamp")
RCSB_ENTRY_API = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
RCSB_ENTRY_PAGE = "https://www.rcsb.org/structure/{pdb_id}"


def build_pdb_metadata_property(blob: Dict[str, Any], description: str, parsed_reference: Optional[Dict[str, Any]],
                                snapshot_base_url: str = DEFAULT_PDB_SNAPSHOT_URL,
                                inline: bool = False,
                                default_retrieved: Optional[str] = None) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Build the 'PDBStructureMetadata' PropertyValue.

    inline=True  : legacy behaviour, the whole record is embedded as a JSON string.
    inline=False : the entry-level record is stored once per PDB entry in the in-repo snapshot
                   (bioschema/pdb_metadata/<PDB_ID>.json); the property links to that file (value) and to
                   the live RCSB record (url), and keeps the small interface-specific context inline.

    Returns (property, snapshot_record_or_None).
    """
    if inline:
        prop = {"@type": "PropertyValue", "name": "PDBStructureMetadata", "description": description,
                "value": json.dumps(blob, default=str)}
        if parsed_reference is not None:
            prop["valueReference"] = parsed_reference
        return prop, None

    pdb_id = str(blob.get("pdb_id") or (blob.get("structure_metadata") or {}).get("pdb_id") or "").upper()
    entry_part = {k: v for k, v in blob.items() if k not in INTERFACE_CONTEXT_KEYS}
    context = {k: blob[k] for k in INTERFACE_CONTEXT_KEYS if k in blob}
    retrieved = str(context.get("enrichment_timestamp", ""))[:10] or default_retrieved
    prop = {
        "@type": "PropertyValue",
        "name": "PDBStructureMetadata",
        "description": description + " Full record: in-repo snapshot (value); live record at RCSB (url).",
        "value": snapshot_base_url + pdb_id + ".json",
        "url": RCSB_ENTRY_API.format(pdb_id=pdb_id),
        "sameAs": RCSB_ENTRY_PAGE.format(pdb_id=pdb_id),
    }
    if parsed_reference is not None:
        prop["valueReference"] = parsed_reference
    if context:
        prop["additionalProperty"] = [{"@type": "PropertyValue", "name": k, "value": v} for k, v in context.items()]
    snapshot = {
        "pdb_id": pdb_id,
        "snapshot": {
            "retrieved": retrieved,
            "source": ("PDB header file of an obsolete entry, processed by scripts/ppi_benchmark_fair.py"
                       if blob.get("found_in_header") else
                       "RCSB PDB Data API (https://data.rcsb.org/rest/v1), processed by scripts/ppi_benchmark_fair.py"),
            "live_record": RCSB_ENTRY_API.format(pdb_id=pdb_id),
            "license": "CC0-1.0 (wwPDB archive data)",
        },
    }
    snapshot.update(entry_part)
    return prop, snapshot


def without_interfaces(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Dataset-level copy without the per-interface items (for the embedded web-discovery markup)."""
    light = {k: v for k, v in dataset.items() if k != "hasPart"}
    light["hasPart"] = {"@type": "DataDownload", "name": "Per-interface items",
                        "description": "The %s interface items are in bioschema/dataset_with_interfaces.json "
                                       "and bioschema/interface_protein_pairs/." % dataset.get("numberOfItems", ""),
                        "contentUrl": FAIR_RAW_BASE + "bioschema/dataset_with_interfaces.json",
                        "encodingFormat": "application/ld+json"}
    return light
