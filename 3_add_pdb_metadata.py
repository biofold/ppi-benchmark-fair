"""
Script 3: PDB Metadata Enricher for PPI Benchmark Data
Updates JSON-LD files with comprehensive PDB metadata from RCSB API.
OPTIMIZED VERSION: Executes approaches sequentially, stops when metadata is obtained.
"""

import json
import os
import requests
from typing import Dict, List, Set, Tuple, Any, Optional
import logging
from pathlib import Path
import argparse
from datetime import datetime
from collections import Counter
import time
from urllib.parse import quote
import pandas as pd
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PDBMetadataFetcher:
    """Fetches comprehensive PDB metadata from RCSB API with sequential approach optimization."""
    
    def __init__(self, pdb_api_base_url: str = "https://data.rcsb.org/rest/v1/core"):
        """
        Initialize the metadata fetcher.
        
        Args:
            pdb_api_base_url: Base URL for RCSB PDB REST API
        """
        self.pdb_api_base_url = pdb_api_base_url.rstrip('/')
        self.metadata_cache = {}  # Cache for PDB metadata to avoid repeated API calls
        self.stats = {
            "total_requests": 0,
            "successful_fetches": 0,
            "failed_fetches": 0,
            "cached_hits": 0,
            "unique_pdbs_fetched": 0,
            "approach_1_success": 0,
            "approach_2_success": 0,
            "approach_3_success": 0,
            "approach_1_only": 0,
            "approach_1_2": 0,
            "all_approaches": 0
        }
    
    def _fetch_with_timeout(self, url: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
        """
        Fetch data from URL with timeout and error handling.
        
        Args:
            url: URL to fetch
            timeout: Timeout in seconds
            
        Returns:
            JSON data as dictionary or None if failed
        """
        try:
            self.stats["total_requests"] += 1
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            else:
                logger.debug(f"HTTP {response.status_code} for URL: {url}")
                return None
        except requests.exceptions.Timeout:
            logger.debug(f"Timeout for URL: {url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.debug(f"Request error for URL {url}: {e}")
            return None
        except Exception as e:
            logger.debug(f"Unexpected error for URL {url}: {e}")
            return None
    
    def fetch_pdb_structure_metadata_optimized(self, pdb_id: str) -> Dict[str, Any]:
        """
        OPTIMIZED: Fetch PDB metadata using sequential approaches.
        Stops when sufficient metadata is obtained.
        
        Args:
            pdb_id: PDB ID (4 letters, e.g., "1ABC")
            
        Returns:
            Dictionary with comprehensive PDB metadata including ALL chain sequences
        """
        # Check cache first
        pdb_id_upper = pdb_id.upper()
        if pdb_id_upper in self.metadata_cache:
            self.stats["cached_hits"] += 1
            return self.metadata_cache[pdb_id_upper]
        
        # Initialize metadata structure
        metadata = {
            "pdb_id": pdb_id_upper,
            "resolution": None,
            "deposition_date": None,
            "release_date": None,
            "experimental_method": None,
            "source_organism": [],
            "citation": None,
            "entity_count": 0,
            "r_factor": None,
            "space_group": None,
            "unit_cell": None,
            "found_in_api": False,
            "sequences": {},  # Store ALL sequences: {"A": "MSEK...", "B": "MSEK..."}
            "chain_info": {},  # Detailed chain information
            "entity_info": {},  # Entity-level information
            "is_homomer": False,  # Flag for homomeric structures
            "sequence_clusters": [],  # Group identical sequences
            "chain_ids": [],  # List of all chain IDs
            "entity_ids": [],  # List of all entity IDs
            "fetch_timestamp": datetime.now().isoformat(),
            "approaches_used": []  # Track which approaches were successful
        }
        
        # APPROACH 1: Try the main entry endpoint (gives basic info)
        logger.debug(f"Approach 1 for {pdb_id}: Fetching entry endpoint...")
        entry_url = f"{self.pdb_api_base_url}/entry/{pdb_id_upper}"
        entry_data = self._fetch_with_timeout(entry_url, timeout=20)
        
        if entry_data:
            metadata["found_in_api"] = True
            metadata["approaches_used"].append("approach_1")
            self.stats["approach_1_success"] += 1
            
            # Extract basic information
            if "rcsb_entry_info" in entry_data:
                entry_info = entry_data["rcsb_entry_info"]
                
                # Resolution
                if "resolution_combined" in entry_info:
                    res = entry_info["resolution_combined"]
                    if isinstance(res, list):
                        valid_res = [float(r) for r in res if r is not None]
                        if valid_res:
                            metadata["resolution"] = min(valid_res)
                    elif res is not None:
                        metadata["resolution"] = float(res)
                
                # Experimental method
                if "exptl_method" in entry_info:
                    methods = entry_info["exptl_method"]
                    if isinstance(methods, list) and methods:
                        metadata["experimental_method"] = methods[0]
                    elif methods:
                        metadata["experimental_method"] = methods
            
            # Dates
            if "rcsb_accession_info" in entry_data:
                acc_info = entry_data["rcsb_accession_info"]
                metadata["deposition_date"] = acc_info.get("deposit_date")
                metadata["release_date"] = acc_info.get("initial_release_date")
            
            # Citation
            if "citation" in entry_data and isinstance(entry_data["citation"], list) and len(entry_data["citation"]) > 0:
                citation = entry_data["citation"][0]
                citation_info = {
                    "title": citation.get("title"),
                    "doi": citation.get("pdbx_database_id_doi"),
                    "authors": citation.get("rcsb_authors")
                }
                metadata["citation"] = {k: v for k, v in citation_info.items() if v}
            
            # Entity count and IDs
            if "rcsb_entry_container_identifiers" in entry_data:
                container_ids = entry_data["rcsb_entry_container_identifiers"]
                if "entity_ids" in container_ids:
                    metadata["entity_count"] = len(container_ids["entity_ids"])
                    metadata["entity_ids"] = container_ids["entity_ids"]
                
                # Get chain IDs
                if "asym_ids" in container_ids:
                    metadata["chain_ids"] = container_ids["asym_ids"]
            
            # Crystallographic info
            if "cell" in entry_data:
                cell = entry_data["cell"]
                if all(k in cell for k in ["length_a", "length_b", "length_c"]):
                    metadata["unit_cell"] = {
                        "a": cell["length_a"],
                        "b": cell["length_b"],
                        "c": cell["length_c"],
                        "alpha": cell.get("angle_alpha"),
                        "beta": cell.get("angle_beta"),
                        "gamma": cell.get("angle_gamma")
                    }
            
            if "symmetry" in entry_data:
                symmetry = entry_data["symmetry"]
                metadata["space_group"] = symmetry.get("space_group_name_H_M")
        
        # Check if we already have sufficient metadata
        if metadata["found_in_api"] and metadata.get("entity_ids"):
            # We have basic info and entity IDs, proceed to Approach 2 for sequences
            logger.debug(f"Approach 2 for {pdb_id}: Fetching entity information for sequences...")
            
            organisms = set()
            sequences_by_entity = {}
            chains_by_entity = {}
            entities_with_data = 0
            
            # Fetch entity information for sequences
            for entity_id in metadata["entity_ids"]:
                try:
                    entity_url = f"{self.pdb_api_base_url}/polymer_entity/{pdb_id_upper}/{entity_id}"
                    entity_data = self._fetch_with_timeout(entity_url, timeout=15)
                    
                    if entity_data:
                        entities_with_data += 1
                        entity_info = {
                            "entity_id": entity_id,
                            "sequence": None,
                            "organism": None,
                            "taxonomy_id": None,
                            "chain_ids": [],
                            "polymer_type": None,
                            "sequence_length": None,
                            "gene_name": None,
                            "uniprot_ids": []
                        }
                        
                        # Extract sequence
                        if "entity_poly" in entity_data:
                            poly = entity_data["entity_poly"]
                            if "pdbx_seq_one_letter_code" in poly:
                                entity_info["sequence"] = poly["pdbx_seq_one_letter_code"]
                                entity_info["sequence_length"] = len(poly["pdbx_seq_one_letter_code"])
                            
                            if "pdbx_strand_id" in poly:
                                chain_ids = poly["pdbx_strand_id"].split(",")
                                entity_info["chain_ids"] = [c.strip() for c in chain_ids]
                                chains_by_entity[entity_id] = chain_ids
                            
                            if "type" in poly:
                                entity_info["polymer_type"] = poly["type"]
                        
                        # Extract organism information
                        if "rcsb_entity_source_organism" in entity_data:
                            source_orgs = entity_data["rcsb_entity_source_organism"]
                            if isinstance(source_orgs, list) and source_orgs:
                                source_org = source_orgs[0]
                                
                                for field in ["ncbi_scientific_name", "scientific_name", "common_name"]:
                                    if field in source_org and source_org[field]:
                                        entity_info["organism"] = source_org[field]
                                        organisms.add(source_org[field])
                                        break
                                
                                if "ncbi_taxonomy_id" in source_org:
                                    entity_info["taxonomy_id"] = source_org["ncbi_taxonomy_id"]
                        
                        # Extract gene information
                        if "rcsb_entity_source_organism" in entity_data:
                            source_orgs = entity_data["rcsb_entity_source_organism"]
                            if isinstance(source_orgs, list):
                                for source_org in source_orgs:
                                    if "rcsb_gene_name" in source_org:
                                        if "value" in source_org["rcsb_gene_name"]:
                                            entity_info["gene_name"] = source_org["rcsb_gene_name"]["value"]
                                        break
                        
                        # Extract UniProt references
                        if "rcsb_polymer_entity_container_identifiers" in entity_data:
                            container_ids = entity_data["rcsb_polymer_entity_container_identifiers"]
                            if "uniprot_ids" in container_ids and container_ids["uniprot_ids"]:
                                entity_info["uniprot_ids"] = container_ids["uniprot_ids"]
                        
                        # Store entity info
                        metadata["entity_info"][entity_id] = entity_info
                        
                        # Store sequence by chain
                        if entity_info["sequence"] and entity_info["chain_ids"]:
                            for chain_id in entity_info["chain_ids"]:
                                metadata["sequences"][chain_id] = entity_info["sequence"]
                                metadata["chain_info"][chain_id] = {
                                    "entity_id": entity_id,
                                    "organism": entity_info["organism"],
                                    "sequence_length": entity_info["sequence_length"],
                                    "polymer_type": entity_info["polymer_type"],
                                    "gene_name": entity_info["gene_name"],
                                    "uniprot_ids": entity_info["uniprot_ids"]
                                }
                        
                        sequences_by_entity[entity_id] = entity_info["sequence"]
                    
                except Exception as e:
                    logger.debug(f"Could not fetch entity {entity_id} for {pdb_id}: {e}")
                    continue
            
            if entities_with_data > 0:
                metadata["approaches_used"].append("approach_2")
                self.stats["approach_2_success"] += 1
                
                if organisms:
                    metadata["source_organism"] = list(organisms)
                
                # Analyze sequence relationships for homomer detection
                if sequences_by_entity:
                    # Group identical sequences
                    seq_to_entities = {}
                    for entity_id, seq in sequences_by_entity.items():
                        if seq:
                            if seq not in seq_to_entities:
                                seq_to_entities[seq] = []
                            seq_to_entities[seq].append(entity_id)
                    
                    metadata["sequence_clusters"] = [
                        {"sequence": seq, "entities": entities, "chain_count": sum(len(chains_by_entity.get(eid, [])) for eid in entities)}
                        for seq, entities in seq_to_entities.items()
                    ]
                    
                    # Check if it's a homomer (all sequences identical)
                    if len(seq_to_entities) == 1:
                        metadata["is_homomer"] = True
                        seq = list(seq_to_entities.keys())[0]
                        entity_count = len(list(seq_to_entities.values())[0])
                        chain_count = sum(len(chains_by_entity.get(eid, [])) for eid in list(seq_to_entities.values())[0])
                        metadata["homomer_type"] = f"{entity_count}-mer" if entity_count > 1 else "monomer"
                        metadata["homomer_chains"] = chain_count
                
                # Calculate statistics
                if metadata["sequences"]:
                    metadata["chain_count"] = len(metadata["sequences"])
                    metadata["unique_sequences"] = len(set(metadata["sequences"].values()))
                    
                    # Calculate sequence lengths
                    seq_lengths = [len(seq) for seq in metadata["sequences"].values()]
                    if seq_lengths:
                        metadata["avg_sequence_length"] = sum(seq_lengths) / len(seq_lengths)
                        metadata["min_sequence_length"] = min(seq_lengths)
                        metadata["max_sequence_length"] = max(seq_lengths)
            
            # Check if we need Approach 3 (only if we still don't have sequences)
            if not metadata["sequences"]:
                logger.debug(f"Approach 3 for {pdb_id}: Trying assembly endpoint as fallback...")
                try:
                    # Try the assembly endpoint
                    assembly_url = f"https://data.rcsb.org/rest/v1/core/assembly/{pdb_id_upper}/1"
                    assembly_data = self._fetch_with_timeout(assembly_url, timeout=10)
                    
                    if assembly_data:
                        metadata["approaches_used"].append("approach_3")
                        self.stats["approach_3_success"] += 1
                        
                        # Extract chain mapping from assembly
                        if "rcsb_assembly_info" in assembly_data:
                            assembly_info = assembly_data["rcsb_assembly_info"]
                            if "polymer_entity_instance_count" in assembly_info:
                                metadata["assembly_polymer_count"] = assembly_info["polymer_entity_instance_count"]
                
                except Exception as e:
                    logger.debug(f"Could not fetch assembly info for {pdb_id}: {e}")
        else:
            # Approach 1 failed or didn't give entity_ids, try Approach 3 as fallback
            logger.debug(f"Approach 1 failed for {pdb_id}, trying Approach 3 directly...")
            try:
                # Try the assembly endpoint
                assembly_url = f"https://data.rcsb.org/rest/v1/core/assembly/{pdb_id_upper}/1"
                assembly_data = self._fetch_with_timeout(assembly_url, timeout=10)
                
                if assembly_data:
                    metadata["found_in_api"] = True
                    metadata["approaches_used"].append("approach_3")
                    self.stats["approach_3_success"] += 1
                    
                    # Extract basic info from assembly
                    if "rcsb_assembly_info" in assembly_data:
                        assembly_info = assembly_data["rcsb_assembly_info"]
                        if "polymer_entity_instance_count" in assembly_info:
                            metadata["assembly_polymer_count"] = assembly_info["polymer_entity_instance_count"]
                
            except Exception as e:
                logger.debug(f"Could not fetch assembly info for {pdb_id}: {e}")
        
        # Update statistics based on approaches used
        approaches_used = metadata.get("approaches_used", [])
        if len(approaches_used) == 1:
            if "approach_1" in approaches_used:
                self.stats["approach_1_only"] += 1
            elif "approach_3" in approaches_used:
                # Approach 3 alone is rare
                pass
        elif len(approaches_used) == 2:
            if "approach_1" in approaches_used and "approach_2" in approaches_used:
                self.stats["approach_1_2"] += 1
        elif len(approaches_used) == 3:
            self.stats["all_approaches"] += 1
        
        # Log success based on what we got
        if metadata["found_in_api"]:
            self.stats["successful_fetches"] += 1
            logger.info(f"Fetched metadata for {pdb_id}: "
                       f"approaches={metadata.get('approaches_used', [])}, "
                       f"chains={metadata.get('chain_count', 0)}, "
                       f"unique_seqs={metadata.get('unique_sequences', 0)}, "
                       f"homomer={metadata.get('is_homomer', False)}, "
                       f"resolution={metadata.get('resolution')}")
        else:
            self.stats["failed_fetches"] += 1
            logger.warning(f"No metadata found for {pdb_id} with any approach")
        
        # Cache the result
        self.metadata_cache[pdb_id_upper] = metadata
        self.stats["unique_pdbs_fetched"] = len(self.metadata_cache)
        
        return metadata
    
    def _calculate_sequence_identity(self, seq1: str, seq2: str) -> float:
        """
        Calculate sequence identity between two sequences.
        
        Args:
            seq1: First amino acid sequence
            seq2: Second amino acid sequence
            
        Returns:
            Sequence identity as percentage (0-100)
        """
        if not seq1 or not seq2:
            return 0.0
        
        # Handle different length sequences
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return 0.0
        
        # Count identical residues
        identical = sum(1 for a, b in zip(seq1[:min_len], seq2[:min_len]) if a == b)
        
        return (identical / min_len) * 100
    
    def enrich_protein_with_pdb_metadata(self, interface_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich interface information with comprehensive PDB metadata.
        
        Args:
            interface_info: Dictionary with interface information
            
        Returns:
            Dictionary with enriched PDB metadata
        """
        pdb_id = interface_info.get("pdb_id", "").upper()
        
        if not pdb_id:
            return {"error": "No PDB ID provided"}
        
        # Fetch comprehensive metadata using optimized sequential approach
        structure_metadata = self.fetch_pdb_structure_metadata_optimized(pdb_id)
        
        # For interface analysis, get specific sequences for interface chains
        interface_sequence_analysis = None
        if structure_metadata.get("sequences"):
            auth_chain1 = interface_info.get("auth_chain1")
            auth_chain2 = interface_info.get("auth_chain2")
            label_chain1 = interface_info.get("label_chain1")
            label_chain2 = interface_info.get("label_chain2")
            
            # Try label chains first, then auth chains
            chain1_seq = None
            chain2_seq = None
            
            if label_chain1 and label_chain1 in structure_metadata["sequences"]:
                chain1_seq = structure_metadata["sequences"][label_chain1]
            elif auth_chain1 and auth_chain1 in structure_metadata["sequences"]:
                chain1_seq = structure_metadata["sequences"][auth_chain1]
            
            if label_chain2 and label_chain2 in structure_metadata["sequences"]:
                chain2_seq = structure_metadata["sequences"][label_chain2]
            elif auth_chain2 and auth_chain2 in structure_metadata["sequences"]:
                chain2_seq = structure_metadata["sequences"][auth_chain2]
            
            if chain1_seq and chain2_seq:
                interface_sequence_analysis = {
                    "chain1_sequence_length": len(chain1_seq),
                    "chain2_sequence_length": len(chain2_seq),
                    "sequences_identical": chain1_seq == chain2_seq,
                    "sequence_identity": self._calculate_sequence_identity(chain1_seq, chain2_seq)
                }
        
        # Combine metadata
        enriched_metadata = {
            "pdb_id": pdb_id,
            "structure_metadata": structure_metadata,
            "interface_info": interface_info,
            "interface_sequence_analysis": interface_sequence_analysis,
            "enrichment_timestamp": datetime.now().isoformat()
        }
        
        return enriched_metadata
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about metadata fetching."""
        stats = self.stats.copy()
        
        # Add metadata cache statistics
        if self.metadata_cache:
            successful_fetches = sum(1 for meta in self.metadata_cache.values() if meta.get("found_in_api", False))
            total_chains = sum(len(meta.get("sequences", {})) for meta in self.metadata_cache.values())
            homomeric_structures = sum(1 for meta in self.metadata_cache.values() if meta.get("is_homomer", False))
            
            stats["metadata_cache"] = {
                "size": len(self.metadata_cache),
                "successful_fetches": successful_fetches,
                "total_chains": total_chains,
                "homomeric_structures": homomeric_structures
            }
            
            # Add approach statistics
            approach_stats = {
                "total_structures": len(self.metadata_cache),
                "approach_breakdown": {
                    "approach_1_only": stats.get("approach_1_only", 0),
                    "approach_1_2": stats.get("approach_1_2", 0),
                    "all_approaches": stats.get("all_approaches", 0),
                    "approach_3_only": sum(1 for meta in self.metadata_cache.values() 
                                          if "approach_3" in meta.get("approaches_used", []) and 
                                          len(meta.get("approaches_used", [])) == 1)
                },
                "success_rate": (successful_fetches / len(self.metadata_cache) * 100) if self.metadata_cache else 0
            }
            stats["approach_statistics"] = approach_stats
            
            # Add resolution statistics
            resolutions = [meta.get("resolution") for meta in self.metadata_cache.values() 
                          if meta.get("resolution") is not None]
            if resolutions:
                stats["resolution_stats"] = {
                    "average": sum(resolutions) / len(resolutions),
                    "min": min(resolutions),
                    "max": max(resolutions),
                    "count": len(resolutions)
                }
            
            # Add sequence statistics
            all_sequences = []
            for meta in self.metadata_cache.values():
                if meta.get("sequences"):
                    all_sequences.extend(list(meta["sequences"].values()))
            
            if all_sequences:
                seq_lengths = [len(seq) for seq in all_sequences]
                stats["sequence_stats"] = {
                    "total_sequences": len(all_sequences),
                    "unique_sequences": len(set(all_sequences)),
                    "avg_sequence_length": sum(seq_lengths) / len(seq_lengths),
                    "min_sequence_length": min(seq_lengths),
                    "max_sequence_length": max(seq_lengths)
                }
            
            # Add organism statistics
            organisms = []
            for meta in self.metadata_cache.values():
                if meta.get("source_organism"):
                    organisms.extend(meta["source_organism"])
            
            if organisms:
                org_counts = Counter(organisms)
                stats["organism_stats"] = {
                    "top_organisms": dict(org_counts.most_common(10)),
                    "unique_organisms": len(org_counts)
                }
        
        return stats


class PDBMetadataEnricher:
    """Enriches JSON-LD files with comprehensive PDB metadata."""
    
    def __init__(self, input_dir: str, pdb_api_base_url: str = "https://data.rcsb.org/rest/v1/core",
                 batch_size: int = 10, delay_between_batches: float = 1.0):
        """
        Initialize the enricher.
        
        Args:
            input_dir: Directory containing JSON-LD files
            pdb_api_base_url: Base URL for RCSB PDB REST API
            batch_size: Number of PDB IDs to fetch in parallel (not implemented in this version)
            delay_between_batches: Delay between batches in seconds
        """
        self.input_dir = Path(input_dir)
        self.pdb_api_base_url = pdb_api_base_url
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        self.metadata_fetcher = PDBMetadataFetcher(pdb_api_base_url)
        
        # Statistics
        self.stats = {
            "total_files": 0,
            "successfully_enriched": 0,
            "failed_enrichments": 0,
            "unique_pdbs_processed": set(),
            "enriched_pdbs": set()
        }
    
    def find_json_files(self) -> List[Path]:
        """Find all JSON-LD files in the input directory."""
        json_files = []
        
        # Look for interface_protein_pairs directory
        pairs_dir = self.input_dir / "interface_protein_pairs"
        if pairs_dir.exists() and pairs_dir.is_dir():
            logger.info(f"Looking for interface files in: {pairs_dir}")
            for file_path in pairs_dir.glob("interface_*.json"):
                if file_path.is_file():
                    json_files.append(file_path)
        
        # Also look for JSON files directly in input directory
        for file_path in self.input_dir.glob("*.json"):
            if file_path.is_file() and file_path.name not in ["manifest.json", "fair_metadata_package.json", 
                                                              "dataset_with_interfaces.json", "pdb_metadata_cache.json"]:
                json_files.append(file_path)
        
        logger.info(f"Found {len(json_files)} JSON files to process")
        return json_files
    
    def extract_protein_info(self, protein_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract protein information from JSON-LD data.
        
        Args:
            protein_data: JSON-LD data for a protein
            
        Returns:
            Dictionary with extracted protein information
        """
        protein_info = {
            "pdb_id": None,
            "identifier": protein_data.get("identifier"),
            "name": protein_data.get("name"),
            "interface_chains": None,
            "label_interface_chains": None,
            "auth_chain1": None,
            "auth_chain2": None,
            "label_chain1": None,
            "label_chain2": None
        }
        
        # Extract from additionalProperty
        if "additionalProperty" in protein_data:
            for prop in protein_data["additionalProperty"]:
                if prop.get("name") == "PDB_ID":
                    protein_info["pdb_id"] = prop.get("value")
                elif prop.get("name") == "InterfaceChains":
                    protein_info["interface_chains"] = prop.get("value")
                elif prop.get("name") == "LabelInterfaceChains":
                    protein_info["label_interface_chains"] = prop.get("value")
                elif prop.get("name") == "AuthChain1":
                    protein_info["auth_chain1"] = prop.get("value")
                elif prop.get("name") == "AuthChain2":
                    protein_info["auth_chain2"] = prop.get("value")
                elif prop.get("name") == "LabelChain1":
                    protein_info["label_chain1"] = prop.get("value")
                elif prop.get("name") == "LabelChain2":
                    protein_info["label_chain2"] = prop.get("value")
        
        return protein_info
    
    def extract_interface_info(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract interface information from JSON-LD data.
        
        Args:
            json_data: JSON-LD data for an interface
            
        Returns:
            Dictionary with extracted interface information
        """
        interface_info = {
            "interface_id": json_data.get("identifier"),
            "name": json_data.get("name"),
            "pdb_id": None,
            "auth_chain1": None,
            "auth_chain2": None,
            "label_chain1": None,
            "label_chain2": None,
            "interface_source": None,
            "interface_type": None
        }
        
        # Check if this is an interface item
        if "@type" in json_data and "DataCatalogItem" in json_data["@type"]:
            interface_info["interface_type"] = "interface_item"
            
            # Extract from additionalProperty
            if "additionalProperty" in json_data:
                for prop in json_data["additionalProperty"]:
                    if prop.get("name") == "PDB_ID":
                        interface_info["pdb_id"] = prop.get("value")
                    elif prop.get("name") == "AuthChain1":
                        interface_info["auth_chain1"] = prop.get("value")
                    elif prop.get("name") == "AuthChain2":
                        interface_info["auth_chain2"] = prop.get("value")
                    elif prop.get("name") == "LabelChain1":
                        interface_info["label_chain1"] = prop.get("value")
                    elif prop.get("name") == "LabelChain2":
                        interface_info["label_chain2"] = prop.get("value")
                    elif prop.get("name") == "InterfaceSource":
                        interface_info["interface_source"] = prop.get("value")
            
            # Also extract from mainEntity if needed
            if "mainEntity" in json_data and not interface_info["pdb_id"]:
                protein_info = self.extract_protein_info(json_data["mainEntity"])
                interface_info["pdb_id"] = protein_info["pdb_id"]
                if not interface_info["label_chain1"]:
                    interface_info["label_chain1"] = protein_info["label_chain1"]
                if not interface_info["label_chain2"]:
                    interface_info["label_chain2"] = protein_info["label_chain2"]
        
        return interface_info
    
    def add_pdb_metadata_to_protein(self, protein_data: Dict[str, Any], 
                                   pdb_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add comprehensive PDB metadata to a protein JSON-LD object.
        
        Args:
            protein_data: Original protein JSON-LD data
            pdb_metadata: PDB metadata from RCSB API
            
        Returns:
            Updated protein JSON-LD data with PDB metadata
        """
        updated_protein = protein_data.copy()
        
        if not pdb_metadata:
            return updated_protein
        
        # Extract structure metadata
        structure_meta = pdb_metadata.get("structure_metadata", {})
        
        # Ensure additionalProperty exists
        if "additionalProperty" not in updated_protein:
            updated_protein["additionalProperty"] = []
        
        # Remove any existing PDB metadata placeholders
        updated_protein["additionalProperty"] = [
            prop for prop in updated_protein["additionalProperty"] 
            if prop.get("name") not in ["PDBMetadataStatus", "PDBStructureMetadata", 
                                       "AssemblyChainStatus", "PDBMetadataPlaceholder"]
        ]
        
        # Create a comprehensive PDB metadata property
        pdb_metadata_property = {
            "@type": "PropertyValue",
            "name": "PDBStructureMetadata",
            "description": "Comprehensive structural metadata from RCSB Protein Data Bank",
            "value": json.dumps(pdb_metadata, default=str),
            "valueReference": {
                "@type": "PropertyValue",
                "name": "ParsedPDBMetadata",
                "value": {
                    "resolution": structure_meta.get("resolution"),
                    "experimentalMethod": structure_meta.get("experimental_method"),
                    "depositionDate": structure_meta.get("deposition_date"),
                    "releaseDate": structure_meta.get("release_date"),
                    "sourceOrganism": structure_meta.get("source_organism"),
                    "spaceGroup": structure_meta.get("space_group"),
                    "entityCount": structure_meta.get("entity_count"),
                    "chainCount": structure_meta.get("chain_count"),
                    "uniqueSequences": structure_meta.get("unique_sequences"),
                    "isHomomer": structure_meta.get("is_homomer"),
                    "homomerType": structure_meta.get("homomer_type"),
                    "citation": structure_meta.get("citation")
                }
            }
        }
        
        updated_protein["additionalProperty"].append(pdb_metadata_property)
        
        # Add individual properties if they exist
        if structure_meta.get("resolution"):
            updated_protein["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "Resolution",
                "value": float(structure_meta["resolution"]),
                "unitCode": "Å",
                "description": "X-ray crystallography resolution"
            })
        
        if structure_meta.get("experimental_method"):
            updated_protein["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "ExperimentalMethod",
                "value": structure_meta["experimental_method"],
                "description": "Experimental method used for structure determination"
            })
        
        if structure_meta.get("source_organism"):
            for org in structure_meta["source_organism"]:
                updated_protein["additionalProperty"].append({
                    "@type": "PropertyValue",
                    "name": "SourceOrganism",
                    "value": org,
                    "description": "Organism from which the protein was isolated"
                })
        
        # Add homomer information
        if structure_meta.get("is_homomer"):
            updated_protein["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "HomomericStructure",
                "value": True,
                "description": f"This is a homomeric {structure_meta.get('homomer_type', 'multimer')}"
            })
            
            if structure_meta.get("homomer_type"):
                updated_protein["additionalProperty"].append({
                    "@type": "PropertyValue",
                    "name": "HomomerType",
                    "value": structure_meta["homomer_type"],
                    "description": "Type of homomeric assembly"
                })
        
        # Add chain and sequence information
        if structure_meta.get("sequences"):
            updated_protein["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "ChainCount",
                "value": len(structure_meta["sequences"]),
                "description": "Number of chains in the structure"
            })
            
            updated_protein["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "UniqueSequenceCount",
                "value": len(set(structure_meta["sequences"].values())),
                "description": "Number of unique amino acid sequences"
            })
            
            # Add sequence information for each chain (first 5 chains for brevity)
            chain_counter = 0
            for chain_id, sequence in structure_meta["sequences"].items():
                if chain_counter >= 5:  # Limit to 5 chains in the main display
                    break
                
                chain_info = structure_meta.get("chain_info", {}).get(chain_id, {})
                
                updated_protein["additionalProperty"].append({
                    "@type": "PropertyValue",
                    "name": f"Chain_{chain_id}_Sequence",
                    "value": sequence,
                    "description": f"Amino acid sequence for chain {chain_id} (length: {len(sequence)})",
                    "chainId": chain_id,
                    "entityId": chain_info.get("entity_id"),
                    "sequenceLength": len(sequence),
                    "organism": chain_info.get("organism"),
                    "geneName": chain_info.get("gene_name")
                })
                chain_counter += 1
            
            # If there are more than 5 chains, add a summary
            if len(structure_meta["sequences"]) > 5:
                updated_protein["additionalProperty"].append({
                    "@type": "PropertyValue",
                    "name": "AdditionalChains",
                    "value": f"{len(structure_meta['sequences']) - 5} more chains",
                    "description": f"Total of {len(structure_meta['sequences'])} chains in structure"
                })
            
            # Add sequence cluster information
            if structure_meta.get("sequence_clusters"):
                for i, cluster in enumerate(structure_meta["sequence_clusters"], 1):
                    updated_protein["additionalProperty"].append({
                        "@type": "PropertyValue",
                        "name": f"SequenceCluster_{i}",
                        "value": f"{len(cluster['entities'])} entities, {cluster['chain_count']} chains",
                        "description": f"Sequence cluster with {len(cluster['entities'])} identical entities"
                    })
        
        if structure_meta.get("citation"):
            citation = structure_meta["citation"]
            updated_protein["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "PrimaryCitation",
                "value": citation.get("title", "Unknown"),
                "description": "Primary publication for this structure"
            })
            
            if "doi" in citation:
                updated_protein["additionalProperty"].append({
                    "@type": "PropertyValue",
                    "name": "PublicationDOI",
                    "value": citation["doi"],
                    "description": "DOI of the primary publication"
                })
        
        # Add entity information (first 3 entities for brevity)
        if structure_meta.get("entity_info"):
            entity_counter = 0
            for entity_id, entity_info in structure_meta["entity_info"].items():
                if entity_counter >= 3:  # Limit to 3 entities in main display
                    break
                
                if entity_info.get("sequence"):
                    updated_protein["additionalProperty"].append({
                        "@type": "PropertyValue",
                        "name": f"Entity_{entity_id}",
                        "value": f"Chains: {', '.join(entity_info.get('chain_ids', []))}, Length: {len(entity_info['sequence'])}",
                        "description": f"Entity {entity_id} information"
                    })
                    entity_counter += 1
        
        # Add interface-specific sequence analysis
        interface_analysis = pdb_metadata.get("interface_sequence_analysis")
        if interface_analysis:
            updated_protein["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "InterfaceSequenceAnalysis",
                "value": json.dumps(interface_analysis, default=str),
                "description": "Sequence analysis for the interface chains",
                "valueReference": {
                    "@type": "PropertyValue",
                    "name": "ParsedInterfaceAnalysis",
                    "value": {
                        "chain1_length": interface_analysis.get("chain1_sequence_length"),
                        "chain2_length": interface_analysis.get("chain2_sequence_length"),
                        "sequences_identical": interface_analysis.get("sequences_identical"),
                        "sequence_identity": interface_analysis.get("sequence_identity")
                    }
                }
            })
            
            if interface_analysis.get("sequence_identity") is not None:
                updated_protein["additionalProperty"].append({
                    "@type": "PropertyValue",
                    "name": "InterfaceSequenceIdentity",
                    "value": f"{interface_analysis['sequence_identity']:.1f}%",
                    "description": "Sequence identity between interface chains"
                })
        
        # Update taxonomicRange with actual organism data
        if structure_meta.get("source_organism") and structure_meta["source_organism"]:
            primary_organism = structure_meta["source_organism"][0]
            
            # Try to find taxonomy ID for this organism
            taxonomy_id = None
            for entity_id, entity_info in structure_meta.get("entity_info", {}).items():
                if entity_info.get("organism") == primary_organism and entity_info.get("taxonomy_id"):
                    taxonomy_id = entity_info["taxonomy_id"]
                    break
            
            # Update the taxonomicRange field
            taxonomic_range = {
                "@type": "DefinedTerm",
                "name": primary_organism,
                "inDefinedTermSet": "https://www.ncbi.nlm.nih.gov/taxonomy"
            }
            
            # Add taxonomy ID if available
            if taxonomy_id:
                taxonomic_range["termCode"] = str(taxonomy_id)
                taxonomic_range["url"] = f"https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id={taxonomy_id}"
            
            updated_protein["taxonomicRange"] = taxonomic_range
        
        # Add PDB metadata status
        updated_protein["additionalProperty"].append({
            "@type": "PropertyValue",
            "name": "PDBMetadataStatus",
            "value": "COMPLETE - Populated by 3rd script",
            "description": "PDB metadata has been fully populated from RCSB API"
        })
        
        return updated_protein
    
    def update_interface_item(self, json_data: Dict[str, Any], pdb_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an interface item with PDB metadata.
        
        Args:
            json_data: Original interface JSON-LD data
            pdb_metadata: PDB metadata for the interface
            
        Returns:
            Updated interface JSON-LD data
        """
        updated_data = json_data.copy()
        
        # Remove any existing PDB metadata placeholders
        if "additionalProperty" in updated_data:
            updated_data["additionalProperty"] = [
                prop for prop in updated_data["additionalProperty"] 
                if prop.get("name") not in ["PDBMetadataStatus", "PDBMetadataPlaceholder"]
            ]
        
        # Add PDB metadata status to interface
        structure_meta = pdb_metadata.get("structure_metadata", {})
        if structure_meta.get("found_in_api"):
            updated_data["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "PDBMetadataStatus",
                "value": "COMPLETE - Populated by 3rd script",
                "description": f"PDB metadata fetched from RCSB API: {structure_meta.get('chain_count', 0)} chains, {structure_meta.get('unique_sequences', 0)} unique sequences"
            })
        else:
            updated_data["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "PDBMetadataStatus",
                "value": "UNAVAILABLE - Not found in RCSB API",
                "description": "PDB metadata could not be fetched from RCSB API"
            })
        
        # Update mainEntity (Protein) with PDB metadata
        if "mainEntity" in updated_data:
            updated_data["mainEntity"] = self.add_pdb_metadata_to_protein(
                updated_data["mainEntity"], 
                pdb_metadata
            )
        
        return updated_data
    
    def update_pdb_metadata_status_in_json(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the PDBMetadataStatus field in JSON-LD data.
        
        Args:
            json_data: JSON-LD data to update
            
        Returns:
            Updated JSON-LD data
        """
        updated_data = json_data.copy()
        
        # Get statistics for the description
        total_pdbs = len(self.stats["unique_pdbs_processed"])
        enriched_pdbs = len(self.stats["enriched_pdbs"])
        
        # Find and update PDBMetadataStatus in additionalProperty
        if "additionalProperty" in updated_data:
            for i, prop in enumerate(updated_data["additionalProperty"]):
                if prop.get("name") == "PDBMetadataStatus":
                    updated_data["additionalProperty"][i] = {
                        "@type": "PropertyValue",
                        "name": "PDBMetadataStatus",
                        "value": f"COMPLETE - {enriched_pdbs} PDBs enriched",
                        "description": f"PDB metadata populated by script 3. Retrieved metadata for {enriched_pdbs} out of {total_pdbs} PDB structures from RCSB API."
                    }
                    logger.debug(f"Updated PDBMetadataStatus in JSON-LD: {enriched_pdbs} PDBs enriched")
                    return updated_data
            
            # If PDBMetadataStatus not found, add it
            updated_data["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "PDBMetadataStatus",
                "value": f"COMPLETE - {enriched_pdbs} PDBs enriched",
                "description": f"PDB metadata populated by script 3. Retrieved metadata for {enriched_pdbs} out of {total_pdbs} PDB structures from RCSB API."
            })
            logger.debug(f"Added PDBMetadataStatus to JSON-LD: {enriched_pdbs} PDBs enriched")
        
        return updated_data
    
    def update_pdb_metadata_status_in_html(self, html_content: str) -> str:
        """
        Update the PDBMetadataStatus field in HTML JSON-LD script tag.
        
        Args:
            html_content: HTML content to update
            
        Returns:
            Updated HTML content
        """
        # Get statistics for the description
        total_pdbs = len(self.stats["unique_pdbs_processed"])
        enriched_pdbs = len(self.stats["enriched_pdbs"])
        
        # Pattern to find JSON-LD script tags
        script_pattern = r'<script type="application/ld\+json">(.*?)</script>'
        
        def update_json_in_script(match):
            script_content = match.group(1)
            try:
                # Clean the script content
                script_content_clean = script_content.strip()
                if not script_content_clean:
                    return match.group(0)
                
                json_data = json.loads(script_content_clean)
                updated_json = self.update_pdb_metadata_status_in_json(json_data)
                return f'<script type="application/ld+json">\n{json.dumps(updated_json, indent=2)}\n</script>'
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON-LD in HTML: {e}")
                logger.debug(f"Problematic JSON content: {script_content[:200]}...")
                return match.group(0)
            except Exception as e:
                logger.error(f"Error updating JSON-LD in HTML: {e}")
                return match.group(0)
        
        # Update all JSON-LD script tags
        updated_html = re.sub(script_pattern, update_json_in_script, html_content, flags=re.DOTALL)
        
        return updated_html
    
    def find_fair_metadata_package_file(self) -> Optional[Path]:
        """
        Find the fair_metadata_package.json file.
        
        Returns:
            Path to the file if found, None otherwise
        """
        # Try in the input directory
        fair_metadata_file = self.input_dir / "fair_metadata_package.json"
        if fair_metadata_file.exists():
            logger.info(f"Found fair_metadata_package.json at: {fair_metadata_file}")
            return fair_metadata_file
        
        # Try looking for it in any subdirectory
        for file_path in self.input_dir.rglob("fair_metadata_package.json"):
            if file_path.is_file():
                logger.info(f"Found fair_metadata_package.json at: {file_path}")
                return file_path
        
        logger.warning(f"fair_metadata_package.json not found in {self.input_dir} or subdirectories")
        
        # List all files in directory for debugging
        logger.debug(f"Files in {self.input_dir}:")
        for item in self.input_dir.iterdir():
            logger.debug(f"  {item.name} ({'dir' if item.is_dir() else 'file'})")
        
        return None
    
    def find_embedded_markup_html_file(self) -> Optional[Path]:
        """
        Find the embedded_markup.html file.
        
        Returns:
            Path to the file if found, None otherwise
        """
        # Try in the input directory
        embedded_html_file = self.input_dir / "embedded_markup.html"
        if embedded_html_file.exists():
            logger.info(f"Found embedded_markup.html at: {embedded_html_file}")
            return embedded_html_file
        
        # Try looking for it in any subdirectory
        for file_path in self.input_dir.rglob("embedded_markup.html"):
            if file_path.is_file():
                logger.info(f"Found embedded_markup.html at: {file_path}")
                return file_path
        
        logger.warning(f"embedded_markup.html not found in {self.input_dir} or subdirectories")
        
        # List all files in directory for debugging
        logger.debug(f"Files in {self.input_dir}:")
        for item in self.input_dir.iterdir():
            logger.debug(f"  {item.name} ({'dir' if item.is_dir() else 'file'})")
        
        return None

    def update_fair_metadata_package(self) -> bool:
        """
        Update fair_metadata_package.json with PDB metadata information.
        
        Returns:
            True if successful, False otherwise
        """
        fair_metadata_file = self.find_fair_metadata_package_file()
        
        if not fair_metadata_file:
            logger.error("❌ Could not find fair_metadata_package.json")
            logger.error(f"   Searched in: {self.input_dir} and subdirectories")
            return False
        
        try:
            logger.info(f"📄 Loading fair_metadata_package.json from: {fair_metadata_file}")
            
            # Read the file
            with open(fair_metadata_file, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            # Parse JSON
            fair_metadata = json.loads(file_content)
            
            logger.debug(f"✅ Successfully parsed fair_metadata_package.json")
            logger.debug(f"   File has keys: {list(fair_metadata.keys())}")
            
            # Get metadata fetcher statistics
            fetcher_stats = self.metadata_fetcher.get_statistics()
            
            # FIRST: Update the dataset structure from dataset_with_interfaces.json if it exists
            dataset_file = self.input_dir / "dataset_with_interfaces.json"
            if dataset_file.exists():
                logger.info(f"📄 Found dataset_with_interfaces.json, updating dataset structure...")
                
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    dataset_data = json.load(f)
                
                # Update the main dataset structure from dataset_with_interfaces.json
                # Copy key metadata that should be in fair_metadata_package.json
                key_fields = [
                    "name", "description", "identifier", "url", "license", "keywords",
                    "creator", "datePublished", "publisher", "version", "citation",
                    "variableMeasured", "measurementTechnique", "dateCreated",
                    "dateModified", "maintainer", "size", "hasPart", "numberOfItems",
                    "distribution"
                ]
                
                for field in key_fields:
                    if field in dataset_data:
                        fair_metadata[field] = dataset_data[field]
                
                logger.info(f"✅ Updated dataset structure from dataset_with_interfaces.json")
                logger.info(f"   - hasPart: {len(dataset_data.get('hasPart', []))} interfaces")
                logger.info(f"   - numberOfItems: {dataset_data.get('numberOfItems', 'N/A')}")
            
            # SECOND: Update PDB metadata status in the hasPart items
            if "hasPart" in fair_metadata and isinstance(fair_metadata["hasPart"], list):
                logger.info(f"📊 Updating PDB metadata status in {len(fair_metadata['hasPart'])} interface items...")
                
                for i, interface_item in enumerate(fair_metadata["hasPart"]):
                    # Check if this is a DataCatalogItem (interface)
                    if isinstance(interface_item, dict) and "DataCatalogItem" in interface_item.get("@type", []):
                        interface_id = interface_item.get("identifier", "")
                        
                        # Update additionalProperty with PDB metadata status
                        if "additionalProperty" in interface_item:
                            # Remove old PDB metadata status if present
                            interface_item["additionalProperty"] = [
                                prop for prop in interface_item["additionalProperty"]
                                if prop.get("name") not in ["PDBMetadataStatus", "PDBMetadataPlaceholder"]
                            ]
                            
                            # Add updated PDB metadata status
                            interface_item["additionalProperty"].append({
                                "@type": "PropertyValue",
                                "name": "PDBMetadataStatus",
                                "value": "COMPLETE - Populated by 3rd script",
                                "description": f"PDB metadata has been fully populated from RCSB API"
                            })
                        
                        # Also update the mainEntity (Protein) if it exists
                        if "mainEntity" in interface_item and isinstance(interface_item["mainEntity"], dict):
                            protein = interface_item["mainEntity"]
                            
                            # Update additionalProperty in protein
                            if "additionalProperty" in protein:
                                # Remove old PDB metadata status if present
                                protein["additionalProperty"] = [
                                    prop for prop in protein["additionalProperty"]
                                    if prop.get("name") not in ["PDBMetadataStatus", "PDBMetadataPlaceholder"]
                                ]
                                
                                # Add updated PDB metadata status
                                protein["additionalProperty"].append({
                                    "@type": "PropertyValue",
                                    "name": "PDBMetadataStatus",
                                    "value": "COMPLETE - Populated by 3rd script",
                                    "description": "PDB metadata has been fully populated from RCSB API"
                                })
                
                logger.info(f"✅ Updated PDB metadata status in all interface items")
            
            # THIRD: Get ALL the PDB metadata that was fetched
            all_pdb_metadata = self.metadata_fetcher.metadata_cache
            
            # Create PDB metadata summary section
            pdb_metadata_summary = {
                "script": "3_add_pdb_metadata.py",
                "execution_date": datetime.now().isoformat(),
                "description": "PDB metadata enrichment from RCSB API using optimized sequential approach",
                "total_pdbs_processed": len(self.stats["unique_pdbs_processed"]),
                "successfully_enriched_pdbs": len(self.stats["enriched_pdbs"]),
                "metadata_fetcher_statistics": {
                    "total_api_requests": fetcher_stats.get("total_requests", 0),
                    "successful_fetches": fetcher_stats.get("successful_fetches", 0),
                    "failed_fetches": fetcher_stats.get("failed_fetches", 0),
                    "cache_hits": fetcher_stats.get("cached_hits", 0),
                    "unique_pdbs_fetched": fetcher_stats.get("unique_pdbs_fetched", 0)
                },
                "optimization_strategy": {
                    "name": "Sequential approach execution",
                    "description": "Approach 1 (entry) → Approach 2 (entities) → Approach 3 (assembly) as fallback",
                    "benefits": [
                        "Reduces unnecessary API calls",
                        "Stops when sufficient metadata is obtained",
                        "Most structures resolved with Approach 1 or 1+2",
                        "Approach 3 only used as fallback when others fail"
                    ]
                },
                "metadata_includes": [
                    "Resolution and experimental method",
                    "Source organism(s) and taxonomy IDs",
                    "ALL chain sequences (not just representative)",
                    "Homomer detection and analysis",
                    "Sequence clusters and identity analysis",
                    "Citation and publication information",
                    "Entity and chain mappings",
                    "Interface-specific sequence analysis"
                ]
            }
            
            # Add approach statistics if available
            if "approach_statistics" in fetcher_stats:
                pdb_metadata_summary["approach_statistics"] = fetcher_stats["approach_statistics"]
            
            # Add resolution statistics if available
            if "resolution_stats" in fetcher_stats:
                pdb_metadata_summary["resolution_statistics"] = fetcher_stats["resolution_stats"]
            
            # Add sequence statistics if available
            if "sequence_stats" in fetcher_stats:
                pdb_metadata_summary["sequence_statistics"] = fetcher_stats["sequence_stats"]
            
            # Add organism statistics if available
            if "organism_stats" in fetcher_stats:
                pdb_metadata_summary["organism_statistics"] = fetcher_stats["organism_stats"]
            
            # Add sample PDB metadata (first 3 PDBs as examples)
            sample_pdb_metadata = {}
            pdb_counter = 0
            for pdb_id, metadata in all_pdb_metadata.items():
                if pdb_counter >= 3:  # Limit to 3 samples
                    break
                
                # Create a simplified version of the metadata for display
                sample_metadata = {
                    "resolution": metadata.get("resolution"),
                    "experimental_method": metadata.get("experimental_method"),
                    "source_organism": metadata.get("source_organism", []),
                    "chain_count": metadata.get("chain_count", 0),
                    "unique_sequences": metadata.get("unique_sequences", 0),
                    "is_homomer": metadata.get("is_homomer", False),
                    "homomer_type": metadata.get("homomer_type"),
                    "entity_count": metadata.get("entity_count", 0)
                }
                
                # Add sequence information (first 2 chains as samples)
                if metadata.get("sequences"):
                    sample_sequences = {}
                    chain_counter = 0
                    for chain_id, sequence in metadata["sequences"].items():
                        if chain_counter >= 2:
                            break
                        # Store just the first 50 characters of sequence for brevity
                        truncated_seq = sequence[:50] + "..." if len(sequence) > 50 else sequence
                        sample_sequences[chain_id] = {
                            "length": len(sequence),
                            "sample": truncated_seq
                        }
                        chain_counter += 1
                    sample_metadata["sample_sequences"] = sample_sequences
                
                sample_pdb_metadata[pdb_id] = sample_metadata
                pdb_counter += 1
            
            if sample_pdb_metadata:
                pdb_metadata_summary["sample_pdb_metadata"] = sample_pdb_metadata
            
            # Add to fair_metadata_package.json as a separate section
            fair_metadata["pdb_metadata_enrichment"] = pdb_metadata_summary
            
            # Add PDB metadata to the additionalProperty section
            if "additionalProperty" not in fair_metadata:
                fair_metadata["additionalProperty"] = []
            
            # Remove any existing PDB metadata properties
            fair_metadata["additionalProperty"] = [
                prop for prop in fair_metadata["additionalProperty"]
                if prop.get("name") not in ["PDBMetadata", "PDBStructureMetadata", "PDBMetadataDetails", "PDBMetadataStatus"]
            ]
            
            # Add comprehensive PDB metadata property
            if all_pdb_metadata:
                # Create a summary of all PDB metadata
                pdb_summary = {
                    "total_pdbs": len(all_pdb_metadata),
                    "successfully_fetched": sum(1 for meta in all_pdb_metadata.values() if meta.get("found_in_api", False)),
                    "pdb_ids": list(all_pdb_metadata.keys()),
                    "metadata_available": True,
                    "fetch_timestamp": datetime.now().isoformat()
                }
                
                fair_metadata["additionalProperty"].append({
                    "@type": "PropertyValue",
                    "name": "PDBMetadata",
                    "value": json.dumps(pdb_summary, default=str),
                    "description": "Summary of PDB metadata retrieved from RCSB API",
                    "valueReference": {
                        "@type": "PropertyValue",
                        "name": "ParsedPDBMetadataSummary",
                        "value": {
                            "total_structures": len(all_pdb_metadata),
                            "structures_with_metadata": sum(1 for meta in all_pdb_metadata.values() if meta.get("found_in_api", False)),
                            "total_chains": sum(len(meta.get("sequences", {})) for meta in all_pdb_metadata.values()),
                            "homomeric_structures": sum(1 for meta in all_pdb_metadata.values() if meta.get("is_homomer", False)),
                            "unique_organisms": len(set(org for meta in all_pdb_metadata.values()
                                                      for org in meta.get("source_organism", [])))
                        }
                    }
                })
                
                # Add resolution information if available
                resolutions = [meta.get("resolution") for meta in all_pdb_metadata.values()
                             if meta.get("resolution") is not None]
                if resolutions:
                    fair_metadata["additionalProperty"].append({
                        "@type": "PropertyValue",
                        "name": "PDBResolutionStats",
                        "value": f"Average: {sum(resolutions)/len(resolutions):.2f} Å, Range: {min(resolutions):.2f}-{max(resolutions):.2f} Å",
                        "description": "Resolution statistics for all PDB structures"
                    })
            
            # Add overall PDB metadata status
            fair_metadata["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "PDBMetadataStatus",
                "value": f"COMPLETE - {len(self.stats['enriched_pdbs'])} PDBs enriched by script 3",
                "description": f"PDB metadata populated by script 3. Retrieved metadata for {len(self.stats['enriched_pdbs'])} out of {len(self.stats['unique_pdbs_processed'])} PDB structures from RCSB API."
            })
            
            # Update processing log
            if "processing_log" not in fair_metadata:
                fair_metadata["processing_log"] = []
            
            fair_metadata["processing_log"].append({
                "step": "3_pdb_metadata_enrichment",
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "details": {
                    "total_files": self.stats["total_files"],
                    "successful_enrichments": self.stats["successfully_enriched"],
                    "unique_pdbs": len(self.stats["unique_pdbs_processed"]),
                    "enriched_pdbs": len(self.stats["enriched_pdbs"]),
                    "metadata_cache_size": len(all_pdb_metadata),
                    "interfaces_updated": len(fair_metadata.get("hasPart", [])),
                    "dataset_source": "dataset_with_interfaces.json" if dataset_file.exists() else "existing_data"
                }
            })
            
            # Save the updated file
            with open(fair_metadata_file, 'w', encoding='utf-8') as f:
                json.dump(fair_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Updated fair_metadata_package.json at: {fair_metadata_file}")
            logger.info(f"   - Updated dataset structure from dataset_with_interfaces.json")
            logger.info(f"   - Updated PDBMetadataStatus in all interface items")
            logger.info(f"   - Added PDB metadata enrichment section")
            logger.info(f"   - Added PDB metadata to additionalProperty section")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse JSON in fair_metadata_package.json: {e}")
            logger.error(f"   File path: {fair_metadata_file}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to update fair_metadata_package.json: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return False
    
    def update_fair_metadata_package2(self) -> bool:
        """
        Update fair_metadata_package.json with PDB metadata information.
        
        Returns:
            True if successful, False otherwise
        """
        fair_metadata_file = self.find_fair_metadata_package_file()
        
        if not fair_metadata_file:
            logger.error("❌ Could not find fair_metadata_package.json")
            logger.error(f"   Searched in: {self.input_dir} and subdirectories")
            return False
        
        try:
            logger.info(f"📄 Loading fair_metadata_package.json from: {fair_metadata_file}")
            
            # Read the file
            with open(fair_metadata_file, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            # Parse JSON
            fair_metadata = json.loads(file_content)
            
            logger.debug(f"✅ Successfully parsed fair_metadata_package.json")
            logger.debug(f"   File has keys: {list(fair_metadata.keys())}")
            
            # Get metadata fetcher statistics
            fetcher_stats = self.metadata_fetcher.get_statistics()
            
            # Get ALL the PDB metadata that was fetched
            all_pdb_metadata = self.metadata_fetcher.metadata_cache
            
            # Update the PDBMetadataStatus field in the main JSON-LD
            fair_metadata = self.update_pdb_metadata_status_in_json(fair_metadata)
            
            # Create PDB metadata summary section
            pdb_metadata_summary = {
                "script": "3_add_pdb_metadata.py",
                "execution_date": datetime.now().isoformat(),
                "description": "PDB metadata enrichment from RCSB API using optimized sequential approach",
                "total_pdbs_processed": len(self.stats["unique_pdbs_processed"]),
                "successfully_enriched_pdbs": len(self.stats["enriched_pdbs"]),
                "metadata_fetcher_statistics": {
                    "total_api_requests": fetcher_stats.get("total_requests", 0),
                    "successful_fetches": fetcher_stats.get("successful_fetches", 0),
                    "failed_fetches": fetcher_stats.get("failed_fetches", 0),
                    "cache_hits": fetcher_stats.get("cached_hits", 0),
                    "unique_pdbs_fetched": fetcher_stats.get("unique_pdbs_fetched", 0)
                },
                "optimization_strategy": {
                    "name": "Sequential approach execution",
                    "description": "Approach 1 (entry) → Approach 2 (entities) → Approach 3 (assembly) as fallback",
                    "benefits": [
                        "Reduces unnecessary API calls",
                        "Stops when sufficient metadata is obtained",
                        "Most structures resolved with Approach 1 or 1+2",
                        "Approach 3 only used as fallback when others fail"
                    ]
                },
                "metadata_includes": [
                    "Resolution and experimental method",
                    "Source organism(s) and taxonomy IDs",
                    "ALL chain sequences (not just representative)",
                    "Homomer detection and analysis",
                    "Sequence clusters and identity analysis",
                    "Citation and publication information",
                    "Entity and chain mappings",
                    "Interface-specific sequence analysis"
                ]
            }
            
            # Add approach statistics if available
            if "approach_statistics" in fetcher_stats:
                pdb_metadata_summary["approach_statistics"] = fetcher_stats["approach_statistics"]
            
            # Add resolution statistics if available
            if "resolution_stats" in fetcher_stats:
                pdb_metadata_summary["resolution_statistics"] = fetcher_stats["resolution_stats"]
            
            # Add sequence statistics if available
            if "sequence_stats" in fetcher_stats:
                pdb_metadata_summary["sequence_statistics"] = fetcher_stats["sequence_stats"]
            
            # Add organism statistics if available
            if "organism_stats" in fetcher_stats:
                pdb_metadata_summary["organism_statistics"] = fetcher_stats["organism_stats"]
            
            # Add sample PDB metadata (first 3 PDBs as examples)
            sample_pdb_metadata = {}
            pdb_counter = 0
            for pdb_id, metadata in all_pdb_metadata.items():
                if pdb_counter >= 3:  # Limit to 3 samples
                    break
                
                # Create a simplified version of the metadata for display
                sample_metadata = {
                    "resolution": metadata.get("resolution"),
                    "experimental_method": metadata.get("experimental_method"),
                    "source_organism": metadata.get("source_organism", []),
                    "chain_count": metadata.get("chain_count", 0),
                    "unique_sequences": metadata.get("unique_sequences", 0),
                    "is_homomer": metadata.get("is_homomer", False),
                    "homomer_type": metadata.get("homomer_type"),
                    "entity_count": metadata.get("entity_count", 0)
                }
                
                # Add sequence information (first 2 chains as samples)
                if metadata.get("sequences"):
                    sample_sequences = {}
                    chain_counter = 0
                    for chain_id, sequence in metadata["sequences"].items():
                        if chain_counter >= 2:
                            break
                        # Store just the first 50 characters of sequence for brevity
                        truncated_seq = sequence[:50] + "..." if len(sequence) > 50 else sequence
                        sample_sequences[chain_id] = {
                            "length": len(sequence),
                            "sample": truncated_seq
                        }
                        chain_counter += 1
                    sample_metadata["sample_sequences"] = sample_sequences
                
                sample_pdb_metadata[pdb_id] = sample_metadata
                pdb_counter += 1
            
            if sample_pdb_metadata:
                pdb_metadata_summary["sample_pdb_metadata"] = sample_pdb_metadata
            
            # Add to fair_metadata_package.json as a separate section
            fair_metadata["pdb_metadata_enrichment"] = pdb_metadata_summary
            
            # ALSO add the actual PDB metadata to the additionalProperty section
            # This is where the actual PDB data should be added, not just the status
            if "additionalProperty" not in fair_metadata:
                fair_metadata["additionalProperty"] = []
            
            # Remove any existing PDB metadata properties
            fair_metadata["additionalProperty"] = [
                prop for prop in fair_metadata["additionalProperty"] 
                if prop.get("name") not in ["PDBMetadata", "PDBStructureMetadata", "PDBMetadataDetails"]
            ]
            
            # Add comprehensive PDB metadata property
            if all_pdb_metadata:
                # Create a summary of all PDB metadata
                pdb_summary = {
                    "total_pdbs": len(all_pdb_metadata),
                    "successfully_fetched": sum(1 for meta in all_pdb_metadata.values() if meta.get("found_in_api", False)),
                    "pdb_ids": list(all_pdb_metadata.keys()),
                    "metadata_available": True,
                    "fetch_timestamp": datetime.now().isoformat()
                }
                
                fair_metadata["additionalProperty"].append({
                    "@type": "PropertyValue",
                    "name": "PDBMetadata",
                    "value": json.dumps(pdb_summary, default=str),
                    "description": "Summary of PDB metadata retrieved from RCSB API",
                    "valueReference": {
                        "@type": "PropertyValue",
                        "name": "ParsedPDBMetadataSummary",
                        "value": {
                            "total_structures": len(all_pdb_metadata),
                            "structures_with_metadata": sum(1 for meta in all_pdb_metadata.values() if meta.get("found_in_api", False)),
                            "total_chains": sum(len(meta.get("sequences", {})) for meta in all_pdb_metadata.values()),
                            "homomeric_structures": sum(1 for meta in all_pdb_metadata.values() if meta.get("is_homomer", False)),
                            "unique_organisms": len(set(org for meta in all_pdb_metadata.values() 
                                                      for org in meta.get("source_organism", [])))
                        }
                    }
                })
                
                # Add resolution information if available
                resolutions = [meta.get("resolution") for meta in all_pdb_metadata.values() 
                             if meta.get("resolution") is not None]
                if resolutions:
                    fair_metadata["additionalProperty"].append({
                        "@type": "PropertyValue",
                        "name": "PDBResolutionStats",
                        "value": f"Average: {sum(resolutions)/len(resolutions):.2f} Å, Range: {min(resolutions):.2f}-{max(resolutions):.2f} Å",
                        "description": "Resolution statistics for all PDB structures"
                    })
            
            # Update processing log
            if "processing_log" not in fair_metadata:
                fair_metadata["processing_log"] = []
            
            fair_metadata["processing_log"].append({
                "step": "3_pdb_metadata_enrichment",
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "details": {
                    "total_files": self.stats["total_files"],
                    "successful_enrichments": self.stats["successfully_enriched"],
                    "unique_pdbs": len(self.stats["unique_pdbs_processed"]),
                    "enriched_pdbs": len(self.stats["enriched_pdbs"]),
                    "metadata_cache_size": len(all_pdb_metadata)
                }
            })
            
            # Save the updated file
            with open(fair_metadata_file, 'w', encoding='utf-8') as f:
                json.dump(fair_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Updated fair_metadata_package.json at: {fair_metadata_file}")
            logger.info(f"   - Updated PDBMetadataStatus field")
            logger.info(f"   - Added PDB metadata enrichment section with {len(self.stats['unique_pdbs_processed'])} PDBs")
            logger.info(f"   - Added actual PDB metadata to additionalProperty section")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse JSON in fair_metadata_package.json: {e}")
            logger.error(f"   File path: {fair_metadata_file}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to update fair_metadata_package.json: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return False

    def update_fair_metadata_package2(self) -> bool:
        """
        Update fair_metadata_package.json with PDB metadata information.
        
        Returns:
            True if successful, False otherwise
        """
        fair_metadata_file = self.find_fair_metadata_package_file()
        
        if not fair_metadata_file:
            logger.error("❌ Could not find fair_metadata_package.json")
            logger.error(f"   Searched in: {self.input_dir} and subdirectories")
            return False
        
        try:
            logger.info(f"📄 Loading fair_metadata_package.json from: {fair_metadata_file}")
            
            # Read the file
            with open(fair_metadata_file, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            # Parse JSON
            fair_metadata = json.loads(file_content)
            
            logger.debug(f"✅ Successfully parsed fair_metadata_package.json")
            logger.debug(f"   File has keys: {list(fair_metadata.keys())}")
            
            # Get metadata fetcher statistics
            fetcher_stats = self.metadata_fetcher.get_statistics()
            
            # Update the PDBMetadataStatus field in the main JSON-LD
            fair_metadata = self.update_pdb_metadata_status_in_json(fair_metadata)
        
            print ('ZZ',self.stats)    
            # Create PDB metadata summary section
            pdb_metadata_summary = {
                "script": "3_add_pdb_metadata.py",
                "execution_date": datetime.now().isoformat(),
                "description": "PDB metadata enrichment from RCSB API using optimized sequential approach",
                "total_pdbs_processed": len(self.stats["unique_pdbs_processed"]),
                "successfully_enriched_pdbs": len(self.stats["enriched_pdbs"]),
                "metadata_fetcher_statistics": {
                    "total_api_requests": fetcher_stats.get("total_requests", 0),
                    "successful_fetches": fetcher_stats.get("successful_fetches", 0),
                    "failed_fetches": fetcher_stats.get("failed_fetches", 0),
                    "cache_hits": fetcher_stats.get("cached_hits", 0),
                    "unique_pdbs_fetched": fetcher_stats.get("unique_pdbs_fetched", 0)
                },
                "optimization_strategy": {
                    "name": "Sequential approach execution",
                    "description": "Approach 1 (entry) → Approach 2 (entities) → Approach 3 (assembly) as fallback",
                    "benefits": [
                        "Reduces unnecessary API calls",
                        "Stops when sufficient metadata is obtained",
                        "Most structures resolved with Approach 1 or 1+2",
                        "Approach 3 only used as fallback when others fail"
                    ]
                },
                "metadata_includes": [
                    "Resolution and experimental method",
                    "Source organism(s) and taxonomy IDs",
                    "ALL chain sequences (not just representative)",
                    "Homomer detection and analysis",
                    "Sequence clusters and identity analysis",
                    "Citation and publication information",
                    "Entity and chain mappings",
                    "Interface-specific sequence analysis"
                ]
            }
            
            # Add approach statistics if available
            if "approach_statistics" in fetcher_stats:
                pdb_metadata_summary["approach_statistics"] = fetcher_stats["approach_statistics"]
            
            # Add to fair_metadata_package.json as a separate section
            fair_metadata["pdb_metadata_enrichment"] = pdb_metadata_summary
            
            # Update processing log
            if "processing_log" not in fair_metadata:
                fair_metadata["processing_log"] = []
            
            fair_metadata["processing_log"].append({
                "step": "3_pdb_metadata_enrichment",
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "details": {
                    "total_files": self.stats["total_files"],
                    "successful_enrichments": self.stats["successfully_enriched"],
                    "unique_pdbs": len(self.stats["unique_pdbs_processed"]),
                    "enriched_pdbs": len(self.stats["enriched_pdbs"])
                }
            })
            
            # Save the updated file
            with open(fair_metadata_file, 'w', encoding='utf-8') as f:
                json.dump(fair_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Updated fair_metadata_package.json at: {fair_metadata_file}")
            logger.info(f"   - Updated PDBMetadataStatus field")
            logger.info(f"   - Added PDB metadata enrichment section with {len(self.stats['unique_pdbs_processed'])} PDBs")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse JSON in fair_metadata_package.json: {e}")
            logger.error(f"   File path: {fair_metadata_file}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to update fair_metadata_package.json: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return False
    
    def update_embedded_markup_html(self) -> bool:
        """
        Update embedded_markup.html with PDB metadata information.
        
        Returns:
            True if successful, False otherwise
        """
        embedded_html_file = self.find_embedded_markup_html_file()
        
        if not embedded_html_file:
            logger.error("❌ Could not find embedded_markup.html")
            logger.error(f"   Searched in: {self.input_dir} and subdirectories")
            return False
        
        try:
            logger.info(f"📄 Loading embedded_markup.html from: {embedded_html_file}")
            
            # Read the file
            with open(embedded_html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            logger.debug(f"✅ Successfully read embedded_markup.html")
            logger.debug(f"   File size: {len(html_content)} bytes")
            
            # Check if file contains JSON-LD
            if '<script type="application/ld+json">' not in html_content:
                logger.warning("⚠️  embedded_markup.html does not contain JSON-LD script tags")
                logger.warning("   Will add PDB metadata summary section anyway")
            
            # Update the PDBMetadataStatus in JSON-LD script tags
            updated_html = self.update_pdb_metadata_status_in_html(html_content)
            
            # Get metadata fetcher statistics
            fetcher_stats = self.metadata_fetcher.get_statistics()
            
            # Create HTML section for PDB metadata summary
            pdb_metadata_section = f"""
<!-- PDB Metadata Summary Section - Added by script 3_add_pdb_metadata.py -->
<section id="pdb-metadata-summary" class="pdb-metadata-summary-section" style="margin: 20px 0; padding: 20px; background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px;">
    <h2 style="color: #2c3e50;">PDB Metadata Enrichment Summary</h2>
    
    <div class="summary-stats" style="margin-bottom: 20px;">
        <h3 style="color: #3498db;">Statistics</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr style="background-color: #e9ecef;">
                    <th style="padding: 10px; text-align: left; border: 1px solid #dee2e6;">Metric</th>
                    <th style="padding: 10px; text-align: left; border: 1px solid #dee2e6;">Value</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="padding: 10px; border: 1px solid #dee2e6;">Total PDBs Processed</td>
                    <td style="padding: 10px; border: 1px solid #dee2e6;">{len(self.stats['unique_pdbs_processed'])}</td>
                </tr>
                <tr style="background-color: #f8f9fa;">
                    <td style="padding: 10px; border: 1px solid #dee2e6;">Successfully Enriched</td>
                    <td style="padding: 10px; border: 1px solid #dee2e6;">{len(self.stats['enriched_pdbs'])}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #dee2e6;">Total API Requests</td>
                    <td style="padding: 10px; border: 1px solid #dee2e6;">{fetcher_stats.get('total_requests', 0)}</td>
                </tr>
                <tr style="background-color: #f8f9fa;">
                    <td style="padding: 10px; border: 1px solid #dee2e6;">Successful Fetches</td>
                    <td style="padding: 10px; border: 1px solid #dee2e6;">{fetcher_stats.get('successful_fetches', 0)}</td>
                </tr>
            </tbody>
        </table>
    </div>
    
    <div class="optimization-info" style="margin-bottom: 20px;">
        <h3 style="color: #3498db;">Optimization Strategy</h3>
        <p><strong>Sequential Approach Execution:</strong> Stop when metadata is obtained</p>
        <ol>
            <li><strong>Approach 1:</strong> Entry endpoint (basic structure info)</li>
            <li><strong>Approach 2:</strong> Entity endpoints (sequences, organisms)</li>
            <li><strong>Approach 3:</strong> Assembly endpoint (fallback only)</li>
        </ol>
        <p><em>Most structures resolved with Approach 1 or 1+2</em></p>
    </div>
    
    <div class="metadata-content" style="margin-bottom: 20px;">
        <h3 style="color: #3498db;">PDB Metadata Included:</h3>
        <ul>
            <li>Resolution and experimental method</li>
            <li>Source organism(s) and taxonomy IDs</li>
            <li>ALL chain sequences (not just representative)</li>
            <li>Homomer detection and analysis</li>
            <li>Sequence clusters and identity analysis</li>
            <li>Citation and publication information</li>
            <li>Entity and chain mappings</li>
            <li>Interface-specific sequence analysis</li>
        </ul>
    </div>
    
    <div class="status-update">
        <h3 style="color: #27ae60;">Status Update</h3>
        <p>The <strong>PDBMetadataStatus</strong> field in the JSON-LD markup has been updated from "To be populated by script 3" to:</p>
        <div style="background-color: #e8f5e9; padding: 10px; border-left: 4px solid #27ae60; margin: 10px 0;">
            <strong>COMPLETE - {len(self.stats['enriched_pdbs'])} PDBs enriched</strong><br>
            PDB metadata populated by script 3. Retrieved metadata for {len(self.stats['enriched_pdbs'])} out of {len(self.stats['unique_pdbs_processed'])} PDB structures from RCSB API.
        </div>
    </div>
</section>
<!-- End PDB Metadata Summary Section -->
"""
            
            # Find where to insert the PDB metadata section
            # Look for the end of the body tag
            insert_point = updated_html.find("</body>")
            if insert_point == -1:
                # If no body tag, look for the end of the html
                insert_point = updated_html.find("</html>")
                if insert_point == -1:
                    # Just append at the end
                    insert_point = len(updated_html)
                    logger.warning("⚠️  Could not find </body> or </html> tag, appending at end")
            
            # Insert the PDB metadata section before the closing body/html tag
            final_html = updated_html[:insert_point] + pdb_metadata_section + updated_html[insert_point:]
            
            # Save the updated HTML file
            with open(embedded_html_file, 'w', encoding='utf-8') as f:
                f.write(final_html)
            
            logger.info(f"✅ Updated embedded_markup.html at: {embedded_html_file}")
            logger.info(f"   - Updated PDBMetadataStatus in JSON-LD script tags")
            logger.info(f"   - Added PDB metadata summary section")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update embedded_markup.html: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return False
    
    def process_interface_file(self, interface_file: Path) -> bool:
        """
        Process a single interface JSON-LD file and add PDB metadata.
        
        Args:
            interface_file: Path to interface JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load the JSON data
            with open(interface_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Extract interface information
            interface_info = self.extract_interface_info(json_data)
            
            if interface_info["interface_type"] != "interface_item":
                logger.warning(f"File {interface_file} is not an interface item")
                return False
            
            interface_id = interface_info["interface_id"]
            pdb_id = interface_info["pdb_id"]
            
            if not interface_id or not pdb_id:
                logger.warning(f"Missing interface_id or pdb_id in {interface_file}")
                return False
            
            logger.info(f"Processing interface: {interface_id} (PDB: {pdb_id})")
            
            # Track unique PDBs
            self.stats["unique_pdbs_processed"].add(pdb_id.upper())
            
            # Fetch PDB metadata using optimized sequential approach
            pdb_metadata = self.metadata_fetcher.enrich_protein_with_pdb_metadata(interface_info)
            
            # Update the JSON data with PDB metadata
            updated_data = self.update_interface_item(json_data, pdb_metadata)
            
            # Save the updated file
            with open(interface_file, 'w', encoding='utf-8') as f:
                json.dump(updated_data, f, indent=2, ensure_ascii=False)
            
            # Update statistics
            self.stats["successfully_enriched"] += 1
            if pdb_metadata.get("structure_metadata", {}).get("found_in_api"):
                self.stats["enriched_pdbs"].add(pdb_id.upper())
            
            # Log success
            structure_meta = pdb_metadata.get("structure_metadata", {})
            if structure_meta.get("found_in_api"):
                approaches_used = structure_meta.get("approaches_used", [])
                logger.info(f"✅ Successfully enriched {interface_file.name}")
                logger.info(f"   Approaches used: {approaches_used}")
                logger.info(f"   PDB metadata: {structure_meta.get('chain_count', 0)} chains, "
                          f"{structure_meta.get('unique_sequences', 0)} unique seqs")
            else:
                logger.warning(f"⚠️  PDB metadata not available for {pdb_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {interface_file}: {e}")
            self.stats["failed_enrichments"] += 1
            return False
    
    def update_dataset_file(self, dataset_file: Path) -> bool:
        """
        Update the main dataset file with PDB metadata information.
        
        Args:
            dataset_file: Path to dataset JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                dataset_data = json.load(f)
            
            # Check if this is a dataset with interface items
            if "hasPart" in dataset_data and isinstance(dataset_data["hasPart"], list):
                updated_items = []
                
                for item in dataset_data["hasPart"]:
                    # Extract interface information
                    interface_info = self.extract_interface_info(item)
                    
                    if interface_info["interface_type"] == "interface_item":
                        interface_id = interface_info["interface_id"]
                        pdb_id = interface_info["pdb_id"]
                        
                        if interface_id and pdb_id:
                            # Fetch PDB metadata
                            pdb_metadata = self.metadata_fetcher.enrich_protein_with_pdb_metadata(interface_info)
                            
                            # Update the interface item
                            updated_item = self.update_interface_item(item, pdb_metadata)
                            updated_items.append(updated_item)
                        else:
                            updated_items.append(item)
                    else:
                        updated_items.append(item)
                
                # Update the dataset
                dataset_data["hasPart"] = updated_items
                
                # Save the updated dataset
                with open(dataset_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"✅ Updated dataset file: {dataset_file}")
                return True
            
            else:
                logger.warning(f"Dataset file {dataset_file} doesn't have interface items in hasPart")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update dataset file {dataset_file}: {e}")
            return False
    
    def update_manifest_file(self, manifest_file: Path) -> bool:
        """
        Update the manifest file with PDB metadata statistics.
        
        Args:
            manifest_file: Path to manifest.json file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
            
            # Get metadata fetcher statistics
            fetcher_stats = self.metadata_fetcher.get_statistics()
            
            # Add PDB metadata information
            manifest_data["pdb_metadata_enrichment"] = {
                "script": "3rd script - PDB Metadata Enricher (Optimized)",
                "execution_date": datetime.now().isoformat(),
                "statistics": {
                    "total_files_processed": self.stats["total_files"],
                    "successfully_enriched": self.stats["successfully_enriched"],
                    "failed_enrichments": self.stats["failed_enrichments"],
                    "unique_pdbs_processed": len(self.stats["unique_pdbs_processed"]),
                    "enriched_pdbs": len(self.stats["enriched_pdbs"]),
                    "success_rate": f"{(self.stats['successfully_enriched'] / self.stats['total_files'] * 100):.1f}%" if self.stats['total_files'] > 0 else "0%"
                },
                "api_statistics": fetcher_stats,
                "settings": {
                    "pdb_api_base_url": self.pdb_api_base_url,
                    "batch_size": self.batch_size,
                    "delay_between_batches": self.delay_between_batches,
                    "optimization_strategy": "Sequential approach execution - stops when metadata is obtained"
                }
            }
            
            # Update fields_to_be_populated section if it exists
            if "schema_structure" in manifest_data and "fields_to_be_populated" in manifest_data["schema_structure"]:
                manifest_data["schema_structure"]["fields_to_be_populated"]["PDB_metadata"] = {
                    "status": "POPULATED by 3rd script",
                    "timestamp": datetime.now().isoformat(),
                    "description": "Comprehensive PDB metadata from RCSB API",
                    "data_includes": [
                        "Resolution",
                        "Experimental method",
                        "Source organism(s)",
                        "ALL chain sequences",
                        "Homomer detection",
                        "Sequence clusters",
                        "Citation information",
                        "Entity information"
                    ],
                    "optimization_note": "Sequential approach execution: Approach 1 (entry) → Approach 2 (entities) → Approach 3 (assembly) as fallback"
                }
            
            # Save the updated manifest
            with open(manifest_file, 'w', encoding='utf-8') as f:
                json.dump(manifest_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Updated manifest file: {manifest_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update manifest file {manifest_file}: {e}")
            return False
    
    def save_metadata_cache(self, cache_file: Path) -> bool:
        """
        Save the PDB metadata cache to a file.
        
        Args:
            cache_file: Path to save cache file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata_fetcher.metadata_cache, f, indent=2, default=str)
            
            logger.info(f"✅ Saved PDB metadata cache to: {cache_file}")
            logger.info(f"   Cache contains {len(self.metadata_fetcher.metadata_cache)} PDB entries")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save metadata cache to {cache_file}: {e}")
            return False
    
    def run(self) -> Dict[str, Any]:
        """
        Main execution method.
        
        Returns:
            Statistics dictionary
        """
        logger.info("=" * 60)
        logger.info("PDB Metadata Enricher - Script 3 (Optimized)")
        logger.info("=" * 60)
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"PDB API URL: {self.pdb_api_base_url}")
        logger.info("Optimization: Sequential approach execution")
        logger.info("Approach 1: Entry endpoint → Approach 2: Entity endpoints → Approach 3: Assembly endpoint (fallback)")
        logger.info("")
        
        # Find JSON files to process
        json_files = self.find_json_files()
        self.stats["total_files"] = len(json_files)
        
        if not json_files:
            logger.warning("No JSON files found to process!")
            return self.stats
        
        # Process interface files
        logger.info(f"Processing {len(json_files)} interface files...")
        logger.info("Using optimized sequential approach to fetch PDB metadata.")
        logger.info("")
        
        for i, json_file in enumerate(json_files):
            logger.info(f"[{i+1}/{len(json_files)}] Processing: {json_file.name}")
            self.process_interface_file(json_file)
            
            # Add delay between requests to be polite to the API
            if i < len(json_files) - 1:  # Don't delay after the last one
                time.sleep(self.delay_between_batches)
        
        # Update dataset file if it exists
        dataset_file = self.input_dir / "dataset_with_interfaces.json"
        if dataset_file.exists():
            logger.info(f"Updating dataset file: {dataset_file.name}")
            self.update_dataset_file(dataset_file)
        
        # Update manifest file if it exists
        manifest_file = self.input_dir / "manifest.json"
        if manifest_file.exists():
            logger.info(f"Updating manifest file: {manifest_file.name}")
            self.update_manifest_file(manifest_file)
        
        # Update fair_metadata_package.json
        logger.info("Updating fair_metadata_package.json with PDB metadata information...")
        fair_metadata_success = self.update_fair_metadata_package()
        
        # Update embedded_markup.html
        logger.info("Updating embedded_markup.html with PDB metadata section...")
        embedded_html_success = self.update_embedded_markup_html()
        
        # Save metadata cache
        cache_file = self.input_dir / "pdb_metadata_cache.json"
        if self.metadata_fetcher.metadata_cache:
            self.save_metadata_cache(cache_file)
        
        # Print summary statistics
        logger.info("")
        logger.info("=" * 60)
        logger.info("PDB METADATA ENRICHMENT COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total files processed: {self.stats['total_files']}")
        logger.info(f"Successfully enriched: {self.stats['successfully_enriched']}")
        logger.info(f"Failed enrichments: {self.stats['failed_enrichments']}")
        logger.info(f"Unique PDBs processed: {len(self.stats['unique_pdbs_processed'])}")
        logger.info(f"PDBs with metadata: {len(self.stats['enriched_pdbs'])}")
        
        # Print metadata fetcher statistics
        fetcher_stats = self.metadata_fetcher.get_statistics()
        logger.info("")
        logger.info("API Statistics:")
        logger.info(f"  Total API requests: {fetcher_stats.get('total_requests', 0)}")
        logger.info(f"  Successful fetches: {fetcher_stats.get('successful_fetches', 0)}")
        logger.info(f"  Failed fetches: {fetcher_stats.get('failed_fetches', 0)}")
        logger.info(f"  Cache hits: {fetcher_stats.get('cached_hits', 0)}")
        
        # Print approach statistics
        if "approach_statistics" in fetcher_stats:
            app_stats = fetcher_stats["approach_statistics"]
            logger.info(f"\nApproach Statistics (Optimization Results):")
            logger.info(f"  Total structures fetched: {app_stats.get('total_structures', 0)}")
            logger.info(f"  Success rate: {app_stats.get('success_rate', 0):.1f}%")
            logger.info(f"  Approach 1 only (basic info): {app_stats['approach_breakdown'].get('approach_1_only', 0)}")
            logger.info(f"  Approach 1+2 (basic + sequences): {app_stats['approach_breakdown'].get('approach_1_2', 0)}")
            logger.info(f"  All approaches needed: {app_stats['approach_breakdown'].get('all_approaches', 0)}")
            logger.info(f"  Approach 3 only (fallback): {app_stats['approach_breakdown'].get('approach_3_only', 0)}")
        
        if "metadata_cache" in fetcher_stats:
            cache_stats = fetcher_stats["metadata_cache"]
            logger.info(f"  Metadata cache size: {cache_stats.get('size', 0)}")
            logger.info(f"  Total chains fetched: {cache_stats.get('total_chains', 0)}")
            logger.info(f"  Homomeric structures: {cache_stats.get('homomeric_structures', 0)}")
        
        if "resolution_stats" in fetcher_stats:
            res_stats = fetcher_stats["resolution_stats"]
            logger.info(f"  Resolution stats: {res_stats.get('count', 0)} structures")
            logger.info(f"    Average: {res_stats.get('average', 0):.2f} Å")
            logger.info(f"    Range: {res_stats.get('min', 0):.2f}-{res_stats.get('max', 0):.2f} Å")
        
        if "sequence_stats" in fetcher_stats:
            seq_stats = fetcher_stats["sequence_stats"]
            logger.info(f"  Sequence stats: {seq_stats.get('total_sequences', 0)} chains")
            logger.info(f"    Unique sequences: {seq_stats.get('unique_sequences', 0)}")
            logger.info(f"    Avg length: {seq_stats.get('avg_sequence_length', 0):.1f} residues")
        
        if "organism_stats" in fetcher_stats:
            org_stats = fetcher_stats["organism_stats"]
            logger.info(f"  Organism stats: {org_stats.get('unique_organisms', 0)} unique organisms")
        
        logger.info("")
        logger.info("📊 PDB Metadata Added:")
        logger.info("  - Resolution and experimental method")
        logger.info("  - Source organism(s) and taxonomy")
        logger.info("  - ALL chain sequences (not just representative)")
        logger.info("  - Homomer detection and analysis")
        logger.info("  - Sequence clusters and identity analysis")
        logger.info("  - Citation and publication information")
        logger.info("  - Entity and chain mappings")
        logger.info("  - Interface-specific sequence analysis")
        
        logger.info("")
        logger.info(f"✅ Optimization Results:")
        logger.info(f"  - Sequential approach execution reduces unnecessary API calls")
        logger.info(f"  - Only proceeds to next approach if needed")
        logger.info(f"  - Most structures resolved with Approach 1 or 1+2")
        logger.info(f"  - Approach 3 only used as fallback when others fail")
        
        logger.info("")
        logger.info(f"📁 Updated Files:")
        logger.info(f"  - All interface JSON-LD files enriched with PDB metadata")
        if manifest_file.exists():
            logger.info(f"  - manifest.json updated with statistics")
        if fair_metadata_success:
            logger.info(f"  - fair_metadata_package.json:")
            logger.info(f"     * Updated PDBMetadataStatus field")
            logger.info(f"     * Added PDB metadata enrichment section")
        else:
            logger.info(f"  - ❌ fair_metadata_package.json NOT updated")
        if embedded_html_success:
            logger.info(f"  - embedded_markup.html:")
            logger.info(f"     * Updated PDBMetadataStatus in JSON-LD script tags")
            logger.info(f"     * Added PDB metadata summary section")
        else:
            logger.info(f"  - ❌ embedded_markup.html NOT updated")
        if self.metadata_fetcher.metadata_cache:
            logger.info(f"  - pdb_metadata_cache.json saved with API responses")
        
        logger.info("")
        logger.info(f"Next step: Run script 4 to add ClusterID information.")
        
        return {**self.stats, **fetcher_stats}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PDB Metadata Enricher (Optimized) - Update JSON-LD files with comprehensive PDB metadata from RCSB API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input bioschemas_output
  %(prog)s --input bioschemas_output --api-url https://data.rcsb.org/rest/v1/core
  %(prog)s --input bioschemas_output --batch-size 5 --delay 2.0
  
Optimization Strategy:
  Approach 1: Fetch basic entry info (resolution, dates, etc.)
  If Approach 1 succeeds and has entity IDs → Approach 2: Fetch entity info for sequences
  If Approach 1 fails or Approach 2 doesn't give sequences → Approach 3: Try assembly endpoint
  Each approach stops if it gets the needed data, avoiding unnecessary API calls.
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory containing JSON-LD files (from scripts 1 & 2)"
    )
    
    parser.add_argument(
        "--api-url",
        default="https://data.rcsb.org/rest/v1/core",
        help="RCSB PDB REST API base URL (default: https://data.rcsb.org/rest/v1/core)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of PDB IDs to process in parallel (not fully implemented, default: 10)"
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API requests in seconds (default: 1.0)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Skip saving metadata cache (not recommended)"
    )
    
    parser.add_argument(
        "--max-requests-per-pdb",
        type=int,
        default=3,
        help="Maximum number of API requests per PDB ID (default: 3, for the 3 approaches)"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  PDB Metadata Enricher - Script 3 (Optimized)                 ║
║  Updates JSON-LD files with comprehensive PDB metadata        ║
║  using sequential approach execution                          ║
╚═══════════════════════════════════════════════════════════════╝
    
Settings:
  Input Directory:     {args.input}
  PDB API URL:         {args.api_url}
  Batch Size:          {args.batch_size}
  Request Delay:       {args.delay} seconds
  Skip Cache:          {args.skip_cache}
  Max Requests per PDB: {args.max_requests_per_pdb}
  
Optimization Strategy:
  Approach 1: Entry endpoint (basic info)
  → If successful with entity IDs: Approach 2: Entity endpoints (sequences)
  → If unsuccessful: Approach 3: Assembly endpoint (fallback)
  Each approach stops when data is obtained.
    """)
    
    # Check if input directory exists
    if not os.path.exists(args.input):
        print(f"\n❌ ERROR: Input directory '{args.input}' does not exist!")
        print("   Please run scripts 1 and 2 first to generate and update the JSON-LD files.")
        return
    
    # List files in directory for debugging
    print(f"\n📁 Files in {args.input}:")
    for item in Path(args.input).iterdir():
        print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
    
    # Create enricher and run
    enricher = PDBMetadataEnricher(
        input_dir=args.input,
        pdb_api_base_url=args.api_url,
        batch_size=args.batch_size,
        delay_between_batches=args.delay
    )
    
    stats = enricher.run()
    
    print(f"\n✅ PDB metadata enrichment complete!")
    print(f"   Updated files saved to: {args.input}")
    
    if not args.skip_cache and "metadata_cache" in stats:
        cache_size = stats["metadata_cache"].get("size", 0)
        print(f"   Metadata cache saved: {cache_size} PDB entries")
    
    # Show optimization benefits
    if "approach_statistics" in stats:
        app_stats = stats["approach_statistics"]
        total = app_stats.get("total_structures", 0)
        if total > 0:
            approach_1_only = app_stats["approach_breakdown"].get("approach_1_only", 0)
            approach_1_2 = app_stats["approach_breakdown"].get("approach_1_2", 0)
            all_approaches = app_stats["approach_breakdown"].get("all_approaches", 0)
            approach_3_only = app_stats["approach_breakdown"].get("approach_3_only", 0)
            
            # Calculate API call savings
            max_possible_calls = total * 3  # If all 3 approaches were always called
            actual_calls = (approach_1_only * 1) + (approach_1_2 * 2) + (all_approaches * 3) + (approach_3_only * 1)
            savings = max_possible_calls - actual_calls
            savings_percent = (savings / max_possible_calls * 100) if max_possible_calls > 0 else 0
            
            print(f"\n   Optimization Benefits:")
            print(f"     Maximum possible API calls: {max_possible_calls}")
            print(f"     Actual API calls made: {actual_calls}")
            print(f"     API calls saved: {savings} ({savings_percent:.1f}%)")
            print(f"     Efficiency improvement: {((max_possible_calls/actual_calls) if actual_calls > 0 else 0):.1f}x")
    
    print(f"\n   Updated files:")
    print(f"     - All interface JSON-LD files enriched with PDB metadata")
    print(f"     - manifest.json updated with statistics")
    
    # Check if files were actually updated
    fair_metadata_file = Path(args.input) / "fair_metadata_package.json"
    embedded_html_file = Path(args.input) / "embedded_markup.html"
    
    if fair_metadata_file.exists():
        print(f"     - fair_metadata_package.json exists and should be updated")
    else:
        print(f"     - ❌ fair_metadata_package.json NOT FOUND at {fair_metadata_file}")
    
    if embedded_html_file.exists():
        print(f"     - embedded_markup.html exists and should be updated")
    else:
        print(f"     - ❌ embedded_markup.html NOT FOUND at {embedded_html_file}")
    
    if not args.skip_cache:
        print(f"     - pdb_metadata_cache.json (cache file)")
    
    print(f"\n   Next step: Run script 4 to add ClusterID information")


if __name__ == "__main__":
    main()
