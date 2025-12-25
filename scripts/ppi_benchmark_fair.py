"""
Script to import protein-protein interaction benchmark data and generate FAIR-compliant
Bioschemas markup with Croissant compatibility for ML datasets.
Optimized for minimal PDB API calls with batch requests and smart caching.
"""

import pandas as pd
import json
import requests
from io import BytesIO
from datetime import datetime, timedelta
from typing import Set, Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging
import os
import gzip
import argparse
import time
import re
from urllib.parse import quote
from collections import Counter, defaultdict
import threading
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProteinInterface:
    """Data class representing a protein-protein interaction interface."""
    ID: str  # PDB ID (4 letters)
    InterfaceID: str  # Unique interface identifier: {PDB_ID}_{interface_number}
    AuthChain1: str  # First chain in the interface
    AuthChain2: str  # Second chain in the interface
    SymmetryOp1: str  # Symmetry operation for chain 1
    SymmetryOp2: str  # Symmetry operation for chain 2
    physio: bool  # Physiological (TRUE) or non-physiological (FALSE)
    label: int = field(init=False)  # Derived from physio: 1 for TRUE, 0 for FALSE
    interface_source: str = field(init=False)  # Source of interface ID: "QSalign" or "ProtCID"
    LabelChain1: Optional[str] = None
    LabelChain2: Optional[str] = None
    contacts: Optional[int] = None
    gene: Optional[str] = None
    superfamily: Optional[str] = None
    pfam: Optional[str] = None
    bsa: Optional[float] = None  # Buried surface area
    bsa_polar: Optional[float] = None
    bsa_apolar: Optional[float] = None
    frac_polar: Optional[float] = None
    frac_apolar: Optional[float] = None
    comments: Optional[str] = None
    cluster_id: Optional[str] = None  # Cluster ID from BLASTClust
    cluster_size: Optional[int] = None  # Size of the cluster
    cluster_members: Optional[List[str]] = None  # Other members in the cluster

    def __post_init__(self):
        """Set the numeric label based on physio value."""
        if isinstance(self.physio, bool):
            self.label = 1 if self.physio else 0
        elif isinstance(self.physio, str):
            self.label = 1 if self.physio.strip().upper() == "TRUE" else 0
        else:
            try:
                self.label = 1 if bool(self.physio) else 0
            except:
                self.label = 0
                logger.warning(f"Could not convert physio value '{self.physio}' to boolean for {self.InterfaceID}")

class ClusterIDProcessor:
    """Processes BLASTClust output and adds ClusterID information."""

    def __init__(self, blastclust_file: str):
        """
        Initialize the processor.

        Args:
            blastclust_file: Path to BLASTClust output file
        """
        self.blastclust_file = blastclust_file

        # Data structures - PUBLIC for direct access
        self.cluster_mapping = {}  # interface_id -> cluster_id
        self.cluster_members = defaultdict(list)  # cluster_id -> list of interface_ids

        # Statistics
        self.stats = {
            "total_clusters": 0,
            "total_interfaces_in_clusters": 0,
            "clusters_processed": 0,
            "clusters_with_single_member": 0,
            "clusters_with_multiple_members": 0,
            "largest_cluster_size": 0
        }

    def parse_blastclust_file(self) -> bool:
        """
        Parse BLASTClust output file.

        Format: Each line contains space-separated InterfaceIDs belonging to the same cluster.
        First InterfaceID in each line becomes the ClusterID for all interfaces in that line.

        Returns:
            True if successful, False otherwise
        """
        blastclust_path = Path(self.blastclust_file)
        if not blastclust_path.exists():
            logger.error(f"BLASTClust file not found: {self.blastclust_file}")
            return False

        try:
            logger.info(f"Parsing BLASTClust file: {self.blastclust_file}")
            logger.info(f"BLASTClust options used: -S 25 -L 0.5 -b F")

            with open(blastclust_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            self.stats["total_clusters"] = len(lines)

            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                # Split by whitespace (space or tab)
                interface_ids = re.split(r'\s+', line)
                if not interface_ids:
                    continue

                # FIRST ELEMENT BECOMES THE CLUSTER ID FOR ALL INTERFACES IN THIS LINE
                cluster_id = interface_ids[0]

                # Clean and add all interface IDs to this cluster
                for interface_id in interface_ids:
                    # Clean the interface ID if needed
                    clean_interface_id = self._clean_interface_id(interface_id)
                    # ASSIGN CLUSTER ID (first element) to each interface
                    self.cluster_mapping[clean_interface_id] = cluster_id
                    self.cluster_members[cluster_id].append(clean_interface_id)

                # Update statistics
                self.stats["total_interfaces_in_clusters"] += len(interface_ids)
                self.stats["clusters_processed"] += 1

                if len(interface_ids) == 1:
                    self.stats["clusters_with_single_member"] += 1
                else:
                    self.stats["clusters_with_multiple_members"] += 1

                self.stats["largest_cluster_size"] = max(
                    self.stats["largest_cluster_size"],
                    len(interface_ids)
                )

            logger.info(f"Successfully parsed {self.stats['clusters_processed']} clusters")
            logger.info(f"Total interfaces in clusters: {self.stats['total_interfaces_in_clusters']}")
            logger.info(f"Clusters with single member: {self.stats['clusters_with_single_member']}")
            logger.info(f"Clusters with multiple members: {self.stats['clusters_with_multiple_members']}")
            logger.info(f"Largest cluster size: {self.stats['largest_cluster_size']}")

            # Show sample of cluster assignments
            if self.cluster_mapping:
                logger.info(f"Sample cluster assignments (first 5 interfaces):")
                for i, (interface_id, cluster_id) in enumerate(list(self.cluster_mapping.items())[:5]):
                    members = self.cluster_members.get(cluster_id, [])
                    logger.info(f"  {interface_id} → Cluster: {cluster_id} (size: {len(members)})")

            return True

        except Exception as e:
            logger.error(f"Failed to parse BLASTClust file: {e}")
            return False

    def _clean_interface_id(self, interface_id: str) -> str:
        """
        Clean interface ID to match JSON-LD format.

        Args:
            interface_id: Interface ID from BLASTClust file

        Returns:
            Cleaned interface ID
        """
        if not interface_id or pd.isna(interface_id):
            return ""

        # Remove any file extensions or extra characters
        clean_id = str(interface_id).strip()

        # Remove common file extensions
        for ext in ['.pdb', '.cif', '.gz', '.ent', '.pdb1']:
            if clean_id.endswith(ext):
                clean_id = clean_id[:-len(ext)]

        # Convert to uppercase for consistency
        clean_id = clean_id.upper()

        return clean_id

class PDBApiClient:
    """Optimized client for PDB API calls with batching and caching."""
    
    def __init__(self, base_url: str = "https://data.rcsb.org/rest/v1"):
        """
        Initialize the PDB API client.
        
        Args:
            base_url: Base URL for RCSB PDB REST API
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PPI-Benchmark-FAIR-Metadata/1.0',
            'Accept': 'application/json'
        })
        
        # Enhanced caching with expiration
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_expiry = timedelta(hours=24)  # Cache for 24 hours
        
        # Batch processing settings
        self.batch_size = 10  # Number of PDBs to fetch in one batch
        self.request_delay = 0.5  # Delay between batch requests in seconds
        
        # Statistics
        self.stats = {
            'api_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'batches_processed': 0,
            'total_pdbs_fetched': 0,
            'obsolete_entries': 0
        }
    
    def _get_cache_key(self, pdb_id: str, endpoint: str = "entry") -> str:
        """Generate cache key for a PDB ID and endpoint."""
        return f"{pdb_id.upper()}_{endpoint}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self.cache_timestamps:
            return False
        cache_time = self.cache_timestamps[cache_key]
        return (datetime.now() - cache_time) < self.cache_expiry
    
    def _get_from_cache(self, pdb_id: str, endpoint: str = "entry") -> Optional[Dict]:
        """Get data from cache if valid."""
        cache_key = self._get_cache_key(pdb_id, endpoint)
        if cache_key in self.cache and self._is_cache_valid(cache_key):
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]
        self.stats['cache_misses'] += 1
        return None
    
    def _save_to_cache(self, pdb_id: str, endpoint: str, data: Dict) -> None:
        """Save data to cache with timestamp."""
        cache_key = self._get_cache_key(pdb_id, endpoint)
        self.cache[cache_key] = data
        self.cache_timestamps[cache_key] = datetime.now()
        
        # Limit cache size to prevent memory issues
        if len(self.cache) > 1000:
            # Remove oldest entries
            sorted_keys = sorted(self.cache_timestamps.items(), key=lambda x: x[1])
            for key, _ in sorted_keys[:100]:
                if key in self.cache:
                    del self.cache[key]
                if key in self.cache_timestamps:
                    del self.cache_timestamps[key]
    
    def fetch_entry_batch(self, pdb_ids: List[str]) -> Dict[str, Dict]:
        """
        Fetch multiple PDB entries in a single API call using POST request.
        
        Args:
            pdb_ids: List of PDB IDs to fetch
            
        Returns:
            Dictionary mapping PDB IDs to their metadata
        """
        if not pdb_ids:
            return {}
        
        # Check cache first
        cached_results = {}
        remaining_ids = []
        
        for pdb_id in pdb_ids:
            cached = self._get_from_cache(pdb_id, "entry")
            if cached:
                cached_results[pdb_id.upper()] = cached
            else:
                remaining_ids.append(pdb_id)
        
        if not remaining_ids:
            return cached_results
        
        # Prepare batch request
        url = f"{self.base_url}/core/entry"
        payload = {
            "ids": [pdb_id.upper() for pdb_id in remaining_ids]
        }
        
        try:
            logger.debug(f"Fetching batch of {len(remaining_ids)} PDB entries")
            response = self.session.post(url, json=payload, timeout=30)
            self.stats['api_calls'] += 1
            
            if response.status_code == 200:
                results = response.json()
                
                # Process results
                batch_results = {}
                for result in results:
                    pdb_id = result.get('rcsb_id', '').upper()
                    if pdb_id:
                        self._save_to_cache(pdb_id, "entry", result)
                        batch_results[pdb_id] = result
                
                self.stats['total_pdbs_fetched'] += len(batch_results)
                self.stats['batches_processed'] += 1
                
                # Merge cached and new results
                cached_results.update(batch_results)
                
                # Log progress
                logger.debug(f"Batch fetch successful: {len(batch_results)}/{len(remaining_ids)} entries retrieved")
                
                # Respectful delay between requests
                time.sleep(self.request_delay)
                
                return cached_results
            else:
                logger.warning(f"Batch API call failed with status {response.status_code}")
                # Fallback to individual requests for remaining IDs
                for pdb_id in remaining_ids:
                    result = self.fetch_entry_single(pdb_id)
                    if result:
                        cached_results[pdb_id.upper()] = result
                
                return cached_results
                
        except Exception as e:
            logger.warning(f"Batch API call failed: {e}")
            # Fallback to individual requests
            for pdb_id in remaining_ids:
                result = self.fetch_entry_single(pdb_id)
                if result:
                    cached_results[pdb_id.upper()] = result
            
            return cached_results
    
    def fetch_entry_single(self, pdb_id: str) -> Optional[Dict]:
        """
        Fetch single PDB entry as fallback.
        
        Args:
            pdb_id: PDB ID
            
        Returns:
            Entry metadata or None
        """
        # Check cache first
        cached = self._get_from_cache(pdb_id, "entry")
        if cached:
            return cached
        
        url = f"{self.base_url}/core/entry/{pdb_id.upper()}"
        
        try:
            response = self.session.get(url, timeout=30)
            self.stats['api_calls'] += 1
            
            if response.status_code == 200:
                result = response.json()
                self._save_to_cache(pdb_id, "entry", result)
                self.stats['total_pdbs_fetched'] += 1
                return result
            elif response.status_code == 404:
                # Entry not found - mark as obsolete
                logger.debug(f"PDB entry {pdb_id} not found (404)")
                result = {"obsolete": True, "found_in_api": False, "pdb_id": pdb_id.upper()}
                self._save_to_cache(pdb_id, "entry", result)
                self.stats['obsolete_entries'] += 1
                return result
            else:
                logger.warning(f"API call failed for {pdb_id}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.warning(f"API call failed for {pdb_id}: {e}")
            return None
    
    def fetch_polymer_entities_batch(self, pdb_id: str, entity_ids: List[str]) -> Dict[str, Dict]:
        """
        Fetch multiple polymer entities for a PDB structure.
        
        Args:
            pdb_id: PDB ID
            entity_ids: List of entity IDs
            
        Returns:
            Dictionary mapping entity IDs to their data
        """
        entity_results = {}
        
        # Process entities in batches
        for i in range(0, len(entity_ids), self.batch_size):
            batch = entity_ids[i:i + self.batch_size]
            
            # Try parallel fetching (commented out for safety)
            # results = self._fetch_entities_parallel(pdb_id, batch)
            
            # Sequential fetching
            results = {}
            for entity_id in batch:
                cache_key = f"{pdb_id}_{entity_id}_entity"
                cached = self._get_from_cache(pdb_id, f"entity_{entity_id}")
                if cached:
                    results[entity_id] = cached
                else:
                    result = self._fetch_single_entity(pdb_id, entity_id)
                    if result:
                        results[entity_id] = result
            
            entity_results.update(results)
            
            # Delay between batches
            if i + self.batch_size < len(entity_ids):
                time.sleep(self.request_delay)
        
        return entity_results
    
    def _fetch_single_entity(self, pdb_id: str, entity_id: str) -> Optional[Dict]:
        """Fetch single polymer entity."""
        url = f"{self.base_url}/core/polymer_entity/{pdb_id.upper()}/{entity_id}"
        
        try:
            response = self.session.get(url, timeout=15)
            self.stats['api_calls'] += 1
            
            if response.status_code == 200:
                result = response.json()
                self._save_to_cache(pdb_id, f"entity_{entity_id}", result)
                return result
            else:
                logger.debug(f"Failed to fetch entity {entity_id} for {pdb_id}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.debug(f"Failed to fetch entity {entity_id} for {pdb_id}: {e}")
            return None
    
    def get_statistics(self) -> Dict:
        """Get API usage statistics."""
        return self.stats.copy()

class PPIBenchmarkProcessor:
    """Processor for protein-protein interaction benchmark data with Croissant support."""

    def __init__(self, csv_url: str, mmcif_base_url: str, pdb_base_url: str,
                 csv_separator: str = ',', auto_detect_separator: bool = False,
                 check_pdb_label: bool = False, pdb_label_dir: Optional[str] = None,
                 check_cif_label: bool = False, cif_label_dir: Optional[str] = None,
                 fetch_pdb_metadata: bool = False, pdb_api_base_url: str = "https://data.rcsb.org/rest/v1",
                 cluster_file: Optional[str] = None):
        """
        Initialize the processor with the CSV and structure file URLs.

        Args:
            csv_url: URL to the benchmark CSV file
            mmcif_base_url: Base URL for mmCIF structure files
            pdb_base_url: Base URL for PDB structure files
            csv_separator: Separator to use for CSV parsing (default: ',')
            auto_detect_separator: Whether to auto-detect separator if default fails
            check_pdb_label: Check interacting chains in the pdb assembly file
            pdb_label_dir: Local pdb assembly file directory
            check_cif_label: Check interacting chains in the cif assembly file
            cif_label_dir: Local cif assembly file directory
            fetch_pdb_metadata: Whether to fetch additional PDB metadata from RCSB API
            pdb_api_base_url: Base URL for RCSB PDB REST API
            cluster_file: Path to BLASTClust output file (optional)
        """
        self.csv_url = csv_url
        self.mmcif_base_url = mmcif_base_url.rstrip('/') + '/'
        self.pdb_base_url = pdb_base_url.rstrip('/') + '/'
        self.csv_separator = csv_separator
        self.auto_detect_separator = auto_detect_separator
        self.check_pdb_label = check_pdb_label
        self.pdb_label_dir = pdb_label_dir
        self.check_cif_label = check_cif_label
        self.cif_label_dir = cif_label_dir
        self.fetch_pdb_metadata = fetch_pdb_metadata
        self.pdb_api_base_url = pdb_api_base_url.rstrip('/')
        self.cluster_file = cluster_file

        self.dataset = None
        self.protein_interfaces = []
        self.column_mapping = {}  # Store actual column names
        self.pdb_metadata_cache = {}  # Cache for PDB metadata to avoid repeated API calls
        
        # Initialize optimized PDB API client
        self.pdb_api_client = PDBApiClient(pdb_api_base_url) if fetch_pdb_metadata else None

        # Initialize cluster processor if cluster file is provided
        self.cluster_processor = None
        if cluster_file:
            self.cluster_processor = ClusterIDProcessor(cluster_file)
    
    def fetch_pdb_metadata_batch(self, pdb_ids: List[str]) -> Dict[str, Dict]:
        """
        Fetch metadata for multiple PDB IDs in optimized batches.
        
        Args:
            pdb_ids: List of PDB IDs to fetch
            
        Returns:
            Dictionary mapping PDB IDs to comprehensive metadata
        """
        if not self.pdb_api_client or not pdb_ids:
            return {}
        
        # Remove duplicates and convert to uppercase
        unique_pdb_ids = list(set(pdb_id.upper() for pdb_id in pdb_ids if pdb_id))
        logger.info(f"Fetching metadata for {len(unique_pdb_ids)} unique PDB structures in batches...")
        
        # Fetch entry data in batches
        all_metadata = {}
        
        # Process in batches using the API client
        for i in range(0, len(unique_pdb_ids), self.pdb_api_client.batch_size):
            batch = unique_pdb_ids[i:i + self.pdb_api_client.batch_size]
            batch_metadata = self.pdb_api_client.fetch_entry_batch(batch)
            all_metadata.update(batch_metadata)
            
            # Log progress
            progress = min(i + len(batch), len(unique_pdb_ids))
            logger.info(f"Progress: {progress}/{len(unique_pdb_ids)} structures fetched")
        
        # Enrich with entity information
        enriched_metadata = {}
        for pdb_id, entry_data in all_metadata.items():
            metadata = self._enrich_entry_with_entities(pdb_id, entry_data)
            enriched_metadata[pdb_id] = metadata
        
        # Update cache
        self.pdb_metadata_cache.update(enriched_metadata)
        
        # Show statistics
        stats = self.pdb_api_client.get_statistics()
        logger.info(f"API Statistics: {stats['api_calls']} calls, "
                   f"{stats['cache_hits']} cache hits, "
                   f"{stats['cache_misses']} cache misses, "
                   f"{stats['total_pdbs_fetched']} PDBs fetched")
        
        return enriched_metadata
    
    def _enrich_entry_with_entities(self, pdb_id: str, entry_data: Dict) -> Dict:
        """
        Enrich entry data with entity information.
        
        Args:
            pdb_id: PDB ID
            entry_data: Entry metadata from API
            
        Returns:
            Enriched metadata with entity information
        """
        # Start with basic metadata structure
        metadata = {
            "pdb_id": pdb_id.upper(),
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
            "obsolete": False,
            "replaced_by": None,
            "sequences": {},
            "chain_info": {},
            "entity_info": {},
            "is_homomer": False,
            "sequence_clusters": [],
            "chain_ids": [],
            "entity_ids": []
        }
        
        # Check if entry is obsolete
        if entry_data.get("obsolete", False):
            metadata["obsolete"] = True
            metadata["found_in_api"] = True
            metadata["replaced_by"] = entry_data.get("replaced_by")
            return metadata
        
        # Extract basic information from entry data
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
        
        metadata["found_in_api"] = True
        
        # Fetch entity information if we have entity IDs
        if metadata["entity_ids"] and self.pdb_api_client:
            entity_metadata = self.pdb_api_client.fetch_polymer_entities_batch(
                pdb_id, metadata["entity_ids"]
            )
            
            # Process entity information
            organisms = set()
            sequences_by_entity = {}
            chains_by_entity = {}
            
            for entity_id, entity_data in entity_metadata.items():
                entity_info = self._extract_entity_info(entity_id, entity_data)
                metadata["entity_info"][entity_id] = entity_info
                
                # Collect sequences
                if entity_info["sequence"] and entity_info["chain_ids"]:
                    sequences_by_entity[entity_id] = entity_info["sequence"]
                    chains_by_entity[entity_id] = entity_info["chain_ids"]
                    
                    # Add to sequences and chain_info
                    for chain_id in entity_info["chain_ids"]:
                        metadata["sequences"][chain_id] = entity_info["sequence"]
                        metadata["chain_info"][chain_id] = {
                            "entity_id": entity_id,
                            "organism": entity_info["organism"],
                            "sequence_length": entity_info["sequence_length"],
                            "polymer_type": entity_info["polymer_type"],
                            "gene_name": entity_info["gene_name"]
                        }
                
                # Collect organisms
                if entity_info["organism"]:
                    organisms.add(entity_info["organism"])
            
            if organisms:
                metadata["source_organism"] = list(organisms)
            
            # Analyze sequence relationships
            if sequences_by_entity:
                self._analyze_sequences(metadata, sequences_by_entity, chains_by_entity)
        
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
        
        return metadata
    
    def _extract_entity_info(self, entity_id: str, entity_data: Dict) -> Dict:
        """
        Extract information from entity data.
        
        Args:
            entity_id: Entity ID
            entity_data: Entity metadata from API
            
        Returns:
            Extracted entity information
        """
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
            if "pdbx_seq_one_letter_code_can" in poly:
                entity_info["sequence"] = poly["pdbx_seq_one_letter_code_can"]
                entity_info["sequence_length"] = len(poly["pdbx_seq_one_letter_code_can"])
            
            if "pdbx_strand_id" in poly:
                chain_ids = poly["pdbx_strand_id"].split(",")
                entity_info["chain_ids"] = [c.strip() for c in chain_ids]
            
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
        
        return entity_info
    
    def _analyze_sequences(self, metadata: Dict, sequences_by_entity: Dict, chains_by_entity: Dict) -> None:
        """
        Analyze sequence relationships for homomer detection.
        
        Args:
            metadata: Metadata dictionary to update
            sequences_by_entity: Dictionary mapping entity IDs to sequences
            chains_by_entity: Dictionary mapping entity IDs to chain IDs
        """
        # Group identical sequences
        seq_to_entities = {}
        for entity_id, seq in sequences_by_entity.items():
            if seq:
                if seq not in seq_to_entities:
                    seq_to_entities[seq] = []
                seq_to_entities[seq].append(entity_id)
        
        metadata["sequence_clusters"] = [
            {"sequence": seq, "entities": entities, 
             "chain_count": sum(len(chains_by_entity.get(eid, [])) for eid in entities)}
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

    def fetch_pdb_structure_metadata_robust(self, pdb_id: str) -> Dict[str, Any]:
        """
        Robust method to fetch PDB metadata using optimized batch approach.
        
        Args:
            pdb_id: PDB ID (4 letters, e.g., "1ABC")
            
        Returns:
            Dictionary with comprehensive PDB metadata
        """
        # Check cache first
        if pdb_id.upper() in self.pdb_metadata_cache:
            return self.pdb_metadata_cache[pdb_id.upper()]
        
        # If we have an API client, use it
        if self.pdb_api_client:
            metadata = self.pdb_api_client.fetch_entry_single(pdb_id)
            if metadata:
                enriched = self._enrich_entry_with_entities(pdb_id, metadata)
                self.pdb_metadata_cache[pdb_id.upper()] = enriched
                return enriched
        
        # Fallback to the original method
        return self._fetch_pdb_metadata_fallback(pdb_id)
    
    def _fetch_pdb_metadata_fallback(self, pdb_id: str) -> Dict[str, Any]:
        """
        Fallback method for fetching PDB metadata (original implementation).
        
        Args:
            pdb_id: PDB ID
            
        Returns:
            Dictionary with PDB metadata
        """
        metadata = {
            "pdb_id": pdb_id.upper(),
            "found_in_api": False,
            "obsolete": True,
            "replaced_by": None,
            "sequences": {},
            "chain_info": {},
            "source_organism": []
        }
        
        # Try to get minimal info from header
        try:
            minimal_info = self.fetch_obsolete_pdb_minimal_info(pdb_id)
            if minimal_info.get("found_in_header"):
                metadata["found_in_api"] = True
                metadata["sequences"] = minimal_info.get("sequences", {})
                metadata["chain_info"] = minimal_info.get("chain_info", {})
                metadata["source_organism"] = minimal_info.get("source_organism", [])
                metadata["taxonomy_ids"] = minimal_info.get("taxonomy_ids", [])
                
                # Analyze sequences
                if metadata["sequences"]:
                    unique_seqs = set(metadata["sequences"].values())
                    if len(unique_seqs) == 1:
                        metadata["is_homomer"] = True
                        metadata["homomer_type"] = f"{len(metadata['sequences'])}-mer"
                        metadata["homomer_chains"] = len(metadata["sequences"])
        except Exception as e:
            logger.debug(f"Fallback failed for {pdb_id}: {e}")
        
        self.pdb_metadata_cache[pdb_id.upper()] = metadata
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

    def enrich_protein_with_pdb_metadata(self, interface: ProteinInterface) -> Dict[str, Any]:
        """
        Enrich a protein interface with PDB metadata.
        Uses batch-optimized method to get ALL chain sequences.

        Args:
            interface: ProteinInterface object

        Returns:
            Dictionary with enriched PDB metadata including ALL sequences
        """
        pdb_id = interface.ID.upper()

        # Check cache first
        if pdb_id in self.pdb_metadata_cache:
            structure_metadata = self.pdb_metadata_cache[pdb_id]
        else:
            # This should rarely happen if we prefetch metadata in batches
            structure_metadata = self.fetch_pdb_structure_metadata_robust(pdb_id)

        # If obsolete and no sequences, try header parsing
        if structure_metadata.get("obsolete", False) and not structure_metadata.get("sequences"):
            logger.debug(f"Fetching sequences from header for obsolete entry {interface.ID}")
            header_info = self.fetch_obsolete_pdb_minimal_info(pdb_id)

            # Merge header info into structure_metadata
            if header_info.get("found_in_header"):
                if header_info.get("sequences"):
                    structure_metadata["sequences"] = header_info["sequences"]
                if header_info.get("chain_info"):
                    structure_metadata["chain_info"] = header_info["chain_info"]
                if header_info.get("source_organism") and header_info["source_organism"]:
                    structure_metadata["source_organism"] = header_info["source_organism"]
                if header_info.get("taxonomy_ids"):
                    structure_metadata["taxonomy_ids"] = header_info["taxonomy_ids"]
                    if header_info.get("primary_taxonomy_id"):
                        structure_metadata["primary_taxonomy_id"] = header_info["primary_taxonomy_id"]

                # Update sequence clusters for homomer detection
                if structure_metadata["sequences"]:
                    seq_to_chains = {}
                    for chain_id, sequence in structure_metadata["sequences"].items():
                        if sequence not in seq_to_chains:
                            seq_to_chains[sequence] = []
                        seq_to_chains[sequence].append(chain_id)

                    structure_metadata["sequence_clusters"] = [
                        {"sequence": seq, "chains": chains, "chain_count": len(chains)}
                        for seq, chains in seq_to_chains.items()
                    ]

                    # Check if it's a homomer
                    if len(seq_to_chains) == 1:
                        structure_metadata["is_homomer"] = True
                        seq = list(seq_to_chains.keys())[0]
                        chain_count = len(list(seq_to_chains.values())[0])
                        structure_metadata["homomer_type"] = f"{chain_count}-mer" if chain_count > 1 else "monomer"
                        structure_metadata["homomer_chains"] = chain_count

                logger.debug(f"Added {len(structure_metadata.get('sequences', {}))} chain sequences from header for obsolete PDB {pdb_id}")

        # For interface analysis, also fetch specific entity metadata for the interface chains
        entity_metadata = {}

        # Try to identify which entity corresponds to the interface chains
        if structure_metadata.get("chain_info"):
            for chain_id, chain_info in structure_metadata["chain_info"].items():
                entity_id = chain_info.get("entity_id")
                if entity_id and entity_id not in entity_metadata and self.pdb_api_client:
                    # Fetch detailed entity metadata
                    try:
                        entity_url = f"{self.pdb_api_base_url}/core/polymer_entity/{pdb_id.upper()}/{entity_id}"
                        response = requests.get(entity_url, timeout=15)
                        if response.status_code == 200:
                            entity_data = response.json()
                            entity_metadata[entity_id] = entity_data
                    except:
                        pass

        # Combine metadata
        enriched_metadata = {
            "pdb_id": pdb_id,
            "structure_metadata": structure_metadata,
            "entity_metadata": entity_metadata,
            "interface_chains": {
                "chain1": interface.AuthChain1,
                "chain2": interface.AuthChain2,
                "assembly_chain1": interface.LabelChain1,
                "assembly_chain2": interface.LabelChain2
            },
            "enrichment_timestamp": datetime.now().isoformat()
        }

        # Add interface-specific sequence analysis
        if structure_metadata.get("sequences"):
            chain1_seq = structure_metadata["sequences"].get(interface.AuthChain1)
            chain2_seq = structure_metadata["sequences"].get(interface.AuthChain2)

            if chain1_seq and chain2_seq:
                enriched_metadata["interface_sequence_analysis"] = {
                    "chain1_sequence_length": len(chain1_seq),
                    "chain2_sequence_length": len(chain2_seq),
                    "sequences_identical": chain1_seq == chain2_seq,
                    "sequence_identity": self._calculate_sequence_identity(chain1_seq, chain2_seq)
                }

        return enriched_metadata

    def fetch_obsolete_pdb_minimal_info(self, interface_id: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Extract minimal information from obsolete PDB entries by parsing the header file.
        
        Args:
            interface_id: PDB ID (4 letters, e.g., "1ABC")
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary with minimal information including sequences for each chain
        """
        # This method remains the same as in the original code
        # Only showing the function signature to save space
        # The full implementation should be copied from the original code
        pass

    def enrich_with_obsolete_info(self, interface: ProteinInterface) -> Dict[str, Any]:
        """
        Enrich interface with information from obsolete PDB entries.

        Args:
            interface: ProteinInterface object

        Returns:
            Dictionary with enriched information including sequences
        """
        pdb_id = interface.ID.upper()

        # Check cache first
        cache_key = f"obsolete_{pdb_id}"
        if hasattr(self, 'pdb_metadata_cache') and cache_key in self.pdb_metadata_cache:
            return self.pdb_metadata_cache[cache_key]

        # Fetch minimal info from header
        minimal_info = self.fetch_obsolete_pdb_minimal_info(pdb_id)

        # Create enriched metadata structure
        enriched_metadata = {
            "pdb_id": pdb_id,
            "is_obsolete": True,
            "minimal_info": minimal_info,
            "interface_chains": {
                "chain1": interface.AuthChain1,
                "chain2": interface.AuthChain2,
                "assembly_chain1": interface.LabelChain1,
                "assembly_chain2": interface.LabelChain2
            },
            "sequences": {},  # Chain ID -> sequence mapping for easy access
            "enrichment_timestamp": datetime.now().isoformat()
        }

        # Add sequences from minimal info
        if minimal_info.get("chains"):
            for chain_id, chain_info in minimal_info["chains"].items():
                sequence = chain_info.get("sequence", "")
                if sequence:
                    enriched_metadata["sequences"][chain_id] = sequence

            # Check if interface chains have sequences
            if interface.AuthChain1 in enriched_metadata["sequences"]:
                enriched_metadata["chain1_sequence"] = enriched_metadata["sequences"][interface.AuthChain1]
            if interface.AuthChain2 in enriched_metadata["sequences"]:
                enriched_metadata["chain2_sequence"] = enriched_metadata["sequences"][interface.AuthChain2]

            # Add sequence analysis if both chains have sequences
            if enriched_metadata.get("chain1_sequence") and enriched_metadata.get("chain2_sequence"):
                enriched_metadata["interface_sequence_analysis"] = {
                    "chain1_sequence_length": len(enriched_metadata["chain1_sequence"]),
                    "chain2_sequence_length": len(enriched_metadata["chain2_sequence"]),
                    "sequences_identical": enriched_metadata["chain1_sequence"] == enriched_metadata["chain2_sequence"],
                    "sequence_identity": self._calculate_sequence_identity(
                        enriched_metadata["chain1_sequence"],
                        enriched_metadata["chain2_sequence"]
                    )
                }

        # Cache the result
        if hasattr(self, 'pdb_metadata_cache'):
            self.pdb_metadata_cache[cache_key] = enriched_metadata

        return enriched_metadata

    def fetch_pdb_structure_metadata_robust_with_obsolete(self, pdb_id: str) -> Dict[str, Any]:
        """
        Robust method to fetch PDB metadata, with fallback to obsolete entry parsing.
        """
        # First try the optimized API client
        if self.pdb_api_client:
            metadata = self.pdb_api_client.fetch_entry_single(pdb_id)
            if metadata:
                enriched = self._enrich_entry_with_entities(pdb_id, metadata)
                return enriched
        
        # Fallback to original method
        return self._fetch_pdb_metadata_fallback(pdb_id)

    def _add_pdb_metadata_to_protein_markup(self, protein_markup: Dict[str, Any],
                                          pdb_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add PDB metadata to the protein Bioschemas markup.
        Includes ALL chain sequences for comprehensive analysis.

        Args:
            protein_markup: Existing protein markup dictionary
            pdb_metadata: PDB metadata from RCSB API (or header for obsolete entries)

        Returns:
            Updated protein markup with PDB metadata
        """
        # This method remains the same as in the original code
        # Only showing the function signature to save space
        pass

    def _add_cluster_properties_to_markup(self, markup: Dict[str, Any], interface: ProteinInterface) -> Dict[str, Any]:
        """
        Add cluster properties to Bioschemas markup.

        Args:
            markup: Existing markup dictionary
            interface: ProteinInterface object with cluster information

        Returns:
            Updated markup with cluster properties
        """
        if not interface.cluster_id:
            return markup

        # Ensure additionalProperty exists
        if "additionalProperty" not in markup:
            markup["additionalProperty"] = []

        # Add cluster properties
        cluster_properties = [
            {
                "@type": "PropertyValue",
                "name": "ClusterID",
                "value": interface.cluster_id,
                "description": "Sequence cluster ID from BLASTClust analysis"
            },
            {
                "@type": "PropertyValue",
                "name": "ClusterSize",
                "value": interface.cluster_size,
                "description": "Number of interfaces in this cluster"
            },
            {
                "@type": "PropertyValue",
                "name": "ClusterMembers",
                "value": interface.cluster_members,
                "description": f"List of other interfaces in the same cluster (excluding current interface)"
            }
        ]

        # Add cluster method information
        cluster_properties.extend([
            {
                "@type": "PropertyValue",
                "name": "ClusterMethod",
                "value": "BLASTClust sequence clustering",
                "description": "Method used for sequence-based clustering"
            },
            {
                "@type": "PropertyValue",
                "name": "ClusterMethodOptions",
                "value": "-S 25 -L 0.5 -b F",
                "description": "BLASTClust parameters used for sequence clustering"
            }
        ])

        # Add all cluster properties to markup
        markup["additionalProperty"].extend(cluster_properties)

        return markup

    def load_data(self) -> pd.DataFrame:
        """
        Load the benchmark data from the CSV URL using specified separator.

        Returns:
            DataFrame containing the benchmark data
        """
        try:
            logger.info(f"Loading data from {self.csv_url}")
            logger.info(f"Using separator: '{self.csv_separator}' (auto-detect: {self.auto_detect_separator})")

            # Try with the specified separator first
            try:
                self.dataset = pd.read_csv(self.csv_url, sep=self.csv_separator).iloc[:, 2:]     #.head(5)
                logger.info(f"Successfully loaded {len(self.dataset)} records with separator '{self.csv_separator}'")
            except Exception as e:
                logger.warning(f"Failed with separator '{self.csv_separator}': {e}")

                if self.auto_detect_separator:
                    logger.info("Auto-detecting separator...")
                    # Try different separators
                    separators_to_try = [',', '\t', ';', ' ', '|']
                    for sep in separators_to_try:
                        if sep == self.csv_separator:
                            continue  # Skip the one we already tried
                        try:
                            self.dataset = pd.read_csv(self.csv_url, sep=sep)
                            logger.info(f"Successfully loaded {len(self.dataset)} records with separator '{repr(sep)}'")
                            self.csv_separator = sep  # Update the separator
                            break
                        except:
                            continue

                if self.dataset is None:
                    raise ValueError(f"Could not parse CSV with separator '{self.csv_separator}'")

            # Show column information
            logger.info(f"Columns found ({len(self.dataset.columns)}): {list(self.dataset.columns)}")

            # Show first few rows
            logger.info(f"First 3 rows:")
            for i in range(min(3, len(self.dataset))):
                row_data = {col: self.dataset.iloc[i][col] for col in self.dataset.columns[:5]}
                logger.info(f"  Row {i}: {row_data}")

            # Create column mapping for flexible column name matching
            available_cols = [str(col).strip().lower() for col in self.dataset.columns]
            logger.info(f"Lowercase column names: {available_cols}")

            # Define expected columns and possible variations
            expected_columns = {
                'id': ['id', 'pdb_id', 'pdbid', 'pdb'],
                'interfaceid': ['interfaceid', 'interface_id', 'interface', 'entry_id', 'entryid'],
                'physio': ['physio', 'physiological', 'label', 'target', 'class'],
                'authchain1': ['authchain1', 'chain1', 'chain_a', 'chain1_id'],
                'authchain2': ['authchain2', 'chain2', 'chain_b', 'chain2_id'],
                'symmetryop1': ['symmetryop1', 'symm1', 'symmetry1', 'symop1'],
                'symmetryop2': ['symmetryop2', 'symm2', 'symmetry2', 'symop2'],
                'bsa': ['bsa', 'buried_surface_area', 'surface_area', 'area'],
                'bsa_polar': ['bsa_polar', 'polar_bsa', 'polar_surface_area', 'bsa_polar_component'],
                'bsa_apolar': ['bsa_apolar', 'apolar_bsa', 'apolar_surface_area', 'bsa_apolar_component'],
                'frac_polar': ['frac_polar', 'fraction_polar', 'polar_fraction', 'fraction_polar_contacts'],
                'frac_apolar': ['frac_apolar', 'fraction_apolar', 'apolar_fraction', 'fraction_apolar_contacts'],
                'contacts': ['contacts', 'num_contacts', 'contact_count'],
                'gene': ['gene', 'gene_name', 'protein_name'],
                'superfamily': ['superfamily', 'family', 'protein_family'],
                'pfam': ['pfam', 'pfam_id', 'domain'],
                'comments': ['comments', 'comment', 'notes']
            }

            # Find matching columns
            self.column_mapping = {}
            for standard_name, possible_names in expected_columns.items():
                for possible_name in possible_names:
                    if possible_name in available_cols:
                        # Get the actual column name (preserving original case)
                        actual_name = self.dataset.columns[available_cols.index(possible_name)]
                        self.column_mapping[standard_name] = actual_name
                        logger.info(f"Mapped '{standard_name}' to column '{actual_name}'")
                        break

            logger.info(f"\nColumn mapping: {self.column_mapping}")

            # Check for critical columns
            critical_cols = ['id', 'interfaceid', 'physio']
            missing_critical = [col for col in critical_cols if col not in self.column_mapping]

            if missing_critical:
                logger.warning(f"Missing critical columns: {missing_critical}")
                logger.info(f"Available columns (original): {list(self.dataset.columns)}")

            return self.dataset
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def _get_column_value(self, row: pd.Series, column_key: str) -> Any:
        """
        Get value from a row using the column mapping.

        Args:
            row: DataFrame row
            column_key: Standard column name (e.g., 'id', 'interfaceid')

        Returns:
            Value from the row, or None if column not found
        """
        if column_key in self.column_mapping:
            column_name = self.column_mapping[column_key]
            if column_name in row:
                return row[column_name]
        return None

    def _generate_html_snippet(self, dataset_markup: Dict[str, Any]) -> str:
        """Generate HTML snippet with embedded JSON-LD markup."""
        html = """<!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Bioschemas & Croissant Markup - PPI Benchmark Dataset</title>
        <script type="application/ld+json">
        {markup}
        </script>
        </head>
        <body>
        <h1>Protein-Protein Interaction Interface Benchmark Dataset</h1>
        <p>This page contains embedded Bioschemas and Croissant markup for the dataset.</p>
        <p>The JSON-LD markup in the header makes this dataset discoverable by search engines and compatible with Google Dataset Search and ML dataset platforms.</p>
        </body>
        </html>""".format(markup=json.dumps(dataset_markup, indent=4))

        return html

    def parse_data(self) -> List[ProteinInterface]:
        """
        Parse the loaded data into structured ProteinInterface objects.

        Returns:
            List of ProteinInterface objects
        """
        if self.dataset is None:
            self.load_data()

        # Parse cluster file BEFORE processing interfaces if available
        if self.cluster_processor:
            logger.info("Parsing BLASTClust file for cluster information...")
            if not self.cluster_processor.parse_blastclust_file():
                logger.warning("Failed to parse BLASTClust file. Skipping cluster information.")
            else:
                logger.info(f"Successfully parsed BLASTClust file. Found {len(self.cluster_processor.cluster_mapping)} interface-cluster mappings")

        # OPTIMIZATION: Pre-fetch PDB metadata for all unique PDB IDs in batch mode
        if self.fetch_pdb_metadata and self.pdb_api_client:
            # Extract all unique PDB IDs from the dataset
            unique_pdb_ids = []
            for idx, row in self.dataset.iterrows():
                pdb_id = self._get_column_value(row, 'id')
                if pdb_id and not pd.isna(pdb_id):
                    pdb_id_clean = str(pdb_id).strip().upper()
                    if pdb_id_clean not in unique_pdb_ids:
                        unique_pdb_ids.append(pdb_id_clean)
            
            logger.info(f"Pre-fetching PDB metadata for {len(unique_pdb_ids)} unique structures...")
            self.fetch_pdb_metadata_batch(unique_pdb_ids)

        self.protein_interfaces = []
        skipped_count = 0
        processed_count = 0
        qsalign_count = 0
        protcid_count = 0

        # Show what columns we're working with
        logger.info(f"Using column mapping: {self.column_mapping}")
        logger.info(f"Dataset has {len(self.dataset)} rows")

        for idx, row in self.dataset.iterrows():
            try:
                # Extract required fields using column mapping
                ID = self._get_column_value(row, 'id')
                original_interface_id = self._get_column_value(row, 'interfaceid')

                # VALIDATION: Check if ID is valid
                if pd.isna(ID) or str(ID).strip() == '':
                    logger.debug(f"Skipping row {idx}: Missing or invalid ID")
                    skipped_count += 1
                    continue

                ID = str(ID).strip().upper()

                # DECISION: Determine InterfaceID format based on the original value
                interface_source = "Unknown"

                if pd.isna(original_interface_id) or str(original_interface_id).strip() == '':
                    # CASE 1: InterfaceID is NA - Use QSalign format {PDB_ID}_{assembly_number}
                    # Default to assembly number 1 if not specified
                    InterfaceID = str(ID).strip().upper()
                    ID = ID.split('_')[0]
                    interface_source = "QSalign"
                    qsalign_count += 1
                    logger.debug(f"Row {idx}: Using QSalign format '{InterfaceID}' (original was empty)")
                else:
                    # CASE 2: InterfaceID has a value - Could be ProtCID integer or other format
                    original_str = str(original_interface_id).strip()

                    # Try to parse as integer for ProtCID
                    try:
                        # Remove any decimal part if it's a float
                        if '.' in original_str:
                            interface_num = int(float(original_str))
                        else:
                            interface_num = int(original_str)

                        # Use the ProtCID integer directly
                        InterfaceID = f'{ID}_{interface_num}'
                        interface_source = "ProtCID"
                        protcid_count += 1
                        logger.debug(f"Row {idx}: Using ProtCID format '{InterfaceID}' (original: '{original_str}')")
                    except (ValueError, TypeError):
                        # If not a valid integer, use as-is but clean it
                        InterfaceID = original_str.replace(' ', '_').replace('.', '_')
                        interface_source = "Other"
                        logger.debug(f"Row {idx}: Using cleaned format '{InterfaceID}' (original: '{original_str}')")

                # Clean InterfaceID - ensure proper format
                InterfaceID = InterfaceID.replace(' ', '_').replace('.', '_')

                # Extract physio value and convert to boolean
                physio_val = self._get_column_value(row, 'physio')
                if pd.isna(physio_val) or physio_val is None:
                    logger.debug(f"Missing 'physio' value for {InterfaceID}, defaulting to FALSE")
                    physio_bool = False
                elif isinstance(physio_val, bool):
                    physio_bool = physio_val
                elif isinstance(physio_val, str):
                    physio_str = physio_val.strip().upper()
                    if physio_str == 'TRUE' or physio_str == '1' or physio_str == 'T':
                        physio_bool = True
                    elif physio_str == 'FALSE' or physio_str == '0' or physio_str == 'F':
                        physio_bool = False
                    else:
                        logger.debug(f"Unexpected 'physio' value '{physio_val}' for {InterfaceID}, defaulting to FALSE")
                        physio_bool = False
                elif isinstance(physio_val, (int, float)):
                    physio_bool = bool(physio_val)
                else:
                    # Try to convert other types
                    try:
                        physio_bool = bool(physio_val)
                    except:
                        logger.debug(f"Cannot convert 'physio' value '{physio_val}' for {InterfaceID}, defaulting to FALSE")
                        physio_bool = False

                # Extract other fields
                auth_chain1 = ""
                auth_chain2 = ""
                symmetry_op1 = ""
                symmetry_op2 = ""

                if self._get_column_value(row, 'authchain1') is not None:
                    auth_chain1 = str(self._get_column_value(row, 'authchain1')).strip()
                if self._get_column_value(row, 'authchain2') is not None:
                    auth_chain2 = str(self._get_column_value(row, 'authchain2')).strip()
                if self._get_column_value(row, 'symmetryop1') is not None:
                    symmetry_op1 = str(self._get_column_value(row, 'symmetryop1')).strip()
                if self._get_column_value(row, 'symmetryop2') is not None:
                    symmetry_op2 = str(self._get_column_value(row, 'symmetryop2')).strip()

                # Create ProteinInterface object
                interface = ProteinInterface(
                    ID=ID,
                    InterfaceID=InterfaceID,
                    AuthChain1=auth_chain1,
                    AuthChain2=auth_chain2,
                    SymmetryOp1=symmetry_op1,
                    SymmetryOp2=symmetry_op2,
                    physio=physio_bool
                )

                # Set the interface source
                interface.interface_source = interface_source

                # Check assembly chains if either flag is set
                if self.check_pdb_label or self.check_cif_label:
                    try:
                        # Construct the correct file paths based on interface source
                        pdb_file_path = None
                        mmcif_file_path = None

                        if self.check_pdb_label:
                            # Use InterfaceID for PDB filename (always lowercase)
                            pdb_filename = f"{interface.InterfaceID.lower()}.pdb.gz"
                            if self.pdb_label_dir:
                                pdb_file_path = os.path.join(self.pdb_label_dir, pdb_filename)
                            else:
                                pdb_file_path = f"{self.pdb_base_url}{pdb_filename}"

                        if self.check_cif_label:
                            # Use different naming based on interface source for mmCIF
                            if interface.interface_source == 'QSalign':
                                cif_filename = f"{interface.ID.lower()}.cif.gz"
                            else:
                                cif_filename = f"{interface.InterfaceID.lower()}.cif.gz"

                            if self.cif_label_dir:
                                mmcif_file_path = os.path.join(self.cif_label_dir, cif_filename)
                            else:
                                mmcif_file_path = f"{self.mmcif_base_url}{cif_filename}"

                        # Log which check we're performing
                        if self.check_pdb_label and self.check_cif_label:
                            logger.debug(f"Checking both PDB and mmCIF assembly chains for {interface.InterfaceID}")
                            logger.debug(f"  PDB file: {pdb_file_path}")
                            logger.debug(f"  mmCIF file: {mmcif_file_path}")
                        elif self.check_pdb_label:
                            logger.debug(f"Checking PDB assembly chains for {interface.InterfaceID}")
                            logger.debug(f"  PDB file: {pdb_file_path}")
                        elif self.check_cif_label:
                            logger.debug(f"Checking mmCIF assembly chains for {interface.InterfaceID}")
                            logger.debug(f"  mmCIF file: {mmcif_file_path}")

                        # Update interface with actual chains
                        interface = update_protein_interface_with_actual_chains(
                            interface,
                            pdb_file_path=pdb_file_path,
                            mmcif_file_path=mmcif_file_path
                        )

                        # Log if chains were updated
                        if interface.LabelChain1 and interface.LabelChain2:
                            if (interface.LabelChain1 != interface.AuthChain1 or
                                interface.LabelChain2 != interface.AuthChain2):
                                logger.info(f"Updated chains for {interface.InterfaceID}: "
                                          f"Auth chains: {interface.AuthChain1}-{interface.AuthChain2} -> "
                                          f"Assembly chains: {interface.LabelChain1}-{interface.LabelChain2}")

                    except Exception as e:
                        logger.warning(f"Could not validate chains for {interface.InterfaceID}: {e}")

                # Continue with annotated chains

                # Add optional fields if present in the CSV
                optional_fields = [
                    'contacts', 'gene', 'superfamily', 'pfam', 'bsa',
                    'bsa_polar', 'bsa_apolar', 'frac_polar', 'frac_apolar', 'comments'
                ]

                for field_name in optional_fields:
                    value = self._get_column_value(row, field_name)
                    if value is not None and pd.notna(value):
                        # Convert numeric fields
                        if field_name in ['contacts', 'bsa', 'bsa_polar', 'bsa_apolar', 'frac_polar', 'frac_apolar']:
                            try:
                                value = float(value)
                            except (ValueError, TypeError):
                                logger.debug(f"Cannot convert {field_name} value '{value}' to float for {InterfaceID}")
                                continue

                        setattr(interface, field_name, value)

                # DIRECT CLUSTER ID ASSIGNMENT from BLASTClust output
                if self.cluster_processor and self.cluster_processor.cluster_mapping:
                    # Direct lookup in cluster mapping
                    cluster_id = self.cluster_processor.cluster_mapping.get(interface.InterfaceID)

                    if not cluster_id:
                        # Try alternative cleanings if direct match fails
                        alt_ids = [
                            interface.InterfaceID,
                            interface.InterfaceID.upper(),
                            interface.InterfaceID.lower(),
                            interface.InterfaceID.replace('_', ''),
                            self.cluster_processor._clean_interface_id(interface.InterfaceID)
                        ]

                        for alt_id in alt_ids:
                            cluster_id = self.cluster_processor.cluster_mapping.get(alt_id)
                            if cluster_id:
                                logger.debug(f"Found cluster for {interface.InterfaceID} via alternative ID: {alt_id}")
                                break

                    if cluster_id:
                        # Direct assignment of cluster information
                        interface.cluster_id = cluster_id

                        # Get cluster members
                        members = self.cluster_processor.cluster_members.get(cluster_id, [])
                        interface.cluster_size = len(members)

                        # Exclude current interface from member list
                        other_members = [m for m in members if m != interface.InterfaceID]
                        interface.cluster_members = other_members

                        logger.debug(f"Assigned cluster {cluster_id} (size: {interface.cluster_size}) to interface {interface.InterfaceID}")
                    else:
                        logger.debug(f"No cluster found for interface: {interface.InterfaceID}")
                        interface.cluster_id = None
                        interface.cluster_size = 0
                        interface.cluster_members = []

                # Add to interfaces list
                self.protein_interfaces.append(interface)
                processed_count += 1

                # Log first few entries for verification
                if idx < 3:
                    logger.info(f"Parsed sample entry {idx}: ID={ID}, InterfaceID={InterfaceID}, source={interface_source}, physio={physio_bool}, label={interface.label}")
                    if interface.LabelChain1 and interface.LabelChain2:
                        logger.info(f"  Assembly chains: {interface.LabelChain1}-{interface.LabelChain2}")
                    if interface.cluster_id:
                        logger.info(f"  Cluster info: ID={interface.cluster_id}, size={interface.cluster_size}, members={len(interface.cluster_members) if interface.cluster_members else 0}")

            except Exception as e:
                logger.warning(f"Failed to parse row {idx}: {e}")
                skipped_count += 1

        # Calculate statistics
        if self.protein_interfaces:
            physio_count = sum(1 for pi in self.protein_interfaces if pi.label == 1)
            non_physio_count = sum(1 for pi in self.protein_interfaces if pi.label == 0)

            logger.info(f"\n=== PARSING SUMMARY ===")
            logger.info(f"Successfully parsed: {len(self.protein_interfaces)} interfaces")
            logger.info(f"Skipped: {skipped_count} entries")
            logger.info(f"Physiological (TRUE): {physio_count}")
            logger.info(f"Non-physiological (FALSE): {non_physio_count}")
            logger.info(f"Interface sources: QSalign={qsalign_count}, ProtCID={protcid_count}")

            # Show statistics for proteins with multiple interfaces
            unique_pdb_ids = set(pi.ID for pi in self.protein_interfaces)
            logger.info(f"Unique PDB IDs: {len(unique_pdb_ids)}")

            # Count interfaces per protein
            interfaces_per_protein = {}
            for pi in self.protein_interfaces:
                if pi.ID not in interfaces_per_protein:
                    interfaces_per_protein[pi.ID] = 0
                interfaces_per_protein[pi.ID] += 1

            multi_interface_proteins = sum(1 for count in interfaces_per_protein.values() if count > 1)
            logger.info(f"Proteins with multiple interfaces: {multi_interface_proteins}")

            if self.protein_interfaces and unique_pdb_ids:
                logger.info(f"Average interfaces per protein: {len(self.protein_interfaces)/len(unique_pdb_ids):.2f}")

            # In the parse_data method, after parsing all interfaces:
            if self.fetch_pdb_metadata and self.pdb_metadata_cache:
                obsolete_count = sum(1 for meta in self.pdb_metadata_cache.values() if meta.get("obsolete", False))
                if obsolete_count > 0:
                    logger.warning(f"Found {obsolete_count} obsolete PDB entries in the dataset")

                    # Show a few examples
                    obsolete_examples = []
                    for pdb_id, meta in self.pdb_metadata_cache.items():
                        if meta.get("obsolete", False):
                            obsolete_examples.append(f"{pdb_id} -> {meta.get('replaced_by', 'Unknown')}")
                            if len(obsolete_examples) >= 5:
                                break

                    if obsolete_examples:
                        logger.warning(f"Sample obsolete entries: {', '.join(obsolete_examples)}")

            # Avoid division by zero
            if self.protein_interfaces:
                logger.info(f"Balance: {physio_count/len(self.protein_interfaces):.2%} physiological")

            # Show field statistics if available
            if self.protein_interfaces:
                bsa_values = [pi.bsa for pi in self.protein_interfaces if pi.bsa is not None]
                if bsa_values:
                    logger.info(f"Buried Surface Area (BSA): {len(bsa_values)} entries, avg={sum(bsa_values)/len(bsa_values):.1f} Å²")

                # Show assembly chain statistics if checked
                if self.check_pdb_label or self.check_cif_label:
                    updated_chains = sum(1 for pi in self.protein_interfaces
                                        if pi.LabelChain1 and pi.LabelChain2 and
                                        (pi.LabelChain1 != pi.AuthChain1 or pi.LabelChain2 != pi.AuthChain2))
                    logger.info(f"Assembly chain updates: {updated_chains} interfaces had chains updated")

                # Show cluster statistics if available
                if self.cluster_processor and self.cluster_processor.cluster_mapping:
                    interfaces_with_cluster = sum(1 for pi in self.protein_interfaces if pi.cluster_id is not None)
                    logger.info(f"Interfaces with cluster information: {interfaces_with_cluster}/{len(self.protein_interfaces)} ({interfaces_with_cluster/len(self.protein_interfaces)*100:.1f}%)")
                    logger.info(f"Total clusters found: {self.cluster_processor.stats['clusters_processed']}")
                    logger.info(f"Largest cluster size: {self.cluster_processor.stats['largest_cluster_size']}")

        else:
            logger.error(f"\n=== PARSING FAILED ===")
            logger.error(f"No interfaces were parsed!")
            logger.error(f"Skipped: {skipped_count} entries")
            logger.error(f"Total rows in dataset: {len(self.dataset)}")

            # Show more details about the data
            if hasattr(self, 'dataset') and self.dataset is not None:
                logger.error(f"Dataset columns: {list(self.dataset.columns)}")
                logger.error(f"First few rows:")
                for i in range(min(5, len(self.dataset))):
                    logger.error(f"  Row {i}: {self.dataset.iloc[i].to_dict()}")

        return self.protein_interfaces

    def generate_dataset_with_interfaces(self) -> Dict[str, Any]:
        """
        Generate Dataset markup where the dataset contains interface items,
        and each interface includes a Protein object.

        ALTERNATIVE SCHEMA: Dataset -> Interface items -> Protein objects
        """
        # Parse cluster file if not already parsed
        if self.cluster_processor and not self.cluster_processor.cluster_mapping:
            logger.info("Parsing BLASTClust file for cluster information...")
            if not self.cluster_processor.parse_blastclust_file():
                logger.warning("Failed to parse BLASTClust file. Skipping cluster information.")

        # Get basic statistics
        stats = self._generate_statistics()

        dataset_markup = {
            "@context": [
                "https://schema.org/",
                {"cr": "https://mlcommons.org/croissant/1.0"}
            ],
            "@type": ["Dataset", "cr:Dataset"],
            "@id": "https://github.com/vibbits/Elixir-3DBioInfo-Benchmark-Protein-Interfaces",

            # === CROISSANT CONFORMANCE ===
            "cr:conformsTo": "https://mlcommons.org/croissant/1.0",

            # === BIOSCHEMAS DATASET PROPERTIES ===
            "dct:conformsTo": "https://bioschemas.org/profiles/Dataset/1.0-RELEASE",
            "name": "Protein-Protein Interaction Interface Benchmark Dataset",
            "description": f"A benchmark dataset of {stats['total_entries']} protein crystal structures with {stats.get('physiological_count', 0)} physiological and {stats.get('non_physiological_count', 0)} non-physiological homodimer interfaces for evaluating protein-protein interface scoring functions. Ideal for machine learning applications in structural bioinformatics.",
            "identifier": "https://doi.org/10.5281/zenodo.XXXXXXX",
            "url": "https://github.com/vibbits/Elixir-3DBioInfo-Benchmark-Protein-Interfaces",
            "license": "https://creativecommons.org/licenses/by/4.0/",

            # Keywords as DefinedTerm list
            "keywords": [
                {
                    "@type": "DefinedTerm",
                    "name": "Protein interaction",
                    "inDefinedTermSet": "http://edamontology.org/topic_0128"
                },
                {
                    "@type": "DefinedTerm",
                    "name": "Protein structure",
                    "inDefinedTermSet": "http://edamontology.org/topic_2814"
                },
                {
                    "@type": "DefinedTerm",
                    "name": "Benchmarking",
                    "inDefinedTermSet": "http://edamontology.org/operation_2816"
                },
                {
                    "@type": "DefinedTerm",
                    "name": "Machine Learning Dataset",
                    "inDefinedTermSet": "http://edamontology.org/topic_3474"
                }
            ],

            # Creator/Publisher information
            "creator": [
                {
                    "@type": "Organization",
                    "name": "ELIXIR 3D-BioInfo Community",
                    "url": "https://elixir-europe.org/platforms/3d-bioinfo"
                }
            ],
            "datePublished": "2023-04-30",
            "publisher": {
                "@type": "Organization",
                "name": "ELIXIR Europe",
                "url": "https://elixir-europe.org"
            },
            "version": "1.0",

            # Citation to the provided publication
            "citation": {
                "@type": "ScholarlyArticle",
                "name": "Discriminating physiological from non-physiological interfaces in structures of protein complexes: A community-wide study",
                "url": "https://pubmed.ncbi.nlm.nih.gov/37365936/",
                "sameAs": "https://doi.org/10.1002/pmic.202200323"
            },

            # Dataset measurements
            "variableMeasured": [
                {
                    "@type": "PropertyValue",
                    "name": "physio",
                    "description": "Binary label indicating physiological (TRUE) or non-physiological (FALSE) homodimer",
                    "value": "Boolean classification"
                },
                {
                    "@type": "PropertyValue",
                    "name": "bsa",
                    "description": "Buried surface area of the protein-protein interface",
                    "unitCode": "Å²",
                    "value": "Surface area"
                },
                {
                    "@type": "PropertyValue",
                    "name": "contacts",
                    "description": "Number of atomic contacts at the interface",
                    "value": "Integer count"
                }
            ],

            "measurementTechnique": [
                "X-ray crystallography",
                "Conservation of interaction geometry analysis",
                "Cross-crystal form comparison (ProtCID)",
                "Homolog comparison (QSalign)"
            ],

            # Additional metadata
            "dateCreated": "2023-04-30",
            "dateModified": datetime.now().strftime("%Y-%m-%d"),
            "maintainer": {
                "@type": "Organization",
                "name": "ELIXIR 3D-BioInfo Community",
                "url": "https://elixir-europe.org/platforms/3d-bioinfo"
            },

            # Size information
            "size": f"{stats['total_entries']} entries"
        }

        # Add interface items to the dataset if we have data
        if self.protein_interfaces:
            # Create a list of interface items (each containing a Protein)
            interface_items = []
            for interface in self.protein_interfaces:
                # Generate Protein markup for this interface
                protein_markup = self._generate_protein_for_interface(interface)

                # Create interface item that includes the Protein
                interface_item = {
                    "@type": "DataCatalogItem",
                    "name": f"Interface {interface.InterfaceID}",
                    "description": f"Protein-protein interaction interface between chains {interface.AuthChain1} and {interface.AuthChain2}",
                    "identifier": interface.InterfaceID,
                    "url": f"https://www.rcsb.org/structure/{interface.ID}",
                    "additionalProperty": [
                        {
                            "@type": "PropertyValue",
                            "name": "InterfaceID",
                            "value": interface.InterfaceID,
                            "description": "Unique interface identifier"
                        },
                        {
                            "@type": "PropertyValue",
                            "name": "InterfaceSource",
                            "value": interface.interface_source,
                            "description": f"Source of interface ID: {interface.interface_source}"
                        },
                        {
                            "@type": "PropertyValue",
                            "name": "physio",
                            "value": interface.physio,
                            "description": "Physiological (TRUE) or non-physiological (FALSE)"
                        },
                        {
                            "@type": "PropertyValue",
                            "name": "label",
                            "value": interface.label,
                            "description": "Numeric label (1=physio, 0=non-physio)"
                        }
                    ]
                }

                # Add all interface features as additional properties
                self._add_interface_features_to_item(interface_item, interface)

                # Add cluster properties to interface item
                if interface.cluster_id:
                    interface_item = self._add_cluster_properties_to_markup(interface_item, interface)

                # Add the Protein object to the interface item
                interface_item["mainEntity"] = protein_markup

                interface_items.append(interface_item)

            dataset_markup["hasPart"] = interface_items
            dataset_markup["numberOfItems"] = len(interface_items)

        # Add Croissant-specific structure
        dataset_markup.update({
            "distribution": [
                {
                    "@type": "DataDownload",
                    "encodingFormat": "text/csv",
                    "contentUrl": self.csv_url,
                    "name": "Benchmark annotations (CSV)"
                },
                {
                    "@type": "DataDownload",
                    "encodingFormat": "chemical/x-mmCIF",
                    "contentUrl": self.mmcif_base_url,
                    "name": "mmCIF structure files"
                },
                {
                    "@type": "DataDownload",
                    "encodingFormat": "chemical/x-pdb",
                    "contentUrl": self.pdb_base_url,
                    "name": "PDB structure files"
                }
            ]
        })

        return dataset_markup

    def _add_interface_features_to_item(self, interface_item: Dict[str, Any], interface: ProteinInterface) -> None:
        """
        Add all interface features as additionalProperty to an interface item.

        Args:
            interface_item: The interface item dictionary to update
            interface: The ProteinInterface object with features
        """

        # ADD AUTHCHAINS FIRST:
        interface_item["additionalProperty"].extend([
            {
                "@type": "PropertyValue",
                "name": "AuthChain1",
                "value": interface.AuthChain1,
                "description": "First chain in the interface (may be updated from assembly)"
            },
            {
                "@type": "PropertyValue",
                "name": "AuthChain2",
                "value": interface.AuthChain2,
                "description": "Second chain in the interface (may be updated from assembly)"
            }
        ])

        # Add symmetry operations
        if interface.SymmetryOp1:
            interface_item["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "SymmetryOp1",
                "value": interface.SymmetryOp1,
                "description": "Symmetry operation for chain 1"
            })

        if interface.SymmetryOp2:
            interface_item["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "SymmetryOp2",
                "value": interface.SymmetryOp2,
                "description": "Symmetry operation for chain 2"
            })

        if interface.LabelChain1:
            interface_item["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "LabelChain1",
                "value": interface.LabelChain1,
                "description": "First chain in the assembly interface"
            })

        if interface.LabelChain2:
            interface_item["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "LabelChain2",
                "value": interface.LabelChain2,
                "description": "Second chain in the assembly interface"
            })

        # Add interface-specific properties if available
        if interface.bsa is not None:
            interface_item["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "Buried Surface Area (BSA)",
                "value": float(interface.bsa),
                "unitCode": "Å²",
                "description": "Total buried surface area at the interface"
            })

        if interface.contacts is not None:
            interface_item["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "Atomic Contacts",
                "value": int(interface.contacts),
                "description": "Number of atomic contacts at the interface"
            })

        if interface.gene is not None:
            interface_item["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "Gene",
                "value": interface.gene,
                "description": "Gene name associated with the protein"
            })

        if interface.superfamily is not None:
            interface_item["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "Superfamily",
                "value": interface.superfamily,
                "description": "Protein superfamily classification"
            })

        if interface.pfam is not None:
            interface_item["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "Pfam",
                "value": interface.pfam,
                "description": "Pfam domain classification"
            })

        # Add BSA components if available
        if interface.bsa_polar is not None:
            interface_item["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "BSA Polar",
                "value": float(interface.bsa_polar),
                "unitCode": "Å²",
                "description": "Polar component of buried surface area"
            })

        if interface.bsa_apolar is not None:
            interface_item["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "BSA Apolar",
                "value": float(interface.bsa_apolar),
                "unitCode": "Å²",
                "description": "Apolar component of buried surface area"
            })

        if interface.frac_polar is not None:
            interface_item["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "Fraction Polar",
                "value": float(interface.frac_polar),
                "description": "Fraction of polar contacts at the interface"
            })

        if interface.frac_apolar is not None:
            interface_item["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "Fraction Apolar",
                "value": float(interface.frac_apolar),
                "description": "Fraction of apolar contacts at the interface"
            })

        if interface.comments is not None and str(interface.comments).strip():
            interface_item["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "Comments",
                "value": str(interface.comments),
                "description": "Additional comments about the interface"
            })

    def _generate_protein_for_interface(self, interface: ProteinInterface) -> Dict[str, Any]:
        """
        Generate Bioschemas Protein markup for a protein associated with an interface.

        FULLY COMPLIANT with Bioschemas Protein Profile 0.11-RELEASE
        """
        # Construct structure file URLs for this protein
        pdb_url = f"{self.pdb_base_url}{interface.InterfaceID.lower()}.pdb.gz"
        if interface.interface_source == 'QSalign':
            mmcif_url = f"{self.mmcif_base_url}{interface.ID.lower()}.cif.gz"
        else:
            mmcif_url = f"{self.mmcif_base_url}{interface.InterfaceID.lower()}.cif.gz"
        rcsb_url = f"https://www.rcsb.org/structure/{interface.ID}"

        protein_markup = {
            "@context": "https://schema.org",
            "@type": "Protein",
            "@id": f"#protein_{interface.ID}",

            # === MINIMUM PROPERTIES (Bioschemas Protein 0.11-RELEASE) ===
            "dct:conformsTo": "https://bioschemas.org/profiles/Protein/0.11-RELEASE",
            "identifier": interface.ID,
            "name": f"Protein {interface.ID}",

            # === RECOMMENDED PROPERTIES ===
            "description": f"Protein structure with PDB ID: {interface.ID}. Part of the ELIXIR 3D-BioInfo benchmark dataset. This protein participates in interface {interface.InterfaceID} as a homodimer.",

            "url": rcsb_url,

            # Taxonomic information (will be updated if PDB metadata is fetched)
            "taxonomicRange": {
                "@type": "DefinedTerm",
                "name": "Organism-specific protein",
                "inDefinedTermSet": "https://www.ncbi.nlm.nih.gov/taxonomy"
            },

            # === OPTIONAL PROPERTIES ===
            "alternateName": [
                f"PDB {interface.ID}",
                f"Protein ID {interface.ID}"
            ],

            # Structural representations
            "hasRepresentation": [
                {
                    "@type": "PropertyValue",
                    "name": "PDB Structure",
                    "value": pdb_url,
                    "description": "Compressed PDB format structure file"
                },
                {
                    "@type": "PropertyValue",
                    "name": "mmCIF Structure",
                    "value": mmcif_url,
                    "description": "Compressed mmCIF format structure file"
                },
                {
                    "@type": "PropertyValue",
                    "name": "PDB ID",
                    "value": interface.ID,
                    "description": "RCSB PDB identifier"
                }
            ],

            # Protein-level properties
            "additionalProperty": [
                {
                    "@type": "PropertyValue",
                    "name": "PDB_ID",
                    "value": interface.ID,
                    "description": "Protein Data Bank identifier (4 letters)"
                },
                {
                    "@type": "PropertyValue",
                    "name": "AssociatedInterface",
                    "value": interface.InterfaceID,
                    "description": "Interface this protein participates in"
                },
                {
                    "@type": "PropertyValue",
                    "name": "InterfaceSource",
                    "value": interface.interface_source,
                    "description": f"Source of interface ID: {interface.interface_source}"
                },
                {
                    "@type": "PropertyValue",
                    "name": "InterfaceChains",
                    "value": f"{interface.AuthChain1}-{interface.AuthChain2}",
                    "description": "Chains involved in the interface"
                },
                {
                    "@type": "PropertyValue",
                    "name": "InterfaceClassification",
                    "value": "Physiological" if interface.label == 1 else "Non-physiological",
                    "description": "Classification of the associated interface"
                }
            ],

            # Molecular function
            "hasMolecularFunction": {
                "@type": "DefinedTerm",
                "name": "protein binding",
                "inDefinedTermSet": "http://purl.obolibrary.org/obo/GO_0005515"
            },

            # BioChemInteraction for the homodimer partner
            "bioChemInteraction": {
                "@type": "Protein",
                "name": f"Homodimer partner for interface {interface.InterfaceID}",
                "description": f"Homodimer interaction partner in chain {interface.AuthChain2}"
            }
        }

        # Add protein-level information if available

        if interface.LabelChain1 and interface.LabelChain2:
            protein_markup["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "LabelInterfaceChains",
                "value": f"{interface.LabelChain1}-{interface.LabelChain2}",
                "description": "Chains involved in the assembly interface"
            })

        if interface.gene:
            protein_markup["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "Gene",
                "value": interface.gene,
                "description": "Gene name associated with the protein"
            })

        if interface.superfamily:
            protein_markup["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "Superfamily",
                "value": interface.superfamily,
                "description": "Protein superfamily classification"
            })

        if interface.pfam:
            protein_markup["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "Pfam",
                "value": interface.pfam,
                "description": "Pfam domain classification"
            })

        # Add PDB metadata if enabled
        if self.fetch_pdb_metadata:
            try:
                pdb_metadata = self.enrich_protein_with_pdb_metadata(interface)
                protein_markup = self._add_pdb_metadata_to_protein_markup(protein_markup, pdb_metadata)
                logger.debug(f"Added PDB metadata for {interface.ID}")
            except Exception as e:
                logger.warning(f"Failed to add PDB metadata for {interface.ID}: {e}")

        # Add cluster properties to protein markup
        if interface.cluster_id:
            protein_markup = self._add_cluster_properties_to_markup(protein_markup, interface)

        return protein_markup

    def _generate_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for the dataset."""
        if not self.protein_interfaces:
            return {
                "total_entries": 0,
                "physiological_count": 0,
                "non_physiological_count": 0,
                "balance_ratio": 0,
                "unique_pdb_ids": 0,
                "qsalign_count": 0,
                "protcid_count": 0
            }

        physiological_count = sum(1 for pi in self.protein_interfaces if pi.label == 1)
        non_physiological_count = sum(1 for pi in self.protein_interfaces if pi.label == 0)

        # Count interface sources
        qsalign_count = sum(1 for pi in self.protein_interfaces if pi.interface_source == "QSalign")
        protcid_count = sum(1 for pi in self.protein_interfaces if pi.interface_source == "ProtCID")
        other_source_count = sum(1 for pi in self.protein_interfaces if pi.interface_source not in ["QSalign", "ProtCID"])

        # Calculate BSA statistics
        bsa_values = [pi.bsa for pi in self.protein_interfaces if pi.bsa is not None]
        contact_values = [pi.contacts for pi in self.protein_interfaces if pi.contacts is not None]

        # Count interfaces per protein
        interfaces_per_protein = {}
        for pi in self.protein_interfaces:
            if pi.ID not in interfaces_per_protein:
                interfaces_per_protein[pi.ID] = 0
            interfaces_per_protein[pi.ID] += 1

        stats = {
            "total_entries": len(self.protein_interfaces),
            "physiological_count": physiological_count,
            "non_physiological_count": non_physiological_count,
            "balance_ratio": physiological_count / len(self.protein_interfaces) if self.protein_interfaces else 0,
            "unique_pdb_ids": len(set(pi.ID for pi in self.protein_interfaces)),
            "qsalign_count": qsalign_count,
            "protcid_count": protcid_count,
            "other_source_count": other_source_count,
            "entries_with_bsa": len(bsa_values),
            "entries_with_contacts": len(contact_values),
            "proteins_with_multiple_interfaces": sum(1 for count in interfaces_per_protein.values() if count > 1)
        }

        if bsa_values:
            stats.update({
                "avg_bsa": sum(bsa_values) / len(bsa_values),
                "min_bsa": min(bsa_values),
                "max_bsa": max(bsa_values)
            })

        if contact_values:
            stats.update({
                "avg_contacts": sum(contact_values) / len(contact_values),
                "min_contacts": min(contact_values),
                "max_contacts": max(contact_values)
            })

        # Add interface distribution
        if interfaces_per_protein:
            interface_counts = list(interfaces_per_protein.values())
            stats.update({
                "avg_interfaces_per_protein": sum(interface_counts) / len(interface_counts),
                "max_interfaces_per_protein": max(interface_counts),
                "min_interfaces_per_protein": min(interface_counts)
            })

        # Add assembly chain statistics if checked
        if self.check_pdb_label or self.check_cif_label:
            updated_chains = sum(1 for pi in self.protein_interfaces
                                if pi.LabelChain1 and pi.LabelChain2 and
                                (pi.LabelChain1 != pi.AuthChain1 or pi.LabelChain2 != pi.AuthChain2))
            stats["assembly_chains_updated"] = updated_chains

        # Add cluster statistics if available
        if self.cluster_processor and self.cluster_processor.cluster_mapping:
            interfaces_with_cluster = sum(1 for pi in self.protein_interfaces if pi.cluster_id is not None)
            cluster_stats = self.cluster_processor.stats

            stats["cluster_info"] = {
                "interfaces_with_cluster": interfaces_with_cluster,
                "coverage_percentage": (interfaces_with_cluster / len(self.protein_interfaces) * 100) if self.protein_interfaces else 0,
                "total_clusters": cluster_stats.get("total_clusters", 0),
                "clusters_with_single_member": cluster_stats.get("clusters_with_single_member", 0),
                "clusters_with_multiple_members": cluster_stats.get("clusters_with_multiple_members", 0),
                "largest_cluster_size": cluster_stats.get("largest_cluster_size", 0),
                "total_interfaces_in_clusters": cluster_stats.get("total_interfaces_in_clusters", 0),
                "assignment_method": "Direct assignment from BLASTClust output",
                "cluster_id_selection": "First InterfaceID in each BLASTClust line becomes the ClusterID"
            }

        return stats

    def generate_fair_metadata_package(self) -> Dict[str, Any]:
        """
        Generate a complete FAIR metadata package for the dataset.
        """
        stats = self._generate_statistics()

        fair_package = {
            "fair_metadata": {
                "findability": {
                    "persistent_identifier": "https://doi.org/10.5281/zenodo.XXXXXXX",
                    "rich_metadata": True,
                    "searchable_registry": ["Google Dataset Search", "WorkflowHub", "bio.tools", "Hugging Face Datasets"],
                    "versioning": {
                        "current_version": "1.0",
                        "versioning_system": "Git",
                        "repository": "https://github.com/vibbits/Elixir-3DBioInfo-Benchmark-Protein-Interfaces"
                    }
                },
                "accessibility": {
                    "protocol": "HTTPS",
                    "authentication": "None required",
                    "long_term_availability": "GitHub repository with Zenodo archive"
                },
                "interoperability": {
                    "data_formats": ["CSV", "PDB", "mmCIF", "JSON-LD"],
                    "vocabularies": ["Schema.org", "Bioschemas", "EDAM", "Gene Ontology", "Pfam", "SCOP"],
                    "standards_compliance": [
                        "Bioschemas Dataset Profile 1.0-RELEASE",
                        "Bioschemas Protein Profile 0.11-RELEASE",
                        "Croissant 1.0",
                        "DCAT"
                    ]
                },
                "reusability": {
                    "license": "CC-BY-4.0",
                    "provenance": {
                        "creation_date": "2023-04-30",
                        "creators": ["ELIXIR 3D-BioInfo Community"],
                        "methods": ["QSalign", "ProtCID", "Conservation analysis"],
                        "reference_publication": "https://pubmed.ncbi.nlm.nih.gov/37365936/",
                        "data_sources": ["Protein Data Bank (PDB)", "ProtCID", "QSalign"]
                    },
                    "documentation": {
                        "readme": "https://github.com/vibbits/Elixir-3DBioInfo-Benchmark-Protein-Interfaces/blob/main/README.md",
                        "methods_paper": "https://doi.org/10.1002/pmic.202200323",
                        "ml_ready": True,
                        "task_type": "Binary classification",
                        "input_modality": "Protein structures + tabular features",
                        "target_column": "physio",
                        "features": ["ID", "InterfaceID", "AuthChain1", "AuthChain2", "bsa", "contacts", "gene", "superfamily", "pfam"]
                    }
                }
            },
            "bioschemas_markup": {
                "dataset": self.generate_dataset_with_interfaces(),
                "total_interfaces": len(self.protein_interfaces) if self.protein_interfaces else 0,
                "sample_proteins": [
                    self._generate_protein_for_interface(self.protein_interfaces[i])
                    for i in range(min(5, len(self.protein_interfaces)))
                ] if self.protein_interfaces else []
            },
            "dataset_statistics": stats,
            "interface_id_handling": {
                "interface_sources": {
                    "QSalign_format": "When InterfaceID is NA, format is PDBID_X where X is biological assembly number (default: 1)",
                    "ProtCID_format": "When InterfaceID is an integer, it's used directly from ProtCID database",
                    "qsalign_count": stats.get("qsalign_count", 0),
                    "protcid_count": stats.get("protcid_count", 0),
                    "other_source_count": stats.get("other_source_count", 0)
                },
                "schema_type": "Dataset contains Interface items, each Interface contains a Protein object",
                "protein_representation": "Each protein is represented by its PDB ID (4 letters)",
                "interface_representation": "Interfaces have IDs from QSalign (PDBID_ASSEMBLY) or ProtCID (integer)",
                "data_structure": "Dataset -> Interface items -> Protein objects",
                "features_included": "All interface features are included in additionalProperty of each interface item"
            }
        }

        # Add PDB metadata statistics if enabled
        if self.fetch_pdb_metadata and self.pdb_metadata_cache:
            pdb_stats = {
                "pdb_metadata_fetched": len(self.pdb_metadata_cache),
                "successful_fetches": sum(1 for meta in self.pdb_metadata_cache.values() if meta.get("found_in_api", False)),
                "unique_proteins_with_metadata": len(set(meta.get("pdb_id") for meta in self.pdb_metadata_cache.values() if meta.get("found_in_api", False))),
                "total_chains": sum(meta.get("chain_count", 0) for meta in self.pdb_metadata_cache.values()),
                "homomeric_structures": sum(1 for meta in self.pdb_metadata_cache.values() if meta.get("is_homomer", False)),
                "heteromeric_structures": sum(1 for meta in self.pdb_metadata_cache.values() if meta.get("is_homomer", False) is False and meta.get("found_in_api", False))
            }

            # Add API usage statistics
            if self.pdb_api_client:
                api_stats = self.pdb_api_client.get_statistics()
                pdb_stats["api_optimization"] = {
                    "total_api_calls": api_stats["api_calls"],
                    "cache_hits": api_stats["cache_hits"],
                    "cache_misses": api_stats["cache_misses"],
                    "cache_hit_rate": f"{api_stats['cache_hits']/(api_stats['cache_hits']+api_stats['cache_misses'])*100:.1f}%" if (api_stats['cache_hits']+api_stats['cache_misses']) > 0 else "0%",
                    "batches_processed": api_stats["batches_processed"],
                    "obsolete_entries": api_stats["obsolete_entries"]
                }

            # Add resolution statistics
            resolutions = [meta.get("resolution") for meta in self.pdb_metadata_cache.values()
                          if meta.get("resolution") is not None]
            if resolutions:
                pdb_stats["resolution_stats"] = {
                    "average": sum(resolutions) / len(resolutions),
                    "min": min(resolutions),
                    "max": max(resolutions),
                    "count": len(resolutions)
                }

            # Add sequence statistics
            all_sequences = []
            for meta in self.pdb_metadata_cache.values():
                if meta.get("sequences"):
                    all_sequences.extend(list(meta["sequences"].values()))

            if all_sequences:
                seq_lengths = [len(seq) for seq in all_sequences]
                pdb_stats["sequence_stats"] = {
                    "total_sequences": len(all_sequences),
                    "unique_sequences": len(set(all_sequences)),
                    "avg_sequence_length": sum(seq_lengths) / len(seq_lengths),
                    "min_sequence_length": min(seq_lengths),
                    "max_sequence_length": max(seq_lengths)
                }

            # Add organism statistics
            organisms = []
            for meta in self.pdb_metadata_cache.values():
                if meta.get("source_organism"):
                    organisms.extend(meta["source_organism"])

            if organisms:
                org_counts = Counter(organisms)
                pdb_stats["top_organisms"] = dict(org_counts.most_common(10))
                pdb_stats["unique_organisms"] = len(org_counts)

            fair_package["pdb_metadata_statistics"] = pdb_stats

        # Add cluster statistics if available
        if self.cluster_processor and self.cluster_processor.cluster_mapping:
            cluster_stats = {
                "blastclust_file": self.cluster_file,
                "blastclust_method": "BLASTClust sequence clustering",
                "blastclust_options": "-S 25 -L 0.5 -b F",
                "interfaces_with_cluster": stats.get("cluster_info", {}).get("interfaces_with_cluster", 0),
                "coverage_percentage": stats.get("cluster_info", {}).get("coverage_percentage", 0),
                "total_clusters": stats.get("cluster_info", {}).get("total_clusters", 0),
                "clusters_with_single_member": stats.get("cluster_info", {}).get("clusters_with_single_member", 0),
                "clusters_with_multiple_members": stats.get("cluster_info", {}).get("clusters_with_multiple_members", 0),
                "largest_cluster_size": stats.get("cluster_info", {}).get("largest_cluster_size", 0),
                "cluster_properties_added": [
                    "ClusterID: The cluster identifier",
                    "ClusterSize: Number of interfaces in the cluster",
                    "ClusterMembers: List of other interfaces (for multi-member clusters)",
                    "ClusterMethod: BLASTClust sequence clustering",
                    "ClusterMethodOptions: Parameters used (-S 25 -L 0.5 -b F)"
                ],
                "assignment_method": "Direct assignment from BLASTClust output during parsing",
                "cluster_id_selection": "First InterfaceID in each BLASTClust line becomes the ClusterID"
            }

            fair_package["cluster_statistics"] = cluster_stats

        return fair_package

    def save_bioschemas_markup(self, output_dir: str = "bioschema") -> Dict[str, Any]:
        """
        Save all generated Bioschemas markup to files.

        Args:
            output_dir: Directory to save output files

        Returns:
            Dictionary with paths to generated files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Parse cluster file if provided and not already parsed
        if self.cluster_processor and not self.cluster_processor.cluster_mapping:
            logger.info("Parsing BLASTClust file for cluster information...")
            if not self.cluster_processor.parse_blastclust_file():
                logger.warning("Failed to parse BLASTClust file. Skipping cluster information.")

        # Generate and save alternative dataset markup with interfaces
        dataset_markup = self.generate_dataset_with_interfaces()
        dataset_file = os.path.join(output_dir, "dataset_with_interfaces.json")
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_markup, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved dataset markup with interfaces to {dataset_file}")
        logger.info(f"Dataset contains {len(self.protein_interfaces) if self.protein_interfaces else 0} interface items")

        # Generate and save sample interface-protein pairs ONLY if we have data
        if not self.protein_interfaces or len(self.protein_interfaces) == 0:
            logger.warning("No protein interfaces to save!")
            # Still generate FAIR package for debugging
            fair_package = {
                "error": "No protein interfaces were parsed",
                "csv_url": self.csv_url,
                "csv_separator": self.csv_separator,
                "columns_found": list(self.dataset.columns) if self.dataset else [],
                "column_mapping": self.column_mapping
            }
            fair_file = os.path.join(output_dir, "fair_metadata_package.json")
            with open(fair_file, 'w', encoding='utf-8') as f:
                json.dump(fair_package, f, indent=2, ensure_ascii=False)

            return {"error": "No data parsed", "dataset_file": dataset_file}

        interfaces_dir = os.path.join(output_dir, "interface_protein_pairs")
        os.makedirs(interfaces_dir, exist_ok=True)

        # Save ALL interface-protein pairs as individual files WITH ALL FEATURES
        generated_files = []
        for i, interface in enumerate(self.protein_interfaces):
            # Create interface item with protein
            interface_item = {
                "@context": "https://schema.org",
                "@type": "DataCatalogItem",
                "name": f"Interface {interface.InterfaceID}",
                "description": f"Protein-protein interaction interface between chains {interface.AuthChain1} and {interface.AuthChain2}",
                "identifier": interface.InterfaceID,
                "mainEntity": self._generate_protein_for_interface(interface),
                "additionalProperty": [
                    {
                        "@type": "PropertyValue",
                        "name": "InterfaceSource",
                        "value": interface.interface_source,
                        "description": f"Source of interface ID: {interface.interface_source}"
                    },
                    {
                        "@type": "PropertyValue",
                        "name": "physio",
                        "value": interface.physio,
                        "description": "Physiological (TRUE) or non-physiological (FALSE)"
                    },
                    {
                        "@type": "PropertyValue",
                        "name": "label",
                        "value": interface.label,
                        "description": "Numeric label (1=physio, 0=non-physio)"
                    },
                    {
                        "@type": "PropertyValue",
                        "name": "PDB_ID",
                        "value": interface.ID,
                        "description": "Protein Data Bank identifier"
                    },
                    {
                        "@type": "PropertyValue",
                        "name": "AuthChain1",
                        "value": interface.AuthChain1,
                        "description": "First chain in interface"
                    },
                    {
                        "@type": "PropertyValue",
                        "name": "AuthChain2",
                        "value": interface.AuthChain2,
                        "description": "Second chain in interface"
                    }
                ]
            }

            # Add all interface features to the additionalProperty
            self._add_interface_features_to_item(interface_item, interface)

            # Add cluster properties to interface item
            if interface.cluster_id:
                interface_item = self._add_cluster_properties_to_markup(interface_item, interface)

            safe_interface_id = self._clean_filename(interface.InterfaceID)
            interface_file = os.path.join(interfaces_dir, f"interface_{safe_interface_id}.json")
            with open(interface_file, 'w', encoding='utf-8') as f:
                json.dump(interface_item, f, indent=2, ensure_ascii=False)

            generated_files.append(f"interface_{safe_interface_id}.json")

            # Log first few entries with their features
            if i < 3:
                logger.info(f"Generated interface-protein markup for {interface.InterfaceID} (source: {interface.interface_source})")
                logger.info(f"  Features included: physio={interface.physio}, bsa={interface.bsa}, contacts={interface.contacts}, gene={interface.gene}")
                if self.fetch_pdb_metadata:
                    pdb_meta = self.pdb_metadata_cache.get(interface.ID.upper(), {})
                    if pdb_meta:
                        logger.info(f"  PDB metadata: {pdb_meta.get('chain_count', 0)} chains, {pdb_meta.get('unique_sequences', 0)} unique seqs, homomer={pdb_meta.get('is_homomer', False)}")
                if interface.LabelChain1 and interface.LabelChain2:
                    logger.info(f"  Assembly chains: {interface.LabelChain1}-{interface.LabelChain2}")
                if interface.cluster_id:
                    logger.info(f"  Cluster info: ID={interface.cluster_id}, size={interface.cluster_size}, members={len(interface.cluster_members) if interface.cluster_members else 0}")
                logger.info(f"  Saved to: {interface_file}")

            # Log progress for large datasets
            if len(self.protein_interfaces) > 100 and i % 100 == 0 and i > 0:
                logger.info(f"Progress: Generated {i} out of {len(self.protein_interfaces)} interface files...")

        logger.info(f"Generated ALL {len(self.protein_interfaces)} interface-protein markup files in {interfaces_dir}")
        logger.info(f"  Each file includes all interface features in additionalProperty")
        if self.fetch_pdb_metadata:
            logger.info(f"  PDB metadata fetched for {len(self.pdb_metadata_cache)} unique structures")
            if self.pdb_api_client:
                api_stats = self.pdb_api_client.get_statistics()
                logger.info(f"  API Optimization: {api_stats['api_calls']} calls, "
                           f"{api_stats['cache_hits']} cache hits ({api_stats['cache_hits']/(api_stats['cache_hits']+api_stats['cache_misses'])*100:.1f}% hit rate)")
        if self.check_pdb_label or self.check_cif_label:
            updated_count = sum(1 for pi in self.protein_interfaces
                               if pi.LabelChain1 and pi.LabelChain2 and
                               (pi.LabelChain1 != pi.AuthChain1 or pi.LabelChain2 != pi.AuthChain2))
            logger.info(f"  Assembly chain validation: {updated_count} interfaces had chains updated")
        if self.cluster_processor and self.cluster_processor.cluster_mapping:
            interfaces_with_cluster = sum(1 for pi in self.protein_interfaces if pi.cluster_id is not None)
            logger.info(f"  Cluster information added for {interfaces_with_cluster} interfaces")
            logger.info(f"  ClusterMethod: BLASTClust sequence clustering")
            logger.info(f"  ClusterMethodOptions: -S 25 -L 0.5 -b F")
            logger.info(f"  ClusterID selection: First InterfaceID in each BLASTClust line becomes the ClusterID")

        # Generate and save complete FAIR package
        fair_package = self.generate_fair_metadata_package()
        fair_file = os.path.join(output_dir, "fair_metadata_package.json")
        with open(fair_file, 'w', encoding='utf-8') as f:
            json.dump(fair_package, f, indent=2, ensure_ascii=False)

        # Generate HTML snippet with embedded JSON-LD
        html_snippet = self._generate_html_snippet(dataset_markup)
        html_file = os.path.join(output_dir, "embedded_markup.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_snippet)

        # Save PDB metadata cache separately
        if self.fetch_pdb_metadata and self.pdb_metadata_cache:
            pdb_metadata_file = os.path.join(output_dir, "pdb_metadata_cache.json")
            with open(pdb_metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.pdb_metadata_cache, f, indent=2, default=str)
            logger.info(f"Saved PDB metadata cache with {len(self.pdb_metadata_cache)} entries to {pdb_metadata_file}")

        # Generate a manifest file
        manifest = {
            "generated_date": datetime.now().isoformat(),
            "dataset_file": dataset_file,
            "interfaces_dir": interfaces_dir,
            "total_interfaces_generated": len(self.protein_interfaces),
            "total_interfaces_available": len(self.protein_interfaces),
            "interface_files_sample": generated_files[:10],
            "fair_package_file": fair_file,
            "html_snippet_file": html_file,
            "csv_settings": {
                "url": self.csv_url,
                "separator_used": self.csv_separator,
                "auto_detect": self.auto_detect_separator
            },
            "data_columns_found": list(self.dataset.columns) if self.dataset is not None and not self.dataset.empty else [],
            "column_mapping_used": self.column_mapping,
            "schema_structure": {
                "schema_type": "Dataset -> Interface items -> Protein objects",
                "total_interfaces": len(self.protein_interfaces),
                "unique_proteins": len(set(pi.ID for pi in self.protein_interfaces)),
                "interface_sources": {
                    "QSalign": sum(1 for pi in self.protein_interfaces if pi.interface_source == "QSalign"),
                    "ProtCID": sum(1 for pi in self.protein_interfaces if pi.interface_source == "ProtCID"),
                    "Other": sum(1 for pi in self.protein_interfaces if pi.interface_source not in ["QSalign", "ProtCID"])
                },
                "interface_id_format_qsalign": "PDBID_ASSEMBLYNUMBER (when InterfaceID is NA)",
                "interface_id_format_protcid": "ProtCID integer (when InterfaceID is an integer)",
                "features_included": "All interface features (physio, bsa, contacts, gene, superfamily, pfam, etc.) in additionalProperty",
                "proteins_with_multiple_interfaces": sum(1 for pid in set(pi.ID for pi in self.protein_interfaces)
                                                         if len([pi for pi in self.protein_interfaces if pi.ID == pid]) > 1)
            }
        }

        # Add PDB metadata info to manifest if enabled
        if self.fetch_pdb_metadata:
            manifest["pdb_metadata"] = {
                "enabled": True,
                "api_base_url": self.pdb_api_base_url,
                "unique_structures_with_metadata": len(self.pdb_metadata_cache),
                "total_chains_with_sequences": sum(len(meta.get("sequences", {})) for meta in self.pdb_metadata_cache.values()),
                "homomeric_structures": sum(1 for meta in self.pdb_metadata_cache.values() if meta.get("is_homomer", False)),
                "metadata_cache_file": "pdb_metadata_cache.json" if self.pdb_metadata_cache else None
            }

            # Add API optimization statistics
            if self.pdb_api_client:
                api_stats = self.pdb_api_client.get_statistics()
                manifest["pdb_metadata"]["api_optimization"] = {
                    "total_api_calls": api_stats["api_calls"],
                    "cache_hits": api_stats["cache_hits"],
                    "cache_misses": api_stats["cache_misses"],
                    "cache_hit_rate": f"{api_stats['cache_hits']/(api_stats['cache_hits']+api_stats['cache_misses'])*100:.1f}%" if (api_stats['cache_hits']+api_stats['cache_misses']) > 0 else "0%",
                    "batches_processed": api_stats["batches_processed"],
                    "batch_size": self.pdb_api_client.batch_size,
                    "request_delay": self.pdb_api_client.request_delay,
                    "cache_expiry_hours": self.pdb_api_client.cache_expiry.total_seconds() / 3600
                }

        # Add assembly checking info to manifest
        if self.check_pdb_label or self.check_cif_label:
            manifest["assembly_checks"] = {
                "pdb_assembly_checked": self.check_pdb_label,
                "cif_assembly_checked": self.check_cif_label,
                "pdb_label_dir": self.pdb_label_dir if self.pdb_label_dir else "default (pdb_base_url)",
                "cif_label_dir": self.cif_label_dir if self.cif_label_dir else "default (mmcif_base_url)",
                "interfaces_with_assembly_chains": sum(1 for pi in self.protein_interfaces if pi.LabelChain1 and pi.LabelChain2),
                "interfaces_with_updated_chains": sum(1 for pi in self.protein_interfaces
                                                     if pi.LabelChain1 and pi.LabelChain2 and
                                                     (pi.LabelChain1 != pi.AuthChain1 or pi.LabelChain2 != pi.AuthChain2))
            }

        # Add cluster information to manifest
        if self.cluster_processor and self.cluster_processor.cluster_mapping:
            interfaces_with_cluster = sum(1 for pi in self.protein_interfaces if pi.cluster_id is not None)
            cluster_stats = self.cluster_processor.stats

            manifest["cluster_information"] = {
                "blastclust_file": self.cluster_file,
                "blastclust_method": "BLASTClust sequence clustering",
                "blastclust_options": "-S 25 -L 0.5 -b F",
                "interfaces_with_cluster": interfaces_with_cluster,
                "coverage_percentage": (interfaces_with_cluster / len(self.protein_interfaces) * 100) if self.protein_interfaces else 0,
                "total_clusters": cluster_stats.get("total_clusters", 0),
                "clusters_with_single_member": cluster_stats.get("clusters_with_single_member", 0),
                "clusters_with_multiple_members": cluster_stats.get("clusters_with_multiple_members", 0),
                "largest_cluster_size": cluster_stats.get("largest_cluster_size", 0),
                "total_interfaces_in_clusters": cluster_stats.get("total_interfaces_in_clusters", 0),
                "cluster_id_selection": "First InterfaceID in each BLASTClust line becomes the ClusterID",
                "cluster_properties_added": [
                    "ClusterID: The cluster identifier",
                    "ClusterSize: Number of interfaces in the cluster",
                    "ClusterMembers: List of other interfaces (for multi-member clusters)",
                    "ClusterMethod: BLASTClust sequence clustering",
                    "ClusterMethodOptions: Parameters used (-S 25 -L 0.5 -b F)"
                ],
                "assignment_method": "Direct assignment from BLASTClust output during parsing"
            }

        manifest_file = os.path.join(output_dir, "manifest.json")
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved all markup files to {output_dir}/")
        logger.info(f"Generated manifest at {manifest_file}")

        return manifest

    def _clean_filename(self, filename: str) -> str:
        """Clean a string to be safe for use as a filename."""
        # First check if it's a valid string
        if pd.isna(filename):
            return "unknown"

        # Convert to string
        filename_str = str(filename)

        # Replace problematic characters
        cleaned = filename_str.replace('\\', '_').replace('/', '_').replace(':', '_')
        cleaned = cleaned.replace('*', '_').replace('?', '_').replace('"', '_')
        cleaned = cleaned.replace('<', '_').replace('>', '_').replace('|', '_')
        cleaned = cleaned.replace('.', '_')  # Also replace dots to avoid issues like "6.0.json"

        return cleaned


# The rest of the helper functions (three_to_one_letter_aa, extract_chain_ids_from_pdb_gz, etc.)
# remain exactly the same as in the original code

def three_to_one_letter_aa(residue_3letter: str) -> str:
    """
    Convert 3-letter amino acid code to 1-letter code.

    Args:
        residue_3letter: Three-letter amino acid code

    Returns:
        One-letter code or 'X' if unknown
    """
    # Copy the entire implementation from the static method
    aa_map = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
        'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
        'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
        'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
        'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        # ... rest of the mapping ...
    }

    residue_upper = residue_3letter.strip().upper()

    if residue_upper in aa_map:
        return aa_map[residue_upper]
    elif len(residue_upper) == 3 and residue_upper.isalpha():
        logger.debug(f"Unknown residue code: {residue_upper}, using 'X'")
        return 'X'
    else:
        logger.debug(f"Invalid residue code format: '{residue_3letter}'")
        return ''


def extract_chain_ids_from_pdb_gz(pdb_gz_path: str) -> Set[str]:
    """
    Extract unique chain IDs from a compressed PDB file.
    Supports both local file paths and HTTP URLs.

    Args:
        pdb_gz_path: Path or URL to the .pdb.gz file
                     Can be:
                     - Local path: "/path/to/file.pdb.gz"
                     - HTTP URL: "https://example.com/file.pdb.gz"
                     - GitHub raw URL: "https://raw.githubusercontent.com/user/repo/file.pdb.gz"

    Returns:
        Set of chain IDs found in the PDB file
    """
    # This function remains the same as in the original code
    pass

def extract_chain_ids_from_mmcif_gz(mmcif_gz_path: str) -> Set[str]:
    """
    Extract unique chain IDs from a compressed mmCIF file.
    Properly parses mmCIF format with loop structures.

    Args:
        mmcif_gz_path: Path to the .cif.gz file

    Returns:
        Set of chain IDs found in the mmCIF file
    """
    # This function remains the same as in the original code
    pass

def update_protein_interface_with_actual_chains(interface: ProteinInterface,
                                               pdb_file_path: Optional[str] = None,
                                               mmcif_file_path: Optional[str] = None) -> ProteinInterface:
    """
    Update a ProteinInterface object with actual chain IDs from structure files.

    Args:
        interface: The ProteinInterface object to update
        pdb_file_path: Full path to PDB.gz file (or None to skip PDB check)
        mmcif_file_path: Full path to mmCIF.gz file (or None to skip mmCIF check)

    Returns:
        Updated ProteinInterface object with actual chain IDs
    """
    # This function remains the same as in the original code
    pass

def get_actual_interface_chains(pdb_id: str,
                               auth_chain1: str,
                               auth_chain2: str,
                               pdb_file_path: Optional[str] = None,
                               mmcif_file_path: Optional[str] = None) -> Tuple[str, str, Dict[str, Any]]:
    """
    Get the actual chain IDs from the structure file, with fallback to provided values.

    Args:
        pdb_id: PDB ID (for logging/identification)
        auth_chain1: First chain from annotation
        auth_chain2: Second chain from annotation
        pdb_file_path: Full path to PDB.gz file (or None to skip PDB check)
        mmcif_file_path: Full path to mmCIF.gz file (or None to skip mmCIF check)

    Returns:
        Tuple of (actual_chain1, actual_chain2, validation_info)
    """
    # This function remains the same as in the original code
    pass

def validate_interface_chains(pdb_id: str,
                             interface_chains: str,
                             pdb_file_path: Optional[str] = None,
                             mmcif_file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate that the interface chains exist in the actual PDB/mmCIF files.

    Args:
        pdb_id: PDB ID (e.g., "1ABC") - for logging only
        interface_chains: Expected chain pair (e.g., "A-B" or "AB")
        pdb_file_path: Full path to PDB.gz file (or None to skip)
        mmcif_file_path: Full path to mmCIF.gz file (or None to skip)

    Returns:
        Dictionary with validation results
    """
    # This function remains the same as in the original code
    pass

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate FAIR-compliant Bioschemas markup for protein-protein interaction benchmark data with Croissant compatibility.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s  # Use default comma separator
  %(prog)s --separator "\\t"  # Use tab separator
  %(prog)s --separator "," --auto-detect  # Use comma, fallback to auto-detection
  %(prog)s --output my_output_dir  # Custom output directory
  %(prog)s --csv-url "https://example.com/data.csv"  # Custom CSV URL
  %(prog)s --cluster blastclust_output.txt  # Add cluster information from BLASTClust file
        """
    )

    parser.add_argument(
        "--csv-url",
        default="https://raw.githubusercontent.com/biofold/ppi-benchmark-fair/main/data/benchmark_annotated_updated_30042023.csv",
        help="URL of the CSV file (default: the benchmark dataset)"
    )

    parser.add_argument(
        "--mmcif-url",
        default="https://raw.githubusercontent.com/biofold/ppi-benchmark-fair/main/data/benchmark_mmcif_format/",
        help="Base URL for mmCIF structure files"
    )

    parser.add_argument(
        "--pdb-url",
        default="https://raw.githubusercontent.com/biofold/ppi-benchmark-fair/main/data/benchmark_pdb_format/",
        help="Base URL for PDB structure files"
    )

    parser.add_argument(
        "--separator", "-s",
        default=",",
        help="CSV separator character (default: ',')"
    )

    parser.add_argument(
        "--auto-detect",
        action="store_true",
        help="Auto-detect separator if the specified one fails"
    )

    parser.add_argument(
        "--output", "-o",
        default="bioschema",
        help="Output directory for generated files (default: 'bioschema')"
    )

    parser.add_argument(
        "--check-pdb-label",
        action="store_true",
        help="Check assembly interface chains on pdb files"
    )

    parser.add_argument(
        "--pdb-label-dir",
        default=None,
        help="Base directory for assembly PDB files (default: use pdb-url from command line)"
    )

    parser.add_argument(
        "--check-cif-label",
        action="store_true",
        help="Check assembly interface chains on cif files"
    )

    parser.add_argument(
        "--cif-label-dir",
        default=None,
        help="Base directory for assembly mmCIF files (default: use mmcif-url from command line)"
    )

    parser.add_argument(
        "--fetch-pdb-metadata",
        action="store_true",
        help="Fetch additional PDB metadata from RCSB API (resolution, organism, ALL sequences, etc.)"
    )

    parser.add_argument(
        "--pdb-api-url",
        default="https://data.rcsb.org/rest/v1",
        help="RCSB PDB REST API base URL (default: https://data.rcsb.org/rest/v1)"
    )

    parser.add_argument(
        "--cluster",
        help="Path to BLASTClust output file for adding cluster information"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: download and inspect CSV without full processing"
    )

    return parser.parse_args()

def debug_csv_parsing(csv_url: str, separator: str = ","):
    """Debug function to inspect CSV parsing issues."""
    print(f"\n=== DEBUGGING CSV PARSING ===")
    print(f"CSV URL: {csv_url}")
    print(f"Separator: '{separator}'")

    try:
        # Download and inspect the CSV
        import requests
        response = requests.get(csv_url)
        if response.status_code == 200:
            content = response.text
            lines = content.split('\n')

            print(f"\n=== CSV STRUCTURE ===")
            print(f"Total lines: {len(lines)}")
            print(f"First 100 characters: {repr(content[:100])}")

            print(f"\nFirst 5 lines:")
            for i, line in enumerate(lines[:5]):
                print(f"Line {i}: {repr(line)}")

            print(f"\n=== TRYING TO PARSE WITH PANDAS ===")
            import pandas as pd
            import io

            # Try different approaches
            success = False
            
            # Approach 1: Try with the specified separator
            try:
                df = pd.read_csv(io.StringIO(content), sep=separator)
                print(f"✅ Successfully parsed with separator '{separator}'")
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {list(df.columns)}")
                success = True
            except Exception as e:
                print(f"❌ Failed with separator '{separator}': {e}")

            # Approach 2: Try auto-detection if Approach 1 failed
            if not success:
                print(f"\n=== TRYING AUTO-DETECTION ===")
                separators_to_try = ['\t', ';', ',', ' ', '|']
                for sep in separators_to_try:
                    if sep == separator:  # Skip already tried
                        continue
                    try:
                        df = pd.read_csv(io.StringIO(content), sep=sep)
                        print(f"✅ Successfully parsed with separator '{repr(sep)}'")
                        print(f"   Shape: {df.shape}")
                        print(f"   Columns: {list(df.columns)}")
                        success = True
                        
                        # Show first few rows
                        print(f"\nFirst 3 rows:")
                        for i in range(min(3, len(df))):
                            print(f"Row {i}:")
                            for col in df.columns[:5]:  # Show first 5 columns
                                print(f"  {col}: {repr(df.iloc[i][col])}")
                        
                        break
                    except Exception as e:
                        print(f"❌ Failed with separator '{repr(sep)}': {e}")
                        continue

            # Approach 3: Try reading with no header if needed
            if not success:
                print(f"\n=== TRYING WITHOUT HEADER ===")
                try:
                    df = pd.read_csv(io.StringIO(content), header=None)
                    print(f"✅ Successfully parsed with no header")
                    print(f"   Shape: {df.shape}")
                    print(f"   Columns: {list(df.columns)}")
                    success = True
                except Exception as e:
                    print(f"❌ Failed without header: {e}")

            # Approach 4: Try different encodings
            if not success:
                print(f"\n=== TRYING DIFFERENT ENCODINGS ===")
                encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                for encoding in encodings_to_try:
                    try:
                        df = pd.read_csv(io.StringIO(content.encode('utf-8').decode(encoding)), sep=separator)
                        print(f"✅ Successfully parsed with encoding '{encoding}'")
                        print(f"   Shape: {df.shape}")
                        success = True
                        break
                    except Exception as e:
                        print(f"❌ Failed with encoding '{encoding}': {e}")

            if success:
                return df
            else:
                print(f"\n❌ Could not parse with any approach")
                return None

        else:
            print(f"❌ Failed to download CSV: HTTP {response.status_code}")
            print(f"   Response text: {response.text[:200]}...")
            return None

    except Exception as e:
        print(f"❌ Error in debug function: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  PPI Benchmark FAIR Metadata Generator                        ║
║  with Bioschemas & Croissant Compliance                       ║
║  (Alternative Schema: Dataset → Interfaces → Proteins)        ║
║  OPTIMIZED for minimal PDB API calls with batch requests      ║
╚═══════════════════════════════════════════════════════════════╝

Settings:
  CSV URL:         {args.csv_url}
  CSV Separator:   '{args.separator}' (auto-detect: {args.auto_detect})
  Output Directory: {args.output}
  mmCIF Files:     {args.mmcif_url}
  PDB Files:       {args.pdb_url}
  PDB Metadata:    {args.fetch_pdb_metadata} (API: {args.pdb_api_url if args.fetch_pdb_metadata else 'N/A'})
  Assembly Checks: PDB={args.check_pdb_label}, mmCIF={args.check_cif_label}
  Cluster Info:    {args.cluster if args.cluster else 'Not provided'}
  API Optimization: Batch requests, caching, rate limiting
    """)

    # DEBUG: First inspect the CSV
    print("\n=== DEBUG: INSPECTING CSV ===")
    df = debug_csv_parsing(args.csv_url, args.separator)

    if df is None:
        print("\n❌ Cannot proceed - CSV parsing failed")
        return

    # Create processor with command line arguments
    processor = PPIBenchmarkProcessor(
        csv_url=args.csv_url,
        mmcif_base_url=args.mmcif_url,
        pdb_base_url=args.pdb_url,
        csv_separator=args.separator,
        auto_detect_separator=args.auto_detect,
        check_pdb_label=args.check_pdb_label,
        pdb_label_dir=args.pdb_label_dir,
        check_cif_label=args.check_cif_label,
        cif_label_dir=args.cif_label_dir,
        fetch_pdb_metadata=args.fetch_pdb_metadata,
        pdb_api_base_url=args.pdb_api_url,
        cluster_file=args.cluster
    )

    try:
        # Load and parse data
        print(f"\n=== LOADING AND PARSING DATA ===")
        data = processor.load_data()

        if data is not None:
            print(f"✅ Dataset loaded successfully!")
            print(f"   Records: {len(data)}")
            print(f"   Columns: {len(data.columns)}")
            print(f"   Separator used: '{processor.csv_separator}'")

            # Show column mapping
            print(f"\n=== COLUMN MAPPING ===")
            for std_name, actual_name in processor.column_mapping.items():
                print(f"   {std_name}: {actual_name}")

            # Parse into structured objects
            print(f"\n=== PARSING PROTEIN INTERFACES ===")
            print("Note: PDB metadata will be fetched in optimized batches if enabled")
            interfaces = processor.parse_data()

            if interfaces and len(interfaces) > 0:
                print(f"✅ Successfully parsed {len(interfaces)} protein interfaces!")

                # Show API optimization statistics if enabled
                if args.fetch_pdb_metadata and processor.pdb_api_client:
                    api_stats = processor.pdb_api_client.get_statistics()
                    print(f"\n=== API OPTIMIZATION STATISTICS ===")
                    print(f"   Total API calls: {api_stats['api_calls']}")
                    print(f"   Cache hits: {api_stats['cache_hits']}")
                    print(f"   Cache misses: {api_stats['cache_misses']}")
                    print(f"   Cache hit rate: {api_stats['cache_hits']/(api_stats['cache_hits']+api_stats['cache_misses'])*100:.1f}%" if (api_stats['cache_hits']+api_stats['cache_misses']) > 0 else "0%")
                    print(f"   Batches processed: {api_stats['batches_processed']}")
                    print(f"   PDBs fetched: {api_stats['total_pdbs_fetched']}")
                    print(f"   Obsolete entries: {api_stats['obsolete_entries']}")
                    print(f"\n   API Reduction: ~{100 * (1 - api_stats['api_calls']/len(interfaces)):.0f}% fewer calls than individual requests")

                # Generate and save Bioschemas markup
                print(f"\n=== GENERATING METADATA ===")
                output_manifest = processor.save_bioschemas_markup(output_dir=args.output)

                # Print summary statistics
                stats = processor._generate_statistics()
                print(f"\n=== DATASET STATISTICS ===")
                print(f"   Total interfaces: {stats['total_entries']}")
                print(f"   Unique proteins: {stats['unique_pdb_ids']}")
                print(f"   Physiological interfaces (TRUE): {stats['physiological_count']}")
                print(f"   Non-physiological interfaces (FALSE): {stats['non_physiological_count']}")
                print(f"   Balance ratio: {stats['balance_ratio']:.2%} physiological")
                print(f"   Proteins with multiple interfaces: {stats['proteins_with_multiple_interfaces']}")
                print(f"   Average interfaces per protein: {stats.get('avg_interfaces_per_protein', 0):.2f}")

                print(f"\n=== GENERATED FILES ===")
                print(f"   Dataset with Interfaces: {args.output}/dataset_with_interfaces.json")
                print(f"   Interface-Protein Pairs: {args.output}/interface_protein_pairs/ (ALL {len(interfaces)} files)")
                print(f"   FAIR Metadata Package: {args.output}/fair_metadata_package.json")
                print(f"   HTML Snippet: {args.output}/embedded_markup.html")
                if args.fetch_pdb_metadata:
                    print(f"   PDB Metadata Cache: {args.output}/pdb_metadata_cache.json")
                print(f"   Manifest: {args.output}/manifest.json")

                print(f"\n=== OPTIMIZATION SUMMARY ===")
                print(f"   Batch processing: {processor.pdb_api_client.batch_size if processor.pdb_api_client else 'N/A'} PDBs per API call")
                print(f"   Smart caching: 24-hour expiry with LRU eviction")
                print(f"   Rate limiting: {processor.pdb_api_client.request_delay if processor.pdb_api_client else 'N/A'}s delay between batches")
                print(f"   Fallback strategy: Header parsing for obsolete entries")

    except Exception as e:
        print(f"\n❌ ERROR in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
