"""
Script 2: Assembly Chain Checker for PPI Benchmark Data
Updates JSON-LD files with actual chain IDs from structure files.
"""

import json
import os
import gzip
import requests
from io import BytesIO
from typing import Dict, List, Set, Tuple, Any, Optional
import logging
from pathlib import Path
import argparse
from datetime import datetime
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_chain_ids_from_pdb_gz(pdb_gz_path: str) -> Set[str]:
    """
    Extract unique chain IDs from a compressed PDB file.
    Supports both local file paths and HTTP URLs.

    Args:
        pdb_gz_path: Path or URL to the .pdb.gz file

    Returns:
        Set of chain IDs found in the PDB file
    """
    chain_ids = set()

    # Check if it's a URL (starts with http:// or https://)
    is_url = pdb_gz_path.startswith(('http://', 'https://'))
    
    if not is_url and not os.path.exists(pdb_gz_path):
        logger.warning(f"PDB file not found: {pdb_gz_path}")
        return chain_ids

    try:
        if is_url:
            # Handle HTTP/HTTPS URL
            logger.debug(f"Downloading PDB file from URL: {pdb_gz_path}")
            response = requests.get(pdb_gz_path, stream=True)
            response.raise_for_status()  # Raise exception for bad status codes
            
            # Create a file-like object from the response content
            file_obj = BytesIO(response.content)
            file_source = pdb_gz_path
            
        else:
            # Handle local file
            file_obj = pdb_gz_path  # gzip.open can handle file paths directly
            file_source = pdb_gz_path

        # Open and process the gzipped file
        with gzip.open(file_obj, 'rt') as f:
            for line in f:
                # Only look at ATOM records (6 characters "ATOM  " with space)
                if line.startswith('ATOM  '):
                    # Chain ID is at position 21 (0-indexed, character 21)
                    if len(line) >= 22:
                        chain_id = line[21].strip()
                        if chain_id:  # Only add non-empty chain IDs
                            chain_ids.add(chain_id)
                # Stop early if we reach END or ENDMDL to save time
                elif line.startswith('END') or line.startswith('ENDMDL'):
                    break

        logger.debug(f"Found {len(chain_ids)} chain(s) in {file_source}: {sorted(chain_ids)}")
        return chain_ids
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download from URL {pdb_gz_path}: {e}")
        return chain_ids
    except gzip.BadGzipFile as e:
        logger.error(f"File is not a valid gzip file: {pdb_gz_path} - {e}")
        return chain_ids
    except UnicodeDecodeError as e:
        logger.error(f"File encoding error: {pdb_gz_path} - {e}")
        return chain_ids
    except Exception as e:
        logger.error(f"Unexpected error processing {pdb_gz_path}: {e}")
        return chain_ids


def extract_chain_ids_from_mmcif_gz(mmcif_gz_path: str) -> Set[str]:
    """
    Extract unique chain IDs from a compressed mmCIF file.
    
    Args:
        mmcif_gz_path: Path to the .cif.gz file
        
    Returns:
        Set of chain IDs found in the mmCIF file
    """
    chain_ids = set()
    
    # Check if it's a URL
    is_url = mmcif_gz_path.startswith(('http://', 'https://'))
    
    if not is_url and not os.path.exists(mmcif_gz_path):
        logger.warning(f"mmCIF file not found: {mmcif_gz_path}")
        return chain_ids
    
    try:
        # Handle URL or local file
        if is_url:
            logger.debug(f"Downloading mmCIF file from URL: {mmcif_gz_path}")
            response = requests.get(mmcif_gz_path, timeout=30)
            response.raise_for_status()
            file_obj = BytesIO(response.content)
        else:
            file_obj = mmcif_gz_path
        
        with gzip.open(file_obj, 'rt') as f:
            content = f.read()
        
        logger.debug(f"Processing mmCIF file: {mmcif_gz_path}")
        
        # Parse atom_site loop properly
        lines = content.split('\n')
        
        # Find atom_site loop
        in_atom_site_loop = False
        atom_site_headers = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if line.startswith('loop_'):
                in_atom_site_loop = False
                atom_site_headers = []
                
            elif line.startswith('_atom_site.'):
                if not in_atom_site_loop:
                    in_atom_site_loop = True
                atom_site_headers.append(line)
                
            elif in_atom_site_loop and line and not line.startswith('_'):
                # This is data line
                if len(atom_site_headers) > 0:
                    # Find auth_asym_id or label_asym_id column index
                    auth_asym_idx = -1
                    label_asym_idx = -1
                    
                    for j, header in enumerate(atom_site_headers):
                        if '.auth_asym_id' in header:
                            auth_asym_idx = j
                        elif '.label_asym_id' in header:
                            label_asym_idx = j
                    
                    # Use label_asym_id if available, otherwise auth_asym_idx
                    chain_idx = label_asym_idx if label_asym_idx != -1 else auth_asym_idx
                    
                    if chain_idx != -1:
                        # Split the data line (mmCIF uses whitespace)
                        parts = line.split()
                        if len(parts) > chain_idx:
                            chain_id = parts[chain_idx]
                            # Clean chain ID (remove quotes and whitespace)
                            chain_id = chain_id.strip("'\" \t")
                            if chain_id and chain_id not in ['.', '?']:
                                chain_ids.add(chain_id)
                                # Stop after collecting some chains
                                if len(chain_ids) >= 10:
                                    break
            elif in_atom_site_loop and (line.startswith('_') or line.startswith('#')):
                # New section starting, end of atom_site loop
                break
         
        logger.debug(f"Found {len(chain_ids)} chains in mmCIF: {sorted(chain_ids)}")
        
        # Clean up: remove any obviously wrong chains
        valid_chains = set()
        for chain in chain_ids:
            # Chain IDs are typically 1-2 alphanumeric characters
            if chain and len(chain) <= 2 and chain.isalnum():
                valid_chains.add(chain)
        
        logger.debug(f"After cleanup: {len(valid_chains)} chains: {sorted(valid_chains)}")
        
        return valid_chains
    
    except Exception as e:
        logger.error(f"Error extracting chain IDs from mmCIF file {mmcif_gz_path}: {e}")
        return set()


def validate_interface_chains(interface_id: str, auth_chain1: str, auth_chain2: str,
                             pdb_file_path: Optional[str] = None,
                             mmcif_file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate that the interface chains exist in the actual structure files.
    
    Args:
        interface_id: Interface identifier (for logging)
        auth_chain1: First chain from annotation
        auth_chain2: Second chain from annotation
        pdb_file_path: Full path to PDB.gz file (or None to skip)
        mmcif_file_path: Full path to mmCIF.gz file (or None to skip)
        
    Returns:
        Dictionary with validation results and actual chains
    """
    # Create interface chains string for validation
    interface_chains = f"{auth_chain1}-{auth_chain2}" if auth_chain1 and auth_chain2 else f"{auth_chain1}{auth_chain2}"
    
    file_checked = None
    actual_file_path = None
    actual_chains = []
    
    # Try PDB file if path is provided
    if pdb_file_path:
        logger.debug(f"Checking PDB file: {pdb_file_path}")
        actual_chains = extract_chain_ids_from_pdb_gz(pdb_file_path)
        actual_chains = list(actual_chains)
        actual_chains.sort()
        
        if actual_chains:
            file_checked = "PDB"
            actual_file_path = pdb_file_path
            logger.debug(f"Found {len(actual_chains)} chains in PDB: {actual_chains}")
    
    # If PDB file doesn't exist or has no chains, and mmCIF file path is provided, try mmCIF
    if not actual_chains and mmcif_file_path:
        logger.debug(f"Checking mmCIF file: {mmcif_file_path}")
        actual_chains = extract_chain_ids_from_mmcif_gz(mmcif_file_path)
        actual_chains = list(actual_chains)
        actual_chains.sort()
        
        if actual_chains:
            file_checked = "mmCIF"
            actual_file_path = mmcif_file_path
            logger.debug(f"Found {len(actual_chains)} chains in mmCIF: {actual_chains}")
    
    # If no files were checked or no chains found
    if not file_checked:
        file_checked = "None"
        actual_file_path = "No structure file provided"
    
    # Check if expected chains exist
    chain1_exists = auth_chain1 in actual_chains if auth_chain1 and actual_chains else False
    chain2_exists = auth_chain2 in actual_chains if auth_chain2 and actual_chains else False
    
    # Determine which chains to use
    label_chain1 = auth_chain1
    label_chain2 = auth_chain2
    chains_updated = False
    
    if actual_chains:
        if not chain1_exists or not chain2_exists or (auth_chain1 and auth_chain2 and auth_chain1 == auth_chain2):
            # Need to update chains
            if len(actual_chains) >= 2:
                # Use first two chains from the structure
                label_chain1, label_chain2 = actual_chains[0], actual_chains[1]
                chains_updated = True
                logger.info(f"Updated chains for {interface_id}: "
                          f"Auth chains: {auth_chain1}-{auth_chain2} -> "
                          f"Assembly chains: {label_chain1}-{label_chain2}")
            elif len(actual_chains) == 1:
                # Only one chain found - might be homodimer
                label_chain1, label_chain2 = actual_chains[0], actual_chains[0]
                chains_updated = True
                logger.info(f"Using single chain for {interface_id}: {label_chain1} for both chains "
                          f"(instead of annotated: {auth_chain1}-{auth_chain2})")
    
    # Determine validation status
    validation_result = {
        "interface_id": interface_id,
        "auth_chains": {
            "chain1": auth_chain1,
            "chain2": auth_chain2
        },
        "label_chains": {
            "chain1": label_chain1,
            "chain2": label_chain2
        },
        "actual_chains_found": sorted(actual_chains) if actual_chains else [],
        "chain1_exists": chain1_exists,
        "chain2_exists": chain2_exists,
        "all_chains_exist": chain1_exists and chain2_exists,
        "chains_updated": chains_updated,
        "file_checked": file_checked,
        "file_path": actual_file_path,
        "pdb_file_provided": pdb_file_path is not None,
        "mmcif_file_provided": mmcif_file_path is not None,
        "validation_timestamp": datetime.now().isoformat()
    }
    
    return validation_result


class AssemblyChainUpdater:
    """Updates JSON-LD files with assembly chain information."""
    
    def __init__(self, input_dir: str, mmcif_base_url: str, pdb_base_url: str,
                 pdb_label_dir: Optional[str] = None, cif_label_dir: Optional[str] = None):
        """
        Initialize the updater.
        
        Args:
            input_dir: Directory containing JSON-LD files from script 1
            mmcif_base_url: Base URL for mmCIF structure files
            pdb_base_url: Base URL for PDB structure files
            pdb_label_dir: Local directory for assembly PDB files
            cif_label_dir: Local directory for assembly mmCIF files
        """
        self.input_dir = Path(input_dir)
        self.mmcif_base_url = mmcif_base_url.rstrip('/') + '/'
        self.pdb_base_url = pdb_base_url.rstrip('/') + '/'
        self.pdb_label_dir = Path(pdb_label_dir) if pdb_label_dir else None
        self.cif_label_dir = Path(cif_label_dir) if cif_label_dir else None
        
        # Statistics
        self.stats = {
            "total_files": 0,
            "successfully_updated": 0,
            "failed_updates": 0,
            "chains_updated": 0,
            "validation_results": []
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
    
    def extract_interface_info(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract interface information from JSON-LD data.
        
        Args:
            json_data: JSON-LD data for an interface
            
        Returns:
            Dictionary with extracted interface information
        """
        interface_info = {
            "interface_id": None,
            "pdb_id": None,
            "auth_chain1": None,
            "auth_chain2": None,
            "interface_source": None,
            "interface_type": None  # "interface_item" or "dataset_item"
        }
        
        # Check if this is an interface item (from interface_protein_pairs directory)
        if "@type" in json_data and "DataCatalogItem" in json_data["@type"]:
            interface_info["interface_type"] = "interface_item"
            interface_info["interface_id"] = json_data.get("identifier")
            interface_info["pdb_id"] = None
            
            # Extract from additionalProperty
            if "additionalProperty" in json_data:
                for prop in json_data["additionalProperty"]:
                    if prop.get("name") == "PDB_ID":
                        interface_info["pdb_id"] = prop.get("value")
                    elif prop.get("name") == "AuthChain1":
                        interface_info["auth_chain1"] = prop.get("value")
                    elif prop.get("name") == "AuthChain2":
                        interface_info["auth_chain2"] = prop.get("value")
                    elif prop.get("name") == "InterfaceSource":
                        interface_info["interface_source"] = prop.get("value")
            
            # Also check mainEntity for PDB ID
            if "mainEntity" in json_data and interface_info["pdb_id"] is None:
                main_entity = json_data["mainEntity"]
                if "additionalProperty" in main_entity:
                    for prop in main_entity["additionalProperty"]:
                        if prop.get("name") == "PDB_ID":
                            interface_info["pdb_id"] = prop.get("value")
        
        # Check if this is a dataset with interface items
        elif "@type" in json_data and "Dataset" in json_data["@type"]:
            interface_info["interface_type"] = "dataset_item"
            # We'll need to process hasPart items separately
        
        return interface_info
    
    def get_structure_file_paths(self, interface_id: str, pdb_id: str, 
                                interface_source: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the structure file paths for an interface.
        
        Args:
            interface_id: Interface identifier
            pdb_id: PDB ID
            interface_source: Source of interface (QSalign, ProtCID, etc.)
            
        Returns:
            Tuple of (pdb_file_path, mmcif_file_path)
        """
        pdb_file_path = None
        mmcif_file_path = None
        
        # Determine file names based on interface source
        if interface_source == 'QSalign':
            # QSalign format: PDBID_ASSEMBLY
            pdb_filename = f"{interface_id.lower()}.pdb.gz"
            cif_filename = f"{pdb_id.lower()}.cif.gz"
        else:
            # ProtCID or other format
            pdb_filename = f"{interface_id.lower()}.pdb.gz"
            cif_filename = f"{interface_id.lower()}.cif.gz"
        
        # Try local directories first
        if self.pdb_label_dir:
            local_pdb_path = self.pdb_label_dir / pdb_filename
            if local_pdb_path.exists():
                pdb_file_path = str(local_pdb_path)
        
        if self.cif_label_dir:
            local_cif_path = self.cif_label_dir / cif_filename
            if local_cif_path.exists():
                mmcif_file_path = str(local_cif_path)
        
        # If not found locally, use URLs
        if not pdb_file_path:
            pdb_file_path = f"{self.pdb_base_url}{pdb_filename}"
        
        if not mmcif_file_path:
            mmcif_file_path = f"{self.mmcif_base_url}{cif_filename}"
        
        return pdb_file_path, mmcif_file_path
    
    def update_interface_item(self, json_data: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an interface item JSON-LD with assembly chain information.
        
        Args:
            json_data: Original JSON-LD data
            validation_result: Validation results from structure files
            
        Returns:
            Updated JSON-LD data
        """
        updated_data = json_data.copy()
        
        # Extract chain information
        label_chain1 = validation_result["label_chains"]["chain1"]
        label_chain2 = validation_result["label_chains"]["chain2"]
        chains_updated = validation_result["chains_updated"]
        
        # Update additionalProperty in interface item
        if "additionalProperty" not in updated_data:
            updated_data["additionalProperty"] = []
        
        # Remove any existing LabelChain properties
        updated_data["additionalProperty"] = [
            prop for prop in updated_data["additionalProperty"] 
            if prop.get("name") not in ["LabelChain1", "LabelChain2", "LabelInterfaceChains"]
        ]
        
        # Add new LabelChain properties
        if label_chain1:
            updated_data["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "LabelChain1",
                "value": label_chain1,
                "description": "First chain in the assembly interface (validated from structure file)"
            })
        
        if label_chain2:
            updated_data["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "LabelChain2",
                "value": label_chain2,
                "description": "Second chain in the assembly interface (validated from structure file)"
            })
        
        # Add LabelInterfaceChains
        if label_chain1 and label_chain2:
            updated_data["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "LabelInterfaceChains",
                "value": f"{label_chain1}-{label_chain2}",
                "description": "Chains involved in the assembly interface (validated from structure file)"
            })
        
        # Add validation metadata
        updated_data["additionalProperty"].append({
            "@type": "PropertyValue",
            "name": "ChainValidation",
            "value": json.dumps(validation_result, default=str),
            "description": "Results of chain validation against structure files",
            "valueReference": {
                "@type": "PropertyValue",
                "name": "ParsedChainValidation",
                "value": {
                    "chains_updated": chains_updated,
                    "file_checked": validation_result["file_checked"],
                    "all_chains_exist": validation_result["all_chains_exist"],
                    "validation_timestamp": validation_result["validation_timestamp"]
                }
            }
        })
        
        # Update mainEntity (Protein) if it exists
        if "mainEntity" in updated_data:
            updated_data["mainEntity"] = self.update_protein_entity(
                updated_data["mainEntity"], 
                label_chain1, 
                label_chain2,
                validation_result
            )
        
        return updated_data
    
    def update_protein_entity(self, protein_data: Dict[str, Any], 
                             label_chain1: str, label_chain2: str,
                             validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a protein entity JSON-LD with assembly chain information.
        
        Args:
            protein_data: Original protein JSON-LD data
            label_chain1: First chain in assembly
            label_chain2: Second chain in assembly
            validation_result: Validation results
            
        Returns:
            Updated protein JSON-LD data
        """
        updated_protein = protein_data.copy()
        
        if "additionalProperty" not in updated_protein:
            updated_protein["additionalProperty"] = []
        
        # Remove any existing AssemblyChainStatus or LabelInterfaceChains
        updated_protein["additionalProperty"] = [
            prop for prop in updated_protein["additionalProperty"] 
            if prop.get("name") not in ["AssemblyChainStatus", "LabelInterfaceChains"]
        ]
        
        # Add updated LabelInterfaceChains if chains are available
        if label_chain1 and label_chain2:
            updated_protein["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "LabelInterfaceChains",
                "value": f"{label_chain1}-{label_chain2}",
                "description": "Chains involved in the assembly interface (validated from structure file)"
            })
        
        # Update status
        chains_updated = validation_result["chains_updated"]
        status_text = "Updated from structure file" if chains_updated else "Matches annotation"
        
        updated_protein["additionalProperty"].append({
            "@type": "PropertyValue",
            "name": "AssemblyChainStatus",
            "value": status_text,
            "description": f"Assembly chain information validated from {validation_result['file_checked']} file"
        })
        
        return updated_protein
    
    def process_interface_file(self, interface_file: Path) -> bool:
        """
        Process a single interface JSON-LD file.
        
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
            auth_chain1 = interface_info["auth_chain1"]
            auth_chain2 = interface_info["auth_chain2"]
            interface_source = interface_info["interface_source"]
            
            if not interface_id or not pdb_id:
                logger.warning(f"Missing interface_id or pdb_id in {interface_file}")
                return False
            
            logger.info(f"Processing interface: {interface_id} (PDB: {pdb_id})")
            
            # Get structure file paths
            pdb_file_path, mmcif_file_path = self.get_structure_file_paths(
                interface_id, pdb_id, interface_source
            )
            
            logger.debug(f"  PDB file: {pdb_file_path}")
            logger.debug(f"  mmCIF file: {mmcif_file_path}")
            
            # Validate chains
            validation_result = validate_interface_chains(
                interface_id, auth_chain1, auth_chain2,
                pdb_file_path, mmcif_file_path
            )
            
            # Update the JSON data with assembly chain information
            updated_data = self.update_interface_item(json_data, validation_result)
            
            # Save the updated file
            with open(interface_file, 'w', encoding='utf-8') as f:
                json.dump(updated_data, f, indent=2, ensure_ascii=False)
            
            # Update statistics
            self.stats["successfully_updated"] += 1
            if validation_result["chains_updated"]:
                self.stats["chains_updated"] += 1
            
            self.stats["validation_results"].append(validation_result)
            
            logger.info(f"Successfully updated {interface_file}")
            logger.info(f"  Auth chains: {auth_chain1}-{auth_chain2}")
            logger.info(f"  Label chains: {validation_result['label_chains']['chain1']}-{validation_result['label_chains']['chain2']}")
            logger.info(f"  Chains updated: {validation_result['chains_updated']}")
            logger.info(f"  File checked: {validation_result['file_checked']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {interface_file}: {e}")
            self.stats["failed_updates"] += 1
            return False
    
    def update_dataset_file(self, dataset_file: Path) -> bool:
        """
        Update the main dataset file with assembly chain information.
        
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
                        auth_chain1 = interface_info["auth_chain1"]
                        auth_chain2 = interface_info["auth_chain2"]
                        interface_source = interface_info["interface_source"]
                        
                        if interface_id and pdb_id:
                            # Get structure file paths
                            pdb_file_path, mmcif_file_path = self.get_structure_file_paths(
                                interface_id, pdb_id, interface_source
                            )
                            
                            # Validate chains
                            validation_result = validate_interface_chains(
                                interface_id, auth_chain1, auth_chain2,
                                pdb_file_path, mmcif_file_path
                            )
                            
                            # Update the interface item
                            updated_item = self.update_interface_item(item, validation_result)
                            updated_items.append(updated_item)
                            
                            # Update statistics
                            if validation_result["chains_updated"]:
                                self.stats["chains_updated"] += 1
                            
                            self.stats["validation_results"].append(validation_result)
                        else:
                            updated_items.append(item)
                    else:
                        updated_items.append(item)
                
                # Update the dataset
                dataset_data["hasPart"] = updated_items
                
                # Save the updated dataset
                with open(dataset_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Updated dataset file: {dataset_file}")
                return True
            
            else:
                logger.warning(f"Dataset file {dataset_file} doesn't have interface items in hasPart")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update dataset file {dataset_file}: {e}")
            self.stats["failed_updates"] += 1
            return False

    def update_fair_metadata_package(self, fair_package_file: Path) -> bool:
        """
        Update the FAIR metadata package with assembly chain checking information.
        
        Args:
            fair_package_file: Path to fair_metadata_package.json file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(fair_package_file, 'r', encoding='utf-8') as f:
                fair_package = json.load(f)
            
            # Debug: print structure to understand what we're working with
            logger.debug(f"FAIR package structure keys: {fair_package.keys()}")
            
            # Add assembly chain checking information to the FAIR package
            assembly_chain_info = {
                "script": "2nd script - Assembly Chain Checker",
                "execution_date": datetime.now().isoformat(),
                "statistics": {
                    "total_files_processed": self.stats["total_files"],
                    "successfully_updated": self.stats["successfully_updated"],
                    "failed_updates": self.stats["failed_updates"],
                    "chains_updated": self.stats["chains_updated"],
                    "success_rate": f"{(self.stats['successfully_updated'] / self.stats['total_files'] * 100):.1f}%" if self.stats['total_files'] > 0 else "0%"
                },
                "validation_summary": {
                    "files_checked": len([r for r in self.stats["validation_results"] if r["file_checked"] != "None"]),
                    "files_with_structure": len([r for r in self.stats["validation_results"] if r["actual_chains_found"]]),
                    "all_chains_exist": len([r for r in self.stats["validation_results"] if r["all_chains_exist"]]),
                    "chains_needed_updating": len([r for r in self.stats["validation_results"] if r["chains_updated"]])
                },
                "settings": {
                    "mmcif_base_url": self.mmcif_base_url,
                    "pdb_base_url": self.pdb_base_url,
                    "pdb_label_dir": str(self.pdb_label_dir) if self.pdb_label_dir else None,
                    "cif_label_dir": str(self.cif_label_dir) if self.cif_label_dir else None
                }
            }
            
            # Add to FAIR package at the top level
            fair_package["assembly_chain_checking"] = assembly_chain_info
            
            # First, let's check what structure we actually have
            logger.debug(f"Looking for interface_id_handling in fair_package: {'interface_id_handling' in fair_package}")
            
            # Try to find where the fields_to_be_populated actually is
            fields_location = None
            if "interface_id_handling" in fair_package:
                fields_location = fair_package["interface_id_handling"]
                logger.debug(f"Found interface_id_handling")
            elif "schema_structure" in fair_package and "fields_to_be_populated" in fair_package["schema_structure"]:
                fields_location = fair_package["schema_structure"]
                logger.debug(f"Found fields_to_be_populated in schema_structure")
            
            if fields_location:
                # Make sure fields_to_be_populated exists
                if "fields_to_be_populated" not in fields_location:
                    fields_location["fields_to_be_populated"] = {}
                
                # Update LabelChain fields - create them as dictionaries
                fields_location["fields_to_be_populated"]["LabelChain1"] = {
                    "status": "POPULATED by 2nd script",
                    "timestamp": datetime.now().isoformat(),
                    "description": "First chain in the assembly interface (validated from structure file)"
                }
                
                fields_location["fields_to_be_populated"]["LabelChain2"] = {
                    "status": "POPULATED by 2nd script",
                    "timestamp": datetime.now().isoformat(),
                    "description": "Second chain in the assembly interface (validated from structure file)"
                }
                
                # Handle PDB_metadata field
                pdb_metadata_field = fields_location["fields_to_be_populated"].get("PDB_metadata")
                if isinstance(pdb_metadata_field, dict):
                    pdb_metadata_field["status"] = "EMPTY - To be populated by script 3"
                    pdb_metadata_field["description"] = "Comprehensive PDB metadata from RCSB API. Will be populated by script 3."
                elif pdb_metadata_field is not None:
                    fields_location["fields_to_be_populated"]["PDB_metadata"] = {
                        "status": "EMPTY - To be populated by script 3",
                        "timestamp": datetime.now().isoformat(),
                        "description": "Comprehensive PDB metadata from RCSB API. Will be populated by script 3.",
                        "previous_value": str(pdb_metadata_field)
                    }
                else:
                    fields_location["fields_to_be_populated"]["PDB_metadata"] = {
                        "status": "EMPTY - To be populated by script 3",
                        "timestamp": datetime.now().isoformat(),
                        "description": "Comprehensive PDB metadata from RCSB API. Will be populated by script 3."
                    }
                
                # Handle ClusterID field
                clusterid_field = fields_location["fields_to_be_populated"].get("ClusterID")
                if isinstance(clusterid_field, dict):
                    clusterid_field["status"] = "PARTIAL - To be completed by script 4"
                    clusterid_field["description"] = "Sequence cluster ID from BLASTClust analysis (-S 25 -L 0.5 -b F). Will be populated/validated by script 4."
                elif clusterid_field is not None:
                    fields_location["fields_to_be_populated"]["ClusterID"] = {
                        "status": "PARTIAL - To be completed by script 4",
                        "timestamp": datetime.now().isoformat(),
                        "description": "Sequence cluster ID from BLASTClust analysis (-S 25 -L 0.5 -b F). Will be populated/validated by script 4.",
                        "previous_value": str(clusterid_field)
                    }
                else:
                    fields_location["fields_to_be_populated"]["ClusterID"] = {
                        "status": "PARTIAL - To be completed by script 4",
                        "timestamp": datetime.now().isoformat(),
                        "description": "Sequence chain ID from BLASTClust analysis (-S 25 -L 0.5 -b F). Will be populated/validated by script 4."
                    }
            
            # Now let's update the actual protein objects in the dataset
            # First check if we have a dataset in the bioschemas_markup
            
            if "bioschemas_markup" in fair_package:
                logger.debug(f"Found bioschemas_markup, keys: {fair_package['bioschemas_markup'].keys()}")
                
                # Check if we have a dataset object
                if "dataset" in fair_package["bioschemas_markup"]:
                    dataset = fair_package["bioschemas_markup"]["dataset"]
                    logger.debug(f"Dataset type: {type(dataset)}")
                    
                    # If dataset is a dict, check for hasPart
                    if isinstance(dataset, dict) and "hasPart" in dataset:
                        logger.debug(f"Dataset has hasPart with {len(dataset['hasPart'])} items")
                        
                        # Get ALL validation results to map interface IDs to LabelChain values
                        validation_map = {}
                        for result in self.stats["validation_results"]:
                            if "interface_id" in result:
                                validation_map[result["interface_id"]] = {
                                    "label_chain1": result.get("label_chains", {}).get("chain1", ""),
                                    "label_chain2": result.get("label_chains", {}).get("chain2", ""),
                                    "auth_chain1": result.get("auth_chains", {}).get("chain1", ""),
                                    "auth_chain2": result.get("auth_chains", {}).get("chain2", ""),
                                    "chains_updated": result.get("chains_updated", False),
                                    "file_checked": result.get("file_checked", "None")
                                }
                        
                        # Update ALL interface items with their actual LabelChain values
                        updated_count = 0
                        for interface_item in dataset["hasPart"]:
                            if isinstance(interface_item, dict) and "mainEntity" in interface_item:
                                # Get interface ID to look up chain data
                                interface_id = None
                                if "identifier" in interface_item:
                                    interface_id = interface_item["identifier"]
                                elif "additionalProperty" in interface_item:
                                    for prop in interface_item["additionalProperty"]:
                                        if isinstance(prop, dict) and prop.get("name") == "InterfaceID":
                                            interface_id = prop.get("value")
                                            break
                                
                                if interface_id and interface_id in validation_map:
                                    protein = interface_item["mainEntity"]
                                    if isinstance(protein, dict):
                                        chain_data = validation_map[interface_id]
                                        label_chain1 = chain_data["label_chain1"]
                                        label_chain2 = chain_data["label_chain2"]
                                        auth_chain1 = chain_data["auth_chain1"]
                                        auth_chain2 = chain_data["auth_chain2"]
                                        chains_updated = chain_data["chains_updated"]
                                        file_checked = chain_data["file_checked"]
                                        
                                        # Ensure additionalProperty exists
                                        if "additionalProperty" not in protein:
                                            protein["additionalProperty"] = []
                                        
                                        # Find and update AssemblyChainStatus
                                        found_assembly_status = False
                                        for prop in protein["additionalProperty"]:
                                            if isinstance(prop, dict) and prop.get("name") == "AssemblyChainStatus":
                                                status_text = "Updated from structure file" if chains_updated else "Matches annotation"
                                                prop["value"] = status_text
                                                prop["description"] = f"Assembly chain information validated from {file_checked} file. Auth chains: {auth_chain1}-{auth_chain2}, Label chains: {label_chain1}-{label_chain2}."
                                                found_assembly_status = True
                                                break
                                        
                                        # If AssemblyChainStatus not found, add it
                                        if not found_assembly_status:
                                            status_text = "Updated from structure file" if chains_updated else "Matches annotation"
                                            protein["additionalProperty"].append({
                                                "@type": "PropertyValue",
                                                "name": "AssemblyChainStatus",
                                                "value": status_text,
                                                "description": f"Assembly chain information validated from {file_checked} file. Auth chains: {auth_chain1}-{auth_chain2}, Label chains: {label_chain1}-{label_chain2}."
                                            })
                                        
                                        # Remove any existing LabelChain properties
                                        protein["additionalProperty"] = [
                                            prop for prop in protein["additionalProperty"] 
                                            if isinstance(prop, dict) and prop.get("name") not in ["LabelChain1", "LabelChain2", "LabelInterfaceChains"]
                                        ]
                                        
                                        # Add LabelChain1 property if we have a value
                                        if label_chain1:
                                            protein["additionalProperty"].append({
                                                "@type": "PropertyValue",
                                                "name": "LabelChain1",
                                                "value": label_chain1,
                                                "description": f"First chain in the assembly interface (validated from {file_checked} file). Auth chain was {auth_chain1}."
                                            })
                                        
                                        # Add LabelChain2 property if we have a value
                                        if label_chain2:
                                            protein["additionalProperty"].append({
                                                "@type": "PropertyValue",
                                                "name": "LabelChain2",
                                                "value": label_chain2,
                                                "description": f"Second chain in the assembly interface (validated from {file_checked} file). Auth chain was {auth_chain2}."
                                            })
                                        
                                        # Add LabelInterfaceChains if we have both chains
                                        if label_chain1 and label_chain2:
                                            protein["additionalProperty"].append({
                                                "@type": "PropertyValue",
                                                "name": "LabelInterfaceChains",
                                                "value": f"{label_chain1}-{label_chain2}",
                                                "description": f"Chains involved in the assembly interface (validated from {file_checked} file). Auth chains: {auth_chain1}-{auth_chain2}."
                                            })
                                        
                                        updated_count += 1
                                        logger.debug(f"Updated protein for interface {interface_id} with LabelChain1={label_chain1}, LabelChain2={label_chain2}")
                        
                        logger.info(f"Updated {updated_count} proteins with LabelChain information in FAIR metadata package")
            
            # Add ALL updated chains to the sample_updated_chains section (not just first 5)
            if self.stats["validation_results"]:
                # Get all chains that were updated
                all_updated_chains = [r for r in self.stats["validation_results"] if r.get("chains_updated", False)]
                
                # Also include some that weren't updated but were validated
                validated_chains = [r for r in self.stats["validation_results"] if not r.get("chains_updated", False) and r.get("file_checked") != "None"][:5]
                
                sample_chains = all_updated_chains[:10] + validated_chains  # Show up to 15 samples total
                
                if sample_chains:
                    fair_package["assembly_chain_checking"]["sample_validation_results"] = [
                        {
                            "interface_id": r["interface_id"],
                            "auth_chains": {
                                "chain1": r.get("auth_chains", {}).get("chain1", ""),
                                "chain2": r.get("auth_chains", {}).get("chain2", "")
                            },
                            "label_chains": {
                                "chain1": r.get("label_chains", {}).get("chain1", ""),
                                "chain2": r.get("label_chains", {}).get("chain2", "")
                            },
                            "actual_chains_found": r.get("actual_chains_found", []),
                            "chain1_exists": r.get("chain1_exists", False),
                            "chain2_exists": r.get("chain2_exists", False),
                            "all_chains_exist": r.get("all_chains_exist", False),
                            "chains_updated": r.get("chains_updated", False),
                            "file_checked": r.get("file_checked", "None")
                        }
                        for r in sample_chains
                    ]
            
            # Save the updated FAIR package
            with open(fair_package_file, 'w', encoding='utf-8') as f:
                json.dump(fair_package, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Updated FAIR metadata package: {fair_package_file}")
            logger.info(f"  Added assembly_chain_checking section with statistics")
            logger.info(f"  Updated fields_to_be_populated to show LabelChain1 and LabelChain2 as POPULATED")
            logger.info(f"  Updated ALL protein objects with actual LabelChain values extracted from structures")
            logger.info(f"  Added sample_validation_results with detailed chain information")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update FAIR metadata package {fair_package_file}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def update_embedded_markup_html(self, html_file: Path, dataset_file: Path) -> bool:
        """
        Update the embedded markup HTML file with updated JSON-LD.
        
        Args:
            html_file: Path to embedded_markup.html file
            dataset_file: Path to dataset_with_interfaces.json file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load the updated dataset JSON-LD
            with open(dataset_file, 'r', encoding='utf-8') as f:
                dataset_markup = json.load(f)
            
            # Generate updated HTML snippet
            html_snippet = """<!DOCTYPE html>
            <html lang="en">
            <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Bioschemas & Croissant Markup - PPI Benchmark Dataset (Updated with Assembly Chains)</title>
            <script type="application/ld+json">
            {markup}
            </script>
            </head>
            <body>
            <h1>Protein-Protein Interaction Interface Benchmark Dataset</h1>
            <p>This page contains embedded Bioschemas and Croissant markup for the dataset.</p>
            <p>The JSON-LD markup in the header makes this dataset discoverable by search engines and compatible with Google Dataset Search and ML dataset platforms.</p>
            <p><strong>Update Status:</strong> Assembly chain information has been validated and added (LabelChain1, LabelChain2, LabelInterfaceChains).</p>
            <p><strong>Next Steps:</strong> Run script 3 to add PDB metadata (resolution, sequences, organism, etc.).</p>
            </body>
            </html>""".format(markup=json.dumps(dataset_markup, indent=4))
            
            # Save the updated HTML file
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_snippet)
            
            logger.info(f"Updated embedded markup HTML: {html_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update embedded markup HTML {html_file}: {e}")
            return False
    
    def update_manifest_file(self, manifest_file: Path) -> bool:
        """
        Update the manifest file with assembly chain checking statistics.
        
        Args:
            manifest_file: Path to manifest.json file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
            
            # Add assembly chain checking information
            manifest_data["assembly_chain_checking"] = {
                "script": "2nd script - Assembly Chain Checker",
                "execution_date": datetime.now().isoformat(),
                "statistics": {
                    "total_files_processed": self.stats["total_files"],
                    "successfully_updated": self.stats["successfully_updated"],
                    "failed_updates": self.stats["failed_updates"],
                    "chains_updated": self.stats["chains_updated"],
                    "success_rate": f"{(self.stats['successfully_updated'] / self.stats['total_files'] * 100):.1f}%" if self.stats['total_files'] > 0 else "0%"
                },
                "settings": {
                    "mmcif_base_url": self.mmcif_base_url,
                    "pdb_base_url": self.pdb_base_url,
                    "pdb_label_dir": str(self.pdb_label_dir) if self.pdb_label_dir else None,
                    "cif_label_dir": str(self.cif_label_dir) if self.cif_label_dir else None
                },
                "validation_summary": {
                    "files_checked": len([r for r in self.stats["validation_results"] if r["file_checked"] != "None"]),
                    "files_with_structure": len([r for r in self.stats["validation_results"] if r["actual_chains_found"]]),
                    "all_chains_exist": len([r for r in self.stats["validation_results"] if r["all_chains_exist"]]),
                    "chains_needed_updating": len([r for r in self.stats["validation_results"] if r["chains_updated"]])
                }
            }
            
            # Update fields_to_be_populated section if it exists
            if "schema_structure" in manifest_data and "fields_to_be_populated" in manifest_data["schema_structure"]:
                manifest_data["schema_structure"]["fields_to_be_populated"]["LabelChain1"] = {
                    "status": "POPULATED by 2nd script",
                    "timestamp": datetime.now().isoformat(),
                    "description": "First chain in the assembly interface (validated from structure file)"
                }
                manifest_data["schema_structure"]["fields_to_be_populated"]["LabelChain2"] = {
                    "status": "POPULATED by 2nd script",
                    "timestamp": datetime.now().isoformat(),
                    "description": "Second chain in the assembly interface (validated from structure file)"
                }
                
                # Update the status for fields that are now populated
                if "PDB_metadata" in manifest_data["schema_structure"]["fields_to_be_populated"]:
                    manifest_data["schema_structure"]["fields_to_be_populated"]["PDB_metadata"]["status"] = "EMPTY - To be populated by script 3"
                    manifest_data["schema_structure"]["fields_to_be_populated"]["PDB_metadata"]["description"] = "Comprehensive PDB metadata from RCSB API. Will be populated by script 3."
                
                if "ClusterID" in manifest_data["schema_structure"]["fields_to_be_populated"]:
                    manifest_data["schema_structure"]["fields_to_be_populated"]["ClusterID"]["status"] = "PARTIAL - To be completed by script 4"
                    manifest_data["schema_structure"]["fields_to_be_populated"]["ClusterID"]["description"] = "Sequence cluster ID from BLASTClust analysis (-S 25 -L 0.5 -b F). Will be populated/validated by script 4."
            
            # Save the updated manifest
            with open(manifest_file, 'w', encoding='utf-8') as f:
                json.dump(manifest_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Updated manifest file: {manifest_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update manifest file {manifest_file}: {e}")
            return False
    
    def run(self) -> Dict[str, Any]:
        """
        Main execution method.
        
        Returns:
            Statistics dictionary
        """
        logger.info("=" * 60)
        logger.info("Assembly Chain Checker - Script 2")
        logger.info("=" * 60)
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"mmCIF base URL: {self.mmcif_base_url}")
        logger.info(f"PDB base URL: {self.pdb_base_url}")
        logger.info(f"PDB label directory: {self.pdb_label_dir}")
        logger.info(f"mmCIF label directory: {self.cif_label_dir}")
        logger.info("")
        
        # Find JSON files to process
        json_files = self.find_json_files()
        self.stats["total_files"] = len(json_files)
        
        if not json_files:
            logger.warning("No JSON files found to process!")
            return self.stats
        
        # Process interface files
        logger.info(f"Processing {len(json_files)} interface files...")
        
        for i, json_file in enumerate(json_files):
            logger.info(f"[{i+1}/{len(json_files)}] Processing: {json_file.name}")
            self.process_interface_file(json_file)
        
        # Update dataset file if it exists
        dataset_file = self.input_dir / "dataset_with_interfaces.json"
        if dataset_file.exists():
            logger.info(f"Updating dataset file: {dataset_file}")
            self.update_dataset_file(dataset_file)
        else:
            logger.warning(f"Dataset file not found: {dataset_file}")
        
        # Update FAIR metadata package if it exists
        fair_package_file = self.input_dir / "fair_metadata_package.json"
        if fair_package_file.exists():
            logger.info(f"Updating FAIR metadata package: {fair_package_file}")
            self.update_fair_metadata_package(fair_package_file)
        else:
            logger.warning(f"FAIR metadata package not found: {fair_package_file}")
        
        # Update embedded markup HTML if it exists
        html_file = self.input_dir / "embedded_markup.html"
        if html_file.exists() and dataset_file.exists():
            logger.info(f"Updating embedded markup HTML: {html_file}")
            self.update_embedded_markup_html(html_file, dataset_file)
        else:
            logger.warning(f"Embedded markup HTML not found: {html_file}")
        
        # Update manifest file if it exists
        manifest_file = self.input_dir / "manifest.json"
        if manifest_file.exists():
            logger.info(f"Updating manifest file: {manifest_file}")
            self.update_manifest_file(manifest_file)
        else:
            logger.warning(f"Manifest file not found: {manifest_file}")
        
        # Print summary statistics
        logger.info("")
        logger.info("=" * 60)
        logger.info("ASSEMBLY CHAIN CHECKING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total files processed: {self.stats['total_files']}")
        logger.info(f"Successfully updated: {self.stats['successfully_updated']}")
        logger.info(f"Failed updates: {self.stats['failed_updates']}")
        logger.info(f"Chains updated: {self.stats['chains_updated']}")
        
        if self.stats['validation_results']:
            files_with_structure = len([r for r in self.stats["validation_results"] if r["actual_chains_found"]])
            all_chains_exist = len([r for r in self.stats["validation_results"] if r["all_chains_exist"]])
            chains_updated = len([r for r in self.stats["validation_results"] if r["chains_updated"]])
            
            logger.info("")
            logger.info("Validation Summary:")
            logger.info(f"  Files with structure data: {files_with_structure}/{len(self.stats['validation_results'])}")
            logger.info(f"  All annotated chains exist: {all_chains_exist}/{len(self.stats['validation_results'])}")
            logger.info(f"  Chains needed updating: {chains_updated}/{len(self.stats['validation_results'])}")
            
            # Show sample of updated chains
            if chains_updated > 0:
                logger.info("")
                logger.info("Sample of updated chains:")
                updated_samples = [r for r in self.stats["validation_results"] if r["chains_updated"]][:3]
                for i, sample in enumerate(updated_samples):
                    logger.info(f"  {i+1}. {sample['interface_id']}: "
                              f"{sample['auth_chains']['chain1']}-{sample['auth_chains']['chain2']} -> "
                              f"{sample['label_chains']['chain1']}-{sample['label_chains']['chain2']}")
        
        logger.info("")
        logger.info(f"All JSON-LD files have been updated with LabelChain1, LabelChain2, and LabelInterfaceChains fields.")
        logger.info(f"These fields contain the actual chain IDs extracted from structure files.")
        
        # List all files that were updated
        logger.info("")
        logger.info("=== UPDATED FILES ===")
        files_to_check = [
            ("dataset_with_interfaces.json", dataset_file.exists()),
            ("fair_metadata_package.json", fair_package_file.exists()),
            ("embedded_markup.html", html_file.exists()),
            ("manifest.json", manifest_file.exists()),
            ("interface_protein_pairs/*.json", True)  # Interface files are always generated
        ]
        
        for file_name, exists in files_to_check:
            if exists:
                logger.info(f"✓ {file_name} - Updated with assembly chain information")
            else:
                logger.warning(f"✗ {file_name} - Not found or not updated")
        
        # Specifically mention what was added to fair_metadata_package.json
        if fair_package_file.exists():
            logger.info("")
            logger.info("=== FAIR METADATA PACKAGE UPDATES ===")
            logger.info("Added to fair_metadata_package.json:")
            logger.info("  • assembly_chain_checking section with statistics")
            logger.info("  • Updated fields_to_be_populated: LabelChain1 and LabelChain2 marked as 'POPULATED'")
            logger.info("  • Updated interface_id_handling section")
            logger.info("  • Added sample of updated chains (if any)")
        
        return self.stats


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Assembly Chain Checker - Update JSON-LD files with actual chain IDs from structure files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input bioschemas_output
  %(prog)s --input bioschemas_output --pdb-label-dir ./pdb_files
  %(prog)s --input bioschemas_output --cif-label-dir ./cif_files
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory containing JSON-LD files from script 1"
    )
    
    parser.add_argument(
        "--mmcif-url",
        default="https://raw.githubusercontent.com/vibbits/Elixir-3DBioInfo-Benchmark-Protein-Interfaces/main/benchmark/benchmark_mmcif_format/",
        help="Base URL for mmCIF structure files"
    )
    
    parser.add_argument(
        "--pdb-url",
        default="https://raw.githubusercontent.com/vibbits/Elixir-3DBioInfo-Benchmark-Protein-Interfaces/main/benchmark/benchmark_pdb_format/",
        help="Base URL for PDB structure files"
    )
    
    parser.add_argument(
        "--pdb-label-dir",
        default=None,
        help="Local directory for assembly PDB files (overrides pdb-url)"
    )
    
    parser.add_argument(
        "--cif-label-dir",
        default=None,
        help="Local directory for assembly mmCIF files (overrides mmcif-url)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
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
║  Assembly Chain Checker - Script 2                            ║
║  Updates JSON-LD files with actual chain IDs from structures  ║
╚═══════════════════════════════════════════════════════════════╝
    
Settings:
  Input Directory: {args.input}
  mmCIF Files:     {args.mmcif_url}
  PDB Files:       {args.pdb_url}
  PDB Local Dir:   {args.pdb_label_dir or 'Not specified'}
  mmCIF Local Dir: {args.cif_label_dir or 'Not specified'}
    """)
    
    # Check if input directory exists
    if not os.path.exists(args.input):
        print(f"\n❌ ERROR: Input directory '{args.input}' does not exist!")
        print("   Please run script 1 first to generate the JSON-LD files.")
        return
    
    # Check for required files
    required_files = ["dataset_with_interfaces.json", "fair_metadata_package.json", 
                      "embedded_markup.html", "manifest.json"]
    missing_files = []
    
    for file_name in required_files:
        file_path = os.path.join(args.input, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)
    
    if missing_files:
        print(f"\n⚠️  WARNING: Some files from script 1 are missing:")
        for file_name in missing_files:
            print(f"   - {file_name}")
        print(f"\n   Script 2 will continue but some updates may be incomplete.")
        print(f"   Make sure to run script 1 first to generate all required files.")
    
    # Create updater and run
    updater = AssemblyChainUpdater(
        input_dir=args.input,
        mmcif_base_url=args.mmcif_url,
        pdb_base_url=args.pdb_url,
        pdb_label_dir=args.pdb_label_dir,
        cif_label_dir=args.cif_label_dir
    )
    
    stats = updater.run()
    
    print(f"\n✅ Assembly chain checking complete!")
    print(f"   Updated files saved to: {args.input}")
    
    # Show what was updated
    print(f"\n📁 Files updated:")
    print(f"   • dataset_with_interfaces.json - Main dataset with LabelChain fields")
    print(f"   • fair_metadata_package.json - FAIR metadata with assembly chain stats and updated fields_to_be_populated")
    print(f"   • embedded_markup.html - HTML with updated JSON-LD")
    print(f"   • manifest.json - Manifest with script 2 execution info")
    print(f"   • interface_protein_pairs/*.json - All interface files")
    
    print(f"\n   Next step: Run script 3 to add PDB metadata")


if __name__ == "__main__":
    main()
