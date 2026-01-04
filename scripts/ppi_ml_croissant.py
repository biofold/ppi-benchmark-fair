"""
Script to import a Croissant-formatted protein-protein interaction dataset
and train/test basic machine learning models using GroupKFold cross-validation.
Accepts local directory or GitHub repository as input.
Includes comprehensive feature evaluation and PDB structural feature extraction.
"""

import json
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc, mutual_info_score
)
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import tempfile
import os
import warnings
import gzip
import shutil
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

# Import BioPython for PDB parsing
try:
    from Bio.PDB import PDBParser, MMCIFParser, Select, StructureBuilder
    from Bio.PDB.Polypeptide import is_aa
    from Bio import SeqIO
    from Bio.SeqUtils import IUPACData
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("Warning: BioPython not available. PDB feature extraction will be limited.")

warnings.filterwarnings('ignore')

# ============================================
# First define all the original classes that were in the file
# ============================================

class ClusterAwareGroupKFold:
    """
    Custom cross-validator that ensures all interfaces from the same ClusterID
    stay together in the same fold using GroupKFold.
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.group_kfold = GroupKFold(n_splits=n_splits)

    def split(self, X, y, cluster_ids):
        """
        Generate indices to split data into training and test sets.

        Args:
            X: Feature matrix (not used directly for splitting)
            y: Target labels
            cluster_ids: ClusterID for each sample (groups)

        Yields:
            (train_indices, test_indices) for each fold
        """
        # Use GroupKFold directly - it ensures all samples from same group stay together
        for train_idx, test_idx in self.group_kfold.split(X, y, groups=cluster_ids):
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, cluster_ids=None):
        return self.n_splits


# ============================================
# UPDATED CLASS: PDBFeatureExtractor
# ============================================

class PDBFeatureExtractor:
    """
    Extract basic structural features from PDB files using BioPython.
    Supports direct parsing of gzipped files and handles case sensitivity.
    """

    def __init__(self, pdb_local_dir: str = None, use_pdb_format: bool = False):
        """
        Initialize the PDB feature extractor.

        Args:
            pdb_local_dir: Local directory with interface structure files
            use_pdb_format: True for PDB format, False for mmCIF format
        """
        self.pdb_local_dir = pdb_local_dir
        self.use_pdb_format = use_pdb_format

        if not BIOPYTHON_AVAILABLE:
            print("⚠️  BioPython is not available. PDB feature extraction will be limited.")
            print("   Install with: pip install biopython")
            return

        # Initialize parsers
        self.pdb_parser = PDBParser(QUIET=True)
        self.mmcif_parser = MMCIFParser(QUIET=True)

        # Create local directory if specified
        if pdb_local_dir:
            Path(pdb_local_dir).mkdir(parents=True, exist_ok=True)
            print(f"Interface structure directory: {pdb_local_dir}")

    def _get_local_interface_file(self, representation: Dict) -> Optional[str]:
        """
        Get local interface structure file from representation information.
        Handles case sensitivity issues (uppercase vs lowercase).
        
        Args:
            representation: Dictionary containing file information from JSON
        
        Returns:
            Path to local interface file or None if not found
        """
        if not self.pdb_local_dir:
            return None
        
        try:
            # Extract interface ID from representation
            interface_id = None
            value = representation.get('value', '')
            
            if isinstance(value, str):
                # Extract filename from URL/path
                filename = value.split('/')[-1]
                # Remove file extension
                interface_id = filename.split('.')[0]
            
            if not interface_id:
                # Try to get from identifier or name
                interface_id = representation.get('identifier') or representation.get('name')
            
            if not interface_id:
                print(f"  Could not extract interface ID from representation")
                return None
            
            print(f"  Looking for local interface file: {interface_id}")
            
            # Handle case sensitivity: try both uppercase and lowercase versions
            interface_variants = [interface_id]
            
            # Add lowercase variant
            if interface_id.isupper():
                interface_variants.append(interface_id.lower())
            # Add uppercase variant
            elif interface_id.islower():
                interface_variants.append(interface_id.upper())
            
            # Determine file extension based on format
            file_ext = '.pdb' if self.use_pdb_format else '.cif'
            alt_ext = '.cif' if self.use_pdb_format else '.pdb'
            
            # Check for various possible file locations and formats
            possible_files = []
            
            for variant in interface_variants:
                # Check for compressed version in pdb_local_dir
                compressed_file = Path(self.pdb_local_dir) / f"{variant}{file_ext}.gz"
                if compressed_file.exists():
                    possible_files.append(str(compressed_file))
                
                # Check uncompressed version in pdb_local_dir
                uncompressed_file = Path(self.pdb_local_dir) / f"{variant}{file_ext}"
                if uncompressed_file.exists():
                    possible_files.append(str(uncompressed_file))
                
                # Check alternative formats
                alt_compressed = Path(self.pdb_local_dir) / f"{variant}{alt_ext}.gz"
                if alt_compressed.exists():
                    possible_files.append(str(alt_compressed))
                
                alt_uncompressed = Path(self.pdb_local_dir) / f"{variant}{alt_ext}"
                if alt_uncompressed.exists():
                    possible_files.append(str(alt_uncompressed))
            
            if possible_files:
                # Prefer compressed files (BioPython can handle gzipped files directly)
                for file_path in possible_files:
                    if file_path.endswith('.gz'):
                        print(f"  Found local gzipped file: {file_path}")
                        # BioPython can parse gzipped files directly
                        return file_path
                
                # If no compressed files, use the first uncompressed file
                print(f"  Found local file: {possible_files[0]}")
                return possible_files[0]
            
            print(f"  No local interface file found for {interface_id} (tried variants: {interface_variants})")
            return None
            
        except Exception as e:
            print(f"  Error looking for local interface file: {e}")
            return None

    def get_pdb_file_from_representation(self, representation: Dict) -> Optional[str]:
        """
        Get structure file path from JSON representation information.
        Prioritizes local interface files, downloads from representation URL if not found.
        
        Args:
            representation: Dictionary containing file information from JSON

        Returns:
            Path to local structure file or None if failed
        """
        try:
            print(f"  Processing representation for structure feature extraction")
            
            # FIRST: Try to get local interface file
            local_file = self._get_local_interface_file(representation)
            if local_file:
                print(f"  ✅ Using local interface file: {Path(local_file).name}")
                return local_file
            
            # SECOND: If no loader available, download directly
            print(f"  No local interface file found, attempting direct download...")
            
            # Extract URL from representation
            content_url = None
            value = representation.get('value', '')
            
            if isinstance(value, str) and value.startswith('http'):
                content_url = value
            else:
                # Try other URL fields
                for url_field in ['contentUrl', 'url']:
                    if url_field in representation:
                        url_value = representation[url_field]
                        if isinstance(url_value, str) and url_value.startswith('http'):
                            content_url = url_value
                            break
            
            if not content_url:
                print(f"  No valid URL found in representation for download")
                return None
            
            # Extract interface ID for filename
            interface_id = None
            if isinstance(value, str):
                filename = value.split('/')[-1]
                interface_id = filename.split('.')[0]
            
            if not interface_id:
                interface_id = representation.get('identifier') or representation.get('name') or 'interface'
            
            print(f"  Downloading interface file from: {content_url}")
            
            # Download using simple requests (without the retry logic from the other method)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(content_url, headers=headers, timeout=60)
            
            if response.status_code == 200:
                # Determine file extension
                if content_url.lower().endswith('.pdb') or content_url.lower().endswith('.pdb.gz'):
                    file_ext = '.pdb'
                elif content_url.lower().endswith('.cif') or content_url.lower().endswith('.cif.gz'):
                    file_ext = '.cif'
                elif content_url.lower().endswith('.mmcif') or content_url.lower().endswith('.mmcif.gz'):
                    file_ext = '.mmcif'
                else:
                    file_ext = '.pdb' if self.use_pdb_format else '.cif'
                
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=f"{file_ext}.gz")
                
                # Compress the downloaded content
                with gzip.GzipFile(fileobj=temp_file, mode='wb') as gz_file:
                    gz_file.write(response.content)
                
                temp_file.close()
                print(f"  Downloaded to temporary gzipped file: {temp_file.name}")
                return temp_file.name
            else:
                print(f"  Failed to download: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"  Error getting structure file: {e}")
            return None

    def extract_interface_features(self, interface_id: str, pdb_file: str, chain_ids: List[str], radius: float = 10.0) -> Dict[str, Any]:
        """
        Extract interface features from a PDB file.
        Supports both compressed (.gz) and uncompressed files.
        Focus on unique residues per chain within 10Å of the other chain.
        Uses only C-alpha atoms for distance calculations.
        If only one chain is found in a monomer mmCIF file, attempts to generate
        biological assembly from the monomer.
        """
        features = {
            'success': False,
            'error': None,
            'assembly_generated': False,
            'original_chain_count': 0
        }

        if not BIOPYTHON_AVAILABLE:
            features['error'] = "BioPython not available"
            return features

        try:
            # Determine parser based on file extension
            is_gzipped = pdb_file.endswith('.gz')
            is_cif = pdb_file.endswith('.cif') or pdb_file.endswith('.cif.gz') or pdb_file.endswith('.mmcif') or pdb_file.endswith('.mmcif.gz')
            
            # BioPython can handle gzipped files directly
            if is_cif:
                parser = self.mmcif_parser
            else:
                parser = self.pdb_parser

            # Parse structure - unified approach for both gzipped and regular files
            open_func = gzip.open if is_gzipped else open
            open_mode = 'rt' if is_gzipped else 'r'

            with open_func(pdb_file, open_mode) as handle:
                structure = parser.get_structure('structure', handle)
                model = structure[0]  # Get first model

            # Get the chains
            chains = {}
            original_chains_found = []
            for chain_id in chain_ids:
                try:
                    chains[chain_id] = model[chain_id]
                    original_chains_found.append(chain_id)
                except KeyError:
                    print(f"  Chain {chain_id} not found in structure")

            print(f"  Found {len(chains)} requested chain(s): {list(chains.keys())}")
            
            # ============================================
            # FIXED: Handle monomer mmCIF files by generating assembly
            # ============================================
            if len(chains) < 2 and is_cif:
                print(f"  Only {len(chains)} chain(s) found in monomer mmCIF file")
                print(f"  Attempting to generate biological assembly from monomer...")
                
                # Try to generate biological assembly (default: assembly from interface_id)
                assembly_structure = self.generate_biological_assembly_from_monomer(interface_id, pdb_file)
                
                if assembly_structure:
                    print(f"  ✅ Successfully generated biological assembly")
                    assembly_model = assembly_structure[0]
                    
                    # Get all chains from the generated assembly
                    available_chain_ids = list(assembly_model.child_dict.keys())
                    print(f"  Assembly has {len(available_chain_ids)} chains: {available_chain_ids}")
                    
                    # For interface analysis, we need at least 2 chains
                    if len(available_chain_ids) >= 2:
                        # Reset chains dictionary
                        chains = {}
                        
                        # Strategy 1: Try to use the original requested chains if they exist in assembly
                        used_chains = []
                        for requested_chain in chain_ids:
                            # Look for exact match or partial match
                            for available_id in available_chain_ids:
                                # Check for exact match
                                if requested_chain == available_id:
                                    chains[available_id] = assembly_model[available_id]
                                    used_chains.append(available_id)
                                    break
                                # Check if requested chain is a prefix (e.g., "A" matches "A_1")
                                elif available_id.startswith(requested_chain + "_") or requested_chain in available_id:
                                    chains[available_id] = assembly_model[available_id]
                                    used_chains.append(available_id)
                                    break
                        
                        # Strategy 2: If we still don't have 2 chains, add more from assembly
                        if len(chains) < 2:
                            for available_id in available_chain_ids:
                                if available_id not in used_chains:
                                    chains[available_id] = assembly_model[available_id]
                                    used_chains.append(available_id)
                                    if len(chains) >= 2:
                                        break
                        
                        # Strategy 3: If still not enough, just use first 2 chains
                        if len(chains) < 2:
                            chains = {}
                            for i in range(min(2, len(available_chain_ids))):
                                chain_id = available_chain_ids[i]
                                chains[chain_id] = assembly_model[chain_id]
                        
                        print(f"  Using {len(chains)} chain(s) for interface analysis: {list(chains.keys())}")
                        
                        # Update the model to use assembly structure
                        model = assembly_model
                        features['assembly_generated'] = True
                        features['original_chain_count'] = len(original_chains_found)
                        print(f"  Original chains found: {original_chains_found}")
                    else:
                        print(f"  ⚠️  Generated assembly has only {len(available_chain_ids)} chain(s), need at least 2")
                        features['error'] = f"Generated assembly has only {len(available_chain_ids)} chain(s)"
                        return features
                else:
                    print(f"  ⚠️  Could not generate biological assembly")
                    features['error'] = f"Need at least 2 chains, found {len(chains)} (and failed to generate assembly)"
                    return features
            
            # If still not enough chains, try alternative approaches
            if len(chains) < 2:
                print(f"  ⚠️  Only {len(chains)} chain(s) available, trying to find alternative chains...")
                
                # Try to get all available chains from the model
                all_chains_in_model = list(model.child_dict.keys())
                print(f"  All chains in structure: {all_chains_in_model}")
                
                if len(all_chains_in_model) >= 2:
                    # Use first 2 available chains
                    chains = {}
                    for i in range(2):
                        chain_id = all_chains_in_model[i]
                        chains[chain_id] = model[chain_id]
                    print(f"  Using first 2 available chains: {list(chains.keys())}")
                else:
                    features['error'] = f"Need at least 2 chains, found {len(chains)} in structure (total: {len(all_chains_in_model)})"
                    return features
            
            # Extract chain names for pairwise analysis
            chain_list = list(chains.keys())
            print(f"  Final chains selected for interface analysis: {chain_list}")

            # Initialize interface features
            interface_features = {}

            # Analyze each pair of chains
            for i, chain1_id in enumerate(chain_list):
                for chain2_id in chain_list[i+1:]:
                    pair_key = f"{chain1_id}_{chain2_id}"

                    try:
                        chain1 = chains[chain1_id]
                        chain2 = chains[chain2_id]

                        # Get C-alpha atoms from both chains
                        chain1_ca_atoms = []
                        chain2_ca_atoms = []

                        for residue in chain1:
                            if is_aa(residue, standard=True):
                                # Get only C-alpha atom if it exists
                                if 'CA' in residue:
                                    chain1_ca_atoms.append(residue['CA'])

                        for residue in chain2:
                            if is_aa(residue, standard=True):
                                # Get only C-alpha atom if it exists
                                if 'CA' in residue:
                                    chain2_ca_atoms.append(residue['CA'])

                        if not chain1_ca_atoms or not chain2_ca_atoms:
                            print(f"  Warning: No C-alpha atoms found for chain pair {pair_key}")
                            continue

                        print(f"  Analyzing chain pair {pair_key}: {len(chain1_ca_atoms)} vs {len(chain2_ca_atoms)} C-alpha atoms")

                        # Find residues in chain1 that are within 10Å of chain2 (using C-alpha)
                        chain1_interface_residues = set()
                        chain2_interface_residues = set()

                        # For each C-alpha in chain1, check if it's close to any C-alpha in chain2
                        for residue1 in chain1:
                            if not is_aa(residue1, standard=True):
                                continue
                            
                            # Check if residue has C-alpha
                            if 'CA' not in residue1:
                                continue
                                
                            ca1 = residue1['CA']
                            
                            # Check distance to all C-alphas in chain2
                            for ca2 in chain2_ca_atoms:
                                if (ca1 - ca2) <= radius:  # 10Å distance threshold
                                    chain1_interface_residues.add(residue1.id[1])  # Use residue number
                                    break

                        # For each C-alpha in chain2, check if it's close to any C-alpha in chain1
                        for residue2 in chain2:
                            if not is_aa(residue2, standard=True):
                                continue
                            
                            # Check if residue has C-alpha
                            if 'CA' not in residue2:
                                continue
                                
                            ca2 = residue2['CA']
                            
                            # Check distance to all C-alphas in chain1
                            for ca1 in chain1_ca_atoms:
                                if (ca2 - ca1) <= radius:  # 10Å distance threshold
                                    chain2_interface_residues.add(residue2.id[1])  # Use residue number
                                    break

                        # Calculate interface features for this chain pair
                        # Count residues with C-alpha atoms only
                        chain1_residue_count = sum(1 for residue in chain1 
                                                  if is_aa(residue, standard=True) and 'CA' in residue)
                        chain2_residue_count = sum(1 for residue in chain2 
                                                  if is_aa(residue, standard=True) and 'CA' in residue)

                        # Calculate interface features
                        chain1_interface_count = len(chain1_interface_residues)
                        chain2_interface_count = len(chain2_interface_residues)
                        total_interface_count = chain1_interface_count + chain2_interface_count
                        
                        interface_features[pair_key] = {
                            'chain1_interface_residues': chain1_interface_count,
                            'chain2_interface_residues': chain2_interface_count,
                            'total_interface_residues': total_interface_count,
                            'interface_residue_ratio': chain1_interface_count / chain2_interface_count 
                                if chain2_interface_count > 0 else 0,
                            'chain1_residue_count': chain1_residue_count,
                            'chain2_residue_count': chain2_residue_count,
                            'chain1_interface_fraction': chain1_interface_count / chain1_residue_count 
                                if chain1_residue_count > 0 else 0,
                            'chain2_interface_fraction': chain2_interface_count / chain2_residue_count 
                                if chain2_residue_count > 0 else 0,
                            'avg_interface_fraction': (chain1_interface_count/chain1_residue_count + 
                                                       chain2_interface_count/chain2_residue_count)/2 
                                if chain1_residue_count > 0 and chain2_residue_count > 0 else 0,
                            'method': 'C-alpha_only',  # Track which method was used
                            'chain_pair': pair_key,
                            'distance_threshold': radius
                        }
                        
                        print(f"    Interface {pair_key}: {chain1_interface_count} + {chain2_interface_count} = {total_interface_count} interface residues")

                    except Exception as e:
                        print(f"  Error analyzing chain pair {pair_key}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

            if not interface_features:
                features['error'] = "No interface features could be extracted"
                return features

            # Aggregate features across all chain pairs
            features['success'] = True
            features['num_chain_pairs'] = len(interface_features)

            # Add aggregated statistics
            all_chain1_residues = [v['chain1_interface_residues'] for v in interface_features.values()]
            all_chain2_residues = [v['chain2_interface_residues'] for v in interface_features.values()]
            all_total_residues = [v['total_interface_residues'] for v in interface_features.values()]
            all_chain1_fractions = [v['chain1_interface_fraction'] for v in interface_features.values()]
            all_chain2_fractions = [v['chain2_interface_fraction'] for v in interface_features.values()]
            all_avg_fractions = [v['avg_interface_fraction'] for v in interface_features.values()]

            if all_chain1_residues:
                features['avg_chain1_interface_residues'] = float(np.mean(all_chain1_residues))
                features['std_chain1_interface_residues'] = float(np.std(all_chain1_residues))
                features['max_chain1_interface_residues'] = float(np.max(all_chain1_residues))
                features['min_chain1_interface_residues'] = float(np.min(all_chain1_residues))

            if all_chain2_residues:
                features['avg_chain2_interface_residues'] = float(np.mean(all_chain2_residues))
                features['std_chain2_interface_residues'] = float(np.std(all_chain2_residues))
                features['max_chain2_interface_residues'] = float(np.max(all_chain2_residues))
                features['min_chain2_interface_residues'] = float(np.min(all_chain2_residues))

            if all_total_residues:
                features['avg_total_interface_residues'] = float(np.mean(all_total_residues))
                features['std_total_interface_residues'] = float(np.std(all_total_residues))
                features['max_total_interface_residues'] = float(np.max(all_total_residues))
                features['min_total_interface_residues'] = float(np.min(all_total_residues))

            if all_chain1_fractions:
                features['avg_chain1_interface_fraction'] = float(np.mean(all_chain1_fractions))
                features['std_chain1_interface_fraction'] = float(np.std(all_chain1_fractions))

            if all_chain2_fractions:
                features['avg_chain2_interface_fraction'] = float(np.mean(all_chain2_fractions))
                features['std_chain2_interface_fraction'] = float(np.std(all_chain2_fractions))

            if all_avg_fractions:
                features['avg_interface_fraction'] = float(np.mean(all_avg_fractions))
                features['std_interface_fraction'] = float(np.std(all_avg_fractions))

            # Add features from the first chain pair (most representative)
            if interface_features:
                first_pair = list(interface_features.values())[0]
                for key, value in first_pair.items():
                    features[f'first_pair_{key}'] = value

            print(f"  ✅ Successfully extracted interface features with {len(interface_features)} chain pairs")
            return features

        except Exception as e:
            features['error'] = f"Error extracting interface features: {str(e)}"
            import traceback
            traceback.print_exc()
            return features

    def generate_biological_assembly_from_monomer(self, interface_id: str, pdb_file: str) -> Optional['Structure']:
        """
        Generate biological assembly from monomer mmCIF file.
        Automatically detects assembly ID from interface_id or mmCIF metadata.
        
        Args:
            pdb_file: Path to mmCIF file (can be .gz)
            interface_id: Interface ID (may contain assembly info, e.g., "7CEI_1")
            
        Returns:
            BioPython Structure object with generated assembly, or None if failed
        """
        if not BIOPYTHON_AVAILABLE:
            print("⚠️  BioPython not available")
            return None
        
        try:
            from Bio.PDB import StructureBuilder
            import numpy as np
            
            # Check if file is mmCIF
            is_gzipped = pdb_file.endswith('.gz')
            is_cif = pdb_file.endswith('.cif') or pdb_file.endswith('.cif.gz') or pdb_file.endswith('.mmcif') or pdb_file.endswith('.mmcif.gz')
            
            if not is_cif:
                print(f"  Not an mmCIF file: {pdb_file}")
                return None
            
            print(f"  Generating biological assembly for interface: {interface_id}")
            
            # Parse mmCIF dictionary for assembly information
            open_func = gzip.open if is_gzipped else open
            open_mode = 'rt' if is_gzipped else 'r'
            
            with open_func(pdb_file, open_mode) as handle:
                mmcif_dict = MMCIF2Dict(handle)
            
            # ============================================
            # 1. DETERMINE WHICH ASSEMBLY TO USE
            # ============================================
            assembly_id = "1"  # Default
            
            if '_pdbx_struct_assembly.id' in mmcif_dict:
                available_assemblies = mmcif_dict['_pdbx_struct_assembly.id']
                print(f"  Available assemblies in mmCIF: {available_assemblies}")
                
                # Strategy 1: Extract from interface_id (e.g., "7CEI_1" -> "1")
                if interface_id:
                    import re
                    # Pattern for PDBID_ASSEMBLY or PDBID_CHAIN_ASSEMBLY
                    assembly_match = re.search(r'_(\d+)$', interface_id)
                    if assembly_match:
                        extracted_id = assembly_match.group(1)
                        if extracted_id in available_assemblies:
                            assembly_id = extracted_id
                            print(f"  Using assembly {assembly_id} extracted from interface_id: {interface_id}")
                        else:
                            print(f"  Assembly {extracted_id} from interface_id not found in mmCIF")
                    else:
                        print(f"  No assembly number in interface_id: {interface_id}")
                
                # Strategy 2: If assembly not determined yet, check for biological assembly
                if assembly_id == "1" and len(available_assemblies) > 1:
                    # Look for biological or recommended assembly
                    if '_pdbx_struct_assembly.details' in mmcif_dict:
                        details = mmcif_dict['_pdbx_struct_assembly.details']
                        for i, detail in enumerate(details):
                            if i < len(available_assemblies):
                                current_id = available_assemblies[i]
                                detail_lower = detail.lower() if detail else ""
                                # Check if this is the biological assembly
                                if 'biological' in detail_lower or 'native' in detail_lower or 'author' in detail_lower:
                                    if current_id in available_assemblies:
                                        assembly_id = current_id
                                        print(f"  Using biological assembly {assembly_id}: {detail}")
                                        break
                    
                    # Strategy 3: If still default, check oligomeric count
                    if assembly_id == "1" and '_pdbx_struct_assembly.oligomeric_count' in mmcif_dict:
                        oligo_counts = mmcif_dict['_pdbx_struct_assembly.oligomeric_count']
                        for i, count in enumerate(oligo_counts):
                            if i < len(available_assemblies):
                                current_id = available_assemblies[i]
                                # Prefer assemblies with oligomeric count > 1 (multimers)
                                if count != "1" and current_id in available_assemblies:
                                    assembly_id = current_id
                                    print(f"  Using assembly {assembly_id} (oligomeric count: {count})")
                                    break
                
                # Strategy 4: If assembly still not found, use first available
                if assembly_id not in available_assemblies and available_assemblies:
                    assembly_id = available_assemblies[0]
                    print(f"  Assembly not found, using first available: {assembly_id}")
            else:
                print(f"  No biological assemblies defined in mmCIF")
                return None
            
            print(f"  Selected assembly ID: {assembly_id}")
            
            # ============================================
            # 2. PARSE THE ORIGINAL STRUCTURE FIRST
            # ============================================
            parser = MMCIFParser(QUIET=True)
            with open_func(pdb_file, open_mode) as handle:
                original_structure = parser.get_structure('original', handle)
            
            original_model = original_structure[0]
            
            # Get available chain IDs from the actual structure (auth_asym_id)
            available_chain_ids = list(original_model.child_dict.keys())
            print(f"  Original structure has chains (auth_asym_id): {available_chain_ids}")
            
            # ============================================
            # 3. CREATE MAPPING BETWEEN LABEL_ASYM_ID AND AUTH_ASYM_ID
            # ============================================
            # This is CRITICAL: mmCIF files use label_asym_id in assembly definitions
            # but auth_asym_id in the actual coordinates
            label_to_auth_map = {}
            auth_to_label_map = {}
            
            if '_atom_site.label_asym_id' in mmcif_dict and '_atom_site.auth_asym_id' in mmcif_dict:
                label_asym_ids = mmcif_dict['_atom_site.label_asym_id']
                auth_asym_ids = mmcif_dict['_atom_site.auth_asym_id']
                
                # Create bidirectional mapping
                for label_id, auth_id in zip(label_asym_ids, auth_asym_ids):
                    if label_id not in label_to_auth_map:
                        label_to_auth_map[label_id] = auth_id
                    if auth_id not in auth_to_label_map:
                        auth_to_label_map[auth_id] = label_id
            
            print(f"  label_asym_id -> auth_asym_id mapping: {label_to_auth_map}")
            print(f"  auth_asym_id -> label_asym_id mapping: {auth_to_label_map}")
            
            # If no mapping found, assume they're the same
            if not label_to_auth_map:
                for chain_id in available_chain_ids:
                    label_to_auth_map[chain_id] = chain_id
                    auth_to_label_map[chain_id] = chain_id
            
            # ============================================
            # 4. GET OLIGOMERIC COUNT FOR SELECTED ASSEMBLY
            # ============================================
            oligo_count = 1
            if '_pdbx_struct_assembly.oligomeric_count' in mmcif_dict:
                oligo_counts = mmcif_dict['_pdbx_struct_assembly.oligomeric_count']
                if assembly_id in available_assemblies:
                    idx = list(available_assemblies).index(assembly_id)
                    if idx < len(oligo_counts):
                        try:
                            oligo_count = int(oligo_counts[idx])
                        except:
                            oligo_count = 1
            
            print(f"  Assembly {assembly_id}: {oligo_count}-mer")
            
            # ============================================
            # 5. FIND GENERATORS FOR THIS ASSEMBLY
            # ============================================
            if '_pdbx_struct_assembly_gen.assembly_id' not in mmcif_dict:
                print(f"  No assembly generators found")
                return None
            
            gen_assembly_ids = mmcif_dict['_pdbx_struct_assembly_gen.assembly_id']
            gen_oper_expr = mmcif_dict.get('_pdbx_struct_assembly_gen.oper_expression', ['1'] * len(gen_assembly_ids))
            gen_asym_ids = mmcif_dict.get('_pdbx_struct_assembly_gen.asym_id_list', [''] * len(gen_assembly_ids))
            
            # Find generators for our assembly
            assembly_generators = []
            for i, gen_assem_id in enumerate(gen_assembly_ids):
                if gen_assem_id == assembly_id:
                    operation_expr = gen_oper_expr[i] if i < len(gen_oper_expr) else '1'
                    chain_list = gen_asym_ids[i] if i < len(gen_asym_ids) else ''
                    
                    # Parse operations (e.g., "1,2" -> ['1', '2'])
                    operations = []
                    for op in operation_expr.split(','):
                        op = op.strip()
                        if op:
                            operations.append(op)
                    
                    # Parse chains (these are LABEL_ASYM_IDs!)
                    chains = [c.strip() for c in chain_list.split(',')] if chain_list else []
                    
                    # Convert label_asym_id to auth_asym_id if possible
                    auth_chains = []
                    for label_chain in chains:
                        auth_chain = label_to_auth_map.get(label_chain, label_chain)
                        auth_chains.append(auth_chain)
                    
                    assembly_generators.append({
                        'operations': operations,
                        'label_chains': chains,  # Original label_asym_id
                        'auth_chains': auth_chains,  # Converted auth_asym_id
                        'operation_expression': operation_expr
                    })
            
            if not assembly_generators:
                print(f"  No generators found for assembly {assembly_id}")
                return None
            
            print(f"  Found {len(assembly_generators)} generator(s) for assembly {assembly_id}")
            for gen in assembly_generators:
                print(f"    Label chains: {gen['label_chains']} -> Auth chains: {gen['auth_chains']}, Operations: {gen['operations']}")
            
            # ============================================
            # 6. GET TRANSFORMATION OPERATIONS
            # ============================================
            if '_pdbx_struct_oper_list.id' not in mmcif_dict:
                print(f"  No transformation operations defined")
                return None
            
            oper_ids = mmcif_dict['_pdbx_struct_oper_list.id']
            transformation_ops = {}
            
            for i, op_id in enumerate(oper_ids):
                # Parse 3x3 rotation matrix
                matrix = []
                for row in range(1, 4):
                    row_vals = []
                    for col in range(1, 4):
                        key = f'_pdbx_struct_oper_list.matrix[{row}][{col}]'
                        if key in mmcif_dict and i < len(mmcif_dict[key]):
                            try:
                                row_vals.append(float(mmcif_dict[key][i]))
                            except:
                                row_vals.append(0.0)
                        else:
                            row_vals.append(0.0)
                    matrix.append(row_vals)
                
                # Parse translation vector
                vector = []
                for comp in range(1, 4):
                    key = f'_pdbx_struct_oper_list.vector[{comp}]'
                    if key in mmcif_dict and i < len(mmcif_dict[key]):
                        try:
                            vector.append(float(mmcif_dict[key][i]))
                        except:
                            vector.append(0.0)
                    else:
                        vector.append(0.0)
                
                transformation_ops[op_id] = {
                    'matrix': np.array(matrix),
                    'vector': np.array(vector)
                }
            
            # ============================================
            # 7. GENERATE THE ASSEMBLY
            # ============================================
            builder = StructureBuilder.StructureBuilder()
            builder.init_structure(f"assembly_{assembly_id}")
            builder.init_model(0)
            
            chain_counter = {}  # Track chain copies
            chains_processed = 0
            
            # Process each generator
            for generator in assembly_generators:
                auth_chains = generator['auth_chains']
                label_chains = generator['label_chains']
                operations = generator['operations']
                
                print(f"  Processing generator: auth_chains={auth_chains}, operations={operations}")
                
                # If no specific chains are listed, use all available chains
                if not auth_chains:
                    auth_chains = available_chain_ids
                    print(f"  No chains specified, using all available: {auth_chains}")
                
                for auth_chain_id, label_chain_id in zip(auth_chains, label_chains):
                    if auth_chain_id not in original_model:
                        print(f"  Warning: Chain '{auth_chain_id}' (from label '{label_chain_id}') not in structure, skipping")
                        continue
                    
                    original_chain = original_model[auth_chain_id]
                    
                    for op_id in operations:
                        if op_id not in transformation_ops:
                            print(f"  Warning: Operation {op_id} not defined, skipping")
                            continue
                        
                        # Determine new chain ID
                        if op_id == '1':  # Identity operation
                            # Keep original chain ID for identity operation
                            new_chain_id = auth_chain_id
                        else:
                            # Count how many times we've used this chain with this operation
                            key = f"{auth_chain_id}_{op_id}"
                            if key not in chain_counter:
                                chain_counter[key] = 0
                            chain_counter[key] += 1
                            
                            # Create unique chain ID
                            if chain_counter[key] == 1:
                                # For first copy, use original chain ID with operation suffix
                                new_chain_id = f"{auth_chain_id}_{op_id}"
                            else:
                                # For additional copies, add number
                                new_chain_id = f"{auth_chain_id}_{op_id}_{chain_counter[key]}"
                        
                        # Apply transformation
                        transformed_chain = self._apply_transformation_to_chain(
                            original_chain, 
                            transformation_ops[op_id],
                            new_chain_id
                        )
                        
                        # Add chain to structure
                        if new_chain_id not in builder.structure[0]:
                            builder.structure[0].add(transformed_chain)
                            chains_processed += 1
                            print(f"    Generated chain {new_chain_id} from {auth_chain_id} (label: {label_chain_id}) + op{op_id}")
            
            # Get the final structure
            assembly_structure = builder.get_structure()
            final_chain_count = len(list(assembly_structure[0].get_chains()))
            
            if final_chain_count == 0:
                print(f"  ⚠️  Generated assembly has 0 chains, trying fallback method...")
                
                # Fallback: Create symmetric copies based on oligo_count
                return self._generate_assembly_fallback(original_structure, assembly_id, oligo_count)
            
            print(f"  ✅ Successfully generated assembly {assembly_id} with {final_chain_count} chains")
            
            return assembly_structure
            
        except Exception as e:
            print(f"  Error generating biological assembly: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_biological_assembly_from_monomer2(self, pdb_file: str, assembly_id: str = "1") -> Optional['Structure']:
        """
        Generate biological assembly from monomer mmCIF file.
        
        Args:
            pdb_file: Path to mmCIF file (can be .gz)
            assembly_id: Which biological assembly to generate (default: "1")
            
        Returns:
            BioPython Structure object with generated assembly, or None if failed
        """
        if not BIOPYTHON_AVAILABLE:
            print("⚠️  BioPython not available")
            return None
        
        try:
            from Bio.PDB import StructureBuilder
            import numpy as np
            
            # Check if file is mmCIF
            is_gzipped = pdb_file.endswith('.gz')
            is_cif = pdb_file.endswith('.cif') or pdb_file.endswith('.cif.gz') or pdb_file.endswith('.mmcif') or pdb_file.endswith('.mmcif.gz')
            
            if not is_cif:
                print(f"  Not an mmCIF file: {pdb_file}")
                return None
            
            print(f"  Generating biological assembly {assembly_id} from monomer...")
            
            # Parse mmCIF dictionary for assembly information
            open_func = gzip.open if is_gzipped else open
            open_mode = 'rt' if is_gzipped else 'r'
            
            with open_func(pdb_file, open_mode) as handle:
                mmcif_dict = MMCIF2Dict(handle)
            
            # ============================================
            # 1. PARSE THE ORIGINAL STRUCTURE FIRST
            # ============================================
            parser = MMCIFParser(QUIET=True)
            with open_func(pdb_file, open_mode) as handle:
                original_structure = parser.get_structure('original', handle)
            
            original_model = original_structure[0]
            
            # Get available chain IDs from the actual structure (auth_asym_id)
            available_chain_ids = list(original_model.child_dict.keys())
            print(f"  Original structure has chains (auth_asym_id): {available_chain_ids}")
            
            # ============================================
            # 2. CREATE MAPPING BETWEEN LABEL_ASYM_ID AND AUTH_ASYM_ID
            # ============================================
            # This is CRITICAL: mmCIF files use label_asym_id in assembly definitions
            # but auth_asym_id in the actual coordinates
            label_to_auth_map = {}
            auth_to_label_map = {}
            
            if '_atom_site.label_asym_id' in mmcif_dict and '_atom_site.auth_asym_id' in mmcif_dict:
                label_asym_ids = mmcif_dict['_atom_site.label_asym_id']
                auth_asym_ids = mmcif_dict['_atom_site.auth_asym_id']
                
                # Create bidirectional mapping
                for label_id, auth_id in zip(label_asym_ids, auth_asym_ids):
                    if label_id not in label_to_auth_map:
                        label_to_auth_map[label_id] = auth_id
                    if auth_id not in auth_to_label_map:
                        auth_to_label_map[auth_id] = label_id
            
            print(f"  label_asym_id -> auth_asym_id mapping: {label_to_auth_map}")
            print(f"  auth_asym_id -> label_asym_id mapping: {auth_to_label_map}")
            
            # If no mapping found, assume they're the same
            if not label_to_auth_map:
                for chain_id in available_chain_ids:
                    label_to_auth_map[chain_id] = chain_id
                    auth_to_label_map[chain_id] = chain_id
            
            # ============================================
            # 3. FIND THE REQUESTED ASSEMBLY
            # ============================================
            if '_pdbx_struct_assembly.id' not in mmcif_dict:
                print(f"  No biological assemblies defined in mmCIF")
                return None
            
            # Get assembly information
            assembly_ids = mmcif_dict['_pdbx_struct_assembly.id']
            
            if assembly_id not in assembly_ids:
                print(f"  Assembly {assembly_id} not found. Available: {assembly_ids}")
                return None
            
            # Get assembly index
            assembly_idx = assembly_ids.index(assembly_id)
            
            # Get oligomeric count
            oligo_counts = mmcif_dict.get('_pdbx_struct_assembly.oligomeric_count', ['1'] * len(assembly_ids))
            oligo_count = int(oligo_counts[assembly_idx]) if assembly_idx < len(oligo_counts) else 1
            
            print(f"  Assembly {assembly_id}: {oligo_count}-mer")
            
            # ============================================
            # 4. FIND GENERATORS FOR THIS ASSEMBLY
            # ============================================
            if '_pdbx_struct_assembly_gen.assembly_id' not in mmcif_dict:
                print(f"  No assembly generators found")
                return None
            
            gen_assembly_ids = mmcif_dict['_pdbx_struct_assembly_gen.assembly_id']
            gen_oper_expr = mmcif_dict.get('_pdbx_struct_assembly_gen.oper_expression', ['1'] * len(gen_assembly_ids))
            gen_asym_ids = mmcif_dict.get('_pdbx_struct_assembly_gen.asym_id_list', [''] * len(gen_assembly_ids))
            
            # Find generators for our assembly
            assembly_generators = []
            for i, gen_assem_id in enumerate(gen_assembly_ids):
                if gen_assem_id == assembly_id:
                    operation_expr = gen_oper_expr[i] if i < len(gen_oper_expr) else '1'
                    chain_list = gen_asym_ids[i] if i < len(gen_asym_ids) else ''
                    
                    # Parse operations (e.g., "1,2" -> ['1', '2'])
                    operations = []
                    for op in operation_expr.split(','):
                        op = op.strip()
                        if op:
                            operations.append(op)
                    
                    # Parse chains (these are LABEL_ASYM_IDs!)
                    chains = [c.strip() for c in chain_list.split(',')] if chain_list else []
                    
                    # Convert label_asym_id to auth_asym_id if possible
                    auth_chains = []
                    for label_chain in chains:
                        auth_chain = label_to_auth_map.get(label_chain, label_chain)
                        auth_chains.append(auth_chain)
                    
                    assembly_generators.append({
                        'operations': operations,
                        'label_chains': chains,  # Original label_asym_id
                        'auth_chains': auth_chains,  # Converted auth_asym_id
                        'operation_expression': operation_expr
                    })
            
            if not assembly_generators:
                print(f"  No generators found for assembly {assembly_id}")
                return None
            
            print(f"  Found {len(assembly_generators)} generator(s) for assembly {assembly_id}")
            for gen in assembly_generators:
                print(f"    Label chains: {gen['label_chains']} -> Auth chains: {gen['auth_chains']}, Operations: {gen['operations']}")
            
            # ============================================
            # 5. GET TRANSFORMATION OPERATIONS
            # ============================================
            if '_pdbx_struct_oper_list.id' not in mmcif_dict:
                print(f"  No transformation operations defined")
                return None
            
            oper_ids = mmcif_dict['_pdbx_struct_oper_list.id']
            transformation_ops = {}
            
            for i, op_id in enumerate(oper_ids):
                # Parse 3x3 rotation matrix
                matrix = []
                for row in range(1, 4):
                    row_vals = []
                    for col in range(1, 4):
                        key = f'_pdbx_struct_oper_list.matrix[{row}][{col}]'
                        if key in mmcif_dict and i < len(mmcif_dict[key]):
                            try:
                                row_vals.append(float(mmcif_dict[key][i]))
                            except:
                                row_vals.append(0.0)
                        else:
                            row_vals.append(0.0)
                    matrix.append(row_vals)
                
                # Parse translation vector
                vector = []
                for comp in range(1, 4):
                    key = f'_pdbx_struct_oper_list.vector[{comp}]'
                    if key in mmcif_dict and i < len(mmcif_dict[key]):
                        try:
                            vector.append(float(mmcif_dict[key][i]))
                        except:
                            vector.append(0.0)
                    else:
                        vector.append(0.0)
                
                transformation_ops[op_id] = {
                    'matrix': np.array(matrix),
                    'vector': np.array(vector)
                }
            
            # ============================================
            # 6. GENERATE THE ASSEMBLY
            # ============================================
            builder = StructureBuilder.StructureBuilder()
            builder.init_structure(f"assembly_{assembly_id}")
            builder.init_model(0)
            
            chain_counter = {}  # Track chain copies
            chains_processed = 0
            
            # Process each generator
            for generator in assembly_generators:
                auth_chains = generator['auth_chains']
                label_chains = generator['label_chains']
                operations = generator['operations']
                
                print(f"  Processing generator: auth_chains={auth_chains}, operations={operations}")
                
                # If no specific chains are listed, use all available chains
                if not auth_chains:
                    auth_chains = available_chain_ids
                    print(f"  No chains specified, using all available: {auth_chains}")
                
                for auth_chain_id, label_chain_id in zip(auth_chains, label_chains):
                    if auth_chain_id not in original_model:
                        print(f"  Warning: Chain '{auth_chain_id}' (from label '{label_chain_id}') not in structure, skipping")
                        continue
                    
                    original_chain = original_model[auth_chain_id]
                    
                    for op_id in operations:
                        if op_id not in transformation_ops:
                            print(f"  Warning: Operation {op_id} not defined, skipping")
                            continue
                        
                        # Determine new chain ID
                        if op_id == '1':  # Identity operation
                            # Keep original chain ID for identity operation
                            new_chain_id = auth_chain_id
                        else:
                            # Count how many times we've used this chain with this operation
                            key = f"{auth_chain_id}_{op_id}"
                            if key not in chain_counter:
                                chain_counter[key] = 0
                            chain_counter[key] += 1
                            
                            # Create unique chain ID
                            if chain_counter[key] == 1:
                                # For first copy, use original chain ID with operation suffix
                                new_chain_id = f"{auth_chain_id}_{op_id}"
                            else:
                                # For additional copies, add number
                                new_chain_id = f"{auth_chain_id}_{op_id}_{chain_counter[key]}"
                        
                        # Apply transformation
                        transformed_chain = self._apply_transformation_to_chain(
                            original_chain, 
                            transformation_ops[op_id],
                            new_chain_id
                        )
                        
                        # Add chain to structure
                        if new_chain_id not in builder.structure[0]:
                            builder.structure[0].add(transformed_chain)
                            chains_processed += 1
                            print(f"    Generated chain {new_chain_id} from {auth_chain_id} (label: {label_chain_id}) + op{op_id}")
            
            # Get the final structure
            assembly_structure = builder.get_structure()
            final_chain_count = len(list(assembly_structure[0].get_chains()))
            
            if final_chain_count == 0:
                print(f"  ⚠️  Generated assembly has 0 chains, trying fallback method...")
                
                # Fallback: Create symmetric copies based on oligo_count
                return self._generate_assembly_fallback(original_structure, assembly_id, oligo_count)
            
            print(f"  ✅ Successfully generated assembly with {final_chain_count} chains")
            
            return assembly_structure
            
        except Exception as e:
            print(f"  Error generating biological assembly: {e}")
            import traceback
            traceback.print_exc()
            return None


    def _generate_assembly_fallback(self, original_structure, assembly_id: str, oligo_count: int):
        """
        Fallback method to generate assembly when standard method fails.
        Creates symmetric copies of the structure.
        """
        try:
            from Bio.PDB import StructureBuilder
            import numpy as np
            
            print(f"  Using fallback method to generate {oligo_count}-mer assembly")
            
            builder = StructureBuilder.StructureBuilder()
            builder.init_structure(f"assembly_{assembly_id}_fallback")
            builder.init_model(0)
            
            original_model = original_structure[0]
            original_chains = list(original_model.get_chains())
            
            if not original_chains:
                print(f"  ⚠️  No chains in original structure")
                return None
            
            print(f"  Original chains: {[c.id for c in original_chains]}")
            
            # For simplicity, create symmetric copies around origin
            chains_added = 0
            
            for i in range(oligo_count):
                for original_chain in original_chains:
                    chain_id = original_chain.id
                    
                    # Create new chain ID for each copy
                    if i == 0:
                        new_chain_id = chain_id  # First copy keeps original ID
                    else:
                        # Use different chain IDs for additional copies
                        # Try letters A-Z, then fallback to numbered IDs
                        if len(chain_id) == 1 and chain_id.isalpha():
                            # Shift to next letter (wrap around Z)
                            base_ord = ord(chain_id.upper())
                            if base_ord >= ord('A') and base_ord <= ord('Z'):
                                offset = (base_ord - ord('A') + i) % 26
                                new_chain_id = chr(offset + ord('A'))
                            else:
                                new_chain_id = f"{chain_id}_{i}"
                        else:
                            new_chain_id = f"{chain_id}_{i}"
                    
                    # Create transformation for this copy
                    if i == 0:
                        # First copy is identity
                        matrix = np.identity(3)
                        vector = np.zeros(3)
                    else:
                        # Create rotation around axis
                        angle = 2 * np.pi * i / oligo_count
                        # Simple rotation around z-axis
                        matrix = np.array([
                            [np.cos(angle), -np.sin(angle), 0],
                            [np.sin(angle), np.cos(angle), 0],
                            [0, 0, 1]
                        ])
                        # Simple translation based on oligo_count
                        if oligo_count == 2:
                            vector = np.array([30.0 * i, 0, 0])  # 30Å spacing for dimer
                        elif oligo_count == 3:
                            # Equilateral triangle
                            vector = np.array([
                                30.0 * np.cos(angle),
                                30.0 * np.sin(angle),
                                0
                            ])
                        elif oligo_count == 4:
                            # Square
                            if i == 1:
                                vector = np.array([30.0, 0, 0])
                            elif i == 2:
                                vector = np.array([0, 30.0, 0])
                            else:  # i == 3
                                vector = np.array([30.0, 30.0, 0])
                        else:
                            # Generic spacing
                            vector = np.array([i * 30.0, 0, 0])
                    
                    # Apply transformation
                    transformed_chain = self._apply_transformation_to_chain(
                        original_chain,
                        {'matrix': matrix, 'vector': vector},
                        new_chain_id
                    )
                    
                    # Add chain to structure
                    builder.structure[0].add(transformed_chain)
                    chains_added += 1
                    print(f"    Generated chain {new_chain_id} (copy {i} of {chain_id})")
            
            assembly_structure = builder.get_structure()
            final_chain_count = len(list(assembly_structure[0].get_chains()))
            
            print(f"  ✅ Fallback generated assembly with {final_chain_count} chains")
            return assembly_structure
            
        except Exception as e:
            print(f"  Error in fallback assembly generation: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _apply_transformation_to_chain(self, original_chain, transformation: Dict, new_chain_id: str):
        """
        Apply transformation matrix to a chain and return new chain.
        """
        try:
            from Bio.PDB import Chain, Residue, Atom
            import numpy as np
            
            matrix = transformation['matrix']
            vector = transformation['vector']
            
            # Create new chain
            new_chain = Chain.Chain(new_chain_id)
            
            # Copy and transform each residue
            for residue in original_chain:
                new_residue = Residue.Residue(
                    residue.id,
                    residue.resname,
                    residue.segid
                )
                
                # Transform each atom
                for atom in residue:
                    original_coord = atom.get_coord()
                    # Apply: new_coord = matrix × original_coord + vector
                    transformed_coord = np.dot(matrix, original_coord) + vector
                    
                    new_atom = Atom.Atom(
                        atom.name,
                        transformed_coord,
                        atom.bfactor,
                        atom.occupancy,
                        atom.altloc,
                        atom.fullname,
                        atom.serial_number,
                        element=atom.element
                    )
                    new_residue.add(new_atom)
                
                new_chain.add(new_residue)
            
            return new_chain
            
        except Exception as e:
            print(f"  Error applying transformation: {e}")
            # Return a simple copy if transformation fails
            return original_chain.copy()

    # Slow calculation not used
    def extract_interface_features_all_atom(self, interface_id: str, pdb_file: str, chain_ids: List[str], radius: float = 5.0) -> Dict[str, Any]:
        """
        Extract interface features from a PDB file.
        Supports both compressed (.gz) and uncompressed files.
        Focus on unique residues per chain within 5Å of the other chain.

        Args:
            pdb_file: Path to PDB/mmCIF file (can be .gz compressed)
            chain_ids: List of chain IDs to analyze

        Returns:
            Dictionary of extracted features
        """
        features = {
            'success': False,
            'error': None
        }

        if not BIOPYTHON_AVAILABLE:
            features['error'] = "BioPython not available"
            return features

        try:
            # Determine parser based on file extension
            is_gzipped = pdb_file.endswith('.gz')
            is_cif = pdb_file.endswith('.cif') or pdb_file.endswith('.cif.gz') or pdb_file.endswith('.mmcif') or pdb_file.endswith('.mmcif.gz')
            
            # BioPython can handle gzipped files directly
            if is_cif:
                parser = self.mmcif_parser
            else:
                parser = self.pdb_parser

            # Parse structure - unified approach for both gzipped and regular files
            open_func = gzip.open if is_gzipped else open
            open_mode = 'rt' if is_gzipped else 'r'

            with open_func(pdb_file, open_mode) as handle:
                structure = parser.get_structure('structure', handle)
                model = structure[0]  # Get first model

            # Get the chains
            chains = {}
            for chain_id in chain_ids:
                try:
                    chains[chain_id] = model[chain_id]
                except KeyError:
                    print(f"  Chain {chain_id} not found in structure")

            if len(chains) < 2:
                features['error'] = f"Need at least 2 chains, found {len(chains)}"
                return features

            # Extract chain names for pairwise analysis
            chain_list = list(chains.keys())

            # Initialize interface features
            interface_features = {}

            # Analyze each pair of chains
            for i, chain1_id in enumerate(chain_list):
                for chain2_id in chain_list[i+1:]:
                    pair_key = f"{chain1_id}_{chain2_id}"

                    try:
                        chain1 = chains[chain1_id]
                        chain2 = chains[chain2_id]

                        # Get all atoms from both chains (for distance calculations)
                        chain1_atoms = []
                        chain2_atoms = []

                        for residue in chain1:
                            if is_aa(residue, standard=True):
                                chain1_atoms.extend(list(residue.get_atoms()))

                        for residue in chain2:
                            if is_aa(residue, standard=True):
                                chain2_atoms.extend(list(residue.get_atoms()))

                        if not chain1_atoms or not chain2_atoms:
                            continue

                        # Find residues in chain1 that are within 5Å of chain2
                        chain1_interface_residues = set()
                        chain2_interface_residues = set()

                        # For each residue in chain1, check if any atom is close to chain2
                        for residue1 in chain1:
                            if not is_aa(residue1, standard=True):
                                continue

                            residue1_atoms = list(residue1.get_atoms())
                            if not residue1_atoms:
                                continue

                            # Check if any atom in this residue is close to chain2
                            for atom1 in residue1_atoms:
                                for atom2 in chain2_atoms:
                                    if (atom1 - atom2) <= radius:  # 5Å distance threshold
                                        chain1_interface_residues.add(residue1.id[1])  # Use residue number
                                        break
                                if residue1.id[1] in chain1_interface_residues:
                                    break

                        # For each residue in chain2, check if any atom is close to chain1
                        for residue2 in chain2:
                            if not is_aa(residue2, standard=True):
                                continue

                            residue2_atoms = list(residue2.get_atoms())
                            if not residue2_atoms:
                                continue

                            # Check if any atom in this residue is close to chain1
                            for atom2 in residue2_atoms:
                                for atom1 in chain1_atoms:
                                    if (atom2 - atom1) <= radius:  # 5Å distance threshold
                                        chain2_interface_residues.add(residue2.id[1])  # Use residue number
                                        break
                                if residue2.id[1] in chain2_interface_residues:
                                    break

                        # Calculate interface features for this chain pair
                        chain1_residue_count = sum(1 for _ in chain1 if is_aa(_, standard=True))
                        chain2_residue_count = sum(1 for _ in chain2 if is_aa(_, standard=True))

                        interface_features[pair_key] = {
                            'chain1_interface_residues': len(chain1_interface_residues),
                            'chain2_interface_residues': len(chain2_interface_residues),
                            'total_interface_residues': len(chain1_interface_residues) + len(chain2_interface_residues),
                            'interface_residue_ratio': len(chain1_interface_residues) / len(chain2_interface_residues) if len(chain2_interface_residues) > 0 else 0,
                            'chain1_residue_count': chain1_residue_count,
                            'chain2_residue_count': chain2_residue_count,
                            'chain1_interface_fraction': len(chain1_interface_residues) / chain1_residue_count if chain1_residue_count > 0 else 0,
                            'chain2_interface_fraction': len(chain2_interface_residues) / chain2_residue_count if chain2_residue_count > 0 else 0,
                            'avg_interface_fraction': (len(chain1_interface_residues)/chain1_residue_count + len(chain2_interface_residues)/chain2_residue_count)/2 if chain1_residue_count > 0 and chain2_residue_count > 0 else 0
                        }

                    except Exception as e:
                        print(f"  Error analyzing chain pair {pair_key}: {e}")
                        continue

            if not interface_features:
                features['error'] = "No interface features could be extracted"
                return features

            # Aggregate features across all chain pairs
            features['success'] = True
            features['num_chain_pairs'] = len(interface_features)

            # Add aggregated statistics
            all_chain1_residues = [v['chain1_interface_residues'] for v in interface_features.values()]
            all_chain2_residues = [v['chain2_interface_residues'] for v in interface_features.values()]
            all_total_residues = [v['total_interface_residues'] for v in interface_features.values()]
            all_chain1_fractions = [v['chain1_interface_fraction'] for v in interface_features.values()]
            all_chain2_fractions = [v['chain2_interface_fraction'] for v in interface_features.values()]
            all_avg_fractions = [v['avg_interface_fraction'] for v in interface_features.values()]

            if all_chain1_residues:
                features['avg_chain1_interface_residues'] = np.mean(all_chain1_residues)
                features['std_chain1_interface_residues'] = np.std(all_chain1_residues)
                features['max_chain1_interface_residues'] = np.max(all_chain1_residues)
                features['min_chain1_interface_residues'] = np.min(all_chain1_residues)

            if all_chain2_residues:
                features['avg_chain2_interface_residues'] = np.mean(all_chain2_residues)
                features['std_chain2_interface_residues'] = np.std(all_chain2_residues)
                features['max_chain2_interface_residues'] = np.max(all_chain2_residues)
                features['min_chain2_interface_residues'] = np.min(all_chain2_residues)

            if all_total_residues:
                features['avg_total_interface_residues'] = np.mean(all_total_residues)
                features['std_total_interface_residues'] = np.std(all_total_residues)
                features['max_total_interface_residues'] = np.max(all_total_residues)
                features['min_total_interface_residues'] = np.min(all_total_residues)

            if all_chain1_fractions:
                features['avg_chain1_interface_fraction'] = np.mean(all_chain1_fractions)
                features['std_chain1_interface_fraction'] = np.std(all_chain1_fractions)

            if all_chain2_fractions:
                features['avg_chain2_interface_fraction'] = np.mean(all_chain2_fractions)
                features['std_chain2_interface_fraction'] = np.std(all_chain2_fractions)

            if all_avg_fractions:
                features['avg_interface_fraction'] = np.mean(all_avg_fractions)
                features['std_interface_fraction'] = np.std(all_avg_fractions)

            # Add features from the first chain pair (most representative)
            if interface_features:
                first_pair = list(interface_features.values())[0]
                for key, value in first_pair.items():
                    features[f'first_pair_{key}'] = value

            return features

        except Exception as e:
            features['error'] = f"Error extracting interface features: {str(e)}"
            return features


class CroissantDatasetLoader:
    """Loader for Croissant-formatted protein interaction datasets."""

    def __init__(self, dataset_source: str, is_github: bool = False):
        """
        Initialize the loader with dataset source.

        Args:
            dataset_source: Path to local directory or GitHub repo URL
            is_github: True if source is GitHub repo, False if local directory
        """
        self.dataset_source = dataset_source
        self.is_github = is_github
        self.dataset = None
        self.interfaces = []
        self.dataframe = None
        self.numerical_features = []
        self.categorical_features = []
        self.all_features_info = {}
        self.cluster_ids = None
        self.pdb_sources = {}  # Store PDB source information
        self.temp_dir = None  # Initialize temp_dir

    def _download_from_github(self, repo_url: str):
        """
        Download dataset files from GitHub repository.

        Args:
            repo_url: GitHub repository URL

        Returns:
            Path to local directory containing downloaded files
        """
        print(f"Downloading from GitHub: {repo_url}")

        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="croissant_ml_")
        print(f"Created temp directory: {self.temp_dir}")

        try:
            # Parse GitHub URL
            if 'github.com' not in repo_url:
                raise ValueError("Not a valid GitHub URL")

            # Extract owner and repo name
            parts = repo_url.replace('https://github.com/', '').strip('/').split('/')
            if len(parts) < 2:
                raise ValueError("Invalid GitHub URL format")

            owner, repo = parts[0], parts[1]

            # Try to download the dataset_with_interfaces.json file directly
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/bioschemas_output/dataset_with_interfaces.json"

            print(f"Attempting to download: {raw_url}")
            response = requests.get(raw_url, timeout=30)

            if response.status_code == 200:
                # Save the file
                local_path = Path(self.temp_dir) / "dataset_with_interfaces.json"
                with open(local_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                print(f"✅ Successfully downloaded dataset file")
                return str(local_path)
            else:
                # Try alternative paths
                print(f"Primary URL failed (HTTP {response.status_code}), trying alternatives...")

                # Try with /master instead of /main
                raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/master/bioschemas_output/dataset_with_interfaces.json"
                response = requests.get(raw_url, timeout=30)

                if response.status_code == 200:
                    local_path = Path(self.temp_dir) / "dataset_with_interfaces.json"
                    with open(local_path, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    print(f"✅ Successfully downloaded dataset file (master branch)")
                    return str(local_path)

                # Last attempt: try the original repo URL structure
                print("Trying original repository structure...")
                return self._download_full_repo(owner, repo)

        except Exception as e:
            print(f"❌ Error downloading from GitHub: {e}")
            if self.temp_dir and Path(self.temp_dir).exists():
                try:
                    shutil.rmtree(self.temp_dir)
                except:
                    pass
            raise

    def _download_full_repo(self, owner: str, repo: str):
        """
        Download the full repository (fallback method).

        Args:
            owner: GitHub owner username
            repo: Repository name

        Returns:
            Path to local directory
        """
        try:
            # Get repository contents
            api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/"
            response = requests.get(api_url, timeout=30)

            if response.status_code != 200:
                raise ValueError(f"GitHub API error: {response.status_code}")

            contents = response.json()

            # Look for bioschemas_output directory
            for item in contents:
                if item['name'] == 'bioschemas_output' and item['type'] == 'dir':
                    # Get contents of bioschemas_output
                    bioschemas_url = f"https://api.github.com/repos/{owner}/{repo}/contents/bioschemas_output"
                    bioschemas_response = requests.get(bioschemas_url, timeout=30)

                    if bioschemas_response.status_code == 200:
                        bioschemas_contents = bioschemas_response.json()

                        # Download dataset file
                        for file_item in bioschemas_contents:
                            if file_item['name'] == 'dataset_with_interfaces.json':
                                download_url = file_item['download_url']
                                file_response = requests.get(download_url, timeout=30)

                                if file_response.status_code == 200:
                                    local_path = Path(self.temp_dir) / "dataset_with_interfaces.json"
                                    with open(local_path, 'w', encoding='utf-8') as f:
                                        f.write(file_response.text)
                                    print(f"✅ Successfully downloaded dataset file via API")
                                    return str(local_path)

            raise FileNotFoundError("Could not find dataset_with_interfaces.json in repository")

        except Exception as e:
            print(f"❌ Error in fallback download: {e}")
            raise

    def load_dataset(self):
        """Load the Croissant dataset from local file or GitHub."""
        try:
            if self.is_github:
                print(f"Loading from GitHub repository: {self.dataset_source}")
                dataset_path = self._download_from_github(self.dataset_source)
            else:
                print(f"Loading from local directory: {self.dataset_source}")
                dataset_path = Path(self.dataset_source) / "dataset_with_interfaces.json"

                if not dataset_path.exists():
                    # Try alternative locations
                    alt_paths = [
                        Path(self.dataset_source) / "bioschemas_output" / "dataset_with_interfaces.json",
                        Path(self.dataset_source) / "dataset_croissant.json",
                        Path(self.dataset_source) / "bioschemas_output" / "dataset_croissant.json"
                    ]

                    for alt_path in alt_paths:
                        if alt_path.exists():
                            dataset_path = alt_path
                            print(f"Found dataset at alternative location: {dataset_path}")
                            break

                    if not dataset_path.exists():
                        raise FileNotFoundError(f"Could not find dataset file in {self.dataset_source}")

            print(f"Loading dataset from: {dataset_path}")

            with open(dataset_path, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)

            print(f"✅ Dataset loaded successfully!")
            print(f"   Dataset name: {self.dataset.get('name', 'Unknown')}")
            print(f"   Description: {self.dataset.get('description', 'No description')[:100]}...")

            # Extract interface items
            if 'hasPart' in self.dataset:
                self.interfaces = self.dataset['hasPart']
                print(f"   Number of interfaces: {len(self.interfaces)}")
            else:
                print("⚠️  Warning: No 'hasPart' field found in dataset")
                self.interfaces = []

            return True

        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False

    def extract_pdb_sources_from_representations(self, pdb_local_dir: str = None, use_pdb_format: bool = False):
        """
        Extract structure source information from the dataset representations.
        
        Args:
            pdb_local_dir: Local directory for structure files
            use_pdb_format: True to extract 'PDB Structure', False for 'mmCIF Structure'
        """
        print("Extracting structure source information from dataset representations...")
        
        if not self.dataset or 'hasPart' not in self.dataset:
            print("  No 'hasPart' found in dataset")
            return
        
        has_part_items = self.dataset['hasPart']
        if not isinstance(has_part_items, list):
            print("  'hasPart' is not a list")
            return
        
        # Look for DataCatalog items in hasPart
        data_catalog_items = []
        for item in has_part_items:
            if isinstance(item, dict):
                # Check if this is a DataCatalog by type
                item_type = item.get('@type', item.get('type', ''))
                if 'DataCatalog' in str(item_type):
                    data_catalog_items.append(item)
        
        if not data_catalog_items:
            print("  No DataCatalog items found in hasPart")
            return
        
        print(f"  Found {len(data_catalog_items)} DataCatalog item(s)")
        
        # Determine which representation type to extract based on use_pdb_format
        target_representation = 'PDB Structure' if use_pdb_format else 'mmCIF Structure'
        print(f"  Looking for '{target_representation}' representations")
        
        # Track downloads
        downloaded_count = 0
        skipped_count = 0
        
        # Process each DataCatalog item
        for catalog_idx, catalog_item in enumerate(data_catalog_items):
            #print(f"  Processing DataCatalog item {catalog_idx + 1}...")
            
            # Check for mainEntity within DataCatalog
            if 'mainEntity' not in catalog_item:
                print(f"    No mainEntity found in DataCatalog item {catalog_idx + 1}")
                continue
            
            main_entity = catalog_item['mainEntity']
            if not isinstance(main_entity, dict):
                print(f"    mainEntity is not a dictionary in DataCatalog item {catalog_idx + 1}")
                continue
            
            # Check for hasRepresentation within mainEntity
            if 'hasRepresentation' not in main_entity:
                print(f"    No hasRepresentation found in mainEntity of DataCatalog item {catalog_idx + 1}")
                continue
            
            representations = main_entity['hasRepresentation']
            if not isinstance(representations, list):
                print(f"    hasRepresentation is not a list in DataCatalog item {catalog_idx + 1}")
                continue
            
            #print(f"    Found {len(representations)} representation(s)")
            
            # Process each representation
            for rep_idx, representation in enumerate(representations):
                if not isinstance(representation, dict):
                    continue
                
                rep_name = representation.get('name', '').strip()
                #print(f"    Representation {rep_idx + 1}: {rep_name}")
                
                # Only process the target representation type based on use_pdb_format
                if rep_name != target_representation:
                    #print(f"      Skipping - not '{target_representation}'")
                    continue
                
                # Get the value field which contains the URL/path
                value = representation.get('value')
                if not value:
                    print(f"      No 'value' field found for {rep_name}")
                    continue
                
                # Extract Interface ID from value field
                interface_id = None
                if isinstance(value, str):
                    # Split by "/" and take the last element
                    filename = value.split('/')[-1]
                    # Remove file extension. file_id is defined because 
                    # mmcif file name exception.
                    file_id = filename.split('.')[0]
                    interface_id = catalog_item.get('identifier','').lower()
                    print(f"      Extracted Interface ID: {interface_id}")
                
                if not interface_id:
                    print(f"      Could not extract Interface ID from value: {value}")
                    continue
                
                # Use the value as the URL
                content_url = value if isinstance(value, str) and value.startswith('http') else None
                
                if not content_url:
                    print(f"      Value is not a valid URL: {value}")
                    continue
                
                # Store the representation with interface ID
                source_id = interface_id
                self.pdb_sources[source_id] = {
                    'interface_id': interface_id,
                    'name': rep_name,
                    'value': value,
                    'url': content_url,
                    'representation': representation,
                    'file_type': 'pdb' if use_pdb_format else 'cif',
                    'format': 'PDB' if use_pdb_format else 'mmCIF',
                    'downloaded': False,
                    'local_path': None
                }
                
                print(f"      Added structure source: {interface_id} ({'PDB' if use_pdb_format else 'mmCIF'})")
                
                # Check if file already exists locally BEFORE downloading
                if pdb_local_dir:
                    file_ext = '.pdb' if use_pdb_format else '.cif'
                    
                    # Handle case sensitivity: try both uppercase and lowercase
                    interface_variants = [file_id]
                    if interface_id.isupper():
                        interface_variants.append(interface_id.lower())
                    elif interface_id.islower():
                        interface_variants.append(interface_id.upper())
                    
                    # Check for various possible file locations

                    file_found = False
                    local_path = None
                    
                    for variant in interface_variants:
                        # Check compressed version
                        compressed_file = Path(pdb_local_dir) / f"{variant}{file_ext}.gz"
                        if compressed_file.exists():
                            file_found = True
                            local_path = str(compressed_file)
                            break
                        
                        # Check uncompressed version
                        uncompressed_file = Path(pdb_local_dir) / f"{variant}{file_ext}"
                        if uncompressed_file.exists():
                            file_found = True
                            local_path = str(uncompressed_file)
                            break
                        
                    
                    if file_found:
                        print(f"      File already exists locally: {Path(local_path).name}")
                        self.pdb_sources[source_id]['local_path'] = local_path
                        self.pdb_sources[source_id]['downloaded'] = True
                        skipped_count += 1
                    else:
                        # File doesn't exist, try to download from the repository URL
                        print(f"      File not found locally, downloading from repository URL...")
                        # Download using file_id
                        self._download_interface_file(file_id, content_url, pdb_local_dir, use_pdb_format)
                        
                        # Check if download was successful (using lowercase variant)
                        lowercase_id = interface_id.lower()
                        downloaded_file = Path(pdb_local_dir) / f"{lowercase_id}{file_ext}.gz"
                        if downloaded_file.exists():
                            self.pdb_sources[source_id]['local_path'] = str(downloaded_file)
                            self.pdb_sources[source_id]['downloaded'] = True
                            downloaded_count += 1
                        else:
                            print(f"      ⚠️  Download failed for {interface_id}")
        
        print(f"  Total structure representations found: {len(self.pdb_sources)}")
        print(f"  Files already available locally: {skipped_count}")
        print(f"  Files downloaded from repository: {downloaded_count}")
        print(f"  Files missing: {len(self.pdb_sources) - skipped_count - downloaded_count}")
        
        # Print summary
        if self.pdb_sources:
            print(f"\n  Interface files status:")
            for source_id, source in list(self.pdb_sources.items())[:10]:
                interface_id = source.get('interface_id', 'unknown')
                format_type = source.get('format', 'unknown')
                has_file = source.get('downloaded', False)
                status = "✓ Available" if has_file else "✗ Missing"
                print(f"    - {interface_id} (Format: {format_type}): {status}")
            
            if len(self.pdb_sources) > 10:
                print(f"    ... and {len(self.pdb_sources) - 10} more interfaces")

    def _download_interface_file(self, interface_id: str, content_url: str,
                               pdb_local_dir: str, use_pdb_format: bool = False):
        """
        Download interface file from the provided URL.
        Always saves as compressed .gz format (BioPython compatible).
        
        Args:
            interface_id: Interface ID extracted from value field
            content_url: URL to download from
            pdb_local_dir: Local directory for files
            use_pdb_format: True for PDB format, False for mmCIF format
        """
        # Determine file extension
        file_ext = '.pdb' if use_pdb_format else '.cif'
        
        # Create local directory
        Path(pdb_local_dir).mkdir(parents=True, exist_ok=True)
        
        # Use lowercase filename
        lowercase_id = interface_id.lower()
        final_filename = f"{lowercase_id}{file_ext}.gz"
        compressed_file = Path(pdb_local_dir) / final_filename
        
        # Check if file already exists
        if compressed_file.exists():
            return
        
        # Download file
        try:
            response = requests.get(content_url, timeout=60)
            
            if response.status_code == 200:
                # Check if URL is already compressed
                if content_url.lower().endswith('.gz'):
                    # Save compressed file directly
                    with open(compressed_file, 'wb') as f:
                        f.write(response.content)
                else:
                    # Compress and save
                    with gzip.open(compressed_file, 'wb') as f:
                        f.write(response.content)
                        
            else:
                print(f"      HTTP Error {response.status_code} for {interface_id}")
                
        except Exception as e:
            print(f"      Error downloading {interface_id}: {e}")

    def _download_interface_file3(self, interface_id: str, content_url: str,
                               pdb_local_dir: str, use_pdb_format: bool = False):
        """
        Download interface file from the provided URL.
        Handles case sensitivity by saving with lowercase filename.
        
        Args:
            interface_id: Interface ID extracted from value field
            content_url: URL to download from (from representation value)
            pdb_local_dir: Local directory for files
            use_pdb_format: True for PDB format, False for mmCIF format
        """
        # Determine file extension based on format
        file_ext = '.pdb' if use_pdb_format else '.cif'
        
        # Create local directory if it doesn't exist
        Path(pdb_local_dir).mkdir(parents=True, exist_ok=True)
        
        # Handle case sensitivity: use lowercase filename
        lowercase_id = interface_id.lower()
        
        # Check for existing files (both compressed and uncompressed)
        uncompressed_file = Path(pdb_local_dir) / f"{lowercase_id}{file_ext}"
        compressed_file = Path(pdb_local_dir) / f"{lowercase_id}{file_ext}.gz"
        
        # Also check with original case
        uncompressed_orig = Path(pdb_local_dir) / f"{interface_id}{file_ext}"
        compressed_orig = Path(pdb_local_dir) / f"{interface_id}{file_ext}.gz"
        
        # Check if any version exists
        existing_files = []
        for file_path in [compressed_file, uncompressed_file,
                         compressed_orig, uncompressed_orig]:
            if file_path.exists():
                existing_files.append(file_path)
        
        if existing_files:
            print(f"      File already exists: {existing_files[0].name}")
            
            # BioPython can handle gzipped files directly, so prefer compressed
            for file_path in existing_files:
                if file_path.endswith('.gz'):
                    print(f"      Using gzipped file (BioPython compatible)")
                    return
            
            # If no compressed files but we have uncompressed, compress it
            for file_path in existing_files:
                if not file_path.endswith('.gz'):
                    # Create compressed version
                    compressed_version = Path(str(file_path) + '.gz')
                    self._compress_file(file_path, compressed_version)
                    print(f"      Created compressed version: {compressed_version.name}")
                    return
            
            return
        
        # File doesn't exist, download from the provided URL
        print(f"      Downloading {interface_id} from repository URL...")
        print(f"      Source URL: {content_url}")
        print(f"      Will save as: {lowercase_id}{file_ext}.gz (lowercase to avoid case issues)")
        
        try:
            import time
            
            # Set headers to mimic browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Try to download with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(content_url, headers=headers, timeout=60)
                    
                    if response.status_code == 200:
                      
                        # Save to local directory (uncompressed first)
                        with open(uncompressed_file, 'wb') as f:
                            f.write(response.content)
                        
                        # Check file size
                        file_size = os.path.getsize(uncompressed_file)
                        if file_size > 1000:  # Reasonable minimum
                            print(f"      ✅ Downloaded {file_size:,} bytes")
                            
                            # Compress the file (BioPython can handle gzipped files)
                            self._compress_file(uncompressed_file, compressed_file)
                            print(f"      ✅ Compressed to: {compressed_file.name}")
                            
                            # Remove uncompressed version
                            uncompressed_file.unlink()
                            
                            return
                        else:
                            print(f"      ⚠️  File too small ({file_size} bytes), might be invalid")
                            uncompressed_file.unlink()
                            
                    elif response.status_code == 404:
                        print(f"      ❌ File not found (404) at URL")
                        break
                    else:
                        print(f"      ❌ HTTP Error {response.status_code}")
                        
                except requests.exceptions.Timeout:
                    print(f"      ⏱️  Timeout on attempt {attempt + 1}/{max_retries}")
                except requests.exceptions.ConnectionError:
                    print(f"      🔌 Connection error on attempt {attempt + 1}/{max_retries}")
                except Exception as e:
                    print(f"      ❌ Error on attempt {attempt + 1}/{max_retries}: {e}")
                
                # Wait before retry
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            print(f"      ❌ Failed to download after {max_retries} attempts")
            
        except ImportError:
            print(f"      ❌ 'requests' library not installed. Cannot download file.")
        except Exception as e:
            print(f"      ❌ Unexpected error: {e}")

    def _compress_file(self, input_path: Path, output_path: Path):
        """
        Compress a file using gzip.
        
        Args:
            input_path: Path to input file
            output_path: Path to output gzipped file
        """
        try:
            with open(input_path, 'rb') as f_in:
                with gzip.open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"      Compressed {input_path.name} to {output_path.name}")
        except Exception as e:
            print(f"      Error compressing file: {e}")

    def _is_pdb_representation(self, rep: Dict) -> bool:
        """
        Check if a representation is a PDB/mmCIF file.

        Args:
            rep: Representation dictionary

        Returns:
            True if it's a PDB representation, False otherwise
        """
        if not isinstance(rep, dict):
            return False

        # Check encodingFormat
        encoding_format = rep.get('encodingFormat', '').lower()
        if any(pdb_term in encoding_format for pdb_term in ['chemical/x-pdb', 'text/pdb', 'pdb', 'cif', 'mmcif']):
            return True

        # Check name
        name = rep.get('name', '').lower()
        if any(pdb_term in name for pdb_term in ['pdb', 'mmcif', 'cif', 'structure']):
            return True

        # Check identifier for PDB pattern
        identifier = rep.get('identifier', '')
        if isinstance(identifier, str):
            import re
            match = re.search(r'([0-9][A-Z0-9]{3})', identifier.upper())
            if match:
                return True

        # Check description
        description = rep.get('description', '').lower()
        if any(pdb_term in description for pdb_term in ['pdb', 'mmcif', 'cif', 'protein data bank']):
            return True

        # Check contentUrl for PDB patterns
        for url_field in ['contentUrl', 'url']:
            if url_field in rep:
                content_url = rep[url_field]
                if isinstance(content_url, str):
                    if '.pdb' in content_url.lower() or '.cif' in content_url.lower():
                        return True
                    # Check for PDB ID pattern in URL
                    import re
                    match = re.search(r'/([0-9][a-z0-9]{3})\.(pdb|cif|mmcif)', content_url.lower())
                    if match:
                        return True

        return False

    def check_interface_file_availability2(self, interface_id: str, pdb_local_dir: str, 
                                        use_pdb_format: bool = False) -> Tuple[bool, Optional[Path]]:
        """
        Check if a specific interface's structure file is available locally.
        Handles case sensitivity issues.
        
        Args:
            interface_id: The interface ID to check
            pdb_local_dir: Local directory to check
            use_pdb_format: True for PDB format, False for mmCIF
            
        Returns:
            Tuple of (is_available, file_path_if_available)
        """
        if not pdb_local_dir:
            return False, None
        
        pdb_local_path = Path(pdb_local_dir)
        if not pdb_local_path.exists():
            return False, None
        
        file_ext = '.pdb' if use_pdb_format else '.cif'
        alt_ext = '.cif' if use_pdb_format else '.pdb'
        
        # Handle case sensitivity: try both uppercase and lowercase
        interface_variants = [interface_id]
        if interface_id.isupper():
            interface_variants.append(interface_id.lower())
        elif interface_id.islower():
            interface_variants.append(interface_id.upper())
        
        for variant in interface_variants:
            # Check for compressed version first (preferred - BioPython compatible)
            compressed_file = pdb_local_path / f"{variant}{file_ext}.gz"
            if compressed_file.exists():
                return True, compressed_file
            
            # Check for uncompressed version
            uncompressed_file = pdb_local_path / f"{variant}{file_ext}"
            if uncompressed_file.exists():
                return True, uncompressed_file
            
            # Check for alternative format
            alt_compressed = pdb_local_path / f"{variant}{alt_ext}.gz"
            if alt_compressed.exists():
                return True, alt_compressed
            
            alt_uncompressed = pdb_local_path / f"{variant}{alt_ext}"
            if alt_uncompressed.exists():
                return True, alt_uncompressed
        
        return False, None

    def get_chain_ids_for_interface(self, interface: Dict, pdb_format: bool = False) -> List[str]:
        """
        Extract chain IDs from interface properties.
        
        Args:
            interface: Interface dictionary
            pdb_format: True/False
        
        Returns:
            List of chain IDs
        """
        chain_labels = []
        chain_auths = []
        additional_props = interface.get('additionalProperty', [])
        chain_prop = 'labelchain' if pdb_format else 'authchain'        

        for prop in additional_props:
            prop_name = prop.get('name', '').lower()
            prop_value = str(prop.get('value', ''))
            
            # Look for chain identifiers
              
            if 'labelchain' in prop_name and prop_value and len(prop_value) <= 2:
                chain_id = prop_value.strip().upper()
                if chain_id and chain_id.isalnum():
                    chain_labels.append(chain_id)

            if 'authchain' in prop_name and prop_value and len(prop_value) <= 2:
                chain_id = prop_value.strip().upper()
                if chain_id and chain_id.isalnum():
                    chain_auths.append(chain_id)

        # Check labels 
        if len(chain_auths) == 0 and not pdb_format : chain_labels=[]  
        return chain_labels

    def get_pdb_representation_for_interface(self, interface: Dict) -> Optional[Dict]:
        """
        Get structure representation for a specific interface.
        
        Args:
            interface: Interface dictionary
        
        Returns:
            Structure representation dictionary or None if not found
        """
        if not self.pdb_sources:
            return None
        
        # Method 1: Direct lookup by interface identifier
        interface_id = interface.get('identifier', '').strip().lower()
        if interface_id in self.pdb_sources:
            return self.pdb_sources[interface_id]
        
        # Method 2: Try interface name
        interface_name = interface.get('name', '').strip().lower()
        if interface_name in self.pdb_sources:
            return self.pdb_sources[interface_name]
        
        # Method 3: Try to find by partial match
        for source_id, source in self.pdb_sources.items():
            if interface_id and interface_id in source_id:
                return source
            if interface_name and interface_name in source_id:
                return source
            
            # Check if interface mentions this source ID
            for field in ['identifier', 'name', 'description']:
                if field in interface:
                    field_value = str(interface[field])
                    if source_id in field_value:
                        return source
        
        # Method 4: Check for direct structure reference
        if 'subjectOf' in interface:
            subject_of = interface['subjectOf']
            if isinstance(subject_of, dict) and self._is_pdb_representation(subject_of):
                return subject_of
        
        return None

    def cleanup(self):
        """Clean up temporary files if downloaded from GitHub."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temp directory: {self.temp_dir}")
            except Exception as e:
                print(f"Warning: Could not clean up temp directory: {e}")

    def _is_numeric(self, value):
        """Check if a value is numeric (int or float) with robust handling."""
        if value is None:
            return False

        # Handle iterables (arrays, lists) - they shouldn't be checked as single values
        if hasattr(value, '__iter__') and not isinstance(value, str):
            return False

        # Handle pandas null values
        try:
            if pd.isna(value):
                return False
        except:
            pass

        # Handle string representations
        if isinstance(value, str):
            # Clean the string
            value = value.strip()

            # Skip empty strings
            if value == '':
                return False

            # Check for special cases like "1_155" which shouldn't be numeric
            if '_' in value or ':' in value or any(c.isalpha() for c in value):
                return False

            # Try to convert to float
            try:
                float(value)
                return True
            except (ValueError, TypeError):
                return False

        # Handle numeric types directly
        if isinstance(value, (int, float, np.integer, np.floating)):
            return True

        return False

    def extract_features_labels(self):
        """
        Extract features and labels from the dataset interfaces.
        Returns ClusterIDs along with features and labels.
        
        Returns:
            Tuple of (features_df, labels_series, cluster_ids_series) or (None, None, None) if extraction fails
        """
        if not self.interfaces:
            print("❌ No interfaces available. Load dataset first.")
            return None, None, None
        
        print("\n=== EXTRACTING FEATURES AND LABELS ===")
        
        features_list = []
        labels_list = []
        cluster_ids_list = []
        
        # First pass: collect all features and sample values
        print("   Scanning interfaces to identify feature types...")
        
        for i, interface in enumerate(self.interfaces[:100]):  # Sample first 100 interfaces
            try:
                additional_props = interface.get('additionalProperty', [])
                
                for prop in additional_props:
                    prop_name = prop.get('name', '')
                    prop_value = prop.get('value')
                    
                    if prop_name in ['physio', 'label', 'ClusterID']:
                        continue
                    
                    if prop_name not in self.all_features_info:
                        self.all_features_info[prop_name] = []
                    
                    # Collect sample values (up to 5 per feature)
                    if len(self.all_features_info[prop_name]) < 5:
                        self.all_features_info[prop_name].append(prop_value)
                        
            except Exception as e:
                continue
        
        # Analyze feature types based on sample values
        print("   Analyzing feature types...")
        for feature_name, samples in self.all_features_info.items():
            if not samples:
                continue
            
            # Clean samples (remove None/NaN)
            clean_samples = []
            for s in samples:
                if s is None:
                    continue
                try:
                    if pd.isna(s):
                        continue
                except:
                    pass
                clean_samples.append(s)
            
            if not clean_samples:
                continue
            
            # Test if all samples are numeric
            all_numeric = True
            for s in clean_samples:
                if not self._is_numeric(s):
                    all_numeric = False
                    break
            
            # Check for chain-like features
            is_chain_feature = any(chain_term in feature_name.lower()
                                  for chain_term in ['chain', 'authchain'])
            
            # Check for symmetry/transformation features
            is_symmetry_feature = any(sym_term in feature_name.lower()
                                     for sym_term in ['symmetry', 'symop', 'symm'])
            
            # Store type info
            self.all_features_info[feature_name] = {
                'type': 'numeric' if all_numeric else 'categorical',
                'is_chain': is_chain_feature,
                'is_symmetry': is_symmetry_feature,
                'samples': samples
            }
        
        # Second pass: extract all data
        print("   Extracting data from all interfaces...")
        
        for i, interface in enumerate(self.interfaces):
            try:
                # Extract interface ID
                interface_id = interface.get('identifier', f'interface_{i}')
                
                # Extract label (physio)
                label = None
                cluster_id = None
                additional_props = interface.get('additionalProperty', [])
                
                # Look for physio and ClusterID properties
                for prop in additional_props:
                    if prop.get('name') == 'physio':
                        label = prop.get('value')
                    elif prop.get('name') == 'ClusterID':
                        cluster_id = prop.get('value')
                
                if label is None:
                    # Try to find label property
                    for prop in additional_props:
                        if prop.get('name') == 'label':
                            label = bool(prop.get('value')) if isinstance(prop.get('value'), (int, float)) else prop.get('value')
                            break
                
                # Skip if no label found
                if label is None:
                    continue
                
                # Initialize features dictionary with interface_id
                features = {
                    'interface_id': interface_id,
                }
                
                # Extract all properties
                for prop in additional_props:
                    prop_name = prop.get('name', '')
                    prop_value = prop.get('value')
                    
                    # Skip physio/label/ClusterID (already used as target/grouping)
                    if prop_name in ['physio', 'label', 'ClusterID']:
                        continue
                    
                    # Get feature info
                    feature_info = self.all_features_info.get(prop_name, {'type': 'unknown'})
                    
                    # Handle based on feature type
                    if feature_info.get('type') == 'numeric' and self._is_numeric(prop_value):
                        # Store as float if numeric
                        try:
                            features[prop_name] = float(prop_value) if prop_value is not None else None
                        except (ValueError, TypeError):
                            features[prop_name] = str(prop_value) if prop_value is not None else None
                    elif feature_info.get('is_chain'):
                        # Handle chain identifiers specially
                        if 'chains' not in features:
                            features['chains'] = str(prop_value) if prop_value is not None else None
                        elif features['chains'] is not None and prop_value is not None:
                            features['chains'] = f"{features['chains']}-{prop_value}"
                    else:
                        # Store as string for categorical features
                        features[prop_name] = str(prop_value) if prop_value is not None else None
                
                features_list.append(features)
                labels_list.append(label)
                cluster_ids_list.append(cluster_id)
                
                if len(features_list) % 100 == 0 and len(features_list) > 0:
                    print(f"   Processed {len(features_list)} interfaces...")
                    
            except Exception as e:
                continue
        
        if not features_list:
            print("❌ No valid features extracted")
            return None, None, None
        
        # Create DataFrame
        self.dataframe = pd.DataFrame(features_list)
        self.cluster_ids = pd.Series(cluster_ids_list)
        
        # Determine numerical and categorical features from actual data
        print("   Determining feature types from extracted data...")
        self.numerical_features = []
        self.categorical_features = []
        
        for col in self.dataframe.columns:
            if col == 'interface_id' or col == 'chains':
                continue
            
            # Skip if all values are None/NaN
            if self.dataframe[col].isna().all():
                continue
            
            # Try to convert to numeric
            try:
                numeric_series = pd.to_numeric(self.dataframe[col], errors='coerce')
                non_nan_count = numeric_series.notna().sum()
                total_count = self.dataframe[col].notna().sum()
                
                if total_count > 0 and (non_nan_count / total_count) > 0.8:
                    # Check for suspicious patterns
                    sample_values = self.dataframe[col].dropna().head(10).astype(str)
                    suspicious_patterns = any('_' in str(v) or ':' in str(v) or
                                             (len(str(v)) > 10 and any(c.isalpha() for c in str(v)))
                                             for v in sample_values)
                    
                    if not suspicious_patterns:
                        self.numerical_features.append(col)
                        self.dataframe[col] = numeric_series
                    else:
                        self.categorical_features.append(col)
                else:
                    self.categorical_features.append(col)
                    
            except Exception as e:
                self.categorical_features.append(col)
        
        # Convert labels to binary (True/False to 1/0)
        labels_series = pd.Series(labels_list)
        
        # Handle different label formats
        if labels_series.dtype == 'bool':
            labels_numeric = labels_series.astype(int)
        elif labels_series.dtype == 'object':
            # Convert string labels
            label_mapping = {
                'true': 1, 'True': 1, 'TRUE': 1, '1': 1, 1: 1,
                'false': 0, 'False': 0, 'FALSE': 0, '0': 0, 0: 0
            }
            labels_numeric = labels_series.map(lambda x: label_mapping.get(str(x).lower(), np.nan))
            labels_numeric = labels_numeric.dropna().astype(int)
        else:
            labels_numeric = labels_series.astype(int)
        
        print(f"✅ Successfully extracted {len(self.dataframe)} samples")
        print(f"   Total features found: {len(self.dataframe.columns) - 1} (excluding interface_id)")
        print(f"   Numerical features ({len(self.numerical_features)}): {sorted(self.numerical_features)}")
        print(f"   Categorical features ({len(self.categorical_features)}): {sorted(self.categorical_features)}")
        
        # Show ClusterID statistics
        cluster_id_counts = self.cluster_ids.value_counts()
        print(f"\n   ClusterID Statistics:")
        print(f"      Unique ClusterIDs: {len(cluster_id_counts)}")
        print(f"      Interfaces without ClusterID: {(self.cluster_ids.isna() | (self.cluster_ids == '')).sum()}")
        print(f"      Cluster size distribution:")
        size_counts = cluster_id_counts.value_counts().sort_index()
        for size, count in size_counts.head(10).items():
            print(f"        Size {size}: {count} clusters")
        if len(size_counts) > 10:
            print(f"        ... and {len(size_counts) - 10} more sizes")
        
        # Show class distribution
        if len(labels_numeric) > 0:
            print(f"\n   Class distribution:")
            print(f"      Physiological (1): {sum(labels_numeric == 1)} samples")
            print(f"      Non-physiological (0): {sum(labels_numeric == 0)} samples")
            if len(labels_numeric) > 0:
                print(f"      Balance ratio: {sum(labels_numeric == 1)/len(labels_numeric):.2%} positive")
        
        return self.dataframe, labels_numeric, self.cluster_ids

    def extract_pdb_contacts(self, pdb_feature_extractor=None, pdb_format=False):
        """
        Extract structural features for interfaces in the dataset.
        Focus on interface residues within 5Å.
        
        Args:
            pdb_feature_extractor: Instance of PDBFeatureExtractor
        
        Returns:
            DataFrame with structural features
        """
        print("\n=== EXTRACTING STRUCTURAL FEATURES ===")
        
        if not BIOPYTHON_AVAILABLE:
            print("⚠️  BioPython not available. Skipping structural feature extraction.")
            print("   Install with: pip install biopython")
            return None
        
        # Check if we have structure sources
        if not self.pdb_sources:
            print("⚠️  No structure sources found. Run extract_pdb_sources_from_representations() first.")
            print("   Or check if dataset contains PDB/mmCIF structure representations.")
            return None
        
        if pdb_feature_extractor is None:
            # Create extractor with default settings
            pdb_feature_extractor = PDBFeatureExtractor()
        
        pdb_features_list = []
        
        print(f"  Processing {len(self.interfaces)} interfaces for structural features...")
        print(f"  Available structure sources: {len(self.pdb_sources)}")
        
        # Track statistics
        processed_count = 0
        skipped_no_file = 0
        skipped_no_rep = 0
        successful = 0
        failed = 0
        
        # Cache for structure files to avoid re-downloading
        structure_cache = {}
        
        for i, interface in enumerate(self.interfaces):
            if i % 100 == 0 and i > 0:
                print(f"    Processed {i} interfaces...")
                print(f"      Status: {successful} successful, {failed} failed")
            
            interface_id = interface.get('identifier', f'interface_{i}')
            processed_count += 1
            
            # Get structure representation for this interface
            pdb_representation = self.get_pdb_representation_for_interface(interface)
            
            if not pdb_representation:
                # No structure representation found for this interface
                pdb_features_list.append({
                    'interface_id': interface_id,
                    'extraction_success': False,
                    'error': 'No structure representation found'
                })
                skipped_no_rep += 1
                continue
            
            # Get the interface ID from the representation (should match pdb_sources key)
            rep_interface_id = pdb_representation.get('interface_id', '')
            
            # Check if we already processed this structure file
            if rep_interface_id in structure_cache:
                pdb_file = structure_cache[rep_interface_id]
            else:
                # Get structure file path from extractor
                pdb_file = pdb_feature_extractor.get_pdb_file_from_representation(pdb_representation)
                
                if pdb_file:
                    structure_cache[rep_interface_id] = pdb_file
                else:
                    # Failed to get structure file
                    pdb_features_list.append({
                        'interface_id': interface_id,
                        'extraction_success': False,
                        'error': f'Failed to get structure file for {rep_interface_id}'
                    })
                    failed += 1
                    continue
            
            # Get chain IDs for this interface
            chain_ids = self.get_chain_ids_for_interface(interface, pdb_format)
            
            # Extract interface features (BioPython can handle gzipped files directly)
            print(f"    Processing {interface_id}: chains {chain_ids}, file: {Path(pdb_file).name}")
            interface_features = pdb_feature_extractor.extract_interface_features(interface_id, pdb_file, chain_ids)
            
            # Clean up temporary downloaded file (only if it's temporary)
            if pdb_file and os.path.exists(pdb_file) and 'tmp' in pdb_file:
                try:
                    os.unlink(pdb_file)
                except:
                    pass
            
            if interface_features['success']:
                # Create feature dictionary
                feature_dict = {
                    'interface_id': interface_id,
                    'extraction_success': True,
                    'pdb_file': Path(pdb_file).name if pdb_file else None,
                    'chain_ids': '-'.join(chain_ids)
                }
                
                # Add interface features
                for key, value in interface_features.items():
                    if key not in ['success', 'error'] and value is not None:
                        feature_dict[f'pdb_{key}'] = value
                
                pdb_features_list.append(feature_dict)
                successful += 1
            else:
                error_msg = interface_features.get('error', 'Unknown error')
                print(f"    Failed to extract features for interface {interface_id}: {error_msg}")
                pdb_features_list.append({
                    'interface_id': interface_id,
                    'extraction_success': False,
                    'error': error_msg,
                    'pdb_file': Path(pdb_file).name if pdb_file else None,
                    'chain_ids': '-'.join(chain_ids) if chain_ids else None
                })
                failed += 1
        
        # Print final statistics
        print(f"\n  Structural Feature Extraction Summary:")
        print(f"    Total interfaces processed: {processed_count}")
        print(f"    Successful extractions: {successful}")
        print(f"    Failed extractions: {failed}")
        print(f"    Skipped (no structure representation): {skipped_no_rep}")
        
        if successful > 0:
            success_rate = (successful / processed_count) * 100
            print(f"    Success rate: {success_rate:.1f}%")
            print(f"    Unique structure files used: {len(structure_cache)}")
        
        if pdb_features_list:
            pdb_features_df = pd.DataFrame(pdb_features_list)
            successful_extractions = pdb_features_df[pdb_features_df['extraction_success']]
            
            if len(successful_extractions) > 0:
                print(f"\n✅ Successfully extracted structural features for {len(successful_extractions)} interfaces")
                
                # Show feature statistics
                pdb_feature_cols = [col for col in successful_extractions.columns
                                  if col.startswith('pdb_') and not col.endswith('_success')]
                
                print(f"  Extracted {len(pdb_feature_cols)} structural feature types:")
                
                # Group features by type for better reporting
                residue_features = [c for c in pdb_feature_cols if 'residue' in c.lower()]
                fraction_features = [c for c in pdb_feature_cols if 'fraction' in c.lower()]
                other_features = [c for c in pdb_feature_cols if c not in residue_features + fraction_features]
                
                if residue_features:
                    print(f"    Residue counts: {len(residue_features)} features")
                if fraction_features:
                    print(f"    Interface fractions: {len(fraction_features)} features")
                if other_features:
                    print(f"    Other features: {len(other_features)}")
                
                # Show sample values for first successful interface
                first_success = successful_extractions.iloc[0]
                print(f"\n  Sample features for interface {first_success['interface_id']}:")
                for col in pdb_feature_cols[:5]:  # Show first 5 features
                    if col in first_success and not pd.isna(first_success[col]):
                        print(f"    {col}: {first_success[col]}")
                
                return pdb_features_df
            else:
                print("❌ No successful structural feature extractions")
                return pdb_features_df
        else:
            print("❌ No structural features extracted")
            return None

    def integrate_pdb_features(self, features_df, pdb_features_df):
        """
        Integrate structural features with existing interface features.
        
        Args:
            features_df: Existing features DataFrame
            pdb_features_df: Structural features DataFrame
        
        Returns:
            Integrated DataFrame
        """
        print("\n=== INTEGRATING STRUCTURAL FEATURES ===")
        
        if pdb_features_df is None or features_df is None:
            print("  No structural features to integrate")
            return features_df
        
        # Create a copy of features
        integrated_df = features_df.copy()
        
        # Merge on interface_id
        merged_df = pd.merge(integrated_df, pdb_features_df, on='interface_id', how='left')
        
        # Count successful integrations
        pdb_feature_cols = [col for col in pdb_features_df.columns
                           if col.startswith('pdb_')]
        
        if pdb_feature_cols:
            # Count interfaces with at least one structural feature
            integrated_count = merged_df[pdb_feature_cols[0]].notna().sum()
            print(f"✅ Successfully integrated structural features for {integrated_count} interfaces")
            print(f"   Added {len(pdb_feature_cols)} new structural features")
        else:
            print("⚠️  No structural features to add")
        
        return merged_df

    def preprocess_features(self, features_df, labels=None):
        """
        Preprocess features for machine learning.
        
        Args:
            features_df: DataFrame with features
            labels: Optional labels for stratified splitting
        
        Returns:
            Preprocessed features DataFrame
        """
        print("\n=== PREPROCESSING FEATURES ===")
        
        # Create a copy to avoid modifying original
        df = features_df.copy()
        
        # Drop identifier columns (not useful as features)
        df = df.drop(['interface_id'], axis=1, errors='ignore')
        
        # Handle chains column if present
        if 'chains' in df.columns:
            print(f"   Chains column: {df['chains'].nunique()} unique values")
            df = df.drop('chains', axis=1)
        
        # First, ensure numerical features are actually numeric
        print(f"   Ensuring numerical features are numeric...")
        for col in self.numerical_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    print(f"      {col}: {missing_count} missing values after conversion, filling with median")
                    df[col] = df[col].fillna(df[col].median())
                else:
                    print(f"      {col}: All values numeric")
        
        # Handle categorical features
        print(f"   Processing categorical features...")
        categorical_cols_in_data = [col for col in self.categorical_features if col in df.columns]
        
        for col in categorical_cols_in_data:
            unique_count = df[col].nunique()
            print(f"      {col}: {unique_count} unique values")
            
            # If reasonable number of categories, one-hot encode
            if unique_count <= 20 and unique_count > 1:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
                print(f"         One-hot encoded {col} → {dummies.shape[1]} features")
            elif unique_count == 1:
                df = df.drop(col, axis=1)
                print(f"         Dropped {col} (only one unique value)")
            else:
                df = df.drop(col, axis=1)
                print(f"         Dropped {col} (too many unique values: {unique_count})")
        
        # Handle structural features (treat most as numeric)
        pdb_feature_cols = [col for col in df.columns if col.startswith('pdb_')]
        print(f"   Processing {len(pdb_feature_cols)} structural features...")
        
        for col in pdb_feature_cols:
            if col in ['pdb_error']:
                df = df.drop(col, axis=1)
                continue
            
            # Try to convert structural features to numeric
            try:
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                non_nan_count = numeric_series.notna().sum()
                total_count = df[col].notna().sum()
                
                if total_count > 0 and (non_nan_count / total_count) > 0.5:
                    # More than 50% numeric, treat as numeric feature
                    df[col] = numeric_series
                    # Fill missing values with median
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    self.numerical_features.append(col)
                else:
                    # Drop columns with too many non-numeric values
                    df = df.drop(col, axis=1)
            except:
                # Drop problematic columns
                df = df.drop(col, axis=1)
        
        # Handle any remaining object columns
        remaining_object_cols = df.select_dtypes(include=['object']).columns.tolist()
        if remaining_object_cols:
            print(f"   Dropping remaining object columns: {remaining_object_cols}")
            df = df.drop(columns=remaining_object_cols)
        
        print(f"✅ Preprocessing complete")
        print(f"   Final feature shape: {df.shape}")
        print(f"   Total features: {len(df.columns)}")
        
        return df

    def analyze_dataset(self):
        """Analyze the dataset and show basic statistics."""
        if self.dataframe is None:
            print("❌ No data available. Load and extract features first.")
            return
        
        print("\n=== DATASET ANALYSIS ===")
        
        # Basic statistics
        print(f"Total samples: {len(self.dataframe)}")
        print(f"Total features extracted: {len(self.dataframe.columns) - 1} (excluding interface_id)")
        
        # Show structure sources found
        print(f"Structure representations found: {len(self.pdb_sources)}")
        
        # Show ClusterID analysis
        if self.cluster_ids is not None:
            print(f"\nClusterID Analysis:")
            unique_clusters = self.cluster_ids.nunique()
            print(f"  Unique ClusterIDs: {unique_clusters}")
            
            # Calculate cluster statistics
            cluster_sizes = self.cluster_ids.value_counts()
            print(f"  Cluster size distribution:")
            print(f"    Min size: {cluster_sizes.min()}")
            print(f"    Max size: {cluster_sizes.max()}")
            print(f"    Average size: {cluster_sizes.mean():.2f}")
            print(f"    Median size: {cluster_sizes.median()}")


# ============================================
# The FeatureEvaluator class (unchanged from original)
# ============================================

class FeatureEvaluator:
    """Comprehensive feature evaluation and analysis."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.feature_stats = {}
        self.correlation_matrix = None
        self.anova_results = {}
        self.mutual_info_results = {}
        self.pca_results = {}
        self.feature_importance = {}

    def compute_basic_statistics(self, X, feature_names=None):
        """
        Compute basic statistics for each feature.

        Args:
            X: Feature matrix
            feature_names: List of feature names

        Returns:
            Dictionary with statistics per feature
        """
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

        stats_dict = {}
        for i, feature in enumerate(feature_names):
            if i < X.shape[1]:
                col_data = X[:, i] if isinstance(X, np.ndarray) else X.iloc[:, i]
                col_data = col_data[~np.isnan(col_data)]  # Remove NaN

                if len(col_data) > 0:
                    stats_dict[feature] = {
                        'mean': np.mean(col_data),
                        'std': np.std(col_data),
                        'min': np.min(col_data),
                        'max': np.max(col_data),
                        'median': np.median(col_data),
                        'skewness': stats.skew(col_data) if len(col_data) > 2 else None,
                        'kurtosis': stats.kurtosis(col_data) if len(col_data) > 3 else None,
                        'missing': np.sum(np.isnan(X[:, i])) if isinstance(X, np.ndarray) else X.iloc[:, i].isna().sum(),
                        'zeros': np.sum(col_data == 0),
                        'unique_values': len(np.unique(col_data)),
                        'data_type': str(type(col_data[0])) if len(col_data) > 0 else 'unknown'
                    }

        self.feature_stats = stats_dict
        return stats_dict

    def compute_correlations(self, X, feature_names=None):
        """
        Compute correlation matrix and identify highly correlated features.

        Args:
            X: Feature matrix
            feature_names: List of feature names

        Returns:
            Correlation matrix and highly correlated feature pairs
        """
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

        # Create DataFrame for easier handling
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=feature_names)
        else:
            df = X.copy()
            if feature_names is not None and len(feature_names) == df.shape[1]:
                df.columns = feature_names

        # Compute correlation matrix
        corr_matrix = df.corr()
        self.correlation_matrix = corr_matrix

        # Find highly correlated feature pairs
        highly_correlated = []
        threshold = 0.8

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    highly_correlated.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })

        return corr_matrix, highly_correlated

    def analyze_feature_target_relationship(self, X, y, feature_names=None):
        """
        Analyze relationship between features and target variable.

        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names

        Returns:
            Dictionary with ANOVA F-values and mutual information scores
        """
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

        # Convert to numpy array for consistency
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        # ANOVA F-test (for continuous features)
        if len(np.unique(y)) > 1:
            f_values, p_values = f_classif(X_array, y)
            self.anova_results = {
                'f_values': f_values,
                'p_values': p_values,
                'significant_features': [feature_names[i] for i in range(len(f_values))
                                       if p_values[i] < 0.05]
            }

        # Mutual information (works for both continuous and categorical)
        mi_scores = mutual_info_classif(X_array, y, random_state=self.random_state)
        self.mutual_info_results = {
            'mi_scores': mi_scores,
            'top_features': [feature_names[i] for i in np.argsort(mi_scores)[::-1][:10]]
        }

        # Combine results
        results = {}
        for i, feature in enumerate(feature_names):
            if i < len(f_values):
                results[feature] = {
                    'anova_f': f_values[i] if i < len(f_values) else None,
                    'anova_p': p_values[i] if i < len(p_values) else None,
                    'mutual_info': mi_scores[i] if i < len(mi_scores) else None,
                    'significant_anova': p_values[i] < 0.05 if i < len(p_values) else False
                }

        return results

    def perform_pca_analysis(self, X, feature_names=None, n_components=None):
        """
        Perform PCA analysis to understand feature variance.

        Args:
            X: Feature matrix
            feature_names: List of feature names
            n_components: Number of PCA components (default: min(n_features, 10))

        Returns:
            PCA results including explained variance and component loadings
        """
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Determine number of components
        if n_components is None:
            n_components = min(10, X.shape[1])

        # Perform PCA
        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = pca.fit_transform(X_scaled)

        # Store results
        self.pca_results = {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'components': pca.components_,
            'n_components': n_components,
            'feature_names': feature_names
        }

        # Get component loadings
        loadings = {}
        for i in range(n_components):
            component_loadings = {}
            for j, feature in enumerate(feature_names):
                if j < len(pca.components_[i]):
                    component_loadings[feature] = pca.components_[i][j]

            # Sort by absolute loading value
            sorted_loadings = sorted(component_loadings.items(),
                                   key=lambda x: abs(x[1]), reverse=True)

            loadings[f'PC{i+1}'] = {
                'top_positive': [(k, v) for k, v in sorted_loadings if v > 0][:5],
                'top_negative': [(k, v) for k, v in sorted_loadings if v < 0][:5],
                'explained_variance': pca.explained_variance_ratio_[i]
            }

        self.pca_results['component_loadings'] = loadings

        return self.pca_results

    def evaluate_feature_importance_models(self, X, y, feature_names=None):
        """
        Evaluate feature importance using multiple models.

        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names

        Returns:
            Dictionary with feature importance from different models
        """
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

        results = {}

        # 1. Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf.fit(X, y)
        if hasattr(rf, 'feature_importances_'):
            rf_importance = rf.feature_importances_
            results['random_forest'] = {
                'importance': rf_importance,
                'top_features': [(feature_names[i], rf_importance[i])
                               for i in np.argsort(rf_importance)[::-1][:10]]
            }

        # 2. Logistic Regression (coefficient magnitude)
        try:
            lr = LogisticRegression(max_iter=1000, random_state=self.random_state, penalty='l2')
            lr.fit(X, y)
            lr_coef = np.abs(lr.coef_[0])
            results['logistic_regression'] = {
                'importance': lr_coef,
                'top_features': [(feature_names[i], lr_coef[i])
                               for i in np.argsort(lr_coef)[::-1][:10]]
            }
        except:
            pass

        # 3. Mutual Information (already computed)
        if self.mutual_info_results:
            results['mutual_information'] = {
                'importance': self.mutual_info_results['mi_scores'],
                'top_features': [(feature, self.mutual_info_results['mi_scores'][i])
                               for i, feature in enumerate(self.mutual_info_results['top_features'])]
            }

        self.feature_importance = results
        return results

    def generate_feature_report(self, X, y, feature_names=None, save_path=None):
        """
        Generate comprehensive feature evaluation report.

        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            save_path: Path to save report (optional)

        Returns:
            Dictionary with all feature evaluation results
        """
        print("\n=== COMPREHENSIVE FEATURE EVALUATION ===")

        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = list(X.columns)
            else:
                feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

        # Run all evaluations
        print("1. Computing basic statistics...")
        stats = self.compute_basic_statistics(X, feature_names)

        print("2. Analyzing feature correlations...")
        corr_matrix, high_corr = self.compute_correlations(X, feature_names)

        print("3. Analyzing feature-target relationships...")
        feature_target = self.analyze_feature_target_relationship(X, y, feature_names)

        print("4. Performing PCA analysis...")
        pca_results = self.perform_pca_analysis(X, feature_names)

        print("5. Evaluating feature importance across models...")
        importance_results = self.evaluate_feature_importance_models(X, y, feature_names)

        # Helper function to convert numpy types to Python native types
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return convert_to_serializable(obj.tolist())
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_serializable(v) for v in obj)
            elif isinstance(obj, np.generic):
                # Handle any other numpy generic types
                return obj.item()
            elif pd.isna(obj):
                return None
            else:
                return obj

        # Convert class distribution
        class_dist = dict(Counter(y))
        serializable_class_dist = {}
        for key, value in class_dist.items():
            if isinstance(key, (np.integer, np.int64)):
                serializable_class_dist[int(key)] = int(value)
            else:
                serializable_class_dist[key] = int(value)

        # Compile report
        report = {
            'dataset_info': {
                'n_samples': int(X.shape[0]),
                'n_features': int(X.shape[1]),
                'feature_names': feature_names,
                'class_distribution': serializable_class_dist
            },
            'basic_statistics': convert_to_serializable(stats),
            'correlation_analysis': {
                'highly_correlated_pairs': convert_to_serializable(high_corr),
                'n_highly_correlated': len(high_corr)
            },
            'feature_target_analysis': {
                'anova_significant_features': convert_to_serializable(self.anova_results.get('significant_features', [])),
                'mutual_info_top_features': convert_to_serializable(self.mutual_info_results.get('top_features', [])),
                'all_scores': convert_to_serializable(feature_target)
            },
            'pca_analysis': {
                'n_components': int(pca_results['n_components']),
                'explained_variance': convert_to_serializable(pca_results['explained_variance_ratio']),
                'cumulative_variance': convert_to_serializable(pca_results['cumulative_variance']),
                'components_needed_95': int(np.argmax(pca_results['cumulative_variance'] > 0.95) + 1),
                'component_loadings': convert_to_serializable(pca_results['component_loadings'])
            },
            'feature_importance': convert_to_serializable(importance_results)
        }

        # Print summary
        self.print_feature_summary(report)

        # Generate visualizations
        if save_path:
            self.generate_feature_visualizations(X, y, feature_names, save_path)

        # Save report if requested
        if save_path:
            report_path = Path(save_path) / "feature_evaluation_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)  # Added default=str as safety
            print(f"\n✅ Feature evaluation report saved to: {report_path}")

        return report

    def print_feature_summary(self, report):
        """Print summary of feature evaluation results."""
        print("\n" + "="*60)
        print("FEATURE EVALUATION SUMMARY")
        print("="*60)

        print(f"\nDataset Information:")
        print(f"  Samples: {report['dataset_info']['n_samples']}")
        print(f"  Features: {report['dataset_info']['n_features']}")
        print(f"  Class distribution: {dict(report['dataset_info']['class_distribution'])}")

        print(f"\nCorrelation Analysis:")
        print(f"  Highly correlated feature pairs (>0.8): {report['correlation_analysis']['n_highly_correlated']}")
        if report['correlation_analysis']['highly_correlated_pairs']:
            print("  Top correlated pairs:")
            for pair in report['correlation_analysis']['highly_correlated_pairs'][:5]:
                print(f"    {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")

        print(f"\nFeature-Target Relationship:")
        print(f"  Significant features (ANOVA p<0.05): {len(report['feature_target_analysis']['anova_significant_features'])}")
        if report['feature_target_analysis']['mutual_info_top_features']:
            print(f"  Top 5 features by mutual information:")
            for i, feature in enumerate(report['feature_target_analysis']['mutual_info_top_features'][:5], 1):
                print(f"    {i}. {feature}")

        print(f"\nPCA Analysis:")
        print(f"  Components needed for 95% variance: {report['pca_analysis']['components_needed_95']}")
        print(f"  Explained variance by top 5 components:")
        for i, var in enumerate(report['pca_analysis']['explained_variance'][:5], 1):
            print(f"    PC{i}: {var:.3f}")

        print(f"\nFeature Importance (Random Forest):")
        if 'random_forest' in report['feature_importance']:
            top_features = report['feature_importance']['random_forest']['top_features']
            for i, (feature, importance) in enumerate(top_features[:5], 1):
                print(f"    {i}. {feature}: {importance:.4f}")

    def generate_feature_visualizations(self, X, y, feature_names, save_path):
        """
        Generate comprehensive feature visualization plots.

        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            save_path: Path to save plots
        """
        print("\nGenerating feature visualizations...")

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))

        # 1. Correlation heatmap
        ax1 = plt.subplot(3, 3, 1)
        if self.correlation_matrix is not None:
            sns.heatmap(self.correlation_matrix, annot=False, cmap='coolwarm',
                       center=0, ax=ax1, cbar_kws={'label': 'Correlation'})
            ax1.set_title('Feature Correlation Matrix')
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            plt.setp(ax1.get_yticklabels(), rotation=0)

        # 2. Feature importance comparison
        ax2 = plt.subplot(3, 3, 2)
        importance_data = []
        for model_name, results in self.feature_importance.items():
            if 'top_features' in results:
                for feature, importance in results['top_features'][:5]:
                    importance_data.append({
                        'Model': model_name.replace('_', ' ').title(),
                        'Feature': feature,
                        'Importance': importance
                    })

        if importance_data:
            importance_df = pd.DataFrame(importance_data)
            if not importance_df.empty:
                pivot_df = importance_df.pivot(index='Feature', columns='Model', values='Importance')
                pivot_df.plot(kind='bar', ax=ax2)
                ax2.set_title('Feature Importance Across Models')
                ax2.set_xlabel('Feature')
                ax2.set_ylabel('Importance Score')
                ax2.legend(title='Model')
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        # 3. PCA explained variance
        ax3 = plt.subplot(3, 3, 3)
        if self.pca_results:
            explained_variance = self.pca_results['explained_variance_ratio']
            cumulative_variance = self.pca_results['cumulative_variance']

            x = range(1, len(explained_variance) + 1)
            ax3.bar(x, explained_variance, alpha=0.6, label='Individual')
            ax3.plot(x, cumulative_variance, 'r-', marker='o', label='Cumulative')
            ax3.axhline(y=0.95, color='g', linestyle='--', alpha=0.5, label='95% threshold')

            ax3.set_xlabel('Principal Component')
            ax3.set_ylabel('Explained Variance Ratio')
            ax3.set_title('PCA Explained Variance')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. ANOVA F-values vs Mutual Information
        ax4 = plt.subplot(3, 3, 4)
        if self.anova_results and self.mutual_info_results:
            f_values = self.anova_results.get('f_values', [])
            mi_scores = self.mutual_info_results.get('mi_scores', [])

            if len(f_values) == len(mi_scores) and len(f_values) > 0:
                # Normalize scores for comparison
                f_norm = f_values / np.max(f_values) if np.max(f_values) > 0 else f_values
                mi_norm = mi_scores / np.max(mi_scores) if np.max(mi_scores) > 0 else mi_scores

                ax4.scatter(f_norm, mi_norm, alpha=0.6)
                ax4.set_xlabel('ANOVA F-value (normalized)')
                ax4.set_ylabel('Mutual Information (normalized)')
                ax4.set_title('Feature Relevance: ANOVA vs Mutual Info')
                ax4.grid(True, alpha=0.3)

                # Add feature labels for top features
                top_indices = np.argsort(f_norm + mi_norm)[-5:]  # Top 5 by combined score
                for idx in top_indices:
                    if idx < len(feature_names):
                        ax4.annotate(feature_names[idx],
                                   (f_norm[idx], mi_norm[idx]),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, alpha=0.8)

        # 5. Feature distribution by class
        ax5 = plt.subplot(3, 3, 5)
        if isinstance(X, pd.DataFrame):
            # Find the most important feature
            if 'random_forest' in self.feature_importance:
                top_feature_name = self.feature_importance['random_forest']['top_features'][0][0]
                if top_feature_name in X.columns:
                    # Check if the feature is numeric
                    if pd.api.types.is_numeric_dtype(X[top_feature_name]):
                        # Create violin plot
                        plot_data = pd.DataFrame({
                            'Feature Value': X[top_feature_name],
                            'Class': y
                        })
                        sns.violinplot(x='Class', y='Feature Value', data=plot_data, ax=ax5)
                        ax5.set_title(f'Distribution of Top Feature: {top_feature_name}')
                        ax5.set_xlabel('Class (0=Non-physio, 1=Physio)')
                        ax5.set_ylabel('Feature Value')
                    else:
                        ax5.text(0.5, 0.5, f'Feature "{top_feature_name}" is not numeric',
                                ha='center', va='center', transform=ax5.transAxes)
                        ax5.set_title(f'Distribution of Top Feature: {top_feature_name}')

        # 6. Missing values heatmap
        ax6 = plt.subplot(3, 3, 6)
        if isinstance(X, pd.DataFrame):
            missing_matrix = X.isna().astype(int)
            if missing_matrix.sum().sum() > 0:
                sns.heatmap(missing_matrix.T, cmap='Reds', cbar_kws={'label': 'Missing (1=Yes)'}, ax=ax6)
                ax6.set_title('Missing Values Pattern')
                ax6.set_xlabel('Sample Index')
                ax6.set_ylabel('Feature')
            else:
                ax6.text(0.5, 0.5, 'No missing values',
                        ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('Missing Values Pattern')

        # 7. Feature statistics boxplot
        ax7 = plt.subplot(3, 3, 7)
        if self.feature_stats:
            stats_df = pd.DataFrame(self.feature_stats).T

            # Only select numeric statistics for plotting
            numeric_stats = ['mean', 'std', 'min', 'max', 'median']
            available_numeric_stats = [s for s in numeric_stats if s in stats_df.columns]

            if available_numeric_stats:
                # Convert to numeric, handling any non-numeric values
                plot_data = stats_df[available_numeric_stats].apply(pd.to_numeric, errors='coerce')

                # Check if we have any numeric data to plot
                if not plot_data.isna().all().all():
                    plot_data.plot(kind='box', ax=ax7)
                    ax7.set_title('Feature Statistics Distribution')
                    ax7.set_ylabel('Value')
                    ax7.grid(True, alpha=0.3)
                else:
                    ax7.text(0.5, 0.5, 'No numeric statistics available',
                            ha='center', va='center', transform=ax7.transAxes)
                    ax7.set_title('Feature Statistics Distribution')
            else:
                ax7.text(0.5, 0.5, 'No numeric statistics available',
                        ha='center', va='center', transform=ax7.transAxes)
                ax7.set_title('Feature Statistics Distribution')

        # 8. Top component loadings
        ax8 = plt.subplot(3, 3, 8)
        if self.pca_results and 'component_loadings' in self.pca_results:
            pc1_loadings = self.pca_results['component_loadings'].get('PC1', {})
            if 'top_positive' in pc1_loadings and 'top_negative' in pc1_loadings:
                # Combine positive and negative loadings
                all_loadings = pc1_loadings.get('top_positive', []) + pc1_loadings.get('top_negative', [])
                features = [item[0] for item in all_loadings]
                loadings = [item[1] for item in all_loadings]

                colors = ['red' if l < 0 else 'blue' for l in loadings]
                ax8.barh(features, loadings, color=colors)
                ax8.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                ax8.set_xlabel('Loading Value')
                ax8.set_title('PC1 Feature Loadings')

        # 9. Cumulative feature importance
        ax9 = plt.subplot(3, 3, 9)
        if 'random_forest' in self.feature_importance:
            importances = self.feature_importance['random_forest']['importance']
            sorted_importances = np.sort(importances)[::-1]
            cumulative_importance = np.cumsum(sorted_importances)

            ax9.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 'b-', marker='o')
            ax9.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% threshold')
            ax9.axhline(y=0.8, color='y', linestyle='--', alpha=0.7, label='80% threshold')

            # Find how many features needed for 80% and 95% importance
            n_features_80 = np.argmax(cumulative_importance >= 0.8) + 1
            n_features_95 = np.argmax(cumulative_importance >= 0.95) + 1

            ax9.axvline(x=n_features_80, color='y', linestyle=':', alpha=0.5)
            ax9.axvline(x=n_features_95, color='r', linestyle=':', alpha=0.5)

            ax9.set_xlabel('Number of Features')
            ax9.set_ylabel('Cumulative Importance')
            ax9.set_title('Cumulative Feature Importance (Random Forest)')
            ax9.legend()
            ax9.grid(True, alpha=0.3)

            # Add text annotations
            ax9.text(n_features_80, 0.82, f'{n_features_80} features\nfor 80%',
                    ha='center', va='bottom', fontsize=8)
            ax9.text(n_features_95, 0.97, f'{n_features_95} features\nfor 95%',
                    ha='center', va='bottom', fontsize=8)

        plt.suptitle('Comprehensive Feature Evaluation', fontsize=16, y=1.02)
        plt.tight_layout()

        # Save the figure
        plot_path = save_path / "feature_evaluation_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Feature evaluation plots saved to: {plot_path}")
        plt.close()

        # Additional: Create detailed feature ranking plot
        self.create_feature_ranking_plot(save_path)

    def create_feature_ranking_plot(self, save_path):
        """Create a detailed feature ranking plot."""
        if not self.feature_importance:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Random Forest feature importance
        ax1 = axes[0, 0]
        if 'random_forest' in self.feature_importance:
            importances = self.feature_importance['random_forest']['importance']
            feature_names = [item[0] for item in self.feature_importance['random_forest']['top_features']]
            importance_values = [item[1] for item in self.feature_importance['random_forest']['top_features']]

            # Plot top 20 features
            n_features = min(20, len(feature_names))
            indices = np.arange(n_features)

            ax1.barh(indices, importance_values[:n_features][::-1])
            ax1.set_yticks(indices)
            ax1.set_yticklabels(feature_names[:n_features][::-1])
            ax1.set_xlabel('Importance')
            ax1.set_title('Random Forest Feature Importance (Top 20)')
            ax1.grid(True, alpha=0.3, axis='x')

        # 2. Mutual Information scores
        ax2 = axes[0, 1]
        if self.mutual_info_results:
            mi_scores = self.mutual_info_results['mi_scores']
            if len(mi_scores) > 0:
                sorted_indices = np.argsort(mi_scores)[::-1]
                sorted_scores = mi_scores[sorted_indices]

                # Plot top 20
                n_features = min(20, len(sorted_scores))
                indices = np.arange(n_features)

                ax2.bar(indices, sorted_scores[:n_features])
                ax2.set_xticks(indices)
                if hasattr(self, 'feature_names') and self.feature_names:
                    top_features = [self.feature_names[i] for i in sorted_indices[:n_features]]
                    ax2.set_xticklabels(top_features, rotation=45, ha='right')
                ax2.set_ylabel('Mutual Information Score')
                ax2.set_title('Mutual Information Feature Ranking')
                ax2.grid(True, alpha=0.3, axis='y')

        # 3. ANOVA F-values
        ax3 = axes[1, 0]
        if self.anova_results:
            f_values = self.anova_results['f_values']
            if len(f_values) > 0:
                sorted_indices = np.argsort(f_values)[::-1]
                sorted_fvalues = f_values[sorted_indices]

                # Plot top 20
                n_features = min(20, len(sorted_fvalues))
                indices = np.arange(n_features)

                ax3.bar(indices, sorted_fvalues[:n_features])
                ax3.set_xticks(indices)
                if hasattr(self, 'feature_names') and self.feature_names:
                    top_features = [self.feature_names[i] for i in sorted_indices[:n_features]]
                    ax3.set_xticklabels(top_features, rotation=45, ha='right')
                ax3.set_ylabel('ANOVA F-value')
                ax3.set_title('ANOVA F-values Feature Ranking')
                ax3.grid(True, alpha=0.3, axis='y')

        # 4. Combined ranking heatmap
        ax4 = axes[1, 1]

        # Collect all rankings
        rankings = {}
        if 'random_forest' in self.feature_importance:
            rf_features = [item[0] for item in self.feature_importance['random_forest']['top_features'][:20]]
            for i, feature in enumerate(rf_features):
                rankings.setdefault(feature, {})['rf_rank'] = i + 1

        if self.mutual_info_results:
            mi_features = self.mutual_info_results.get('top_features', [])[:20]
            for i, feature in enumerate(mi_features):
                rankings.setdefault(feature, {})['mi_rank'] = i + 1

        if self.anova_results:
            anova_features = self.anova_results.get('significant_features', [])[:20]
            for i, feature in enumerate(anova_features):
                rankings.setdefault(feature, {})['anova_rank'] = i + 1

        if rankings:
            # Create DataFrame for heatmap
            heatmap_data = []
            for feature, ranks in rankings.items():
                row = {'feature': feature}
                row.update({k: ranks.get(k, np.nan) for k in ['rf_rank', 'mi_rank', 'anova_rank']})
                heatmap_data.append(row)

            heatmap_df = pd.DataFrame(heatmap_data).set_index('feature')

            # Plot heatmap
            sns.heatmap(heatmap_df, annot=True, fmt='.0f', cmap='YlOrRd_r',
                       cbar_kws={'label': 'Rank (lower is better)'}, ax=ax4)
            ax4.set_title('Feature Rankings Across Different Methods')
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
            plt.setp(ax4.get_yticklabels(), rotation=0)

        plt.suptitle('Feature Ranking Analysis', fontsize=16, y=1.02)
        plt.tight_layout()

        ranking_path = save_path / "feature_ranking_analysis.png"
        plt.savefig(ranking_path, dpi=300, bbox_inches='tight')
        print(f"✅ Feature ranking analysis saved to: {ranking_path}")
        plt.close()


# ============================================
# ProteinInteractionClassifier class (unchanged from original)
# ============================================

class ProteinInteractionClassifier:
    """Machine learning classifier for protein interaction prediction with GroupKFold CV and feature evaluation."""

    def __init__(self, random_state=42, n_splits=5):
        """Initialize the classifier with multiple models and feature evaluator."""
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'SVM': SVC(kernel='rbf', probability=True, random_state=random_state),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state)
        }

        self.scaler = StandardScaler()
        self.feature_evaluator = FeatureEvaluator(random_state=random_state)
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.random_state = random_state
        self.n_splits = n_splits
        self.cv_splits = None
        self.feature_evaluation_report = None
        self.feature_names = None

    def evaluate_features(self, X, y, feature_names=None, save_path=None):
        """
        Perform comprehensive feature evaluation.

        Args:
            X: Feature matrix
            y: Labels
            feature_names: List of feature names
            save_path: Path to save evaluation results

        Returns:
            Feature evaluation report
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE FEATURE EVALUATION")
        print("="*60)

        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = list(X.columns)
            else:
                feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

        # Store feature names for later use
        self.feature_names = feature_names

        # Perform feature evaluation
        self.feature_evaluation_report = self.feature_evaluator.generate_feature_report(
            X, y, feature_names, save_path
        )

        return self.feature_evaluation_report

    def create_group_kfold_splits(self, X, y, cluster_ids):
        """
        Create GroupKFold cross-validation splits.

        Args:
            X: Features
            y: Labels
            cluster_ids: ClusterID for each sample (groups)

        Returns:
            List of (train_indices, test_indices) for each fold
        """
        print(f"\n=== CREATING GROUPKFOLD CROSS-VALIDATION SPLITS ===")
        print(f"   Number of folds: {self.n_splits}")
        print(f"   Using ClusterID as groups to ensure same-cluster samples stay together")

        # Initialize GroupKFold
        gkf = GroupKFold(n_splits=self.n_splits)

        # Generate splits
        self.cv_splits = list(gkf.split(X, y, groups=cluster_ids))

        # Analyze the splits
        print(f"   Split analysis:")
        for fold_idx, (train_idx, test_idx) in enumerate(self.cv_splits):
            # Calculate class distribution in this fold
            train_labels = y.iloc[train_idx] if hasattr(y, 'iloc') else [y[i] for i in train_idx]
            test_labels = y.iloc[test_idx] if hasattr(y, 'iloc') else [y[i] for i in test_idx]

            train_clusters = cluster_ids.iloc[train_idx] if hasattr(cluster_ids, 'iloc') else [cluster_ids[i] for i in train_idx]
            test_clusters = cluster_ids.iloc[test_idx] if hasattr(cluster_ids, 'iloc') else [cluster_ids[i] for i in test_idx]

            print(f"   Fold {fold_idx + 1}:")
            print(f"     Training: {len(train_idx)} samples, {len(set(train_clusters))} unique clusters, "
                  f"{sum(train_labels == 1)} physio ({sum(train_labels == 1)/len(train_labels):.1%})")
            print(f"     Testing:  {len(test_idx)} samples, {len(set(test_clusters))} unique clusters, "
                  f"{sum(test_labels == 1)} physio ({sum(test_labels == 1)/len(test_labels):.1%})")

            # Check for cluster overlap between train and test
            train_cluster_set = set(train_clusters)
            test_cluster_set = set(test_clusters)
            overlap = train_cluster_set.intersection(test_cluster_set)
            if overlap:
                print(f"     ⚠️  Warning: {len(overlap)} clusters appear in both train and test sets")

        return self.cv_splits

    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler."""
        print("\n=== SCALING FEATURES ===")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("✅ Features scaled")
        return X_train_scaled, X_test_scaled

    def train_with_cross_validation(self, X, y, cluster_ids, feature_names=None):
        """
        Train and evaluate models using GroupKFold cross-validation.

        Args:
            X: Features (DataFrame or array)
            y: Labels
            cluster_ids: ClusterID for each sample (groups)
            feature_names: Optional list of feature names

        Returns:
            Dictionary with cross-validation results
        """
        print(f"\n=== TRAINING WITH {self.n_splits}-FOLD GROUPKFOLD CROSS-VALIDATION ===")

        # Store feature names if provided
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)

        # Create CV splits
        cv_splits = self.create_group_kfold_splits(X, y, cluster_ids)

        # Initialize results storage
        self.results = {}
        all_predictions = {}
        all_true_labels = {}
        all_proba_predictions = {}

        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"\n--- Training {name} with GroupKFold CV ---")

            # Store per-fold results
            fold_accuracies = []
            fold_precisions = []
            fold_recalls = []
            fold_f1s = []
            fold_roc_aucs = []

            # Initialize arrays for overall predictions
            all_predictions[name] = []
            all_true_labels[name] = []
            all_proba_predictions[name] = []

            # Perform cross-validation
            for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
                print(f"  Fold {fold_idx + 1}/{self.n_splits}...")

                # Split data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)

                # Train model
                try:
                    model.fit(X_train_scaled, y_train)
                except Exception as e:
                    print(f"    Error training model: {e}")
                    continue

                # Make predictions
                y_pred = model.predict(X_test_scaled)

                # Get probability predictions if available
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_pred_proba = None

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                # Calculate ROC-AUC if probability estimates are available
                roc_auc = None
                if y_pred_proba is not None:
                    try:
                        roc_auc = roc_auc_score(y_test, y_pred_proba)
                    except:
                        roc_auc = None

                # Store fold results
                fold_accuracies.append(accuracy)
                fold_precisions.append(precision)
                fold_recalls.append(recall)
                fold_f1s.append(f1)
                if roc_auc is not None:
                    fold_roc_aucs.append(roc_auc)

                # Store predictions for overall evaluation
                all_predictions[name].extend(y_pred)
                all_true_labels[name].extend(y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test))
                if y_pred_proba is not None:
                    all_proba_predictions[name].extend(y_pred_proba.tolist() if hasattr(y_pred_proba, 'tolist') else list(y_pred_proba))

            # Calculate average metrics across folds
            if fold_accuracies:
                avg_accuracy = np.mean(fold_accuracies)
                avg_precision = np.mean(fold_precisions)
                avg_recall = np.mean(fold_recalls)
                avg_f1 = np.mean(fold_f1s)
                avg_roc_auc = np.mean(fold_roc_aucs) if fold_roc_aucs else None
            else:
                avg_accuracy = avg_precision = avg_recall = avg_f1 = avg_roc_auc = 0

            # Calculate overall metrics
            if all_true_labels[name]:
                overall_accuracy = accuracy_score(all_true_labels[name], all_predictions[name])
                overall_precision = precision_score(all_true_labels[name], all_predictions[name], zero_division=0)
                overall_recall = recall_score(all_true_labels[name], all_predictions[name], zero_division=0)
                overall_f1 = f1_score(all_true_labels[name], all_predictions[name], zero_division=0)

                overall_roc_auc = None
                if all_proba_predictions[name]:
                    try:
                        overall_roc_auc = roc_auc_score(all_true_labels[name], all_proba_predictions[name])
                    except:
                        overall_roc_auc = None
            else:
                overall_accuracy = overall_precision = overall_recall = overall_f1 = 0
                overall_roc_auc = None

            # Store results
            self.results[name] = {
                'cv_metrics': {
                    'accuracy': avg_accuracy,
                    'precision': avg_precision,
                    'recall': avg_recall,
                    'f1': avg_f1,
                    'roc_auc': avg_roc_auc
                },
                'overall_metrics': {
                    'accuracy': overall_accuracy,
                    'precision': overall_precision,
                    'recall': overall_recall,
                    'f1': overall_f1,
                    'roc_auc': overall_roc_auc
                },
                'fold_metrics': {
                    'accuracies': fold_accuracies,
                    'precisions': fold_precisions,
                    'recalls': fold_recalls,
                    'f1_scores': fold_f1s,
                    'roc_aucs': fold_roc_aucs
                },
                'predictions': all_predictions[name],
                'true_labels': all_true_labels[name],
                'probabilities': all_proba_predictions[name] if all_proba_predictions[name] else None
            }

            # Print results
            print(f"  CV Results for {name}:")
            print(f"    Average Accuracy:  {avg_accuracy:.4f} (±{np.std(fold_accuracies) if fold_accuracies else 0:.4f})")
            print(f"    Average Precision: {avg_precision:.4f} (±{np.std(fold_precisions) if fold_precisions else 0:.4f})")
            print(f"    Average Recall:    {avg_recall:.4f} (±{np.std(fold_recalls) if fold_recalls else 0:.4f})")
            print(f"    Average F1-Score:  {avg_f1:.4f} (±{np.std(fold_f1s) if fold_f1s else 0:.4f})")
            if avg_roc_auc is not None:
                print(f"    Average ROC-AUC:   {avg_roc_auc:.4f} (±{np.std(fold_roc_aucs) if fold_roc_aucs else 0:.4f})")

            print(f"  Overall Results (all folds combined):")
            print(f"    Accuracy:  {overall_accuracy:.4f}")
            print(f"    Precision: {overall_precision:.4f}")
            print(f"    Recall:    {overall_recall:.4f}")
            print(f"    F1-Score:  {overall_f1:.4f}")
            if overall_roc_auc is not None:
                print(f"    ROC-AUC:   {overall_roc_auc:.4f}")

        # Determine best model based on average F1-score
        best_f1 = -1
        for name, metrics in self.results.items():
            if metrics['cv_metrics']['f1'] > best_f1:
                best_f1 = metrics['cv_metrics']['f1']
                self.best_model_name = name
                self.best_model = self.models[name]

        if self.best_model_name:
            print(f"\n✅ Best model: {self.best_model_name} (Average F1-Score: {best_f1:.4f})")

        return self.results

    def train_test_split(self, X, y, cluster_ids, test_size=0.2, random_state=42):
        """
        Split data into train and test sets while keeping clusters together.

        Args:
            X: Features
            y: Labels
            cluster_ids: Cluster IDs
            test_size: Size of test set
            random_state: Random seed

        Returns:
            X_train, X_test, y_train, y_test, cluster_ids_train, cluster_ids_test
        """
        print(f"\n=== CREATING TRAIN/TEST SPLIT (Test size: {test_size}) ===")

        # Get unique clusters
        unique_clusters = cluster_ids.unique()
        n_clusters = len(unique_clusters)
        n_test_clusters = int(n_clusters * test_size)

        # Randomly select test clusters
        np.random.seed(random_state)
        test_clusters = np.random.choice(unique_clusters, n_test_clusters, replace=False)

        # Create masks
        train_mask = ~cluster_ids.isin(test_clusters)
        test_mask = cluster_ids.isin(test_clusters)

        # Split data
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        cluster_ids_train = cluster_ids[train_mask]
        cluster_ids_test = cluster_ids[test_mask]

        print(f"  Training set: {len(X_train)} samples, {len(cluster_ids_train.unique())} clusters")
        print(f"  Test set: {len(X_test)} samples, {len(cluster_ids_test.unique())} clusters")

        return X_train, X_test, y_train, y_test, cluster_ids_train, cluster_ids_test

    def train_and_evaluate(self, X, y, cluster_ids, feature_names=None, test_size=0.2):
        """
        Train and evaluate models using train/test split.

        Args:
            X: Features
            y: Labels
            cluster_ids: Cluster IDs
            feature_names: Feature names
            test_size: Test set size

        Returns:
            Dictionary with evaluation results
        """
        print(f"\n=== TRAINING AND EVALUATING MODELS (Test size: {test_size}) ===")

        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)

        # Split data
        X_train, X_test, y_train, y_test, cluster_ids_train, cluster_ids_test = self.train_test_split(
            X, y, cluster_ids, test_size=test_size, random_state=self.random_state
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train and evaluate each model
        self.results = {}
        for name, model in self.models.items():
            print(f"\n--- Training {name} ---")

            try:
                # Train model
                model.fit(X_train_scaled, y_train)

                # Make predictions
                y_pred = model.predict(X_test_scaled)

                # Get probability predictions if available
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_pred_proba = None

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                # Calculate ROC-AUC if probability estimates are available
                roc_auc = None
                if y_pred_proba is not None:
                    try:
                        roc_auc = roc_auc_score(y_test, y_pred_proba)
                    except:
                        roc_auc = None

                # Calculate confusion matrix
                cm = confusion_matrix(y_test, y_pred)

                # Store results
                self.results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc,
                    'confusion_matrix': cm.tolist(),
                    'predictions': y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred),
                    'true_labels': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test),
                    'probabilities': y_pred_proba.tolist() if y_pred_proba is not None and hasattr(y_pred_proba, 'tolist') else list(y_pred_proba) if y_pred_proba is not None else None
                }

                # Print results
                print(f"  Test Results for {name}:")
                print(f"    Accuracy:  {accuracy:.4f}")
                print(f"    Precision: {precision:.4f}")
                print(f"    Recall:    {recall:.4f}")
                print(f"    F1-Score:  {f1:.4f}")
                if roc_auc is not None:
                    print(f"    ROC-AUC:   {roc_auc:.4f}")

                # Print classification report
                print(f"    Classification Report:")
                report = classification_report(y_test, y_pred, target_names=['Non-physio', 'Physio'])
                for line in report.split('\n'):
                    print(f"      {line}")

            except Exception as e:
                print(f"  Error training {name}: {e}")
                self.results[name] = {
                    'accuracy': 0,
                    'precision': 0,
                    'recall': 0,
                    'f1': 0,
                    'roc_auc': None,
                    'confusion_matrix': [[0, 0], [0, 0]],
                    'predictions': [],
                    'true_labels': [],
                    'probabilities': None
                }

        # Determine best model based on F1-score
        best_f1 = -1
        for name, metrics in self.results.items():
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                self.best_model_name = name
                self.best_model = self.models[name]

        if self.best_model_name:
            print(f"\n✅ Best model: {self.best_model_name} (F1-Score: {best_f1:.4f})")

        return self.results

    def plot_cv_results(self, save_plots=False, output_dir="."):
        """Plot cross-validation results."""
        if not self.results:
            print("❌ No results to plot. Train models first.")
            return

        print("\n=== PLOTTING CROSS-VALIDATION RESULTS ===")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Model Performance with {self.n_splits}-Fold GroupKFold Cross-Validation', fontsize=16)

        # Plot 1: Average metrics comparison
        ax = axes[0, 0]
        model_names = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        x = np.arange(len(model_names))
        width = 0.2

        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [self.results[name]['cv_metrics'][metric] for name in model_names]
            ax.bar(x + i*width - width*1.5, values, width, label=label)

        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Average CV Metrics by Model')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()
        ax.set_ylim(0, 1.1)

        # Plot 2: Per-fold F1 scores
        ax = axes[0, 1]
        for name in model_names:
            fold_f1s = self.results[name]['fold_metrics']['f1_scores']
            if fold_f1s:
                ax.plot(range(1, len(fold_f1s) + 1), fold_f1s, marker='o', label=name)

        ax.set_xlabel('Fold')
        ax.set_ylabel('F1-Score')
        ax.set_title('F1-Score per Fold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: ROC curves (if available)
        ax = axes[0, 2]
        for name in model_names:
            if self.results[name]['probabilities']:
                fpr, tpr, _ = roc_curve(self.results[name]['true_labels'],
                                       self.results[name]['probabilities'])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves (All Folds Combined)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Confusion matrix for best model
        if self.best_model_name:
            ax = axes[1, 0]
            y_true = self.results[self.best_model_name]['true_labels']
            y_pred = self.results[self.best_model_name]['predictions']

            if y_true and y_pred:
                cm = confusion_matrix(y_true, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['Non-physio', 'Physio'],
                           yticklabels=['Non-physio', 'Physio'])
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title(f'Confusion Matrix - {self.best_model_name}')

        # Plot 5: Metric distributions across folds
        ax = axes[1, 1]
        metrics_to_plot = ['accuracies', 'precisions', 'recalls', 'f1_scores']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1']

        data = []
        for name in model_names:
            for metric, label in zip(metrics_to_plot, metric_labels):
                values = self.results[name]['fold_metrics'][metric]
                if values:
                    for value in values:
                        data.append({'Model': name, 'Metric': label, 'Value': value})

        if data:
            df_plot = pd.DataFrame(data)
            if not df_plot.empty:
                sns.boxplot(x='Model', y='Value', hue='Metric', data=df_plot, ax=ax)
                ax.set_title('Metric Distributions Across Folds')
                ax.set_ylabel('Score')
                ax.set_ylim(0, 1.1)
                ax.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Plot 6: Class distribution in each fold (empty for now)
        ax = axes[1, 2]
        ax.axis('off')
        if self.cv_splits:
            # Create a simple text summary
            text_content = "CV Split Summary:\n"
            for fold_idx, (train_idx, test_idx) in enumerate(self.cv_splits):
                text_content += f"\nFold {fold_idx+1}:\n"
                text_content += f"  Train: {len(train_idx)} samples\n"
                text_content += f"  Test:  {len(test_idx)} samples\n"

            ax.text(0.1, 0.5, text_content, transform=ax.transAxes,
                   verticalalignment='center', fontsize=10)

        plt.tight_layout()

        if save_plots:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            plot_path = output_path / "group_kfold_cv_results.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved CV results plot to: {plot_path}")

        plt.show()
        print("✅ Cross-validation visualizations generated")

    def plot_test_results(self, save_plots=False, output_dir="."):
        """Plot test results."""
        if not self.results:
            print("❌ No results to plot. Train models first.")
            return

        print("\n=== PLOTTING TEST RESULTS ===")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Performance on Test Set', fontsize=16)

        # Plot 1: Metrics comparison
        ax1 = axes[0, 0]
        model_names = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        x = np.arange(len(model_names))
        width = 0.2

        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [self.results[name][metric] for name in model_names]
            ax1.bar(x + i*width - width*1.5, values, width, label=label)

        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Test Set Metrics by Model')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45)
        ax1.legend()
        ax1.set_ylim(0, 1.1)

        # Plot 2: ROC curves
        ax2 = axes[0, 1]
        for name in model_names:
            if self.results[name]['probabilities']:
                fpr, tpr, _ = roc_curve(self.results[name]['true_labels'],
                                       self.results[name]['probabilities'])
                roc_auc = auc(fpr, tpr)
                ax2.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

        ax2.plot([0, 1], [0, 1], 'k--', label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves on Test Set')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Confusion matrix for best model
        if self.best_model_name:
            ax3 = axes[1, 0]
            y_true = self.results[self.best_model_name]['true_labels']
            y_pred = self.results[self.best_model_name]['predictions']

            if y_true and y_pred:
                cm = confusion_matrix(y_true, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                           xticklabels=['Non-physio', 'Physio'],
                           yticklabels=['Non-physio', 'Physio'])
                ax3.set_xlabel('Predicted')
                ax3.set_ylabel('True')
                ax3.set_title(f'Confusion Matrix - {self.best_model_name}')

        # Plot 4: Model comparison table
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Create comparison table
        if self.results:
            comparison_data = []
            for name in model_names:
                comparison_data.append([
                    name,
                    f"{self.results[name]['accuracy']:.4f}",
                    f"{self.results[name]['precision']:.4f}",
                    f"{self.results[name]['recall']:.4f}",
                    f"{self.results[name]['f1']:.4f}",
                    f"{self.results[name]['roc_auc']:.4f}" if self.results[name]['roc_auc'] is not None else "N/A"
                ])

            # Create table
            col_labels = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            table = ax4.table(cellText=comparison_data, colLabels=col_labels,
                             cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)

        plt.tight_layout()

        if save_plots:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            plot_path = output_path / "test_results.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved test results plot to: {plot_path}")

        plt.show()
        print("✅ Test results visualizations generated")

    def feature_importance_analysis(self, X_train, y_train, save_plots=False, output_dir="."):
        """
        Analyze feature importance (for tree-based models).

        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")

        # Only for Random Forest
        if 'Random Forest' in self.models:
            model_type = 'Random Forest'
            model = self.models[model_type]

            # Train on all data for feature importance
            X_scaled = self.scaler.fit_transform(X_train)
            model.fit(X_scaled, y_train)

            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_

                # Get feature names
                if self.feature_names is not None:
                    feature_names = self.feature_names
                elif hasattr(X_train, 'columns'):
                    feature_names = list(X_train.columns)
                else:
                    feature_names = [f'Feature_{i}' for i in range(len(importances))]

                # Create DataFrame for visualization
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)

                print(f"\nMost Important Features ({model_type}):")
                for idx, row in importance_df.head(10).iterrows():
                    print(f"   {row['feature']}: {row['importance']:.4f}")

                # Plot feature importance
                plt.figure(figsize=(10, 6))
                plt.barh(importance_df['feature'][:15], importance_df['importance'][:15])
                plt.xlabel('Importance')
                plt.title(f'Top 15 Most Important Features ({model_type})')
                plt.gca().invert_yaxis()
                plt.tight_layout()

                if save_plots:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    fi_path = output_path / "feature_importance.png"
                    plt.savefig(fi_path, dpi=300, bbox_inches='tight')
                    print(f"Saved feature importance plot to: {fi_path}")

                plt.show()

        print("✅ Feature importance analysis complete")

    def save_results(self, output_dir="."):
        """Save model results to files."""
        if not self.results:
            print("❌ No results to save.")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

        # Prepare results for saving
        results_dict = {}
        for name, metrics in self.results.items():
            results_dict[name] = {
                'cv_metrics': {
                    k: (float(v) if isinstance(v, (np.float32, np.float64)) else v)
                    for k, v in metrics.get('cv_metrics', {}).items()
                },
                'overall_metrics': {
                    k: (float(v) if isinstance(v, (np.float32, np.float64)) else v)
                    for k, v in metrics.get('overall_metrics', {}).items()
                },
                'fold_metrics': {
                    k: ([float(val) for val in v] if isinstance(v, list) else v)
                    for k, v in metrics.get('fold_metrics', {}).items()
                },
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1': metrics.get('f1', 0),
                'roc_auc': metrics.get('roc_auc'),
                'confusion_matrix': metrics.get('confusion_matrix', [[0, 0], [0, 0]])
            }

        # Add cross-validation settings
        results_dict['cross_validation_settings'] = {
            'n_splits': self.n_splits,
            'method': 'GroupKFold cross-validation (ClusterID as groups)',
            'random_state': self.random_state,
            'best_model': self.best_model_name
        }

        # Save as JSON
        results_file = output_path / "group_kfold_cv_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"✅ Results saved to: {results_file}")


# ============================================
# UPDATED parse_arguments and main functions
# ============================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train ML models on Croissant-formatted protein interaction dataset with GroupKFold cross-validation, feature evaluation, and structural feature extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use local directory with structural feature extraction (mmCIF format)
  python ppi_ml_croissant.py --local ./bioschemas_output --extract-pdb-features --pdb-local-dir ./structure_files

  # Use PDB format instead of mmCIF
  python ppi_ml_croissant.py --local ./bioschemas_output --extract-pdb-features --pdb-local-dir ./structure_files --pdb-format

  # Full pipeline with all features
  python ppi_ml_croissant.py --local ./bioschemas_output --extract-pdb-features --evaluate-features --feature-analysis --save-plots

  # Structural features only (no model training)
  python ppi_ml_croissant.py --local ./bioschemas_output --extract-pdb-features --pdb-features-only --pdb-local-dir ./structure_files

  # Quick mode with structural features
  python ppi_ml_croissant.py --local ./bioschemas_output --extract-pdb-features --quick --pdb-local-dir ./structure_files

Note: Files are stored in gzipped format (.pdb.gz or .cif.gz)
      BioPython can parse gzipped files directly - no need to uncompress
      Interface IDs are handled case-insensitively (uppercase/lowercase)
        """
    )

    parser.add_argument(
        '--local',
        type=str,
        help='Path to local directory containing dataset_with_interfaces.json'
    )

    parser.add_argument(
        '--github',
        type=str,
        help='GitHub repository URL containing the dataset'
    )

    parser.add_argument(
        '--folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )

    parser.add_argument(
        '--test-split',
        type=float,
        default=0.0,
        help='Fraction of data to use as test set (0.0 for CV only, >0 for train/test split)'
    )

    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save plots to files'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./group_kfold_results',
        help='Directory to save results and plots (default: ./group_kfold_results)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick demo with basic output'
    )

    parser.add_argument(
        '--feature-analysis',
        action='store_true',
        help='Run feature importance analysis'
    )

    parser.add_argument(
        '--evaluate-features',
        action='store_true',
        help='Perform comprehensive feature evaluation before training'
    )

    parser.add_argument(
        '--feature-report-only',
        action='store_true',
        help='Only generate feature evaluation report (skip model training)'
    )

    parser.add_argument(
        '--extract-pdb-features',
        action='store_true',
        help='Extract structural features (interface residues within 5Å)'
    )

    parser.add_argument(
        '--pdb-local-dir',
        type=str,
        default='./structure_files',
        help='Local directory for interface structure files (acts as cache, default: ./structure_files)'
    )

    parser.add_argument(
        '--pdb-format',
        action='store_true',
        help='Use PDB format (default: mmCIF format)'
    )

    parser.add_argument(
        '--pdb-features-only',
        action='store_true',
        help='Only extract structural features (skip model training)'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    print("""
╔═══════════════════════════════════════════════════════════╗
║  Protein Interaction ML Classifier                        ║
║  with GroupKFold Cross-Validation                         ║
║  and Interface Structural Feature Extraction (5Å distance)║
║  Using local interface structure files                    ║
║  BioPython gzipped file support + Case sensitivity fix    ║
╚═══════════════════════════════════════════════════════════╝
    """)

    # Validate arguments
    if not args.local and not args.github:
        print("❌ Error: Must specify either --local or --github")
        print("   Use --help for usage information")
        return

    if args.local and args.github:
        print("⚠️  Warning: Both --local and --github specified. Using --local.")

    # Determine dataset source
    if args.local:
        dataset_source = args.local
        is_github = False
        print(f"Using local directory: {dataset_source}")
    else:
        dataset_source = args.github
        is_github = True
        print(f"Using GitHub repository: {dataset_source}")

    # Create output directory upfront
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    try:
        # Step 1: Load Croissant dataset
        print("\n" + "="*60)
        print("Step 1: Loading Croissant dataset")
        print("="*60)

        loader = CroissantDatasetLoader(dataset_source, is_github)

        if not loader.load_dataset():
            print("❌ Failed to load dataset. Exiting.")
            return

        # Step 2: Extract features, labels, and ClusterIDs
        print("\n" + "="*60)
        print("Step 2: Extracting features, labels, and ClusterIDs")
        print("="*60)

        features_df, labels, cluster_ids = loader.extract_features_labels()

        if features_df is None or labels is None or cluster_ids is None:
            print("❌ Failed to extract features/labels/ClusterIDs. Exiting.")
            loader.cleanup()
            return

        # Store labels in loader for analysis
        loader.labels = labels

        # Step 3: Process structure files if requested
        if args.extract_pdb_features:
            print("\n" + "="*60)
            print("Step 3: Processing Interface Structure Files")
            print("="*60)
            
            # First, extract structure sources from representations
            print("  Extracting structure source information from dataset representations...")
            
            # This extracts URLs and interface IDs from the dataset
            loader.extract_pdb_sources_from_representations(
                pdb_local_dir=args.pdb_local_dir,
                use_pdb_format=args.pdb_format
            )
      
            print(f"\n  Found {len(loader.pdb_sources)} structure sources in dataset")
            
            # Check which interface files we actually have locally
            print(f"  Checking local directory: {args.pdb_local_dir}")
            
            # Ensure the local directory exists
            pdb_local_path = Path(args.pdb_local_dir)
            pdb_local_path.mkdir(parents=True, exist_ok=True)
            
            # Check for each interface file (handling case sensitivity)
            file_ext = '.pdb' if args.pdb_format else '.cif'
            alt_ext = '.cif' if args.pdb_format else '.pdb'
            missing_files = []
            available_files = []
            
            for source_id, source in loader.pdb_sources.items():
                interface_id = source.get('interface_id', source_id)
                
                # Handle case sensitivity: try both uppercase and lowercase
                interface_variants = [interface_id]
                if interface_id.isupper():
                    interface_variants.append(interface_id.lower())
                elif interface_id.islower():
                    interface_variants.append(interface_id.upper())
                
                file_found = False
                for variant in interface_variants:
                    # Check compressed version first
                    compressed_file = pdb_local_path / f"{variant}{file_ext}.gz"
                    uncompressed_file = pdb_local_path / f"{variant}{file_ext}"
                    
                    # Also check for alternative format
                    alt_compressed = pdb_local_path / f"{variant}{alt_ext}.gz"
                    alt_uncompressed = pdb_local_path / f"{variant}{alt_ext}"
                    
                    if (compressed_file.exists() or uncompressed_file.exists() or 
                        alt_compressed.exists() or alt_uncompressed.exists()):
                        available_files.append(f"{interface_id} (found as {variant})")
                        file_found = True
                        break
                
                if not file_found:
                    missing_files.append(interface_id)
            
            print(f"  Files available locally: {len(available_files)}")
            print(f"  Files missing locally: {len(missing_files)}")
            
            if available_files:
                print(f"  Available files (first 10):")
                for file_info in available_files[:10]:
                    print(f"    - {file_info}")
                if len(available_files) > 10:
                    print(f"    ... and {len(available_files) - 10} more")
            
            if missing_files:
                print(f"  Missing files (first 10): {', '.join(missing_files[:10])}")
                if len(missing_files) > 10:
                    print(f"    ... and {len(missing_files) - 10} more")
                
                # Ask user if they want to download missing files
                if not args.pdb_features_only:
                    print(f"\n  ⚠️  Warning: {len(missing_files)} interface structure files are missing locally.")
                    print(f"     These files are needed for structural feature extraction.")
                    print(f"     You can:")
                    print(f"     1. Run with --pdb-features-only to download all missing files first")
                    print(f"     2. Download them manually to {args.pdb_local_dir}")
                    print(f"     3. Skip structural feature extraction")
                    
                    if len(missing_files) > 0 and len(missing_files) <= 20:
                        print(f"\n     Missing files: {', '.join(missing_files)}")
            
            # If pdb-features-only flag is set, we can exit after reporting
            if args.pdb_features_only:
                print("\n✅ Interface structure file inventory completed!")
                print(f"   Total structure sources in dataset: {len(loader.pdb_sources)}")
                print(f"   Available locally: {len(available_files)}")
                print(f"   Missing: {len(missing_files)}")
                print(f"   Local directory: {args.pdb_local_dir}")
                print(f"   File format: {'PDB' if args.pdb_format else 'mmCIF'}")
                print(f"   Note: BioPython can parse gzipped files directly")
                loader.cleanup()
                return
                
            # If we have no files locally and user wants to proceed, warn them
            if len(available_files) == 0 and args.extract_pdb_features:
                print(f"\n  ⚠️  Warning: No interface structure files found locally!")
                print(f"     Structural feature extraction will fail without structure files.")
                print(f"     Consider running with --pdb-features-only first to download files.")
                response = input("     Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    print("Exiting...")
                    loader.cleanup()
                    return

        # Step 4: Extract structural features (if requested AND we have files)
        if args.extract_pdb_features:
            print("\n" + "="*60)
            print("Step 4: Extracting Structural Features")
            print("="*60)

            if not BIOPYTHON_AVAILABLE:
                print("⚠️  BioPython not available. Skipping structural feature extraction.")
                print("   Install with: pip install biopython")
            elif len(available_files) == 0:
                print("⚠️  No interface structure files available locally. Skipping structural feature extraction.")
                print(f"   Run with --pdb-features-only to download files first.")
            else:
                # Create PDB feature extractor
                print(f"  Using {len(available_files)} available interface structure files")
                print(f"  File format: {'PDB' if args.pdb_format else 'mmCIF (default)'}")
                print(f"  Local directory: {args.pdb_local_dir}")
                print(f"  Note: BioPython can parse gzipped files directly")

                pdb_extractor = PDBFeatureExtractor(
                    pdb_local_dir=args.pdb_local_dir,
                    use_pdb_format=args.pdb_format
                )

                # Extract structural features (interface residues within 5Å)
                pdb_features_df = loader.extract_pdb_contacts(
                    pdb_feature_extractor=pdb_extractor, 
                    pdb_format=args.pdb_format
                )

                # Integrate structural features with existing features
                if pdb_features_df is not None:
                    features_df = loader.integrate_pdb_features(features_df, pdb_features_df)

                    # Ensure output directory exists before saving
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Save structural features to file
                    pdb_output_file = output_dir / "structural_features.csv"
                    pdb_features_df.to_csv(pdb_output_file, index=False)
                    print(f"  Structural features saved to: {pdb_output_file}")
                    
                    # Also save a summary
                    successful_extractions = pdb_features_df[pdb_features_df['extraction_success']]
                    print(f"  Successfully extracted structural features for {len(successful_extractions)} interfaces")
                    
                    # Show some extracted features
                    if len(successful_extractions) > 0:
                        pdb_feature_cols = [col for col in successful_extractions.columns 
                                          if col.startswith('pdb_') and col not in ['pdb_success', 'pdb_error']]
                        print(f"  Extracted {len(pdb_feature_cols)} structural feature types")
                        for col in pdb_feature_cols[:5]:
                            non_null = successful_extractions[col].notna().sum()
                            if non_null > 0:
                                print(f"    {col}: {non_null} non-null values")

        # Step 5: Preprocess features
        print("\n" + "="*60)
        print("Step 5: Preprocessing features")
        print("="*60)

        X_processed = loader.preprocess_features(features_df, labels)

        # Step 6: Feature Evaluation (if requested)
        if args.evaluate_features or args.feature_report_only:
            print("\n" + "="*60)
            print("Step 6: Performing Comprehensive Feature Evaluation")
            print("="*60)

            # Initialize classifier with feature evaluator
            classifier = ProteinInteractionClassifier(
                random_state=args.random_state,
                n_splits=args.folds
            )

            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)

            # Perform feature evaluation
            feature_report = classifier.evaluate_features(
                X_processed, labels,
                feature_names=list(X_processed.columns),
                save_path=args.output_dir if args.save_plots else None
            )

            # If only feature report is requested, exit here
            if args.feature_report_only:
                print("\n✅ Feature evaluation completed successfully!")
                print(f"   Feature report generated with {X_processed.shape[1]} features")
                print(f"   Feature statistics saved to: {args.output_dir}")
                loader.cleanup()
                return

        # Quick mode: minimal processing
        if args.quick:
            print("\n=== QUICK MODE ===")

            # Initialize classifier
            classifier = ProteinInteractionClassifier(
                random_state=args.random_state,
                n_splits=min(args.folds, 3)  # Use fewer folds for quick mode
            )

            # Train with cross-validation
            results = classifier.train_with_cross_validation(
                X_processed, labels, cluster_ids,
                feature_names=list(X_processed.columns)
            )

            print(f"\nQuick Results Summary:")
            print(f"  Dataset: {loader.dataset.get('name', 'Unknown')}")
            print(f"  Samples: {len(features_df)}")
            print(f"  Features: {X_processed.shape[1]}")
            if args.extract_pdb_features:
                pdb_feature_count = sum(1 for col in X_processed.columns if col.startswith('pdb_'))
                print(f"  Structural features: {pdb_feature_count}")
                print(f"  File format: {'PDB' if args.pdb_format else 'mmCIF'}")
            print(f"  Unique ClusterIDs: {cluster_ids.nunique()}")
            print(f"  Cross-validation folds: {classifier.n_splits}")

            if classifier.best_model_name and results:
                best_result = results[classifier.best_model_name]
                print(f"\n  Best model: {classifier.best_model_name}")
                print(f"    Average CV F1-Score: {best_result['cv_metrics']['f1']:.4f}")
                print(f"    Overall Accuracy:    {best_result['overall_metrics']['accuracy']:.4f}")

            loader.cleanup()
            return

        # Step 7: Analyze dataset
        print("\n" + "="*60)
        print("Step 7: Analyzing dataset")
        print("="*60)

        loader.analyze_dataset()

        # Step 8: Initialize classifier with GroupKFold CV
        print("\n" + "="*60)
        print("Step 8: Initializing classifier with GroupKFold CV")
        print("="*60)

        classifier = ProteinInteractionClassifier(
            random_state=args.random_state,
            n_splits=args.folds
        )

        # Step 9: Train and evaluate models
        print("\n" + "="*60)
        print("Step 9: Training and evaluating models")
        print("="*60)

        if args.test_split > 0:
            # Use train/test split
            results = classifier.train_and_evaluate(
                X_processed, labels, cluster_ids,
                feature_names=list(X_processed.columns),
                test_size=args.test_split
            )
        else:
            # Use cross-validation
            results = classifier.train_with_cross_validation(
                X_processed, labels, cluster_ids,
                feature_names=list(X_processed.columns)
            )

        # Step 10: Save results
        print("\n" + "="*60)
        print("Step 10: Saving results")
        print("="*60)

        classifier.save_results(output_dir=args.output_dir)

        if args.save_plots:
            # Step 11: Generate visualizations
            print("\n" + "="*60)
            print("Step 11: Generating visualizations")
            print("="*60)

            if args.test_split > 0:
                classifier.plot_test_results(save_plots=args.save_plots, output_dir=args.output_dir)
            else:
                classifier.plot_cv_results(save_plots=args.save_plots, output_dir=args.output_dir)

            # Step 12: Feature importance analysis
            if args.feature_analysis:
                print("\n" + "="*60)
                print("Step 12: Analyzing feature importance")
                print("="*60)

                classifier.feature_importance_analysis(
                    X_processed, labels,
                    save_plots=args.save_plots,
                    output_dir=args.output_dir
                )

        # Step 13: Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Dataset: {loader.dataset.get('name', 'Unknown')}")
        print(f"Source: {'GitHub: ' + dataset_source if is_github else 'Local: ' + dataset_source}")
        print(f"Total samples: {len(features_df)}")
        print(f"Unique ClusterIDs: {cluster_ids.nunique()}")
        print(f"Features used: {X_processed.shape[1]}")

        if args.extract_pdb_features:
            pdb_features_count = sum(1 for col in X_processed.columns if col.startswith('pdb_'))
            print(f"Structural interface features extracted: {pdb_features_count}")
            print(f"  Focus: Unique residues per chain within 5Å of other chain")
            print(f"  File format: {'PDB' if args.pdb_format else 'mmCIF (default)'}")
            print(f"  Local directory: {args.pdb_local_dir}")
            print(f"  Files: Gzipped format (.pdb.gz or .cif.gz)")
            print(f"  BioPython: Can parse gzipped files directly")
            print(f"  Case sensitivity: Handled (uppercase/lowercase interface IDs)")

        if args.test_split > 0:
            print(f"Evaluation method: Train/test split (Test size: {args.test_split})")
        else:
            print(f"Cross-validation folds: {args.folds}")

        print(f"Random state: {args.random_state}")

        if args.evaluate_features:
            print(f"\nFeature Evaluation Performed:")
            if classifier.feature_evaluation_report:
                report = classifier.feature_evaluation_report
                print(f"  - Highly correlated feature pairs: {report['correlation_analysis']['n_highly_correlated']}")
                print(f"  - Significant features (ANOVA p<0.05): {len(report['feature_target_analysis']['anova_significant_features'])}")
                print(f"  - PCA components for 95% variance: {report['pca_analysis']['components_needed_95']}")

        if args.test_split <= 0:
            print(f"\nCross-validation method: GroupKFold CV")
            print(f"  - Ensures all interfaces from same cluster stay together")
            print(f"  - Uses ClusterID as grouping variable")
            print(f"  - Prevents data leakage between training and testing")

        if classifier.best_model_name and results:
            if args.test_split > 0:
                best_result = results[classifier.best_model_name]
                print(f"\nBest model: {classifier.best_model_name}")
                print(f"  Test Set Metrics:")
                print(f"    F1-Score:  {best_result['f1']:.4f}")
                print(f"    Accuracy:  {best_result['accuracy']:.4f}")
                print(f"    Precision: {best_result['precision']:.4f}")
                print(f"    Recall:    {best_result['recall']:.4f}")
                if best_result['roc_auc'] is not None:
                    print(f"    ROC-AUC:   {best_result['roc_auc']:.4f}")
            else:
                best_result = results[classifier.best_model_name]
                print(f"\nBest model: {classifier.best_model_name}")
                print(f"  Average CV Metrics:")
                print(f"    F1-Score:  {best_result['cv_metrics']['f1']:.4f}")
                print(f"    Accuracy:  {best_result['cv_metrics']['accuracy']:.4f}")
                print(f"    Precision: {best_result['cv_metrics']['precision']:.4f}")
                print(f"    Recall:    {best_result['cv_metrics']['recall']:.4f}")
                if best_result['cv_metrics']['roc_auc'] is not None:
                    print(f"    ROC-AUC:   {best_result['cv_metrics']['roc_auc']:.4f}")

                print(f"\n  Overall Metrics (all folds combined):")
                print(f"    F1-Score:  {best_result['overall_metrics']['f1']:.4f}")
                print(f"    Accuracy:  {best_result['overall_metrics']['accuracy']:.4f}")

        if args.save_plots:
            print(f"\nResults saved to: {args.output_dir}")

        print("\n✅ ML pipeline with GroupKFold cross-validation and structural feature extraction completed successfully!")

    except Exception as e:
        print(f"\n❌ Error in ML pipeline: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup temporary files
        if 'loader' in locals():
            loader.cleanup()


def example_usage():
    """Show example usage of the script with structural feature extraction."""
    print("""
Example Usage with Structural Feature Extraction:
===============================================

1. Extract structural features only (mmCIF format):
   python ppi_ml_croissant.py --local ./bioschemas_output --extract-pdb-features --pdb-features-only

2. Use PDB format instead of mmCIF:
   python ppi_ml_croissant.py --local ./bioschemas_output --extract-pdb-features --pdb-features-only --pdb-format

3. Full pipeline with structural features:
   python ppi_ml_croissant.py --local ./bioschemas_output --extract-pdb-features --evaluate-features --feature-analysis --save-plots

4. Custom structure local directory:
   python ppi_ml_croissant.py --local ./bioschemas_output --extract-pdb-features --pdb-local-dir ./my_structure_files

5. Combine structural features with quick evaluation:
   python ppi_ml_croissant.py --local ./bioschemas_output --extract-pdb-features --quick --pdb-local-dir ./structure_files

Note: Structural feature extraction requires BioPython. Install with: pip install biopython
Files are stored in gzipped format (.pdb.gz or .cif.gz) - BioPython can parse them directly
Interface IDs are handled case-insensitively (uppercase/lowercase)
    """)


if __name__ == "__main__":
    # Check if no arguments provided
    import sys
    if len(sys.argv) == 1:
        print("No arguments provided. Showing usage examples:")
        example_usage()
        sys.exit(1)

    # Run main pipeline
    main()
