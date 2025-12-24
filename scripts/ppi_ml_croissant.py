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
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

# Import BioPython for PDB parsing
try:
    from Bio.PDB import PDBParser, MMCIFParser, Select
    from Bio.PDB.Polypeptide import is_aa
    from Bio import SeqIO
    from Bio.SeqUtils import IUPACData
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
# NEW CLASS: PDBFeatureExtractor
# ============================================

class PDBFeatureExtractor:
    """
    Extract basic structural features from PDB files using BioPython.
    Can fetch PDB files from RCSB web service or local files.
    """
    
    def __init__(self, cache_dir: str = None, use_mmcif: bool = True):
        """
        Initialize the PDB feature extractor.
        
        Args:
            cache_dir: Directory to cache downloaded PDB files
            use_mmcif: Whether to use mmCIF format (recommended)
        """
        self.cache_dir = cache_dir
        self.use_mmcif = use_mmcif
        self.parser = None
        
        if not BIOPYTHON_AVAILABLE:
            print("⚠️  BioPython is not available. PDB feature extraction will be limited.")
            print("   Install with: pip install biopython")
            return
        
        # Initialize parser
        if use_mmcif:
            self.parser = MMCIFParser(QUIET=True)
        else:
            self.parser = PDBParser(QUIET=True)
        
        # Create cache directory if specified
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    def get_pdb_file(self, pdb_source: Dict) -> Optional[str]:
        """
        Get PDB file path from JSON source information.
        
        Args:
            pdb_source: Dictionary containing PDB file information from JSON
            
        Returns:
            Path to local PDB file or None if failed
        """
        try:
            # Try different possible locations for PDB file information
            pdb_file_path = None
            
            # Option 1: Direct file path in contentUrl
            if 'contentUrl' in pdb_source:
                content_url = pdb_source['contentUrl']
                if isinstance(content_url, str):
                    # Check if it's a local file path
                    if content_url.startswith('file://'):
                        pdb_file_path = content_url[7:]  # Remove 'file://' prefix
                    elif os.path.exists(content_url):
                        pdb_file_path = content_url
                    elif content_url.startswith('http'):
                        # Download from URL
                        pdb_id = None
                        # Extract PDB ID from URL
                        import re
                        match = re.search(r'/([0-9][a-z0-9]{3})\.(pdb|cif)', content_url.lower())
                        if match:
                            pdb_id = match.group(1).upper()
                            return self.fetch_pdb_file(pdb_id)
            
            # Option 2: Check for encodingFormat that might indicate PDB file
            if not pdb_file_path and 'encodingFormat' in pdb_source:
                encoding_format = pdb_source['encodingFormat']
                if encoding_format in ['chemical/x-pdb', 'text/pdb', 'pdb', 'cif', 'mmcif']:
                    # Look for actual file path
                    if 'contentUrl' in pdb_source and os.path.exists(pdb_source['contentUrl']):
                        pdb_file_path = pdb_source['contentUrl']
            
            # Option 3: Check for name or identifier that looks like a PDB ID
            if not pdb_file_path:
                for field in ['name', 'identifier', 'description']:
                    if field in pdb_source:
                        value = str(pdb_source[field])
                        import re
                        match = re.search(r'([0-9][A-Z0-9]{3})', value.upper())
                        if match:
                            pdb_id = match.group(1)
                            return self.fetch_pdb_file(pdb_id)
            
            if pdb_file_path and os.path.exists(pdb_file_path):
                return pdb_file_path
            
            return None
            
        except Exception as e:
            print(f"  Error getting PDB file: {e}")
            return None
    
    def fetch_pdb_file(self, pdb_id: str) -> Optional[str]:
        """
        Fetch PDB file from RCSB web service.
        
        Args:
            pdb_id: 4-character PDB ID
            
        Returns:
            Path to local PDB file or None if failed
        """
        pdb_id = pdb_id.lower().strip()
        
        if self.cache_dir:
            cache_path = Path(self.cache_dir) / f"{pdb_id}.{'cif' if self.use_mmcif else 'pdb'}"
            if cache_path.exists():
                print(f"  Using cached PDB file for {pdb_id.upper()}")
                return str(cache_path)
        
        try:
            # Try different URL formats
            if self.use_mmcif:
                # Try mmCIF format first
                url = f"https://files.rcsb.org/download/{pdb_id}.cif"
            else:
                # Try PDB format
                url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            
            print(f"  Fetching PDB file for {pdb_id.upper()} from {url}")
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                if self.cache_dir:
                    # Save to cache
                    with open(cache_path, 'w') as f:
                        f.write(response.text)
                    return str(cache_path)
                else:
                    # Save to temporary file
                    temp_file = tempfile.NamedTemporaryFile(mode='w', 
                                                          suffix=f".{pdb_id}.{'cif' if self.use_mmcif else 'pdb'}",
                                                          delete=False)
                    temp_file.write(response.text)
                    temp_file.close()
                    return temp_file.name
            else:
                print(f"  Failed to fetch PDB {pdb_id.upper()}: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"  Error fetching PDB {pdb_id.upper()}: {e}")
            return None
    
    def extract_basic_features(self, pdb_source: Dict, chain_ids: List[str] = None) -> Dict[str, Any]:
        """
        Extract basic structural features from a PDB file.
        
        Args:
            pdb_source: Dictionary containing PDB file information from JSON
            chain_ids: List of chain IDs to analyze (if None, analyze all chains)
            
        Returns:
            Dictionary of extracted features
        """
        features = {
            'pdb_source': pdb_source.get('name', pdb_source.get('identifier', 'unknown')),
            'success': False,
            'error': None
        }
        
        if not BIOPYTHON_AVAILABLE:
            features['error'] = "BioPython not available"
            return features
        
        # Get PDB file path
        pdb_file = self.get_pdb_file(pdb_source)
        if not pdb_file:
            features['error'] = "Failed to get PDB file"
            return features
        
        try:
            # Parse structure
            if self.use_mmcif and pdb_file.endswith('.cif'):
                structure = self.parser.get_structure('structure', pdb_file)
            else:
                structure = self.parser.get_structure('structure', pdb_file)
            
            # Extract basic metadata
            model = structure[0]  # Get first model
            
            # Count chains
            all_chains = list(model.get_chains())
            features['total_chains'] = len(all_chains)
            
            # If chain_ids specified, filter chains
            if chain_ids:
                chains_to_analyze = [chain for chain in all_chains if chain.id in chain_ids]
            else:
                chains_to_analyze = all_chains
            
            features['analyzed_chains'] = len(chains_to_analyze)
            
            if not chains_to_analyze:
                features['error'] = "No chains to analyze"
                return features
            
            # Initialize feature accumulators
            total_residues = 0
            total_atoms = 0
            aa_counts = {}
            secondary_structure_counts = {'helix': 0, 'sheet': 0, 'coil': 0}
            chain_lengths = []
            
            # Analyze each chain
            for chain in chains_to_analyze:
                chain_residues = []
                chain_atoms = []
                
                # Count residues and atoms in this chain
                for residue in chain:
                    if is_aa(residue, standard=True):
                        chain_residues.append(residue)
                        chain_atoms.extend(list(residue.get_atoms()))
                        
                        # Count amino acid types
                        resname = residue.get_resname()
                        aa_counts[resname] = aa_counts.get(resname, 0) + 1
                
                chain_length = len(chain_residues)
                chain_lengths.append(chain_length)
                total_residues += chain_length
                total_atoms += len(chain_atoms)
            
            # Calculate basic statistics
            features['total_residues'] = total_residues
            features['total_atoms'] = total_atoms
            features['residues_per_chain_avg'] = np.mean(chain_lengths) if chain_lengths else 0
            features['residues_per_chain_std'] = np.std(chain_lengths) if len(chain_lengths) > 1 else 0
            features['residues_per_chain_min'] = min(chain_lengths) if chain_lengths else 0
            features['residues_per_chain_max'] = max(chain_lengths) if chain_lengths else 0
            
            # Calculate amino acid composition percentages
            if total_residues > 0:
                # Group by amino acid properties
                hydrophobic_aa = ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO']
                polar_aa = ['SER', 'THR', 'CYS', 'ASN', 'GLN', 'TYR']
                charged_aa = ['ASP', 'GLU', 'LYS', 'ARG', 'HIS']
                special_aa = ['GLY']
                
                hydrophobic_count = sum(aa_counts.get(aa, 0) for aa in hydrophobic_aa)
                polar_count = sum(aa_counts.get(aa, 0) for aa in polar_aa)
                charged_count = sum(aa_counts.get(aa, 0) for aa in charged_aa)
                special_count = sum(aa_counts.get(aa, 0) for aa in special_aa)
                
                features['hydrophobic_percent'] = hydrophobic_count / total_residues * 100
                features['polar_percent'] = polar_count / total_residues * 100
                features['charged_percent'] = charged_count / total_residues * 100
                features['special_percent'] = special_count / total_residues * 100
                
                # Add individual AA percentages for top AAs
                for aa, count in sorted(aa_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    features[f'aa_{aa}_percent'] = count / total_residues * 100
            
            # Try to extract resolution from header
            try:
                if self.use_mmcif and pdb_file.endswith('.cif'):
                    # For mmCIF, resolution might be in different locations
                    # Try common mmCIF tags
                    resolution = None
                    if hasattr(structure, 'header'):
                        header = structure.header
                        if 'resolution' in header:
                            resolution = header['resolution']
                    
                    if resolution is None:
                        # Try to get from _refine.ls_d_res_high
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            try:
                                # Access mmCIF data directly
                                mmcif_dict = structure.header
                                if '_refine.ls_d_res_high' in mmcif_dict:
                                    resolution = float(mmcif_dict['_refine.ls_d_res_high'][0])
                            except:
                                pass
                    
                    features['resolution'] = resolution
                else:
                    # For PDB format, resolution is in structure header
                    if hasattr(structure, 'header') and 'resolution' in structure.header:
                        features['resolution'] = float(structure.header['resolution'])
                    else:
                        features['resolution'] = None
            except:
                features['resolution'] = None
            
            # Calculate atom/residue ratio
            if total_residues > 0:
                features['atoms_per_residue'] = total_atoms / total_residues
            else:
                features['atoms_per_residue'] = 0
            
            # Add success flag
            features['success'] = True
            
            # Clean up temporary file if we downloaded it and not cached
            if not self.cache_dir and pdb_file and os.path.exists(pdb_file) and 'temp' in pdb_file:
                try:
                    os.unlink(pdb_file)
                except:
                    pass
            
            return features
            
        except Exception as e:
            features['error'] = f"Error parsing PDB: {str(e)}"
            # Clean up temporary file if we downloaded it
            if not self.cache_dir and pdb_file and os.path.exists(pdb_file) and 'temp' in pdb_file:
                try:
                    os.unlink(pdb_file)
                except:
                    pass
            return features
    
    def extract_interface_features(self, pdb_source: Dict, chain_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Extract features specific to protein-protein interfaces.
        
        Args:
            pdb_source: Dictionary containing PDB file information from JSON
            chain_pairs: List of (chain1, chain2) tuples to analyze as interfaces
            
        Returns:
            Dictionary of interface features
        """
        interface_features = {
            'pdb_source': pdb_source.get('name', pdb_source.get('identifier', 'unknown')),
            'success': False,
            'error': None,
            'interfaces_analyzed': 0
        }
        
        if not BIOPYTHON_AVAILABLE:
            interface_features['error'] = "BioPython not available"
            return interface_features
        
        # Get PDB file path
        pdb_file = self.get_pdb_file(pdb_source)
        if not pdb_file:
            interface_features['error'] = "Failed to get PDB file"
            return interface_features
        
        try:
            # Parse structure
            if self.use_mmcif and pdb_file.endswith('.cif'):
                structure = self.parser.get_structure('structure', pdb_file)
            else:
                structure = self.parser.get_structure('structure', pdb_file)
            
            model = structure[0]
            
            interface_count = 0
            all_distances = []
            
            for chain1_id, chain2_id in chain_pairs:
                try:
                    chain1 = model[chain1_id]
                    chain2 = model[chain2_id]
                    
                    # Get CA atoms from both chains
                    ca_atoms_chain1 = []
                    ca_atoms_chain2 = []
                    
                    for residue in chain1:
                        if is_aa(residue, standard=True) and 'CA' in residue:
                            ca_atoms_chain1.append(residue['CA'])
                    
                    for residue in chain2:
                        if is_aa(residue, standard=True) and 'CA' in residue:
                            ca_atoms_chain2.append(residue['CA'])
                    
                    if not ca_atoms_chain1 or not ca_atoms_chain2:
                        continue
                    
                    # Calculate minimum distance between chains
                    min_distance = float('inf')
                    for ca1 in ca_atoms_chain1:
                        for ca2 in ca_atoms_chain2:
                            distance = ca1 - ca2
                            if distance < min_distance:
                                min_distance = distance
                    
                    if min_distance < float('inf'):
                        all_distances.append(min_distance)
                        interface_count += 1
                        
                except KeyError:
                    # Chain not found
                    continue
            
            interface_features['interfaces_analyzed'] = interface_count
            
            if interface_count > 0:
                interface_features['min_interface_distance'] = min(all_distances) if all_distances else None
                interface_features['avg_interface_distance'] = np.mean(all_distances) if all_distances else None
                interface_features['max_interface_distance'] = max(all_distances) if all_distances else None
                
                # Count close contacts (< 8Å)
                close_contacts = sum(1 for d in all_distances if d < 8.0)
                interface_features['close_interfaces_percent'] = (close_contacts / interface_count * 100) if interface_count > 0 else 0
            
            interface_features['success'] = True
            
            # Clean up temporary file if we downloaded it
            if not self.cache_dir and pdb_file and os.path.exists(pdb_file) and 'temp' in pdb_file:
                try:
                    os.unlink(pdb_file)
                except:
                    pass
            
            return interface_features
            
        except Exception as e:
            interface_features['error'] = f"Error analyzing interfaces: {str(e)}"
            # Clean up temporary file if we downloaded it
            if not self.cache_dir and pdb_file and os.path.exists(pdb_file) and 'temp' in pdb_file:
                try:
                    os.unlink(pdb_file)
                except:
                    pass
            return interface_features


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
                import shutil
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
            
            # Extract PDB sources if available
            self.extract_pdb_sources()
            
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
    
    def extract_pdb_sources(self):
        """Extract PDB source information from the dataset."""
        print("Extracting PDB source information from dataset...")
        
        # Look for PDB-related sources in the dataset
        if 'hasPart' in self.dataset:
            for item in self.dataset['hasPart']:
                # Check if this item is a PDB file source
                if 'encodingFormat' in item:
                    encoding_format = item['encodingFormat']
                    if encoding_format in ['chemical/x-pdb', 'text/pdb', 'pdb', 'cif', 'mmcif']:
                        # This is likely a PDB file
                        source_id = item.get('identifier', item.get('name', 'unknown'))
                        self.pdb_sources[source_id] = item
                        print(f"  Found PDB source: {source_id}")
        
        print(f"  Total PDB sources found: {len(self.pdb_sources)}")
    
    def get_pdb_source_for_interface(self, interface: Dict) -> Optional[Dict]:
        """
        Get PDB source information for a specific interface.
        
        Args:
            interface: Interface dictionary
            
        Returns:
            PDB source dictionary or None if not found
        """
        # Check if interface has a workExample or citation pointing to PDB
        for field in ['workExample', 'citation', 'subjectOf']:
            if field in interface:
                example = interface[field]
                if isinstance(example, dict):
                    example_id = example.get('identifier', example.get('name', ''))
                    if example_id in self.pdb_sources:
                        return self.pdb_sources[example_id]
                elif isinstance(example, list):
                    for ex in example:
                        if isinstance(ex, dict):
                            ex_id = ex.get('identifier', ex.get('name', ''))
                            if ex_id in self.pdb_sources:
                                return self.pdb_sources[ex_id]
        
        # Try to find PDB source by matching identifier patterns
        interface_id = interface.get('identifier', '')
        interface_name = interface.get('name', '')
        
        # Look for PDB ID in interface identifier or name
        import re
        
        for text in [interface_id, interface_name]:
            if isinstance(text, str):
                # Look for 4-character PDB ID pattern
                match = re.search(r'([0-9][A-Z0-9]{3})', text.upper())
                if match:
                    pdb_id = match.group(1)
                    # Check if we have this PDB in our sources
                    for source_id, source in self.pdb_sources.items():
                        if pdb_id in source_id.upper():
                            return source
        
        return None
    
    def cleanup(self):
        """Clean up temporary files if downloaded from GitHub."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
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
    
    def extract_pdb_features(self, pdb_feature_extractor=None, extract_interface_features=False):
        """
        Extract PDB structural features for interfaces in the dataset.
        
        Args:
            pdb_feature_extractor: Instance of PDBFeatureExtractor
            extract_interface_features: Whether to extract interface-specific features
            
        Returns:
            DataFrame with PDB features
        """
        print("\n=== EXTRACTING PDB STRUCTURAL FEATURES ===")
        
        if not BIOPYTHON_AVAILABLE:
            print("⚠️  BioPython not available. Skipping PDB feature extraction.")
            print("   Install with: pip install biopython")
            return None
        
        if pdb_feature_extractor is None:
            # Create extractor with caching
            cache_dir = Path("pdb_cache")
            pdb_feature_extractor = PDBFeatureExtractor(cache_dir=str(cache_dir))
        
        pdb_features_list = []
        
        print(f"  Processing {len(self.interfaces)} interfaces for PDB features...")
        
        for i, interface in enumerate(self.interfaces):
            if i % 100 == 0 and i > 0:
                print(f"    Processed {i} interfaces...")
            
            interface_id = interface.get('identifier', f'interface_{i}')
            
            # Get PDB source for this interface
            pdb_source = self.get_pdb_source_for_interface(interface)
            
            if not pdb_source:
                # No PDB source found for this interface
                pdb_features_list.append({
                    'interface_id': interface_id,
                    'extraction_success': False,
                    'error': 'No PDB source found'
                })
                continue
            
            # Extract chain information from interface properties
            chain_ids = []
            additional_props = interface.get('additionalProperty', [])
            
            for prop in additional_props:
                prop_name = prop.get('name', '').lower()
                prop_value = str(prop.get('value', ''))
                
                if 'chain' in prop_name and prop_value and len(prop_value) <= 2:
                    # Clean chain identifier
                    chain_id = prop_value.strip().upper()
                    if chain_id and chain_id.isalnum():
                        chain_ids.append(chain_id)
            
            # Extract basic PDB features
            basic_features = pdb_feature_extractor.extract_basic_features(pdb_source, chain_ids)
            
            if basic_features['success']:
                # Create feature dictionary
                feature_dict = {
                    'interface_id': interface_id,
                    'pdb_source': basic_features['pdb_source'],
                    'extraction_success': True
                }
                
                # Add basic features
                for key, value in basic_features.items():
                    if key not in ['pdb_source', 'success', 'error'] and value is not None:
                        feature_dict[f'pdb_{key}'] = value
                
                # Extract interface features if requested and we have chain information
                if extract_interface_features and len(chain_ids) >= 2:
                    # Create chain pairs (all combinations)
                    chain_pairs = []
                    for i in range(len(chain_ids)):
                        for j in range(i+1, len(chain_ids)):
                            chain_pairs.append((chain_ids[i], chain_ids[j]))
                    
                    if chain_pairs:
                        interface_features = pdb_feature_extractor.extract_interface_features(
                            pdb_source, chain_pairs
                        )
                        
                        if interface_features['success']:
                            for key, value in interface_features.items():
                                if key not in ['pdb_source', 'success', 'error', 'interfaces_analyzed'] and value is not None:
                                    feature_dict[f'pdb_{key}'] = value
                
                pdb_features_list.append(feature_dict)
            else:
                print(f"    Failed to extract features for interface {interface_id}: {basic_features.get('error', 'Unknown error')}")
                pdb_features_list.append({
                    'interface_id': interface_id,
                    'extraction_success': False,
                    'error': basic_features.get('error', 'Unknown error')
                })
        
        if pdb_features_list:
            pdb_features_df = pd.DataFrame(pdb_features_list)
            successful_extractions = pdb_features_df[pdb_features_df['extraction_success']]
            
            print(f"✅ Successfully extracted PDB features for {len(successful_extractions)} interfaces")
            
            if len(successful_extractions) > 0:
                # Show feature statistics
                print(f"  Available PDB features:")
                pdb_feature_cols = [col for col in successful_extractions.columns 
                                  if col.startswith('pdb_')]
                
                for col in pdb_feature_cols[:10]:  # Show first 10 features
                    non_null = successful_extractions[col].notna().sum()
                    if non_null > 0:
                        print(f"    {col}: {non_null} non-null values")
                
                if len(pdb_feature_cols) > 10:
                    print(f"    ... and {len(pdb_feature_cols) - 10} more features")
            
            return pdb_features_df
        else:
            print("❌ No PDB features extracted")
            return None
    
    def integrate_pdb_features(self, features_df, pdb_features_df):
        """
        Integrate PDB structural features with existing interface features.
        
        Args:
            features_df: Existing features DataFrame
            pdb_features_df: PDB features DataFrame
        
        Returns:
            Integrated DataFrame
        """
        print("\n=== INTEGRATING PDB FEATURES ===")
        
        if pdb_features_df is None or features_df is None:
            print("  No PDB features to integrate")
            return features_df
        
        # Create a copy of features
        integrated_df = features_df.copy()
        
        # Merge on interface_id
        merged_df = pd.merge(integrated_df, pdb_features_df, on='interface_id', how='left')
        
        # Count successful integrations
        pdb_feature_cols = [col for col in pdb_features_df.columns 
                           if col.startswith('pdb_')]
        
        if pdb_feature_cols:
            # Count interfaces with at least one PDB feature
            integrated_count = merged_df[pdb_feature_cols[0]].notna().sum()
            print(f"✅ Successfully integrated PDB features for {integrated_count} interfaces")
            print(f"   Added {len(pdb_feature_cols)} new PDB features")
        else:
            print("⚠️  No PDB features to add")
        
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
        
        # Handle PDB features (treat most as numeric)
        pdb_feature_cols = [col for col in df.columns if col.startswith('pdb_')]
        print(f"   Processing {len(pdb_feature_cols)} PDB features...")
        
        for col in pdb_feature_cols:
            if col in ['pdb_source', 'pdb_error']:
                df = df.drop(col, axis=1)
                continue
                
            # Try to convert PDB features to numeric
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
        
        # Show PDB sources found
        print(f"PDB sources found: {len(self.pdb_sources)}")
        
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
            
            # Show clusters by class
            if hasattr(self, 'labels'):
                # Create a DataFrame for analysis
                analysis_df = pd.DataFrame({
                    'cluster_id': self.cluster_ids,
                    'label': self.labels if hasattr(self, 'labels') else [0] * len(self.cluster_ids)
                })
                
                # Calculate purity of each cluster
                cluster_purities = []
                for cluster_id in cluster_sizes.index:
                    if pd.notna(cluster_id) and cluster_id != '':
                        cluster_data = analysis_df[analysis_df['cluster_id'] == cluster_id]
                        if len(cluster_data) > 0:
                            majority_class = cluster_data['label'].mode()[0]
                            purity = (cluster_data['label'] == majority_class).mean()
                            cluster_purities.append(purity)
                
                if cluster_purities:
                    print(f"  Cluster purity (homogeneity within clusters):")
                    print(f"    Average: {np.mean(cluster_purities):.3f}")
                    print(f"    Min: {np.min(cluster_purities):.3f}")
                    print(f"    Max: {np.max(cluster_purities):.3f}")


# ============================================
# Now define the new FeatureEvaluator class
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
        
        # 7. Feature statistics boxplot (FIXED)
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
# Update the ProteinInteractionClassifier class
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
# Function definitions (parse_arguments, main, example_usage)
# ============================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train ML models on Croissant-formatted protein interaction dataset with GroupKFold cross-validation, feature evaluation, and PDB structural feature extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use local directory with PDB feature extraction
  python ppi_ml_croissant.py --local ./bioschemas_output --extract-pdb-features
  
  # Extract PDB features with interface analysis
  python ppi_ml_croissant.py --local ./bioschemas_output --extract-pdb-features --extract-interface-features
  
  # Full pipeline with all features
  python ppi_ml_croissant.py --local ./bioschemas_output --evaluate-features --extract-pdb-features --feature-analysis --save-plots
  
  # PDB features only (no model training)
  python ppi_ml_croissant.py --local ./bioschemas_output --extract-pdb-features --pdb-features-only
  
  # Use custom PDB cache directory
  python ppi_ml_croissant.py --local ./bioschemas_output --extract-pdb-features --pdb-cache-dir ./my_pdb_cache
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
        help='Extract basic structural features from PDB files'
    )
    
    parser.add_argument(
        '--extract-interface-features',
        action='store_true',
        help='Extract interface-specific features from PDB files (requires --extract-pdb-features)'
    )
    
    parser.add_argument(
        '--pdb-cache-dir',
        type=str,
        default='./pdb_cache',
        help='Directory to cache downloaded PDB files (default: ./pdb_cache)'
    )
    
    parser.add_argument(
        '--pdb-features-only',
        action='store_true',
        help='Only extract PDB features (skip model training)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("""
╔══════════════════════════════════════════════════════════╗
║  Protein Interaction ML Classifier                       ║
║  with GroupKFold Cross-Validation                        ║
║  Comprehensive Feature Evaluation                        ║
║  and PDB Structural Feature Extraction                   ║
╚══════════════════════════════════════════════════════════╝
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
        
        # Step 3: Extract PDB features (if requested)
        if args.extract_pdb_features:
            print("\n" + "="*60)
            print("Step 3: Extracting PDB Structural Features")
            print("="*60)
            
            if not BIOPYTHON_AVAILABLE:
                print("⚠️  BioPython not available. Skipping PDB feature extraction.")
                print("   Install with: pip install biopython")
            else:
                # Create PDB feature extractor
                pdb_extractor = PDBFeatureExtractor(
                    cache_dir=args.pdb_cache_dir,
                    use_mmcif=True
                )
                
                # Extract PDB features
                pdb_features_df = loader.extract_pdb_features(
                    pdb_feature_extractor=pdb_extractor,
                    extract_interface_features=args.extract_interface_features
                )
                
                # Integrate PDB features with existing features
                if pdb_features_df is not None:
                    features_df = loader.integrate_pdb_features(features_df, pdb_features_df)
                    
                    # Ensure output directory exists before saving
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save PDB features to file
                    pdb_output_file = output_dir / "pdb_features.csv"
                    pdb_features_df.to_csv(pdb_output_file, index=False)
                    print(f"  PDB features saved to: {pdb_output_file}")
                
                # If only PDB features requested, exit here
                if args.pdb_features_only:
                    print("\n✅ PDB feature extraction completed successfully!")
                    print(f"   PDB cache directory: {args.pdb_cache_dir}")
                    if pdb_features_df is not None:
                        successful_extractions = pdb_features_df[pdb_features_df['extraction_success']]
                        print(f"   Extracted features for {len(successful_extractions)} interfaces")
                    loader.cleanup()
                    return
        
        # Step 4: Preprocess features
        print("\n" + "="*60)
        print("Step 4: Preprocessing features")
        print("="*60)
        
        X_processed = loader.preprocess_features(features_df, labels)
        
        # Step 5: Feature Evaluation (if requested)
        if args.evaluate_features or args.feature_report_only:
            print("\n" + "="*60)
            print("Step 5: Performing Comprehensive Feature Evaluation")
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
            print(f"  Unique ClusterIDs: {cluster_ids.nunique()}")
            print(f"  Cross-validation folds: {classifier.n_splits}")
            
            if classifier.best_model_name and results:
                best_result = results[classifier.best_model_name]
                print(f"\n  Best model: {classifier.best_model_name}")
                print(f"    Average CV F1-Score: {best_result['cv_metrics']['f1']:.4f}")
                print(f"    Overall Accuracy:    {best_result['overall_metrics']['accuracy']:.4f}")
            
            loader.cleanup()
            return
        
        # Step 6: Analyze dataset
        print("\n" + "="*60)
        print("Step 6: Analyzing dataset")
        print("="*60)
        
        loader.analyze_dataset()
        
        # Step 7: Initialize classifier with GroupKFold CV
        print("\n" + "="*60)
        print("Step 7: Initializing classifier with GroupKFold CV")
        print("="*60)
        
        classifier = ProteinInteractionClassifier(
            random_state=args.random_state,
            n_splits=args.folds
        )
        
        # Step 8: Train and evaluate models
        print("\n" + "="*60)
        print("Step 8: Training and evaluating models")
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
        
        # Step 9: Save results
        print("\n" + "="*60)
        print("Step 9: Saving results")
        print("="*60)
        
        classifier.save_results(output_dir=args.output_dir)
        
        if args.save_plots:
            # Step 10: Generate visualizations
            print("\n" + "="*60)
            print("Step 10: Generating visualizations")
            print("="*60)
            
            if args.test_split > 0:
                classifier.plot_test_results(save_plots=args.save_plots, output_dir=args.output_dir)
            else:
                classifier.plot_cv_results(save_plots=args.save_plots, output_dir=args.output_dir)
            
            # Step 11: Feature importance analysis
            if args.feature_analysis:
                print("\n" + "="*60)
                print("Step 11: Analyzing feature importance")
                print("="*60)
                
                classifier.feature_importance_analysis(
                    X_processed, labels,
                    save_plots=args.save_plots,
                    output_dir=args.output_dir
                )
        
        # Step 12: Summary
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
            print(f"PDB features extracted: {pdb_features_count}")
        
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
        
        if args.extract_pdb_features:
            print(f"\nPDB Feature Extraction:")
            print(f"  - PDB cache directory: {args.pdb_cache_dir}")
            print(f"  - Interface features extracted: {'Yes' if args.extract_interface_features else 'No'}")
        
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
        
        print("\n✅ ML pipeline with GroupKFold cross-validation and PDB feature extraction completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error in ML pipeline: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup temporary files
        if 'loader' in locals():
            loader.cleanup()


def example_usage():
    """Show example usage of the script with PDB feature extraction."""
    print("""
Example Usage with PDB Feature Extraction:
==========================================

1. Extract PDB features only:
   python ppi_ml_croissant.py --local ./bioschemas_output --extract-pdb-features --pdb-features-only

2. Extract PDB features with interface analysis:
   python ppi_ml_croissant.py --local ./bioschemas_output --extract-pdb-features --extract-interface-features

3. Full pipeline with PDB features:
   python ppi_ml_croissant.py --local ./bioschemas_output --extract-pdb-features --evaluate-features --feature-analysis

4. Use custom PDB cache directory:
   python ppi_ml_croissant.py --local ./bioschemas_output --extract-pdb-features --pdb-cache-dir ./my_pdb_cache

5. Combine PDB features with quick evaluation:
   python ppi_ml_croissant.py --local ./bioschemas_output --extract-pdb-features --quick

Note: PDB feature extraction requires BioPython. Install with: pip install biopython
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
