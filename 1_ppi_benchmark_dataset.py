"""
Script to import protein-protein interaction benchmark data and generate FAIR-compliant 
Bioschemas markup with Croissant compatibility for ML datasets.

Script 1: Initial JSON-LD generation with empty fields for subsequent scripts to populate.
"""

import pandas as pd
import json
import requests
from io import BytesIO
from datetime import datetime
from typing import Set, Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import logging
import os
import gzip
import argparse
import time
from urllib.parse import quote
from collections import Counter

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
    ClusterID: Optional[str] = None  # BLASTClust cluster ID (to be populated by 4th script)
    LabelChain1: Optional[str] = None  # To be populated by assembly checking script (script 2)
    LabelChain2: Optional[str] = None  # To be populated by assembly checking script (script 2)
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

class PPIBenchmarkProcessor:
    """Processor for protein-protein interaction benchmark data with Croissant support."""
    
    def __init__(self, csv_url: str, mmcif_base_url: str, pdb_base_url: str, 
                 csv_separator: str = ',', auto_detect_separator: bool = False):
        """
        Initialize the processor with the CSV and structure file URLs.
        
        Args:
            csv_url: URL to the benchmark CSV file
            mmcif_base_url: Base URL for mmCIF structure files
            pdb_base_url: Base URL for PDB structure files
            csv_separator: Separator to use for CSV parsing (default: ',')
            auto_detect_separator: Whether to auto-detect separator if default fails
        """
        self.csv_url = csv_url
        self.mmcif_base_url = mmcif_base_url.rstrip('/') + '/'
        self.pdb_base_url = pdb_base_url.rstrip('/') + '/'
        self.csv_separator = csv_separator
        self.auto_detect_separator = auto_detect_separator
        self.dataset = None
        self.protein_interfaces = []
        self.column_mapping = {}  # Store actual column names
    
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
                self.dataset = pd.read_csv(self.csv_url, sep=self.csv_separator).iloc[:, 2:].head(5)
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
                'comments': ['comments', 'comment', 'notes'],
                'clusterid': ['clusterid', 'cluster_id', 'cluster', 'blastclust_id']  # Added ClusterID
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
                
                # NOTE: Assembly chain checking is skipped - LabelChain1 and LabelChain2 will remain empty
                # They will be populated by script 2
                
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
                
                # Extract ClusterID if present
                clusterid = self._get_column_value(row, 'clusterid')
                if clusterid is not None and pd.notna(clusterid):
                    interface.ClusterID = str(clusterid).strip()
                else:
                    # Set to placeholder - will be populated by script 4
                    interface.ClusterID = None
                
                # Add to interfaces list
                self.protein_interfaces.append(interface)
                processed_count += 1
                
                # Log first few entries for verification
                if idx < 3:
                    logger.info(f"Parsed sample entry {idx}: ID={ID}, InterfaceID={InterfaceID}, source={interface_source}, physio={physio_bool}, label={interface.label}")
                    if interface.ClusterID:
                        logger.info(f"  ClusterID: {interface.ClusterID}")
                
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
            
            # Count ClusterID statistics
            clusterid_count = sum(1 for pi in self.protein_interfaces if pi.ClusterID is not None)
            logger.info(f"Interfaces with ClusterID: {clusterid_count} (will be populated by script 4)")
            
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
            
            # Avoid division by zero
            if self.protein_interfaces:
                logger.info(f"Balance: {physio_count/len(self.protein_interfaces):.2%} physiological")
            
            # Show field statistics if available
            if self.protein_interfaces:
                bsa_values = [pi.bsa for pi in self.protein_interfaces if pi.bsa is not None]
                if bsa_values:
                    logger.info(f"Buried Surface Area (BSA): {len(bsa_values)} entries, avg={sum(bsa_values)/len(bsa_values):.1f} Å²")
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
                },
                {
                    "@type": "PropertyValue",
                    "name": "ClusterID",
                    "description": "Sequence cluster ID from BLASTClust analysis (options: -S 25 -L 0.5 -b F)",
                    "value": "String identifier"
                }
            ],
            
            "measurementTechnique": [
                "X-ray crystallography",
                "Conservation of interaction geometry analysis",
                "Cross-crystal form comparison (ProtCID)",
                "Homolog comparison (QSalign)",
                "Sequence clustering (BLASTClust)"
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
                
                # Add ClusterID if available
                if interface.ClusterID:
                    interface_item["additionalProperty"].append({
                        "@type": "PropertyValue",
                        "name": "ClusterID",
                        "value": interface.ClusterID,
                        "description": f"Sequence cluster ID from BLASTClust analysis (-S 25 -L 0.5 -b F). Will be populated by script 4."
                    })
                else:
                    interface_item["additionalProperty"].append({
                        "@type": "PropertyValue",
                        "name": "ClusterID",
                        "value": "To be populated by script 4",
                        "description": "Sequence cluster ID from BLASTClust analysis (-S 25 -L 0.5 -b F). Will be populated by script 4."
                    })
                
                # Add all interface features as additional properties
                self._add_interface_features_to_item(interface_item, interface)
                
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
        
        # NOTE: LabelChain1 and LabelChain2 are intentionally omitted - they will be populated by script 2
        
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
            mmcif_url = f"{self.mmcif_base_url}{interface.InterfaceID}.cif.gz"
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
            
            # Taxonomic information (will be updated by script 3 with PDB metadata)
            "taxonomicRange": {
                "@type": "DefinedTerm",
                "name": "Organism-specific protein",
                "inDefinedTermSet": "https://www.ncbi.nlm.nih.gov/taxonomy",
                "placeholder_note": "Organism information will be populated from PDB metadata by script 3"
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
                },
                {
                    "@type": "PropertyValue",
                    "name": "AssemblyChainStatus",
                    "value": "To be populated by script 2",
                    "description": "Assembly chain information (LabelChain1, LabelChain2) will be added by script 2"
                },
                {
                    "@type": "PropertyValue",
                    "name": "PDBMetadataStatus",
                    "value": "To be populated by script 3",
                    "description": "PDB metadata (resolution, sequences, organism, etc.) will be added by script 3"
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
        
        # Add ClusterID if available
        if interface.ClusterID:
            protein_markup["additionalProperty"].append({
                "@type": "PropertyValue",
                "name": "ClusterID",
                "value": interface.ClusterID,
                "description": "Sequence cluster ID from BLASTClust analysis (-S 25 -L 0.5 -b F). Will be populated by script 4."
            })
        
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
                "protcid_count": 0,
                "clusterid_count": 0
            }
        
        physiological_count = sum(1 for pi in self.protein_interfaces if pi.label == 1)
        non_physiological_count = sum(1 for pi in self.protein_interfaces if pi.label == 0)
        
        # Count interface sources
        qsalign_count = sum(1 for pi in self.protein_interfaces if pi.interface_source == "QSalign")
        protcid_count = sum(1 for pi in self.protein_interfaces if pi.interface_source == "ProtCID")
        other_source_count = sum(1 for pi in self.protein_interfaces if pi.interface_source not in ["QSalign", "ProtCID"])
        
        # Count ClusterID statistics
        clusterid_count = sum(1 for pi in self.protein_interfaces if pi.ClusterID is not None)
        
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
            "clusterid_count": clusterid_count,
            "clusterid_missing": len(self.protein_interfaces) - clusterid_count,
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
                        "features": ["ID", "InterfaceID", "AuthChain1", "AuthChain2", "bsa", "contacts", "gene", "superfamily", "pfam", "ClusterID"]
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
                "features_included": "All interface features are included in additionalProperty of each interface item",
                "fields_to_be_populated": {
                    "LabelChain1": "Will be populated by script 2 (assembly chain checking)",
                    "LabelChain2": "Will be populated by script 2 (assembly chain checking)",
                    "PDB_metadata": "Will be populated by script 3 (PDB metadata fetching)",
                    "ClusterID": "Will be populated by script 4 (BLASTClust clustering with -S 25 -L 0.5 -b F)"
                }
            }
        }
        
        return fair_package
    
    def save_bioschemas_markup(self, output_dir: str = "bioschemas_output") -> Dict[str, Any]:
        """
        Save all generated Bioschemas markup to files.
        
        Args:
            output_dir: Directory to save output files
            
        Returns:
            Dictionary with paths to generated files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
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
            
            # Add ClusterID if available
            if interface.ClusterID:
                interface_item["additionalProperty"].append({
                    "@type": "PropertyValue",
                    "name": "ClusterID",
                    "value": interface.ClusterID,
                    "description": f"Sequence cluster ID from BLASTClust analysis (-S 25 -L 0.5 -b F). Will be populated by script 4."
                })
            else:
                interface_item["additionalProperty"].append({
                    "@type": "PropertyValue",
                    "name": "ClusterID",
                    "value": "To be populated by script 4",
                    "description": "Sequence cluster ID from BLASTClust analysis (-S 25 -L 0.5 -b F). Will be populated by script 4."
                })
            
            # Add all interface features to the additionalProperty
            self._add_interface_features_to_item(interface_item, interface)
            
            safe_interface_id = self._clean_filename(interface.InterfaceID)
            interface_file = os.path.join(interfaces_dir, f"interface_{safe_interface_id}.json")
            with open(interface_file, 'w', encoding='utf-8') as f:
                json.dump(interface_item, f, indent=2, ensure_ascii=False)
            
            generated_files.append(f"interface_{safe_interface_id}.json")
            
            # Log first few entries with their features
            if i < 3:
                logger.info(f"Generated interface-protein markup for {interface.InterfaceID} (source: {interface.interface_source})")
                logger.info(f"  Features included: physio={interface.physio}, bsa={interface.bsa}, contacts={interface.contacts}, gene={interface.gene}")
                if interface.ClusterID:
                    logger.info(f"  ClusterID: {interface.ClusterID} (will be populated by script 4)")
                logger.info(f"  Assembly chains: LabelChain1 and LabelChain2 will be populated by script 2")
                logger.info(f"  PDB metadata: Will be populated by script 3")
                logger.info(f"  Saved to: {interface_file}")
            
            # Log progress for large datasets
            if len(self.protein_interfaces) > 100 and i % 100 == 0 and i > 0:
                logger.info(f"Progress: Generated {i} out of {len(self.protein_interfaces)} interface files...")
        
        logger.info(f"Generated ALL {len(self.protein_interfaces)} interface-protein markup files in {interfaces_dir}")
        logger.info(f"  Each file includes all interface features in additionalProperty")
        logger.info(f"  NOTE: The following fields are placeholders to be populated by subsequent scripts:")
        logger.info(f"    - LabelChain1/LabelChain2: Will be populated by script 2 (assembly checking)")
        logger.info(f"    - PDB metadata: Will be populated by script 3")
        logger.info(f"    - ClusterID: Will be populated by script 4 (BLASTClust with -S 25 -L 0.5 -b F)")
        
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
                                                         if len([pi for pi in self.protein_interfaces if pi.ID == pid]) > 1),
                "fields_to_be_populated": {
                    "LabelChain1": {
                        "status": "EMPTY - To be populated by script 2",
                        "description": "First chain in the assembly interface (will be validated from structure file)"
                    },
                    "LabelChain2": {
                        "status": "EMPTY - To be populated by script 2",
                        "description": "Second chain in the assembly interface (will be validated from structure file)"
                    },
                    "PDB_metadata": {
                        "status": "EMPTY - To be populated by script 3",
                        "description": "Comprehensive PDB metadata from RCSB API",
                        "data_will_include": [
                            "Resolution",
                            "Experimental method",
                            "Source organism(s)",
                            "ALL chain sequences",
                            "Homomer detection",
                            "Sequence clusters",
                            "Citation information",
                            "Entity information"
                        ]
                    },
                    "ClusterID": {
                        "status": "PARTIAL - To be completed by script 4",
                        "description": "Sequence cluster ID from BLASTClust analysis",
                        "blastclust_options": "-S 25 -L 0.5 -b F",
                        "script": "Will be populated/validated by script 4"
                    }
                }
            }
        }
        
        # Add ClusterID statistics to manifest
        clusterid_count = sum(1 for pi in self.protein_interfaces if pi.ClusterID is not None)
        manifest["clusterid_stats"] = {
            "interfaces_with_clusterid": clusterid_count,
            "interfaces_needing_clusterid": len(self.protein_interfaces) - clusterid_count,
            "blastclust_options": "-S 25 -L 0.5 -b F",
            "population_script": "script 4"
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
        """
    )
    
    parser.add_argument(
        "--csv-url",
        default="https://raw.githubusercontent.com/vibbits/Elixir-3DBioInfo-Benchmark-Protein-Interfaces/refs/heads/main/benchmark/benchmark_annotated_updated_30042023.csv",
        help="URL of the CSV file (default: the benchmark dataset)"
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
        default="bioschemas_output",
        help="Output directory for generated files (default: 'bioschemas_output')"
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
            
            print(f"\nFirst 5 lines:")
            for i, line in enumerate(lines[:5]):
                print(f"Line {i}: {repr(line)}")
            
            print(f"\n=== TRYING TO PARSE WITH PANDAS ===")
            import pandas as pd
            
            # Try with the specified separator
            try:
                df = pd.read_csv(csv_url, sep=separator)
                print(f"✅ Successfully parsed with separator '{separator}'")
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {list(df.columns)}")
                
                print(f"\nFirst 3 rows:")
                for i in range(min(3, len(df))):
                    print(f"Row {i}:")
                    for col in df.columns[:5]:  # Show first 5 columns
                        print(f"  {col}: {repr(df.iloc[i][col])}")
                
                # Check for our expected columns
                expected_columns = ['ID', 'InterfaceID', 'physio']
                available_columns = [str(col).strip() for col in df.columns]
                
                print(f"\n=== COLUMN ANALYSIS ===")
                print(f"Looking for: {expected_columns}")
                print(f"Available: {available_columns}")
                
                for expected in expected_columns:
                    found = False
                    for available in available_columns:
                        if expected.lower() in available.lower():
                            print(f"  ✅ Found '{expected}' as '{available}'")
                            found = True
                            break
                    if not found:
                        print(f"  ❌ Missing '{expected}'")
                
                return df
                
            except Exception as e:
                print(f"❌ Failed with separator '{separator}': {e}")
                
                # Try other separators
                separators_to_try = ['\t', ';', ' ', '|']
                for sep in separators_to_try:
                    try:
                        df = pd.read_csv(csv_url, sep=sep)
                        print(f"\n✅ Successfully parsed with separator '{repr(sep)}'")
                        print(f"   Shape: {df.shape}")
                        print(f"   Columns: {list(df.columns)}")
                        return df
                    except:
                        continue
                
                print(f"\n❌ Could not parse with any separator")
                return None
                
        else:
            print(f"❌ Failed to download CSV: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error in debug function: {e}")
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
║  PPI Benchmark FAIR Metadata Generator - Script 1             ║
║  Initial JSON-LD creation with placeholders for scripts 2-4   ║
╚═══════════════════════════════════════════════════════════════╝
    
Settings:
  CSV URL:         {args.csv_url}
  CSV Separator:   '{args.separator}' (auto-detect: {args.auto_detect})
  Output Directory: {args.output}
  mmCIF Files:     {args.mmcif_url}
  PDB Files:       {args.pdb_url}
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
        auto_detect_separator=args.auto_detect
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
            
            # DEBUG: Show what columns we actually have
            print(f"\n=== ACTUAL COLUMNS ===")
            for i, col in enumerate(data.columns):
                print(f"   {i}: {col} (type: {data[col].dtype})")
                
                # Show sample value
                if len(data) > 0:
                    sample = data.iloc[0][col]
                    print(f"     Sample: {repr(sample)}")
            
            # Check for critical columns
            critical_cols = ['id', 'interfaceid', 'physio']
            missing = [col for col in critical_cols if col not in processor.column_mapping]
            if missing:
                print(f"\n⚠️  WARNING: Missing critical columns: {missing}")
                print(f"   Looking for lowercase versions of: ID, InterfaceID, physio")
                print(f"   Available columns: {list(data.columns)}")
                
                # Try to find similar columns
                print(f"\n=== SEARCHING FOR SIMILAR COLUMNS ===")
                for col in data.columns:
                    col_lower = str(col).lower()
                    if 'id' in col_lower and 'interface' not in col_lower:
                        print(f"   Possible ID column: '{col}'")
                    elif 'interface' in col_lower:
                        print(f"   Possible InterfaceID column: '{col}'")
                    elif 'physio' in col_lower or 'label' in col_lower:
                        print(f"   Possible physio column: '{col}'")
            
            # Parse into structured objects
            print(f"\n=== PARSING PROTEIN INTERFACES ===")
            interfaces = processor.parse_data()
            
            if interfaces and len(interfaces) > 0:
                print(f"✅ Successfully parsed {len(interfaces)} protein interfaces!")
                
                # Show interface source statistics
                qsalign_count = sum(1 for pi in interfaces if pi.interface_source == "QSalign")
                protcid_count = sum(1 for pi in interfaces if pi.interface_source == "ProtCID")
                other_count = len(interfaces) - qsalign_count - protcid_count
                
                print(f"   Interface sources: QSalign={qsalign_count}, ProtCID={protcid_count}, Other={other_count}")
                
                # Show ClusterID statistics
                clusterid_count = sum(1 for pi in interfaces if pi.ClusterID is not None)
                print(f"   ClusterID field: {clusterid_count} entries have values (will be populated/validated by script 4)")
                
                # Show placeholder status
                print(f"\n=== PLACEHOLDER FIELDS ===")
                print(f"   The following fields are intentionally left empty for subsequent scripts:")
                print(f"   1. LabelChain1/LabelChain2: Will be populated by script 2 (assembly checking)")
                print(f"   2. PDB metadata: Will be populated by script 3 (RCSB API)")
                print(f"   3. ClusterID: Will be populated/validated by script 4 (BLASTClust with -S 25 -L 0.5 -b F)")
                
                # Show sample of the structure
                print(f"\n=== SAMPLE DATA STRUCTURE ===")
                print(f"Schema: Dataset → Interface items → Protein objects")
                for i, interface in enumerate(interfaces[:3]):
                    print(f"   Interface {i+1}: {interface.InterfaceID}")
                    print(f"     Source: {interface.interface_source}")
                    print(f"     Protein: {interface.ID}")
                    print(f"     Chains: {interface.AuthChain1}-{interface.AuthChain2}")
                    print(f"     Assembly Chains: LabelChain1/LabelChain2 will be populated by script 2")
                    print(f"     Classification: {'Physiological' if interface.label == 1 else 'Non-physiological'}")
                    print(f"     Features: physio={interface.physio}, bsa={interface.bsa}, contacts={interface.contacts}")
                    if interface.ClusterID:
                        print(f"     ClusterID: {interface.ClusterID} (will be validated by script 4)")
                    else:
                        print(f"     ClusterID: Will be populated by script 4 (BLASTClust -S 25 -L 0.5 -b F)")
                
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
                print(f"\n   Interface ID Sources:")
                print(f"     QSalign format (PDBID_X): {stats.get('qsalign_count', 0)}")
                print(f"     ProtCID integer: {stats.get('protcid_count', 0)}")
                print(f"     Other formats: {stats.get('other_source_count', 0)}")
                print(f"\n   ClusterID Status:")
                print(f"     Interfaces with ClusterID: {stats.get('clusterid_count', 0)}")
                print(f"     Interfaces needing ClusterID: {stats.get('clusterid_missing', 0)}")
                print(f"     ClusterID source: BLASTClust analysis with options: -S 25 -L 0.5 -b F")
                print(f"     ClusterID population: Will be done by script 4")
                
                print(f"\n=== NEXT STEPS ===")
                print(f"   1. Run script 2 to add assembly chain information (LabelChain1, LabelChain2)")
                print(f"   2. Run script 3 to add PDB metadata (resolution, sequences, organism, etc.)")
                print(f"   3. Run script 4 to add/validate ClusterID information")
                
                print(f"\n=== GENERATED FILES ===")
                print(f"   Dataset with Interfaces: {args.output}/dataset_with_interfaces.json")
                print(f"   Interface-Protein Pairs: {args.output}/interface_protein_pairs/ (ALL {len(interfaces)} files)")
                print(f"   FAIR Metadata Package: {args.output}/fair_metadata_package.json")
                print(f"   HTML Snippet: {args.output}/embedded_markup.html")
                print(f"   Manifest: {args.output}/manifest.json")
                
                print(f"\n=== SCHEMA STRUCTURE ===")
                print(f"   Dataset → hasPart → Interface items (ALL {len(interfaces)} interfaces)")
                print(f"   Each Interface item → mainEntity → Protein object")
                print(f"   Each Protein object represents a PDB structure (4 letters)")
                print(f"   Each interface includes ALL features in additionalProperty")
                print(f"   NOTE: The following fields are placeholders to be populated later:")
                print(f"     - LabelChain1/LabelChain2: Empty (script 2)")
                print(f"     - PDB metadata: Empty (script 3)")
                print(f"     - ClusterID: May be empty or contain preliminary values (script 4)")
                
            else:
                print(f"\n❌ ERROR: No interfaces were parsed!")
                print(f"   Parsed {len(interfaces)} interfaces")
                print(f"\n   Possible issues:")
                print(f"   1. Column names don't match expected patterns")
                print(f"   2. CSV separator is incorrect")
                print(f"   3. Data format is unexpected")
                print(f"\n   Try running with:")
                print(f"      python {__file__} --test (to inspect CSV)")
                print(f"      python {__file__} --auto-detect (to auto-detect separator)")
                print(f"      python {__file__} --separator \"\\t\" (try tab separator)")
        
    except Exception as e:
        print(f"\n❌ ERROR in main execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
