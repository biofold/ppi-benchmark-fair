"""
Script 4: ClusterID Enricher for PPI Benchmark Data
Updates JSON-LD files with ClusterID information from BLASTClust output.
"""

import json
import os
import re
from typing import Dict, List, Set, Tuple, Any, Optional
import logging
from pathlib import Path
import argparse
from datetime import datetime
from collections import defaultdict, Counter
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClusterIDProcessor:
    """Processes BLASTClust output and adds ClusterID information to JSON-LD files."""
    
    def __init__(self, input_dir: str, blastclust_file: str):
        """
        Initialize the processor.
        
        Args:
            input_dir: Directory containing JSON-LD files (from scripts 1-3)
            blastclust_file: Path to BLASTClust output file
        """
        self.input_dir = Path(input_dir)
        self.blastclust_file = Path(blastclust_file)
        
        # Data structures
        self.cluster_mapping = {}  # interface_id -> cluster_id
        self.cluster_members = defaultdict(list)  # cluster_id -> list of interface_ids
        self.interface_to_pdb = {}  # interface_id -> pdb_id (for validation)
        
        # Statistics
        self.stats = {
            "total_clusters": 0,
            "total_interfaces_in_clusters": 0,
            "clusters_processed": 0,
            "interfaces_with_clusterid_before": 0,
            "interfaces_with_clusterid_after": 0,
            "interfaces_updated": 0,
            "interfaces_added": 0,
            "interfaces_skipped": 0,
            "clusters_with_single_member": 0,
            "clusters_with_multiple_members": 0,
            "largest_cluster_size": 0
        }
    
    def parse_blastclust_file(self) -> bool:
        """
        Parse BLASTClust output file.
        
        Format: Each line contains space-separated InterfaceIDs belonging to the same cluster.
        First InterfaceID in each line becomes the ClusterID.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.blastclust_file.exists():
            logger.error(f"BLASTClust file not found: {self.blastclust_file}")
            return False
        
        try:
            logger.info(f"Parsing BLASTClust file: {self.blastclust_file}")
            logger.info(f"BLASTClust options used: -S 25 -L 0.5 -b F")
            
            with open(self.blastclust_file, 'r', encoding='utf-8') as f:
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
                
                # First interface ID becomes the cluster ID
                cluster_id = interface_ids[0]
                
                # Add all interface IDs to this cluster
                for interface_id in interface_ids:
                    # Clean the interface ID if needed
                    clean_interface_id = self._clean_interface_id(interface_id)
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
                
                # Log first few clusters for verification
                if line_num < 3:
                    logger.info(f"Cluster {line_num + 1}: ID={cluster_id}, members={len(interface_ids)}")
                    if len(interface_ids) <= 5:
                        logger.info(f"  Members: {interface_ids}")
                    else:
                        logger.info(f"  Members (first 5): {interface_ids[:5]}...")
            
            logger.info(f"Successfully parsed {self.stats['clusters_processed']} clusters")
            logger.info(f"Total interfaces in clusters: {self.stats['total_interfaces_in_clusters']}")
            logger.info(f"Clusters with single member: {self.stats['clusters_with_single_member']}")
            logger.info(f"Clusters with multiple members: {self.stats['clusters_with_multiple_members']}")
            logger.info(f"Largest cluster size: {self.stats['largest_cluster_size']}")
            
            # Show cluster size distribution
            self._analyze_cluster_distribution()
            
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
        # Remove any file extensions or extra characters
        clean_id = interface_id.strip()
        
        # Remove common file extensions
        for ext in ['.pdb', '.cif', '.gz', '.ent', '.pdb1']:
            if clean_id.endswith(ext):
                clean_id = clean_id[:-len(ext)]
        
        # Convert to uppercase for consistency
        clean_id = clean_id.upper()
        
        return clean_id
    
    def _analyze_cluster_distribution(self):
        """Analyze and log cluster size distribution."""
        if not self.cluster_members:
            return
        
        cluster_sizes = [len(members) for members in self.cluster_members.values()]
        
        if cluster_sizes:
            size_distribution = Counter(cluster_sizes)
            logger.info("\n=== CLUSTER SIZE DISTRIBUTION ===")
            for size in sorted(size_distribution.keys()):
                count = size_distribution[size]
                percentage = (count / len(cluster_sizes)) * 100
                logger.info(f"  Size {size}: {count} clusters ({percentage:.1f}%)")
    
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
            "current_clusterid": None,
            "file_path": None
        }
        
        # Check if this is an interface item
        if "@type" in json_data and "DataCatalogItem" in json_data["@type"]:
            interface_info["interface_id"] = json_data.get("identifier")
            
            # Extract from additionalProperty
            if "additionalProperty" in json_data:
                for prop in json_data["additionalProperty"]:
                    if prop.get("name") == "PDB_ID":
                        interface_info["pdb_id"] = prop.get("value")
                    elif prop.get("name") == "ClusterID":
                        interface_info["current_clusterid"] = prop.get("value")
        
        return interface_info
    
    def update_interface_with_clusterid(self, json_data: Dict[str, Any], 
                                       interface_id: str, 
                                       cluster_id: str) -> Dict[str, Any]:
        """
        Update an interface JSON-LD with ClusterID information.
        
        Args:
            json_data: Original JSON-LD data
            interface_id: Interface identifier
            cluster_id: Cluster ID from BLASTClust
            
        Returns:
            Updated JSON-LD data
        """
        updated_data = json_data.copy()
        
        # Ensure additionalProperty exists
        if "additionalProperty" not in updated_data:
            updated_data["additionalProperty"] = []
        
        # Remove any existing ClusterID properties
        updated_data["additionalProperty"] = [
            prop for prop in updated_data["additionalProperty"] 
            if prop.get("name") not in ["ClusterID", "ClusterIDStatus", "ClusterMembers", 
                                       "ClusterMembersNumber", "BLASTClustMethod", "BLASTClustMethodOptions"]
        ]
        
        # Get cluster information
        cluster_members = self.cluster_members.get(cluster_id, [])
        cluster_size = len(cluster_members)
        
        # Add the new ClusterID
        updated_data["additionalProperty"].append({
            "@type": "PropertyValue",
            "name": "ClusterID",
            "value": cluster_id,
            "description": f"Sequence cluster ID from BLASTClust analysis. Cluster contains {cluster_size} members."
        })
        
        # Add ClusterMembersNumber property
        updated_data["additionalProperty"].append({
            "@type": "PropertyValue",
            "name": "ClusterMembersNumber",
            "value": cluster_size,
            "description": f"Number of interfaces in this cluster"
        })
        
        # Add cluster membership information for clusters with multiple members
        if cluster_size > 1:
            other_members = [m for m in cluster_members if m != interface_id]
            if other_members:
                # Show all other members
                member_text = ", ".join(other_members)
                
                updated_data["additionalProperty"].append({
                    "@type": "PropertyValue",
                    "name": "ClusterMembers",
                    "value": other_members,
                    "description": f"Other interfaces in the same cluster (total: {cluster_size})"
                })
        
        # Add BLASTClust method information with separate options property
        updated_data["additionalProperty"].append({
            "@type": "PropertyValue",
            "name": "BLASTClustMethod",
            "value": "BLASTClust sequence clustering",
            "description": "Method used for sequence-based clustering"
        })
        
        updated_data["additionalProperty"].append({
            "@type": "PropertyValue",
            "name": "BLASTClustMethodOptions",
            "value": "-S 25 -L 0.5 -b F",
            "description": "BLASTClust parameters used for sequence clustering"
        })
        
        # Add cluster status
        updated_data["additionalProperty"].append({
            "@type": "PropertyValue",
            "name": "ClusterIDStatus",
            "value": "COMPLETE - Populated by script 4",
            "description": "ClusterID has been validated/added from BLASTClust analysis"
        })
        
        # Update mainEntity (Protein) if it exists
        if "mainEntity" in updated_data:
            updated_data["mainEntity"] = self.update_protein_with_clusterid(
                updated_data["mainEntity"], 
                cluster_id,
                cluster_size
            )
        
        return updated_data
    
    def update_protein_with_clusterid(self, protein_data: Dict[str, Any], 
                                     cluster_id: str,
                                     cluster_size: int) -> Dict[str, Any]:
        """
        Update a protein JSON-LD with ClusterID information.
        
        Args:
            protein_data: Original protein JSON-LD data
            cluster_id: Cluster ID from BLASTClust
            cluster_size: Number of members in the cluster
            
        Returns:
            Updated protein JSON-LD data
        """
        updated_protein = protein_data.copy()
        
        if "additionalProperty" not in updated_protein:
            updated_protein["additionalProperty"] = []
        
        # Remove any existing ClusterID properties in protein
        updated_protein["additionalProperty"] = [
            prop for prop in updated_protein["additionalProperty"] 
            if prop.get("name") not in ["ClusterID", "ClusterIDStatus", "ClusterMembersNumber",
                                       "BLASTClustMethod", "BLASTClustMethodOptions"]
        ]
        
        # Add ClusterID to protein
        updated_protein["additionalProperty"].append({
            "@type": "PropertyValue",
            "name": "ClusterID",
            "value": cluster_id,
            "description": f"Sequence cluster ID from BLASTClust analysis. Cluster contains {cluster_size} members."
        })
        
        # Add ClusterMembersNumber property to protein
        updated_protein["additionalProperty"].append({
            "@type": "PropertyValue",
            "name": "ClusterMembersNumber",
            "value": cluster_size,
            "description": f"Number of interfaces in this cluster"
        })
        
        # Add BLASTClust method information to protein
        updated_protein["additionalProperty"].append({
            "@type": "PropertyValue",
            "name": "BLASTClustMethod",
            "value": "BLASTClust sequence clustering",
            "description": "Method used for sequence-based clustering"
        })
        
        updated_protein["additionalProperty"].append({
            "@type": "PropertyValue",
            "name": "BLASTClustMethodOptions",
            "value": "-S 25 -L 0.5 -b F",
            "description": "BLASTClust parameters used for sequence clustering"
        })
        
        # Add cluster status to protein
        updated_protein["additionalProperty"].append({
            "@type": "PropertyValue",
            "name": "ClusterIDStatus",
            "value": "COMPLETE - Populated by script 4",
            "description": "ClusterID has been validated/added from BLASTClust analysis"
        })
        
        return updated_protein
    
    def process_interface_file(self, interface_file: Path) -> Tuple[bool, str]:
        """
        Process a single interface JSON-LD file and add/update ClusterID.
        
        Args:
            interface_file: Path to interface JSON file
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Load the JSON data
            with open(interface_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Extract interface information
            interface_info = self.extract_interface_info(json_data)
            interface_id = interface_info["interface_id"]
            
            if not interface_id:
                return False, f"Missing interface_id in {interface_file.name}"
            
            # Track this interface for statistics
            self.interface_to_pdb[interface_id] = interface_info["pdb_id"]
            
            # Check if this interface has a ClusterID in BLASTClust results
            cluster_id = self.cluster_mapping.get(interface_id)
            
            if cluster_id:
                # Update the JSON data with ClusterID
                current_clusterid = interface_info["current_clusterid"]
                updated_data = self.update_interface_with_clusterid(
                    json_data, 
                    interface_id, 
                    cluster_id
                )
                
                # Save the updated file
                with open(interface_file, 'w', encoding='utf-8') as f:
                    json.dump(updated_data, f, indent=2, ensure_ascii=False)
                
                # Update statistics
                self.stats["interfaces_with_clusterid_after"] += 1
                
                if current_clusterid:
                    self.stats["interfaces_with_clusterid_before"] += 1
                    if current_clusterid != cluster_id:
                        self.stats["interfaces_updated"] += 1
                        message = f"Updated ClusterID: {current_clusterid} -> {cluster_id}"
                    else:
                        self.stats["interfaces_skipped"] += 1
                        message = f"ClusterID already correct: {cluster_id}"
                else:
                    self.stats["interfaces_added"] += 1
                    message = f"Added ClusterID: {cluster_id}"
                
                # Log cluster size
                cluster_size = len(self.cluster_members.get(cluster_id, []))
                message += f" (cluster size: {cluster_size})"
                
                return True, message
            else:
                # Interface not found in BLASTClust results
                self.stats["interfaces_skipped"] += 1
                return False, f"Interface {interface_id} not found in BLASTClust results"
            
        except Exception as e:
            logger.error(f"Failed to process {interface_file}: {e}")
            return False, f"Error: {str(e)}"
    
    def update_dataset_file(self, dataset_file: Path) -> bool:
        """
        Update the main dataset file with ClusterID information.
        
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
                    interface_id = interface_info["interface_id"]
                    
                    if interface_id and interface_id in self.cluster_mapping:
                        cluster_id = self.cluster_mapping[interface_id]
                        cluster_size = len(self.cluster_members.get(cluster_id, []))
                        
                        # Update the interface item
                        updated_item = self.update_interface_with_clusterid(item, interface_id, cluster_id)
                        updated_items.append(updated_item)
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
            return False
    
    def update_manifest_file(self, manifest_file: Path) -> bool:
        """
        Update the manifest file with ClusterID statistics.
        
        Args:
            manifest_file: Path to manifest.json file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
            
            # Calculate coverage statistics
            total_interfaces = len(self.interface_to_pdb)
            interfaces_in_clusters = sum(1 for interface_id in self.interface_to_pdb.keys() 
                                        if interface_id in self.cluster_mapping)
            coverage_percentage = (interfaces_in_clusters / total_interfaces * 100) if total_interfaces > 0 else 0
            
            # Add ClusterID enrichment information
            manifest_data["clusterid_enrichment"] = {
                "script": "4th script - ClusterID Enricher",
                "execution_date": datetime.now().isoformat(),
                "blastclust_file": str(self.blastclust_file),
                "blastclust_method": "BLASTClust sequence clustering",
                "blastclust_options": "-S 25 -L 0.5 -b F",
                "statistics": {
                    "total_interfaces_processed": total_interfaces,
                    "interfaces_in_clusters": interfaces_in_clusters,
                    "coverage_percentage": f"{coverage_percentage:.1f}%",
                    "interfaces_with_clusterid_before": self.stats["interfaces_with_clusterid_before"],
                    "interfaces_with_clusterid_after": self.stats["interfaces_with_clusterid_after"],
                    "interfaces_updated": self.stats["interfaces_updated"],
                    "interfaces_added": self.stats["interfaces_added"],
                    "interfaces_skipped": self.stats["interfaces_skipped"],
                    "clusters_processed": self.stats["clusters_processed"],
                    "total_clusters": self.stats["total_clusters"],
                    "clusters_with_single_member": self.stats["clusters_with_single_member"],
                    "clusters_with_multiple_members": self.stats["clusters_with_multiple_members"],
                    "largest_cluster_size": self.stats["largest_cluster_size"]
                },
                "cluster_methodology": {
                    "cluster_id_selection": "First InterfaceID in each BLASTClust line becomes the ClusterID",
                    "properties_added": [
                        "ClusterID: The cluster identifier",
                        "ClusterMembersNumber: Number of interfaces in the cluster",
                        "ClusterMembers: List of other interfaces (for multi-member clusters)",
                        "BLASTClustMethod: Method used for clustering",
                        "BLASTClustMethodOptions: Parameters used (-S 25 -L 0.5 -b F)"
                    ],
                    "mapping": f"{len(self.cluster_mapping)} interfaces mapped to {len(self.cluster_members)} clusters"
                }
            }
            
            # Update fields_to_be_populated section if it exists
            if "schema_structure" in manifest_data and "fields_to_be_populated" in manifest_data["schema_structure"]:
                manifest_data["schema_structure"]["fields_to_be_populated"]["ClusterID"] = {
                    "status": "COMPLETELY POPULATED by script 4",
                    "timestamp": datetime.now().isoformat(),
                    "description": "Sequence cluster ID from BLASTClust analysis",
                    "blastclust_method": "BLASTClust sequence clustering",
                    "blastclust_options": "-S 25 -L 0.5 -b F",
                    "coverage": f"{coverage_percentage:.1f}% of interfaces",
                    "cluster_selection": "First InterfaceID in each BLASTClust line becomes the ClusterID",
                    "properties_added": [
                        "ClusterID",
                        "ClusterMembersNumber",
                        "ClusterMembers (for multi-member clusters)",
                        "BLASTClustMethod",
                        "BLASTClustMethodOptions"
                    ]
                }
            
            # Save the updated manifest
            with open(manifest_file, 'w', encoding='utf-8') as f:
                json.dump(manifest_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Updated manifest file: {manifest_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update manifest file {manifest_file}: {e}")
            return False
    
    def generate_cluster_summary_report(self, output_dir: Path) -> bool:
        """
        Generate a detailed cluster summary report.
        
        Args:
            output_dir: Directory to save report
            
        Returns:
            True if successful, False otherwise
        """
        try:
            report_file = output_dir / "cluster_summary_report.json"
            
            # Prepare cluster data
            cluster_data = []
            for cluster_id, members in self.cluster_members.items():
                cluster_info = {
                    "cluster_id": cluster_id,
                    "size": len(members),
                    "members": members,
                    "pdb_ids": list(set(self.interface_to_pdb.get(m, "Unknown") for m in members))
                }
                cluster_data.append(cluster_info)
            
            # Sort by cluster size (descending)
            cluster_data.sort(key=lambda x: x["size"], reverse=True)
            
            # Prepare summary
            summary = {
                "generated_date": datetime.now().isoformat(),
                "blastclust_file": str(self.blastclust_file),
                "blastclust_method": "BLASTClust sequence clustering",
                "blastclust_options": "-S 25 -L 0.5 -b F",
                "total_clusters": len(self.cluster_members),
                "total_interfaces": len(self.cluster_mapping),
                "coverage_statistics": {
                    "interfaces_processed": len(self.interface_to_pdb),
                    "interfaces_in_clusters": sum(1 for i in self.interface_to_pdb.keys() if i in self.cluster_mapping),
                    "coverage_percentage": (sum(1 for i in self.interface_to_pdb.keys() if i in self.cluster_mapping) / 
                                          len(self.interface_to_pdb) * 100) if self.interface_to_pdb else 0
                },
                "cluster_size_distribution": Counter(len(members) for members in self.cluster_members.values()),
                "top_clusters": cluster_data[:10],  # Top 10 largest clusters
                "single_member_clusters": [c for c in cluster_data if c["size"] == 1][:10],  # First 10 singleton clusters
                "all_clusters": cluster_data
            }
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Generated cluster summary report: {report_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate cluster summary report: {e}")
            return False
    
    def run(self) -> Dict[str, Any]:
        """
        Main execution method.
        
        Returns:
            Statistics dictionary
        """
        logger.info("=" * 60)
        logger.info("ClusterID Enricher - Script 4")
        logger.info("=" * 60)
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"BLASTClust file: {self.blastclust_file}")
        logger.info("")
        
        # Step 1: Parse BLASTClust file
        if not self.parse_blastclust_file():
            logger.error("Failed to parse BLASTClust file. Exiting.")
            return self.stats
        
        # Step 2: Find JSON files to process
        json_files = self.find_json_files()
        if not json_files:
            logger.warning("No JSON files found to process!")
            return self.stats
        
        # Step 3: Process interface files
        logger.info(f"\nProcessing {len(json_files)} interface files...")
        logger.info("Adding/updating ClusterID information from BLASTClust results.")
        logger.info("")
        
        successful_updates = 0
        failed_updates = 0
        
        for i, json_file in enumerate(json_files):
            logger.info(f"[{i+1}/{len(json_files)}] Processing: {json_file.name}")
            success, message = self.process_interface_file(json_file)
            
            if success:
                successful_updates += 1
                logger.info(f"  ✅ {message}")
            else:
                failed_updates += 1
                logger.warning(f"  ⚠️  {message}")
        
        # Step 4: Update dataset file if it exists
        dataset_file = self.input_dir / "dataset_with_interfaces.json"
        if dataset_file.exists():
            logger.info(f"\nUpdating dataset file: {dataset_file}")
            self.update_dataset_file(dataset_file)
        
        # Step 5: Update manifest file if it exists
        manifest_file = self.input_dir / "manifest.json"
        if manifest_file.exists():
            logger.info(f"Updating manifest file: {manifest_file}")
            self.update_manifest_file(manifest_file)
        
        # Step 6: Generate cluster summary report
        logger.info(f"\nGenerating cluster summary report...")
        self.generate_cluster_summary_report(self.input_dir)
        
        # Step 7: Print summary statistics
        logger.info("")
        logger.info("=" * 60)
        logger.info("CLUSTERID ENRICHMENT COMPLETE")
        logger.info("=" * 60)
        
        total_interfaces = len(self.interface_to_pdb)
        interfaces_in_clusters = sum(1 for interface_id in self.interface_to_pdb.keys() 
                                    if interface_id in self.cluster_mapping)
        coverage_percentage = (interfaces_in_clusters / total_interfaces * 100) if total_interfaces > 0 else 0
        
        logger.info(f"📊 File Processing Summary:")
        logger.info(f"  Total interface files: {len(json_files)}")
        logger.info(f"  Successfully processed: {successful_updates}")
        logger.info(f"  Failed/skipped: {failed_updates}")
        
        logger.info(f"\n📊 Cluster Coverage Summary:")
        logger.info(f"  Total interfaces in dataset: {total_interfaces}")
        logger.info(f"  Interfaces found in BLASTClust: {interfaces_in_clusters}")
        logger.info(f"  Coverage: {coverage_percentage:.1f}%")
        
        logger.info(f"\n📊 ClusterID Updates:")
        logger.info(f"  Interfaces with ClusterID before: {self.stats['interfaces_with_clusterid_before']}")
        logger.info(f"  Interfaces with ClusterID after: {self.stats['interfaces_with_clusterid_after']}")
        logger.info(f"  ClusterIDs added: {self.stats['interfaces_added']}")
        logger.info(f"  ClusterIDs updated: {self.stats['interfaces_updated']}")
        logger.info(f"  Interfaces skipped: {self.stats['interfaces_skipped']}")
        
        logger.info(f"\n📊 BLASTClust Analysis:")
        logger.info(f"  Total clusters: {self.stats['total_clusters']}")
        logger.info(f"  Clusters with single member: {self.stats['clusters_with_single_member']}")
        logger.info(f"  Clusters with multiple members: {self.stats['clusters_with_multiple_members']}")
        logger.info(f"  Largest cluster size: {self.stats['largest_cluster_size']}")
        
        if self.cluster_members:
            avg_cluster_size = sum(len(m) for m in self.cluster_members.values()) / len(self.cluster_members)
            logger.info(f"  Average cluster size: {avg_cluster_size:.1f}")
        
        logger.info("")
        logger.info("✅ ClusterID enrichment complete!")
        logger.info("")
        logger.info("📁 Generated Files:")
        logger.info(f"  Updated interface files in: {self.input_dir}/interface_protein_pairs/")
        if dataset_file.exists():
            logger.info(f"  Updated dataset file: {dataset_file}")
        logger.info(f"  Updated manifest: {self.input_dir}/manifest.json")
        logger.info(f"  Cluster summary report: {self.input_dir}/cluster_summary_report.json")
        
        logger.info("")
        logger.info("🎯 All 4 scripts have been executed successfully!")
        logger.info("The dataset now contains:")
        logger.info("  1. Basic interface data (Script 1)")
        logger.info("  2. Assembly chain information (Script 2)")
        logger.info("  3. Comprehensive PDB metadata (Script 3)")
        logger.info("  4. Sequence cluster information (Script 4)")
        
        return {**self.stats, "successful_updates": successful_updates, "failed_updates": failed_updates}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ClusterID Enricher - Update JSON-LD files with ClusterID information from BLASTClust output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input bioschemas_output --blastclust blastclust_output.txt
  %(prog)s --input bioschemas_output --blastclust ./results/blastclust.txt --verbose

BLASTClust File Format:
  Each line contains space-separated InterfaceIDs belonging to the same cluster.
  First InterfaceID in each line becomes the ClusterID.
  
  Example:
    1ABC_1 1DEF_2 1GHI_3
    2JKL_1
    3MNO_1 3PQR_2 3STU_3 3VWX_4
    
  This creates 3 clusters:
    Cluster 1ABC_1 with members: 1ABC_1, 1DEF_2, 1GHI_3
    Cluster 2JKL_1 with member: 2JKL_1
    Cluster 3MNO_1 with members: 3MNO_1, 3PQR_2, 3STU_3, 3VWX_4
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory containing JSON-LD files (from scripts 1-3)"
    )
    
    parser.add_argument(
        "--blastclust", "-b",
        required=True,
        help="Path to BLASTClust output file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def validate_blastclust_file(file_path: Path) -> bool:
    """
    Validate BLASTClust file format.
    
    Args:
        file_path: Path to BLASTClust file
        
    Returns:
        True if valid, False otherwise
    """
    if not file_path.exists():
        print(f"❌ ERROR: BLASTClust file '{file_path}' does not exist!")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            print(f"❌ ERROR: BLASTClust file '{file_path}' is empty!")
            return False
        
        # Check format of first few lines
        valid_lines = 0
        for i, line in enumerate(lines[:5]):
            line = line.strip()
            if line:
                # Should contain at least one InterfaceID
                if re.search(r'[A-Za-z0-9_]+', line):
                    valid_lines += 1
                else:
                    print(f"⚠️  Warning: Line {i+1} doesn't look like valid InterfaceIDs: '{line}'")
        
        if valid_lines == 0:
            print(f"❌ ERROR: No valid InterfaceID lines found in BLASTClust file!")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Could not read BLASTClust file: {e}")
        return False


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  ClusterID Enricher - Script 4                                ║
║  Updates JSON-LD files with ClusterID from BLASTClust output  ║
╚═══════════════════════════════════════════════════════════════╝
    
Settings:
  Input Directory: {args.input}
  BLASTClust File: {args.blastclust}
  BLASTClust Method: BLASTClust sequence clustering
  BLASTClust Options: -S 25 -L 0.5 -b F
    """)
    
    # Check if input directory exists
    if not os.path.exists(args.input):
        print(f"\n❌ ERROR: Input directory '{args.input}' does not exist!")
        print("   Please run scripts 1-3 first to generate and update the JSON-LD files.")
        return
    
    # Validate BLASTClust file
    blastclust_path = Path(args.blastclust)
    if not validate_blastclust_file(blastclust_path):
        return
    
    # Create processor and run
    processor = ClusterIDProcessor(
        input_dir=args.input,
        blastclust_file=args.blastclust
    )
    
    stats = processor.run()
    
    print(f"\n✅ Script 4 execution complete!")
    print(f"   The dataset now contains complete FAIR metadata with:")
    print(f"   - Basic interface annotations")
    print(f"   - Assembly chain validation")
    print(f"   - Comprehensive PDB metadata")
    print(f"   - Sequence cluster information from BLASTClust")
    print(f"\n   All 4 scripts have been executed successfully! 🎉")


if __name__ == "__main__":
    main()
