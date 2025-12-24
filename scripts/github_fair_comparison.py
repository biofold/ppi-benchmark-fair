#!/usr/bin/env python3
"""
GitHub Repository FAIRness Evaluator with Metadata File Detection
Evaluates repositories by cloning them locally and parsing metadata files
"""

import os
import json
import tempfile
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict
import pandas as pd
import yaml  # For YAML metadata files
import xml.etree.ElementTree as ET  # For XML metadata files
import re  # For pattern matching

# Try to import rdflib for RDF parsing (optional)
try:
    import rdflib
    RDFLIB_AVAILABLE = True
except ImportError:
    RDFLIB_AVAILABLE = False
    print("Note: rdflib not installed. RDF metadata parsing will be limited.")

@dataclass
class FAIRScoreWithImprovements:
    """FAIR score with specific improvement suggestions"""
    repository: str
    stars: int = 0
    last_updated: str = ""
    
    # Scores
    findable: float = 0.0
    accessible: float = 0.0
    interoperable: float = 0.0
    reusable: float = 0.0
    total: float = 0.0
    
    # Missing elements
    missing_findable: List[str] = field(default_factory=list)
    missing_accessible: List[str] = field(default_factory=list)
    missing_interoperable: List[str] = field(default_factory=list)
    missing_reusable: List[str] = field(default_factory=list)
    
    # Improvement actions
    improvement_actions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Detailed breakdown
    detailed_scores: Dict[str, Dict] = field(default_factory=dict)
    
    # Extracted metadata
    extracted_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MetadataFile:
    """Represents a detected metadata file"""
    path: str
    format: str  # 'json', 'yaml', 'xml', 'rdf', 'text'
    content: Dict[str, Any]
    raw_content: str
    standard: Optional[str] = None  # 'schema.org', 'datacite', 'bioschemas', 'dcat', 'dublin-core'

class GitHubFAIRComparator:
    """Compare FAIRness of GitHub repositories by cloning them locally"""
    
    def __init__(self, cache_dir: str = None, max_repo_size_mb: int = 500):
        """
        Initialize the comparator
        
        Args:
            cache_dir: Directory to cache cloned repositories (optional)
            max_repo_size_mb: Maximum repository size to clone in MB
        """
        self.cache_dir = cache_dir
        self.max_repo_size_mb = max_repo_size_mb
        self.clone_timeout = 300  # 5 minutes
        
        # Metadata file patterns to look for
        self.metadata_patterns = [
            # Common metadata file names
            ('*.json', 'json'),
            ('*.yaml', 'yaml'),
            ('*.yml', 'yaml'),
            ('*.xml', 'xml'),
            ('*.rdf', 'rdf'),
            ('*.ttl', 'rdf'),  # Turtle RDF
            ('*.n3', 'rdf'),   # Notation3
            ('*.nt', 'rdf'),   # N-Triples
            ('*.jsonld', 'json'),  # JSON-LD
        ]
        
        # Directory patterns where metadata is often found
        self.metadata_directories = [
            'bioschema',
            'bioschemas',
            'metadata',
            'meta',
            '.metadata',
            'data',
            'docs',
            'documentation',
            'config',
            'configuration'
        ]
        
        # Known metadata standards and their indicators
        self.metadata_standards = {
            'schema.org': ['@type', '@context', 'schema.org', 'https://schema.org'],
            'datacite': ['datacite.org', 'resourceTypeGeneral', 'creators', 'titles'],
            'bioschemas': ['bioschemas.org', 'bioschemas:', 'BioschemasDataset'],
            'dcat': ['dcat:', 'Dataset', 'Catalog', 'dcat:Dataset'],
            'dublin-core': ['dc:', 'dcterms:', 'dct:', 'title', 'creator', 'subject'],
            'fair': ['fair', 'FAIR', 'findable', 'accessible', 'interoperable', 'reusable'],
            'codemeta': ['codemeta', 'CodeMeta'],
            'zenodo': ['zenodo', '.zenodo.json'],
        }
        
        # Define scoring criteria with enhanced weights for metadata
        self.criteria = {
            'findable': [
                ('has_readme', 'README file', 10, "Add a comprehensive README.md file"),
                ('has_metadata_file', 'Metadata file', 25, "Add metadata.json or other metadata file"),
                ('has_structured_metadata', 'Structured metadata', 20, "Add structured metadata (JSON/XML/YAML)"),
                ('has_doi_or_pid', 'DOI or Persistent ID', 15, "Add DOI or other persistent identifier"),
                ('has_topics', 'Repository topics', 5, "Add relevant topics to repository"),
                ('has_description', 'Repository description', 5, "Add a clear description"),
                ('has_website', 'Project website', 5, "Add homepage URL"),
                ('has_releases', 'GitHub releases', 5, "Create GitHub releases"),
                ('has_wiki', 'Documentation wiki', 5, "Enable or populate wiki"),
                ('has_license_in_metadata', 'License in metadata', 5, "Include license information in metadata"),
            ],
            'accessible': [
                ('is_public', 'Public repository', 30, "Make repository public"),
                ('has_open_license', 'Open license', 25, "Add an open-source license"),
                ('has_access_rights', 'Access rights in metadata', 20, "Specify access rights in metadata"),
                ('has_download_instructions', 'Download instructions', 15, "Add download/access instructions"),
                ('has_data_files', 'Data in repository', 10, "Include actual data files, not just links"),
            ],
            'interoperable': [
                ('uses_standard_formats', 'Standard formats', 25, "Use standard formats like CSV, JSON, HDF5"),
                ('has_schema', 'Data schema', 20, "Add schema.json or data dictionary"),
                ('has_standard_metadata', 'Standard metadata schema', 20, "Use standard metadata schema (Schema.org, DataCite)"),
                ('has_vocabularies', 'Controlled vocabularies', 15, "Use controlled vocabularies/ontologies"),
                ('has_usage_scripts', 'Usage scripts', 10, "Add Python/R scripts for data usage"),
                ('has_format_info', 'Format information', 10, "Include format information in metadata"),
            ],
            'reusable': [
                ('has_license_file', 'LICENSE file', 25, "Add LICENSE file with clear terms"),
                ('has_citation_file', 'Citation file', 20, "Add CITATION.cff or citation file"),
                ('has_provenance_info', 'Provenance information', 15, "Add provenance/creation information"),
                ('has_examples', 'Examples/notebooks', 15, "Add examples/notebooks directory"),
                ('has_contact_info', 'Contact information', 10, "Include contact information in metadata"),
                ('has_detailed_readme', 'Detailed README', 10, "Expand README with usage examples"),
                ('has_issue_templates', 'Issue templates', 5, "Add issue templates for bug reports"),
            ]
        }
        
        # Create cache directory if specified
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def compare_repos(self, repo_urls: List[str]) -> Dict:
        """Compare multiple repositories with improvement suggestions"""
        results = []
        all_improvements = []
        
        print(f"🔍 Starting evaluation of {len(repo_urls)} repositories...")
        
        for i, repo_url in enumerate(repo_urls, 1):
            try:
                print(f"\n[{i}/{len(repo_urls)}] Evaluating {repo_url}...")
                score_obj = self.evaluate_repo(repo_url)
                
                if score_obj:
                    results.append(score_obj)
                    
                    # Extract improvements
                    repo_improvements = self._extract_improvements(score_obj, repo_url)
                    all_improvements.extend(repo_improvements)
                    
                    # Print metadata summary if found
                    if score_obj.extracted_metadata:
                        meta_count = len(score_obj.extracted_metadata.get('files', []))
                        print(f"   ✓ Score: {score_obj.total:.1f}/100 | Metadata files: {meta_count}")
                    else:
                        print(f"   ✓ Score: {score_obj.total:.1f}/100")
                else:
                    print(f"   ✗ Failed to evaluate repository")
                
            except Exception as e:
                print(f"   ✗ Error: {str(e)[:100]}...")
                continue
        
        # Generate comprehensive report
        report = self._generate_detailed_report(results, all_improvements)
        
        return report
    
    def evaluate_repo(self, repo_url: str) -> Optional[FAIRScoreWithImprovements]:
        """Evaluate a single repository by cloning it locally"""
        try:
            # Parse repository URL
            owner, repo = self._parse_repo_url(repo_url)
            
            # Clone repository to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                clone_path = Path(temp_dir) / repo
                
                # Clone the repository (shallow clone for speed)
                print(f"   Cloning repository...", end="", flush=True)
                try:
                    clone_cmd = [
                        'git', 'clone', '--depth', '1',
                        '--filter=blob:none',  # Don't clone file contents initially
                        repo_url, str(clone_path)
                    ]
                    
                    result = subprocess.run(
                        clone_cmd,
                        capture_output=True,
                        text=True,
                        timeout=self.clone_timeout
                    )
                    
                    if result.returncode != 0:
                        print(f" ✗ Failed: {result.stderr[:100]}")
                        return None
                    
                    print(" ✓ Done")
                    
                except subprocess.TimeoutExpired:
                    print(" ✗ Timeout")
                    return None
                except Exception as e:
                    print(f" ✗ Error: {str(e)[:100]}")
                    return None
                
                # Get repository contents from local clone
                print(f"   Scanning files...", end="", flush=True)
                contents = self._scan_local_contents(clone_path)
                print(f" ✓ Found {len(contents)} files")
                
                # Scan for metadata files
                print(f"   Scanning metadata...", end="", flush=True)
                metadata_files = self._scan_metadata_files(clone_path, contents)
                print(f" ✓ Found {len(metadata_files)} metadata files")
                
                # Parse metadata files
                parsed_metadata = self._parse_metadata_files(metadata_files)
                
                # Extract basic repository info from git config
                repo_info = self._get_local_repo_info(clone_path, repo_url)
                
                # Initialize score object with metadata
                score_obj = FAIRScoreWithImprovements(
                    repository=repo_url,
                    stars=0,  # Can't get stars without API
                    last_updated=datetime.now().isoformat(),
                    extracted_metadata=parsed_metadata
                )
                
                # Evaluate each FAIR principle (passing metadata for evaluation)
                score_obj.detailed_scores = {}
                
                for principle, criteria_list in self.criteria.items():
                    principle_score = 0
                    max_score = sum([weight for _, _, weight, _ in criteria_list])
                    missing = []
                    
                    for criterion, description, weight, improvement in criteria_list:
                        # Check if criterion is met
                        checker_name = f"_check_{criterion}"
                        checker = getattr(self, checker_name, None)
                        
                        if checker:
                            # Pass metadata files to checkers that need them
                            is_met, details = checker(repo_info, contents, clone_path, metadata_files, parsed_metadata)
                            
                            if is_met:
                                principle_score += weight
                            else:
                                missing.append(description)
                                score_obj.improvement_actions.append({
                                    'principle': principle,
                                    'criterion': description,
                                    'action': improvement,
                                    'points': weight,
                                    'repository': repo_url
                                })
                    
                    # Calculate percentage score
                    percentage_score = (principle_score / max_score * 100) if max_score > 0 else 0
                    
                    # Store score
                    setattr(score_obj, principle, percentage_score)
                    
                    # Store missing elements
                    missing_attr = f"missing_{principle}"
                    setattr(score_obj, missing_attr, missing)
                    
                    # Store detailed breakdown
                    score_obj.detailed_scores[principle] = {
                        'score': principle_score,
                        'max_score': max_score,
                        'percentage': percentage_score,
                        'missing': missing
                    }
                
                # Calculate total score
                principles = ['findable', 'accessible', 'interoperable', 'reusable']
                total = sum(getattr(score_obj, p) for p in principles) / len(principles)
                score_obj.total = total
                
                return score_obj
                
        except Exception as e:
            print(f"   ✗ Evaluation failed: {str(e)[:100]}")
            return None
    
    def _scan_metadata_files(self, clone_path: Path, contents: List[Dict]) -> List[Dict]:
        """Scan for metadata files in the repository"""
        metadata_files = []
        
        # Look for files matching metadata patterns
        for file_item in contents:
            file_path = Path(file_item['local_path'])
            
            # Check file name patterns
            for pattern, file_type in self.metadata_patterns:
                if file_path.match(pattern):
                    # Check if in metadata directory or has metadata in name
                    rel_path = str(file_path.relative_to(clone_path))
                    path_lower = rel_path.lower()
                    
                    # Check if file is in a metadata directory or has metadata-related name
                    is_metadata_dir = any(f'/{dir}/' in f'/{path_lower}/' for dir in self.metadata_directories)
                    has_metadata_name = any(term in path_lower for term in [
                        'metadata', 'meta', 'schema', 'bioschema', 'datacite', 
                        'dcat', 'dublin', 'codemeta', 'citation', 'license'
                    ])
                    
                    if is_metadata_dir or has_metadata_name:
                        metadata_files.append({
                            'path': rel_path,
                            'local_path': str(file_path),
                            'type': file_type,
                            'size': file_item.get('size', 0)
                        })
        
        return metadata_files
    
    def _parse_metadata_files(self, metadata_files: List[Dict]) -> Dict[str, Any]:
        """Parse metadata files and extract structured information"""
        parsed_files = []
        all_content = {}
        
        for meta_file in metadata_files:
            try:
                file_path = meta_file['local_path']
                file_type = meta_file['type']
                
                # Read file content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_content = f.read()
                
                # Parse based on file type
                if file_type == 'json':
                    try:
                        content = json.loads(raw_content)
                        standard = self._detect_metadata_standard(content, raw_content)
                        parsed_files.append(MetadataFile(
                            path=meta_file['path'],
                            format='json',
                            content=content,
                            raw_content=raw_content,
                            standard=standard
                        ))
                        all_content[meta_file['path']] = content
                    except json.JSONDecodeError:
                        # Try to parse as JSON-LD or fix common issues
                        content = self._parse_json_with_fixes(raw_content)
                        if content:
                            standard = self._detect_metadata_standard(content, raw_content)
                            parsed_files.append(MetadataFile(
                                path=meta_file['path'],
                                format='json',
                                content=content,
                                raw_content=raw_content,
                                standard=standard
                            ))
                            all_content[meta_file['path']] = content
                
                elif file_type == 'yaml':
                    try:
                        content = yaml.safe_load(raw_content)
                        standard = self._detect_metadata_standard(content, raw_content)
                        parsed_files.append(MetadataFile(
                            path=meta_file['path'],
                            format='yaml',
                            content=content if content else {},
                            raw_content=raw_content,
                            standard=standard
                        ))
                        if content:
                            all_content[meta_file['path']] = content
                    except yaml.YAMLError:
                        pass
                
                elif file_type == 'xml':
                    try:
                        root = ET.fromstring(raw_content)
                        content = self._xml_to_dict(root)
                        standard = self._detect_metadata_standard(content, raw_content)
                        parsed_files.append(MetadataFile(
                            path=meta_file['path'],
                            format='xml',
                            content=content,
                            raw_content=raw_content,
                            standard=standard
                        ))
                        all_content[meta_file['path']] = content
                    except ET.ParseError:
                        pass
                
                elif file_type == 'rdf':
                    content = self._parse_rdf(raw_content, meta_file['path'])
                    if content:
                        standard = self._detect_metadata_standard(content, raw_content)
                        parsed_files.append(MetadataFile(
                            path=meta_file['path'],
                            format='rdf',
                            content=content,
                            raw_content=raw_content,
                            standard=standard
                        ))
                        all_content[meta_file['path']] = content
            
            except Exception as e:
                # Skip files that can't be parsed
                continue
        
        # Extract combined metadata information
        extracted_info = self._extract_metadata_info(parsed_files)
        
        return {
            'files': [{'path': f.path, 'format': f.format, 'standard': f.standard} 
                     for f in parsed_files],
            'parsed_files': parsed_files,
            'combined_info': extracted_info,
            'all_content': all_content
        }
    
    def _parse_json_with_fixes(self, raw_content: str) -> Optional[Dict]:
        """Try to parse JSON with common fixes"""
        # Remove BOM if present
        if raw_content.startswith('\ufeff'):
            raw_content = raw_content[1:]
        
        # Try to fix common JSON issues
        fixes = [
            # Fix trailing commas
            (r',\s*}', '}'),
            (r',\s*]', ']'),
            # Fix single quotes
            (r"'([^']*)'", r'"\1"'),
        ]
        
        fixed_content = raw_content
        for pattern, replacement in fixes:
            fixed_content = re.sub(pattern, replacement, fixed_content)
        
        try:
            return json.loads(fixed_content)
        except json.JSONDecodeError:
            # Try to extract JSON objects from text
            json_pattern = r'\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}'
            matches = re.findall(json_pattern, raw_content, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _xml_to_dict(self, element: ET.Element) -> Dict:
        """Convert XML element to dictionary"""
        result = {}
        
        # Add element tag
        result['_tag'] = element.tag
        
        # Add attributes
        if element.attrib:
            result['_attributes'] = dict(element.attrib)
        
        # Add text content
        if element.text and element.text.strip():
            result['_text'] = element.text.strip()
        
        # Process children
        children = {}
        for child in element:
            child_dict = self._xml_to_dict(child)
            child_tag = child.tag
            
            # Handle multiple children with same tag
            if child_tag in children:
                if not isinstance(children[child_tag], list):
                    children[child_tag] = [children[child_tag]]
                children[child_tag].append(child_dict)
            else:
                children[child_tag] = child_dict
        
        if children:
            result.update(children)
        
        return result
    
    def _parse_rdf(self, raw_content: str, file_path: str) -> Optional[Dict]:
        """Parse RDF content"""
        if not RDFLIB_AVAILABLE:
            # Simple text extraction for RDF without rdflib
            return {'_format': 'rdf', '_content': raw_content[:1000]}
        
        try:
            g = rdflib.Graph()
            
            # Try different RDF formats
            formats = ['xml', 'n3', 'turtle', 'nt', 'json-ld']
            for fmt in formats:
                try:
                    g.parse(data=raw_content, format=fmt)
                    break
                except:
                    continue
            
            # Extract basic triples as dictionary
            result = {'_format': 'rdf', '_triples': []}
            for subj, pred, obj in g:
                result['_triples'].append({
                    'subject': str(subj),
                    'predicate': str(pred),
                    'object': str(obj)
                })
            
            return result
        
        except Exception:
            return {'_format': 'rdf', '_content': raw_content[:1000]}
    
    def _detect_metadata_standard(self, content: Any, raw_content: str) -> Optional[str]:
        """Detect which metadata standard is being used"""
        if isinstance(content, dict):
            content_str = json.dumps(content).lower()
        elif isinstance(content, str):
            content_str = content.lower()
        else:
            content_str = str(content).lower()
        
        raw_lower = raw_content.lower()
        
        for standard, indicators in self.metadata_standards.items():
            for indicator in indicators:
                if indicator.lower() in content_str or indicator.lower() in raw_lower:
                    return standard
        
        # Check for specific patterns
        if '@context' in content_str and ('schema.org' in content_str or 'https://schema.org' in content_str):
            return 'schema.org'
        elif 'resourceTypeGeneral' in content_str or 'datacite.org' in content_str:
            return 'datacite'
        elif 'bioschemas' in content_str:
            return 'bioschemas'
        
        return None
    
    def _extract_metadata_info(self, metadata_files: List[MetadataFile]) -> Dict[str, Any]:
        """Extract key information from parsed metadata files"""
        info = {
            'has_license': False,
            'has_doi': False,
            'has_contact': False,
            'has_provenance': False,
            'has_access_rights': False,
            'has_format_info': False,
            'has_vocabularies': False,
            'standards_used': set(),
            'license_types': set(),
            'formats_mentioned': set(),
        }
        
        for meta_file in metadata_files:
            if meta_file.standard:
                info['standards_used'].add(meta_file.standard)
            
            content = meta_file.content
            if not isinstance(content, dict):
                continue
            
            # Check for license information
            license_fields = ['license', 'licence', 'rights', 'rightsHolder', 'accessRights']
            for field in license_fields:
                if field in content:
                    info['has_license'] = True
                    license_val = content[field]
                    if isinstance(license_val, str):
                        info['license_types'].add(license_val.lower())
            
            # Check for DOI
            doi_fields = ['doi', 'identifier', 'pid', 'persistentId']
            for field in doi_fields:
                if field in content:
                    field_val = content[field]
                    if isinstance(field_val, str) and ('10.' in field_val or 'doi.org' in field_val):
                        info['has_doi'] = True
                    elif isinstance(field_val, dict) and 'value' in field_val:
                        val_str = str(field_val['value'])
                        if '10.' in val_str or 'doi.org' in val_str:
                            info['has_doi'] = True
            
            # Check for contact information
            contact_fields = ['contact', 'creator', 'author', 'maintainer', 'publisher']
            for field in contact_fields:
                if field in content:
                    info['has_contact'] = True
            
            # Check for provenance
            provenance_fields = ['created', 'dateCreated', 'provenance', 'version', 'history']
            for field in provenance_fields:
                if field in content:
                    info['has_provenance'] = True
            
            # Check for access rights
            access_fields = ['accessRights', 'availability', 'access', 'rights']
            for field in access_fields:
                if field in content:
                    info['has_access_rights'] = True
            
            # Check for format information
            format_fields = ['format', 'encodingFormat', 'mediaType', 'fileFormat']
            for field in format_fields:
                if field in content:
                    info['has_format_info'] = True
                    fmt_val = content[field]
                    if isinstance(fmt_val, str):
                        info['formats_mentioned'].add(fmt_val.lower())
            
            # Check for vocabularies/ontologies
            vocab_fields = ['subject', 'keywords', 'theme', 'category', 'about']
            for field in vocab_fields:
                if field in content:
                    field_val = content[field]
                    if isinstance(field_val, list) and len(field_val) > 0:
                        info['has_vocabularies'] = True
                    elif isinstance(field_val, str) and field_val.strip():
                        info['has_vocabularies'] = True
        
        # Convert sets to lists for JSON serialization
        info['standards_used'] = list(info['standards_used'])
        info['license_types'] = list(info['license_types'])
        info['formats_mentioned'] = list(info['formats_mentioned'])
        
        return info
    
    # ===== ENHANCED CHECKER METHODS (with metadata support) =====
    
    def _check_has_metadata_file(self, repo_info, contents, clone_path, metadata_files, parsed_metadata) -> Tuple[bool, str]:
        """Check for metadata files"""
        if metadata_files:
            return True, f"Found {len(metadata_files)} metadata files"
        return False, "No metadata files"
    
    def _check_has_structured_metadata(self, repo_info, contents, clone_path, metadata_files, parsed_metadata) -> Tuple[bool, str]:
        """Check for structured metadata (JSON/XML/YAML)"""
        structured_files = [f for f in metadata_files if f['type'] in ['json', 'yaml', 'xml']]
        if structured_files:
            return True, f"Found {len(structured_files)} structured metadata files"
        return False, "No structured metadata files"
    
    def _check_has_doi_or_pid(self, repo_info, contents, clone_path, metadata_files, parsed_metadata) -> Tuple[bool, str]:
        """Check for DOI or persistent identifier"""
        if parsed_metadata.get('combined_info', {}).get('has_doi', False):
            return True, "DOI found in metadata"
        
        # Also check README for DOI
        for f in contents:
            if 'readme' in f['name'].lower():
                try:
                    content = self._get_file_content(f)
                    if re.search(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', content, re.IGNORECASE):
                        return True, "DOI found in README"
                except:
                    pass
        
        return False, "No DOI or persistent identifier"
    
    def _check_has_license_in_metadata(self, repo_info, contents, clone_path, metadata_files, parsed_metadata) -> Tuple[bool, str]:
        """Check for license information in metadata"""
        if parsed_metadata.get('combined_info', {}).get('has_license', False):
            licenses = parsed_metadata['combined_info'].get('license_types', [])
            if licenses:
                return True, f"License in metadata: {', '.join(licenses[:3])}"
            return True, "License mentioned in metadata"
        return False, "No license information in metadata"
    
    def _check_has_access_rights(self, repo_info, contents, clone_path, metadata_files, parsed_metadata) -> Tuple[bool, str]:
        """Check for access rights in metadata"""
        if parsed_metadata.get('combined_info', {}).get('has_access_rights', False):
            return True, "Access rights specified in metadata"
        return False, "No access rights information in metadata"
    
    def _check_has_standard_metadata(self, repo_info, contents, clone_path, metadata_files, parsed_metadata) -> Tuple[bool, str]:
        """Check for standard metadata schema usage"""
        standards = parsed_metadata.get('combined_info', {}).get('standards_used', [])
        if standards:
            return True, f"Uses metadata standards: {', '.join(standards)}"
        return False, "No standard metadata schema used"
    
    def _check_has_vocabularies(self, repo_info, contents, clone_path, metadata_files, parsed_metadata) -> Tuple[bool, str]:
        """Check for controlled vocabularies"""
        if parsed_metadata.get('combined_info', {}).get('has_vocabularies', False):
            return True, "Uses controlled vocabularies/ontologies"
        return False, "No controlled vocabularies used"
    
    def _check_has_format_info(self, repo_info, contents, clone_path, metadata_files, parsed_metadata) -> Tuple[bool, str]:
        """Check for format information in metadata"""
        if parsed_metadata.get('combined_info', {}).get('has_format_info', False):
            formats = parsed_metadata['combined_info'].get('formats_mentioned', [])
            if formats:
                return True, f"Format info in metadata: {', '.join(formats[:3])}"
            return True, "Format information in metadata"
        return False, "No format information in metadata"
    
    def _check_has_provenance_info(self, repo_info, contents, clone_path, metadata_files, parsed_metadata) -> Tuple[bool, str]:
        """Check for provenance information"""
        if parsed_metadata.get('combined_info', {}).get('has_provenance', False):
            return True, "Provenance information in metadata"
        return False, "No provenance information"
    
    def _check_has_contact_info(self, repo_info, contents, clone_path, metadata_files, parsed_metadata) -> Tuple[bool, str]:
        """Check for contact information in metadata"""
        if parsed_metadata.get('combined_info', {}).get('has_contact', False):
            return True, "Contact information in metadata"
        return False, "No contact information in metadata"
    
    # ===== EXISTING CHECKER METHODS (updated to accept metadata parameters) =====
    
    def _check_has_readme(self, repo_info, contents, clone_path, metadata_files=None, parsed_metadata=None) -> Tuple[bool, str]:
        """Check if repository has README"""
        readme_patterns = ['readme.md', 'readme.rst', 'readme.txt', 'readme']
        for f in contents:
            if f['name'].lower() in readme_patterns:
                return True, "README found"
        return False, "No README file"
    
    def _check_has_topics(self, repo_info, contents, clone_path, metadata_files=None, parsed_metadata=None) -> Tuple[bool, str]:
        """Check if repository has topics (from local scan, we can't get actual topics)"""
        # Since we can't get topics without API, check for topic-like files
        topic_files = ['topics.md', 'tags.txt', 'keywords.txt']
        for f in contents:
            if any(tf in f['name'].lower() for tf in topic_files):
                return True, "Topic-related file found"
        return False, "No topic information"
    
    def _check_has_description(self, repo_info, contents, clone_path, metadata_files=None, parsed_metadata=None) -> Tuple[bool, str]:
        """Check if repository has description"""
        desc = repo_info.get('description', '')
        return bool(desc and len(desc.strip()) > 10), f"Description: {desc[:50]}..."
    
    def _check_has_website(self, repo_info, contents, clone_path, metadata_files=None, parsed_metadata=None) -> Tuple[bool, str]:
        """Check for project website"""
        # Check README for URLs
        for f in contents:
            if 'readme' in f['name'].lower():
                try:
                    content = self._get_file_content(f)
                    if 'http://' in content or 'https://' in content:
                        return True, "URLs found in README"
                except:
                    pass
        return False, "No website URL detected"
    
    def _check_has_releases(self, repo_info, contents, clone_path, metadata_files=None, parsed_metadata=None) -> Tuple[bool, str]:
        """Check for GitHub releases"""
        return repo_info.get('has_releases', False), "Has releases" if repo_info.get('has_releases') else "No releases"
    
    def _check_has_wiki(self, repo_info, contents, clone_path, metadata_files=None, parsed_metadata=None) -> Tuple[bool, str]:
        """Check for wiki"""
        return repo_info.get('has_wiki', False), "Has wiki" if repo_info.get('has_wiki') else "No wiki"
    
    def _check_is_public(self, repo_info, contents, clone_path, metadata_files=None, parsed_metadata=None) -> Tuple[bool, str]:
        """Check if repository is public (assume true if we can clone it)"""
        return True, "Public (successfully cloned)"
    
    def _check_has_open_license(self, repo_info, contents, clone_path, metadata_files=None, parsed_metadata=None) -> Tuple[bool, str]:
        """Check for open license"""
        license_files = ['license', 'license.txt', 'license.md', 'copying', 'copying.txt']
        open_license_indicators = ['mit', 'apache', 'gpl', 'bsd', 'cc-by', 'cc0', 
                                  'artistic', 'eclipse', 'mpl', 'lgpl', 'agpl']
        
        # Check metadata first
        if parsed_metadata and parsed_metadata.get('combined_info', {}).get('has_license', False):
            licenses = parsed_metadata['combined_info'].get('license_types', [])
            for license_type in licenses:
                if any(indicator in license_type for indicator in open_license_indicators):
                    return True, f"Open license in metadata: {license_type}"
        
        # Check files
        for f in contents:
            if any(lf in f['name'].lower() for lf in license_files):
                try:
                    content = self._get_file_content(f).lower()
                    if any(oli in content for oli in open_license_indicators):
                        return True, "Open license detected in file"
                    return True, "License file found (type unknown)"
                except:
                    return True, "License file found"
        
        # Check for license in README
        for f in contents:
            if 'readme' in f['name'].lower():
                try:
                    content = self._get_file_content(f).lower()
                    if 'license' in content or 'licence' in content:
                        return True, "License mentioned in README"
                except:
                    pass
        
        return False, "No license detected"
    
    def _check_has_download_instructions(self, repo_info, contents, clone_path, metadata_files=None, parsed_metadata=None) -> Tuple[bool, str]:
        """Check for download/access instructions"""
        download_keywords = ['download', 'install', 'clone', 'usage', 'quick start', 
                           'getting started', 'how to use', 'installation']
        
        for f in contents:
            if 'readme' in f['name'].lower():
                try:
                    content = self._get_file_content(f).lower()
                    if any(keyword in content for keyword in download_keywords):
                        return True, "Download/usage instructions found"
                except:
                    pass
        
        return False, "No download instructions"
    
    def _check_has_data_files(self, repo_info, contents, clone_path, metadata_files=None, parsed_metadata=None) -> Tuple[bool, str]:
        """Check for actual data files"""
        data_extensions = ['.csv', '.json', '.tsv', '.txt', '.h5', '.hdf5', 
                          '.cif', '.pdb', '.fasta', '.fastq', '.parquet', 
                          '.feather', '.arrow', '.xlsx', '.xls', '.sqlite',
                          '.db', '.sql', '.rds', '.rdata', '.mat', '.nc']
        
        data_files = [f for f in contents if f['extension'] in data_extensions]
        return len(data_files) > 0, f"Has {len(data_files)} data files"
    
    def _check_uses_standard_formats(self, repo_info, contents, clone_path, metadata_files=None, parsed_metadata=None) -> Tuple[bool, str]:
        """Check for standard data formats"""
        std_formats = ['.csv', '.json', '.tsv', '.h5', '.hdf5', '.parquet', 
                      '.cif', '.pdb', '.fasta', '.fastq']
        
        std_files = [f for f in contents if f['extension'] in std_formats]
        return len(std_files) >= 2, f"Uses {len(std_files)} standard format files"
    
    def _check_has_schema(self, repo_info, contents, clone_path, metadata_files=None, parsed_metadata=None) -> Tuple[bool, str]:
        """Check for data schema"""
        schema_files = ['schema.json', 'data_dictionary.md', 'README_data.md', 
                       'data_schema.json', 'table_schema.json', 'schema.yaml',
                       'schema.yml']
        
        for f in contents:
            if any(sf in f['name'].lower() for sf in schema_files):
                return True, "Schema/documentation found"
        
        # Check for schema in filenames
        for f in contents:
            if 'schema' in f['name'].lower() or 'dictionary' in f['name'].lower():
                return True, "Schema-related file found"
        
        return False, "No data schema"
    
    def _check_has_usage_scripts(self, repo_info, contents, clone_path, metadata_files=None, parsed_metadata=None) -> Tuple[bool, str]:
        """Check for usage scripts"""
        script_extensions = ['.py', '.r', '.ipynb', '.sh', '.jl', '.m', '.java',
                           '.cpp', '.c', '.js', '.ts']
        
        script_files = [f for f in contents if f['extension'] in script_extensions]
        return len(script_files) >= 2, f"Has {len(script_files)} usage scripts"
    
    def _check_has_license_file(self, repo_info, contents, clone_path, metadata_files=None, parsed_metadata=None) -> Tuple[bool, str]:
        """Check for LICENSE file"""
        license_files = ['license', 'license.txt', 'license.md', 'copying', 'copying.txt']
        
        for f in contents:
            if any(lf in f['name'].lower() for lf in license_files):
                return True, "LICENSE file found"
        return False, "No LICENSE file"
    
    def _check_has_citation_file(self, repo_info, contents, clone_path, metadata_files=None, parsed_metadata=None) -> Tuple[bool, str]:
        """Check for citation file"""
        citation_files = ['citation.cff', 'citation.json', 'cite.bib', 'citation.bib',
                         'CITATION', 'CITATION.txt', 'CITATION.md']
        
        for f in contents:
            if any(cf in f['name'].lower() for cf in citation_files):
                return True, "Citation file found"
        return False, "No citation file"
    
    def _check_has_examples(self, repo_info, contents, clone_path, metadata_files=None, parsed_metadata=None) -> Tuple[bool, str]:
        """Check for examples/notebooks"""
        example_dirs = ['examples', 'notebooks', 'demo', 'tutorials', 'scripts',
                       'demos', 'tutorial', 'example']
        
        example_patterns = ['example', 'demo', 'tutorial', 'notebook']
        
        # Check for example directories
        dirs_found = []
        for f in contents:
            path_parts = f['path'].lower().split('/')
            if len(path_parts) > 1:  # Has directory component
                dir_name = path_parts[0]
                if dir_name in example_dirs:
                    return True, f"Example directory found: {dir_name}"
        
        # Check for example files
        example_files = [f for f in contents if 
                        any(ep in f['name'].lower() for ep in example_patterns)]
        
        if example_files:
            return True, f"Has {len(example_files)} example files"
        
        return False, "No examples/notebooks"
    
    def _check_has_detailed_readme(self, repo_info, contents, clone_path, metadata_files=None, parsed_metadata=None) -> Tuple[bool, str]:
        """Check for detailed README"""
        for f in contents:
            if 'readme' in f['name'].lower():
                try:
                    content = self._get_file_content(f)
                    lines = content.split('\n')
                    
                    # Check for common README sections
                    sections = ['##', 'installation', 'usage', 'example', 'license',
                               'contributing', 'authors', 'acknowledgements']
                    
                    section_count = 0
                    for line in lines:
                        line_lower = line.lower()
                        if any(sec in line_lower for sec in sections if sec != '##'):
                            section_count += 1
                        elif line.startswith('##'):
                            section_count += 1
                    
                    return len(lines) > 30 and section_count >= 3, \
                           f"README has {len(lines)} lines, {section_count} sections"
                except:
                    pass
        
        return False, "README not detailed enough"
    
    def _check_has_issue_templates(self, repo_info, contents, clone_path, metadata_files=None, parsed_metadata=None) -> Tuple[bool, str]:
        """Check for issue templates"""
        issue_patterns = ['.github/issue_template', '.github/ISSUE_TEMPLATE',
                         'ISSUE_TEMPLATE.md', 'issue_template.md']
        
        for f in contents:
            if any(ip in f['path'].lower() for ip in issue_patterns):
                return True, "Issue templates found"
        
        return False, "No issue templates"
    
    # ===== EXISTING HELPER METHODS =====
    
    def _scan_local_contents(self, directory_path: Path) -> List[Dict]:
        """Scan local directory for files and return structure similar to GitHub API"""
        contents = []
        
        try:
            for root, dirs, files in os.walk(directory_path):
                # Skip .git directory
                dirs[:] = [d for d in dirs if d != '.git']
                
                for file in files:
                    file_path = Path(root) / file
                    try:
                        rel_path = file_path.relative_to(directory_path)
                        
                        # Get file stats
                        stat = file_path.stat()
                        
                        contents.append({
                            'name': file,
                            'path': str(rel_path),
                            'type': 'file',
                            'size': stat.st_size,
                            'local_path': str(file_path),
                            'extension': file_path.suffix.lower()
                        })
                    except (ValueError, OSError):
                        continue
                        
        except Exception as e:
            print(f"   Warning: Error scanning directory: {e}")
        
        return contents
    
    def _get_file_content(self, file_item: Dict) -> str:
        """Get content from a file item"""
        local_path = file_item.get('local_path')
        if local_path and os.path.exists(local_path):
            try:
                with open(local_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except:
                return ""
        return ""
    
    def _get_local_repo_info(self, clone_path: Path, repo_url: str) -> Dict:
        """Get basic repository info from local clone"""
        info = {
            'url': repo_url,
            'name': clone_path.name,
            'private': False,  # Assume public if we can clone it
            'has_wiki': False,
            'has_releases': False,
            'description': '',
            'homepage': '',
            'topics': []
        }
        
        # Try to read description from README
        readme_files = ['README.md', 'README.rst', 'README.txt', 'README']
        for readme in readme_files:
            readme_path = clone_path / readme
            if readme_path.exists():
                try:
                    with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(500)  # First 500 chars
                        # Extract first line as potential description
                        lines = content.strip().split('\n')
                        for line in lines:
                            if line.strip() and not line.strip().startswith('#'):
                                info['description'] = line.strip()[:200]
                                break
                except:
                    pass
                break
        
        # Check for wiki directory
        wiki_path = clone_path / 'wiki'
        if wiki_path.exists() and wiki_path.is_dir():
            info['has_wiki'] = True
        
        # Check for releases (Git tags)
        try:
            result = subprocess.run(
                ['git', '-C', str(clone_path), 'tag'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.stdout.strip():
                info['has_releases'] = True
        except:
            pass
        
        return info
    
    def _parse_repo_url(self, url: str) -> Tuple[str, str]:
        """Parse GitHub URL to owner and repo"""
        url = url.rstrip('/')
        
        # Handle various URL formats
        if 'github.com/' in url:
            parts = url.split('github.com/')[1].split('/')
            if len(parts) >= 2:
                return parts[0], parts[1].replace('.git', '')
        elif 'git@github.com:' in url:
            parts = url.split('git@github.com:')[1].split('/')
            if len(parts) >= 2:
                return parts[0], parts[1].replace('.git', '')
        else:
            # Assume format is "owner/repo"
            parts = url.split('/')
            if len(parts) >= 2:
                return parts[-2], parts[-1].replace('.git', '')
        
        raise ValueError(f"Invalid GitHub URL: {url}")
    
    def _extract_improvements(self, score_obj: FAIRScoreWithImprovements, repo_url: str) -> List[Dict]:
        """Extract improvement suggestions from score object"""
        improvements = []
        
        # Check each principle for missing elements
        for principle in ['findable', 'accessible', 'interoperable', 'reusable']:
            missing_attr = f"missing_{principle}"
            missing = getattr(score_obj, missing_attr, [])
            
            for missing_item in missing:
                improvements.append({
                    'repository': repo_url,
                    'principle': principle.capitalize(),
                    'missing': missing_item,
                    'action': self._get_improvement_action(principle, missing_item),
                    'priority': self._get_priority(principle, missing_item),
                    'potential_points': self._get_points_for_criterion(principle, missing_item)
                })
        
        return improvements
    
    def _get_improvement_action(self, principle: str, missing_item: str) -> str:
        """Get specific improvement action for missing item"""
        actions = {
            'README file': "Create README.md with project overview, installation, usage examples",
            'Metadata file': "Add metadata.json following schema.org or DataCite schema",
            'Structured metadata': "Add structured metadata file (JSON/YAML/XML) in bioschema/ or metadata/ directory",
            'DOI or Persistent ID': "Register dataset with Zenodo/Figshare to get DOI, include in metadata",
            'Repository topics': "Add relevant topics to repository settings on GitHub",
            'Repository description': "Add clear description (min 50 chars) in repository settings",
            'Project website': "Add homepage URL in repository settings or README",
            'GitHub releases': "Create a release with version tag for stable versions",
            'Documentation wiki': "Enable wiki and add basic documentation on GitHub",
            'License in metadata': "Include license information in metadata files",
            'Public repository': "Make repository public in settings (if currently private)",
            'Open license': "Add LICENSE file with MIT, Apache, CC-BY, or other open license",
            'Access rights in metadata': "Specify access rights (open, restricted, closed) in metadata",
            'Download instructions': "Add 'Getting Started' section to README with clear steps",
            'Data in repository': "Include actual data files, not just links to external sources",
            'Standard formats': "Convert data to CSV, JSON, HDF5, or other standard formats",
            'Data schema': "Add schema.json or data_dictionary.md describing data structure",
            'Standard metadata schema': "Use standard metadata schema like Schema.org, DataCite, or Bioschemas",
            'Controlled vocabularies': "Use controlled vocabularies/ontologies for metadata fields",
            'Usage scripts': "Add Python/R scripts demonstrating how to load and use the data",
            'Format information': "Include format information (MIME types, extensions) in metadata",
            'LICENSE file': "Add explicit LICENSE file with clear terms of use",
            'Citation file': "Add CITATION.cff file for proper academic citation",
            'Provenance information': "Add provenance information (creation date, version, history) in metadata",
            'Examples/notebooks': "Create examples/ directory with Jupyter notebooks or scripts",
            'Contact information': "Include contact information (email, ORCID) in metadata",
            'Detailed README': "Expand README with sections: Installation, Usage, Examples, API, Contributing",
            'Issue templates': "Add .github/ISSUE_TEMPLATE/ for structured bug reports and feature requests",
        }
        
        return actions.get(missing_item, f"Improve {missing_item} for better {principle}")
    
    def _get_priority(self, principle: str, missing_item: str) -> str:
        """Determine priority level"""
        high_priority = ['README file', 'Open license', 'LICENSE file', 'Data in repository', 
                        'Structured metadata', 'DOI or Persistent ID']
        medium_priority = ['Metadata file', 'Standard formats', 'Download instructions', 
                          'Data schema', 'Standard metadata schema', 'Contact information']
        
        if missing_item in high_priority:
            return "High"
        elif missing_item in medium_priority:
            return "Medium"
        else:
            return "Low"
    
    def _get_points_for_criterion(self, principle: str, missing_item: str) -> int:
        """Get potential points for fixing this criterion"""
        for criterion, description, weight, _ in self.criteria.get(principle, []):
            if description == missing_item:
                return weight
        return 0
    
    def _generate_detailed_report(self, scores: List[FAIRScoreWithImprovements], improvements: List[Dict]) -> Dict:
        """Generate comprehensive comparison report"""
        
        # Convert scores to dictionaries
        scores_dict = []
        for score in scores:
            if score:  # Only include valid scores
                score_dict = {
                    'repository': score.repository,
                    'stars': score.stars,
                    'findable': score.findable,
                    'accessible': score.accessible,
                    'interoperable': score.interoperable,
                    'reusable': score.reusable,
                    'total': score.total,
                    'missing_findable': score.missing_findable,
                    'missing_accessible': score.missing_accessible,
                    'missing_interoperable': score.missing_interoperable,
                    'missing_reusable': score.missing_reusable,
                    'detailed_scores': score.detailed_scores,
                    'metadata_summary': score.extracted_metadata.get('combined_info', {}),
                    'metadata_files_count': len(score.extracted_metadata.get('files', [])),
                }
                scores_dict.append(score_dict)
        
        # Create improvement summary by repository
        improvement_summary = defaultdict(list)
        for imp in improvements:
            improvement_summary[imp['repository']].append(imp)
        
        # Create DataFrame for statistics
        df_data = []
        for score in scores:
            if score:
                df_data.append({
                    'repository': score.repository,
                    'total': score.total,
                    'findable': score.findable,
                    'accessible': score.accessible,
                    'interoperable': score.interoperable,
                    'reusable': score.reusable,
                    'metadata_files': len(score.extracted_metadata.get('files', []))
                })
        
        if df_data:
            df = pd.DataFrame(df_data)
            
            # Create ranking
            if 'total' in df.columns and not df.empty:
                ranking = df.sort_values('total', ascending=False).to_dict('records')
            else:
                ranking = []
            
            stats = {
                'average_total': float(df['total'].mean()),
                'median_total': float(df['total'].median()),
                'std_total': float(df['total'].std()),
                'highest_total': float(df['total'].max()),
                'lowest_total': float(df['total'].min()),
                'average_metadata_files': float(df['metadata_files'].mean()),
                'average_by_principle': {
                    'findable': float(df['findable'].mean()),
                    'accessible': float(df['accessible'].mean()),
                    'interoperable': float(df['interoperable'].mean()),
                    'reusable': float(df['reusable'].mean()),
                }
            }
        else:
            ranking = []
            stats = {
                'average_total': 0,
                'median_total': 0,
                'std_total': 0,
                'highest_total': 0,
                'lowest_total': 0,
                'average_metadata_files': 0,
                'average_by_principle': {
                    'findable': 0,
                    'accessible': 0,
                    'interoperable': 0,
                    'reusable': 0,
                }
            }
            print("Warning: No valid scores to generate statistics")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'repositories_compared': len(scores_dict),
            'scores': scores_dict,
            'improvements': improvements,
            'improvement_summary': dict(improvement_summary),
            'ranking': ranking,
            'statistics': stats
        }
        
        return report

def save_report(report: Dict, output_dir: str = "fair_reports"):
    """Save the report to JSON and generate summary files"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full report
    report_file = os.path.join(output_dir, f"fair_comparison_{timestamp}.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Save summary CSV
    if report['scores']:
        df_scores = pd.DataFrame(report['scores'])
        csv_file = os.path.join(output_dir, f"fair_scores_{timestamp}.csv")
        df_scores.to_csv(csv_file, index=False)
    
    # Save improvements CSV
    if report['improvements']:
        df_improvements = pd.DataFrame(report['improvements'])
        improvements_file = os.path.join(output_dir, f"fair_improvements_{timestamp}.csv")
        df_improvements.to_csv(improvements_file, index=False)
    
    # Save metadata summary
    metadata_summary = []
    for score in report['scores']:
        if 'metadata_summary' in score:
            metadata_summary.append({
                'repository': score['repository'],
                'metadata_files': score.get('metadata_files_count', 0),
                'has_doi': score['metadata_summary'].get('has_doi', False),
                'has_license': score['metadata_summary'].get('has_license', False),
                'has_contact': score['metadata_summary'].get('has_contact', False),
                'standards_used': ', '.join(score['metadata_summary'].get('standards_used', [])),
            })
    
    if metadata_summary:
        df_metadata = pd.DataFrame(metadata_summary)
        metadata_file = os.path.join(output_dir, f"metadata_summary_{timestamp}.csv")
        df_metadata.to_csv(metadata_file, index=False)
    
    return report_file

def print_summary(report: Dict):
    """Print a summary of the comparison results"""
    print("\n" + "="*80)
    print("📊 FAIRNESS COMPARISON SUMMARY WITH METADATA ANALYSIS")
    print("="*80)
    
    if not report['ranking']:
        print("No repositories were successfully evaluated.")
        return
    
    for i, repo in enumerate(report['ranking'], 1):
        repo_name = repo['repository'].split('/')[-1]
        print(f"\n{i}. {repo_name}")
        print(f"   URL: {repo['repository']}")
        print(f"   Total Score: {repo['total']:.1f}/100")
        print(f"   Breakdown: F{repo['findable']:.0f} A{repo['accessible']:.0f} "
              f"I{repo['interoperable']:.0f} R{repo['reusable']:.0f}")
        
        # Show metadata info if available
        if 'metadata_files_count' in repo and repo['metadata_files_count'] > 0:
            print(f"   Metadata: {repo['metadata_files_count']} files found")
            if 'metadata_summary' in repo and repo['metadata_summary']:
                meta = repo['metadata_summary']
                if meta.get('standards_used'):
                    print(f"   Standards: {', '.join(meta['standards_used'])}")
                if meta.get('has_doi'):
                    print(f"   ✓ Includes DOI")
                if meta.get('has_license'):
                    print(f"   ✓ License in metadata")
        
        # Show top improvements
        improvements = report['improvement_summary'].get(repo['repository'], [])
        if improvements:
            print(f"   Top improvements needed:")
            for imp in improvements[:3]:  # Show only top 3
                print(f"   • [{imp['priority']}] {imp['missing']} (+{imp['potential_points']} pts)")
    
    # Print statistics
    stats = report['statistics']
    print("\n" + "="*80)
    print("📈 OVERALL STATISTICS")
    print("="*80)
    print(f"Average Total Score: {stats['average_total']:.1f}/100")
    print(f"Median Total Score:  {stats['median_total']:.1f}/100")
    print(f"Score Range:         {stats['lowest_total']:.1f} - {stats['highest_total']:.1f}")
    print(f"Avg Metadata Files:  {stats.get('average_metadata_files', 0):.1f}")
    
    print(f"\nAverage by Principle:")
    for principle, score in stats['average_by_principle'].items():
        print(f"  {principle.capitalize()}: {score:.1f}/100")

def main():
    """Main function to run the comparison"""
    import sys
    
    print("🚀 GitHub Repository FAIRness Evaluator with Metadata Analysis")
    print("="*80)
    
    # List of repositories to compare (add your bioschema repository here)
    REPOS_TO_COMPARE = [
        "https://github.com/biofold/ppi-benchmark-fair",
        "https://github.com/vibbits/Elixir-3DBioInfo-Benchmark-Protein-Interfaces",
        "https://github.com/google-deepmind/alphafold"
        # Add more repositories with metadata files
        # "https://github.com/vibbits/Elixir-3DBioInfo-Benchmark-Protein-Interfaces",
        # "https://github.com/example/repo-with-bioschema",
    ]
    
    # You can also read repositories from a file
    if len(sys.argv) > 1:
        repo_file = sys.argv[1]
        try:
            with open(repo_file, 'r') as f:
                REPOS_TO_COMPARE = [line.strip() for line in f if line.strip()]
            print(f"Read {len(REPOS_TO_COMPARE)} repositories from {repo_file}")
        except FileNotFoundError:
            print(f"Warning: File {repo_file} not found, using default list")
    
    if not REPOS_TO_COMPARE:
        print("Error: No repositories to compare")
        sys.exit(1)
    
    # Check for optional dependencies
    try:
        import yaml
        YAML_AVAILABLE = True
    except ImportError:
        YAML_AVAILABLE = False
        print("Note: PyYAML not installed. YAML metadata files will be skipped.")
        print("      Install with: pip install pyyaml")
    
    if not RDFLIB_AVAILABLE:
        print("Note: rdflib not installed. RDF metadata parsing will be limited.")
        print("      Install with: pip install rdflib")
    
    # Create comparator
    comparator = GitHubFAIRComparator(
        cache_dir="./repo_cache",  # Optional cache directory
        max_repo_size_mb=1000      # Max 1GB per repo
    )
    
    # Run comparison
    report = comparator.compare_repos(REPOS_TO_COMPARE)
    
    # Save reports
    report_file = save_report(report)
    
    # Print summary
    print_summary(report)
    
    print(f"\n✅ Reports saved to: {report_file}")
    print(f"   Total repositories evaluated: {report['repositories_compared']}")

if __name__ == "__main__":
    main()        


