
# DHF_Extractor.py

import os
import fitz
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import logging
import json
from datetime import datetime
import gc
import sys
from dataclasses import dataclass
from .LLM_Engine import ISO11135CategoryMapper
import logging_setup
logger = logging_setup.get_logger(__name__)

# Create a single mapper instance to use throughout the class
# Use DHF-specific categories from inputs if available; falls back to built-ins otherwise
# Try multiple possible locations for the config file
_config_paths = [
    Path(__file__).parent.parent.parent / "inputs" / "dhf_categories.json",  # Project root
    Path(__file__).parent.parent / "inputs" / "dhf_categories.json",  # Backend/inputs
    Path("inputs") / "dhf_categories.json",  # Current working directory
]

_config_path = None
for path in _config_paths:
    if path.exists():
        _config_path = path
        logger.info(f"Found category config at: {_config_path}")
        break

mapper = ISO11135CategoryMapper(config_path=str(_config_path) if _config_path else None)

# Your comprehensive parameter categories (same as original)
COMPREHENSIVE_PARAMETER_CATEGORIES = {
    'concentration': ['EO concentration', 'ethylene oxide', 'ppm', 'mg/L', 'concentration', 'gas concentration'],
    'humidity': ['relative humidity', 'RH', '%RH', 'moisture', 'water vapor', 'humidity level'],
    'temperature': ['temperature', 'Â°C', 'Â°F', 'celsius', 'fahrenheit', 'thermal', 'heat'],
    'time': ['exposure time', 'dwell time', 'contact time', 'cycle time', 'duration', 'treatment time'],
    'pressure': ['pressure', 'kPa', 'mmHg', 'torr', 'vacuum', 'atmospheric pressure', 'gauge pressure'],
    'biological': ['D-value', 'Z-value', 'SLR', 'spore log reduction', 'biological indicator', 'BI', 'microbial'],
    'validation': ['validation', 'qualification', 'IQ', 'OQ', 'PQ', 'performance qualification'],
    'monitoring': ['monitoring', 'control', 'parametric release', 'process control', 'measurement'],
    'materials': ['material', 'polymer', 'resin', 'coating', 'substrate', 'biocompatibility', 'USP Class VI', 'cytotoxicity'],
    'dimensions': ['diameter', 'length', 'width', 'height', 'thickness', 'tolerance', 'specification', 'dimension'],
    'mechanical_properties': ['tensile strength', 'burst pressure', 'flexibility', 'durability', 'fatigue', 'mechanical'],
    'electrical': ['voltage', 'current', 'resistance', 'impedance', 'insulation', 'electrical safety', 'EMC'],
    'software': ['software version', 'firmware', 'algorithm', 'verification', 'software validation', 'cybersecurity'],
    'packaging': ['sterile barrier', 'shelf life', 'packaging validation', 'seal integrity', 'package'],
    'labeling': ['label', 'instructions for use', 'IFU', 'warnings', 'contraindications', 'indications'],
    'risk_management': ['risk analysis', 'FMEA', 'hazard', 'risk control', 'ISO 14971', 'risk assessment'],
    'regulatory': ['FDA', '510k', 'CE mark', 'ISO 13485', 'compliance', 'regulatory submission', 'predicate device'],
    'clinical': ['clinical data', 'clinical evaluation', 'clinical study', 'patient safety', 'efficacy'],
    'manufacturing': ['manufacturing process', 'production', 'assembly', 'quality control', 'batch record'],
    'testing': ['performance testing', 'bench testing', 'simulation', 'accelerated aging', 'test protocol'],
    'traceability': ['lot number', 'serial number', 'batch', 'traceability', 'device history record', 'DHR'],
    'change_control': ['change control', 'design change', 'ECO', 'engineering change', 'revision'],
    'suppliers': ['supplier', 'vendor', 'component', 'raw material', 'supplier qualification'],
    'design_controls': ['design input', 'design output', 'design review', 'design verification', 'design validation'],
    'quality_system': ['quality manual', 'quality policy', 'quality objectives', 'management review', 'internal audit']
}

# Unit patterns (same as original)
COMPREHENSIVE_UNIT_PATTERNS = {
    'concentration': r'\d+\.?\d*\s*(ppm|mg/L|%|ppb|Âµg/mL|ng/mL)',
    'humidity': r'\d+\.?\d*\s*(%|%RH|relative humidity)',
    'temperature': r'\d+\.?\d*\s*(Â°C|Â°F|celsius|fahrenheit|K|kelvin)',
    'time': r'\d+\.?\d*\s*(min|hour|sec|day|week|month|year|h|hr|s)',
    'pressure': r'\d+\.?\d*\s*(kPa|mmHg|torr|psi|bar|Pa|atm)',
    'biological': r'\d+\.?\d*\s*(log|D-value|Z-value|CFU|colony)',
    'dimensions': r'\d+\.?\d*\s*(mm|cm|m|inch|mil|Âµm|nm|ft)',
    'mechanical': r'\d+\.?\d*\s*(MPa|psi|N|kg|lb|kN|GPa)',
    'electrical': r'\d+\.?\d*\s*(V|A|Î©|ohm|mA|ÂµA|kV|Hz|MHz|GHz)',
}

class SinglePDFExtractor:
    """Extract DHF data from a single specified PDF file"""
    
    def __init__(self, pdf_filename: str, output_file: str = "Single_PDF_Extraction.txt"):
        self.pdf_filename = pdf_filename
        self.output_file = Path(output_file)
        self.logger = logger
        
    def detect_document_type(self, text: str) -> List[str]:
        """Detect document type from your original code"""
        doc_types = {
            'design_control': ['design control', 'design input', 'design output', 'design review'],
            'risk_management': ['risk management file', 'risk analysis', 'FMEA', 'hazard analysis'],
            'validation': ['design validation', 'design verification', 'V&V', 'verification protocol'],
            'clinical': ['clinical evaluation', 'clinical data', 'clinical study', 'clinical investigation'],
            'manufacturing': ['device master record', 'manufacturing', 'production', 'batch record'],
            'testing': ['test protocol', 'test report', 'performance testing', 'bench testing'],
            'regulatory': ['510k', 'regulatory submission', 'FDA', 'CE mark', 'compliance'],
            'labeling': ['labeling', 'IFU', 'instructions for use', 'package insert'],
            'sterilization': ['sterilization', 'ethylene oxide', 'EO', 'sterile', 'bioburden'],
            'packaging': ['packaging', 'shelf life', 'stability', 'package validation'],
            'software': ['software', 'firmware', 'algorithm', 'cybersecurity', 'software validation']
        }
        
        detected = []
        text_lower = text.lower()
        
        for doc_type, keywords in doc_types.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            if score >= 1:
                detected.append((doc_type, score))
        
        return [doc_type for doc_type, score in sorted(detected, key=lambda x: x[1], reverse=True)]
    
    def extract_pdf_content(self) -> Optional[Dict]:
        """Extract content from the specified PDF"""
        pdf_path = Path(self.pdf_filename)
        
        # Check if file exists
        if not pdf_path.exists():
            self.logger.error(f"PDF file not found: {self.pdf_filename}")
            return None
        
        try:
            with open(pdf_path, "rb") as f:
                doc = fitz.open(stream=f.read(), filetype="pdf")
            
            full_text = ""
            pages_text = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                pages_text.append({
                    'page_number': page_num + 1,
                    'text': page_text,
                    'char_count': len(page_text)
                })
                full_text += f"\n--- PAGE {page_num + 1} ---\n{page_text}"
            
            # Enhanced metadata
            file_stat = os.stat(pdf_path)
            content = {
                'filename': pdf_path.name,
                'filepath': str(pdf_path),
                'full_text': full_text,
                'pages': pages_text,
                'document_types': self.detect_document_type(full_text),
                'metadata': {
                    'total_pages': len(doc),
                    'total_characters': len(full_text),
                    'title': doc.metadata.get('title', ''),
                    'author': doc.metadata.get('author', ''),
                    'subject': doc.metadata.get('subject', ''),
                    'creator': doc.metadata.get('creator', ''),
                    'producer': doc.metadata.get('producer', ''),
                    'creation_date': doc.metadata.get('creationDate', ''),
                    'modification_date': doc.metadata.get('modDate', ''),
                    'file_size_bytes': file_stat.st_size,
                    'file_size_mb': round(file_stat.st_size / (1024*1024), 2),
                    'extraction_time': datetime.now().isoformat()
                }
            }
            
            doc.close()
            gc.collect()
            return content
            
        except Exception as e:
            self.logger.error(f"Error extracting {pdf_path}: {e}")
            return None
    
    def calculate_relevance_score(self, context: str, keyword: str, category: str) -> float:
        """Calculate relevance score (from your original code)"""
        score = 1.0
        context_lower = context.lower()
        
        # Check for numerical values
        has_numbers = bool(re.search(r'\d+\.?\d*', context))
        if has_numbers:
            score += 2.0
        
        # Specification language
        spec_words = ['shall', 'must', 'required', 'minimum', 'maximum', 'specified', 'maintain']
        score += sum(1.5 for word in spec_words if word in context_lower)
        
        # Range detection
        range_patterns = [r'\d+\s*[-â€"to]\s*\d+', r'between\s+\d+\s+and\s+\d+', r'\d+\s*Â±\s*\d+']
        for pattern in range_patterns:
            if re.search(pattern, context_lower):
                score += 2.0
                break
        
        # Test/measurement context
        test_words = ['test', 'measure', 'verify', 'validate', 'confirm']
        score += sum(0.5 for word in test_words if word in context_lower)
        
        return score
    
    def extract_parameters_with_context(self, text: str, context_window: int = 400, extra_percent: float = 0.3) -> List[Dict]:
        """Extract parameters with context.

        Adds an `extra_percent` multiplier to expand the context window (default +30%).
        """
        results = []

        # Calculate effective context window (increase by extra_percent)
        try:
            effective_window = int(context_window * (1.0 + float(extra_percent)))
        except Exception:
            effective_window = context_window

        self.logger.debug("Using context window: %d (base=%d, extra=%s)", effective_window, context_window, extra_percent)
        
        for category, keywords in COMPREHENSIVE_PARAMETER_CATEGORIES.items():
            for keyword in keywords:
                pattern = rf'\b{re.escape(keyword)}\b'
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                
                for match in matches:
                    start = max(0, match.start() - effective_window)
                    end = min(len(text), match.end() + effective_window)
                    context = text[start:end].strip()
                    context = ' '.join(context.split())

                    relevance_score = self.calculate_relevance_score(context, keyword, category)
                    
                    # Extract numerical data
                    numerical_data = []
                    for num_cat, pattern_num in COMPREHENSIVE_UNIT_PATTERNS.items():
                        matches_num = re.finditer(pattern_num, context, re.IGNORECASE)
                        for num_match in matches_num:
                            value_match = re.search(r'\d+\.?\d*', num_match.group())
                            if value_match:
                                numerical_data.append({
                                    'category': num_cat,
                                    'value': float(value_match.group()),
                                    'unit': num_match.group().replace(value_match.group(), '').strip(),
                                    'full_text': num_match.group()
                                })
                    
                    results.append({
                        'category': category,
                        'keyword': keyword,
                        'context': context,
                        'relevance_score': relevance_score,
                        'numerical_data': numerical_data,
                        'has_numerical': bool(numerical_data),
                        'position': match.start()
                    })
        
        self.logger.info("Extracted %d raw parameter occurrences before deduplication", len(results))

        # Remove duplicates
        deduped = self.deduplicate_results(results)
        self.logger.info("%d parameter occurrences after deduplication", len(deduped))
        return deduped

    def map_parameters_to_iso_categories(self, parameters: List[Dict]) -> Dict[str, List[Dict]]:
        """Map extracted DHF parameters to standardized ISO11135 categories"""
        mapped_sections = {cat: [] for cat in mapper.target_categories}
        for param in parameters:
            text_block = param['context']
            mapped_cat = mapper.categorize_content(text_block, param['category'])
            mapped_sections[mapped_cat].append(param)
        return mapped_sections
    
    def deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicates based on context similarity"""
        unique_results = []
        seen_contexts = set()
        
        for result in sorted(results, key=lambda x: x['relevance_score'], reverse=True):
            context_key = result['context'][:150]
            if context_key not in seen_contexts:
                seen_contexts.add(context_key)
                unique_results.append(result)
        
        return unique_results
    
    def process_pdf(self) -> None:
        """Process the specified PDF and create output"""
        self.logger.info(f"Processing PDF: {self.pdf_filename}")
        
        # Extract content
        content = self.extract_pdf_content()
        if not content:
            self.logger.error("Failed to extract PDF content")
            return
        
        # Extract parameters
        parameters = self.extract_parameters_with_context(content['full_text'])
        high_relevance = [p for p in parameters if p['relevance_score'] >= 2.0]
        
        # Map parameters to ISO categories
        mapped_params = self.map_parameters_to_iso_categories(high_relevance)
        grouped_params = mapped_params
        
        # Create output
        with open(self.output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("#" * 100 + "\n")
            f.write("#" + " " * 98 + "#\n")
            f.write("#" + " DHF EXTRACTION REPORT".center(98) + "#\n")
            f.write("#" + " Design History File Parameter Analysis".center(98) + "#\n")
            f.write("#" + " " * 98 + "#\n")
            f.write("#" * 100 + "\n\n")
            
            f.write("┌─ DOCUMENT INFORMATION " + "─" * 76 + "\n")
            f.write(f"│  PDF File: {content['filename']}\n")
            f.write(f"│  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"│  Document Types: {', '.join(content['document_types'])}\n")
            f.write(f"│  Pages: {content['metadata']['total_pages']}\n")
            f.write(f"│  Size: {content['metadata']['file_size_mb']} MB\n")
            f.write(f"│  Parameters Found: {len(high_relevance)}\n")
            f.write("└" + "─" * 99 + "\n\n")
            
            # Write parameters by category
            for category, params in grouped_params.items():
                if params:  # Only show categories that have parameters
                    f.write("\n" + "#" * 100 + "\n")
                    f.write(f"# CATEGORY: {category.upper().replace('_', ' ')} ({len(params)} items)\n")
                    f.write("#" * 100 + "\n\n")
                    
                    for j, param in enumerate(params, 1):
                        if param['relevance_score'] >= 4.0:
                            score_badge = "[HIGH]"
                        elif param['relevance_score'] >= 3.0:
                            score_badge = "[MED ]"
                        else:
                            score_badge = "[LOW ]"
                        
                        num_badge = "[NUMERICAL]" if param['has_numerical'] else ""
                        
                        f.write(f"┌─ PARAMETER {j:02d} {score_badge} {num_badge} " + "─" * (100 - 25 - len(score_badge) - len(num_badge)) + "\n")
                        f.write(f"│\n")
                        f.write(f"│  Keyword: {param['keyword']}\n")
                        f.write(f"│  Relevance Score: {param['relevance_score']:.2f}/5.0\n")
                        f.write(f"│\n")
                        f.write(f"│  Context:\n")
                        
                        # Wrap context
                        context_words = param['context'].split()
                        current_line = ""
                        for word in context_words:
                            if len(current_line) + len(word) + 1 <= 90:
                                current_line += word + " "
                            else:
                                f.write(f"│    {current_line.strip()}\n")
                                current_line = word + " "
                        if current_line:
                            f.write(f"│    {current_line.strip()}\n")
                        
                        if param['numerical_data']:
                            f.write(f"│\n")
                            f.write(f"│  Numerical Data:\n")
                            for num_item in param['numerical_data']:
                                f.write(f"│    → {num_item['value']} {num_item['unit']} ({num_item['category']})\n")
                        
                        f.write(f"│\n")
                        f.write("└" + "─" * 99 + "\n\n")
            
            # Full text section
            f.write("\n" + "#" * 100 + "\n")
            f.write("#" + " COMPLETE DOCUMENT TEXT".center(98) + "#\n")
            f.write("#" * 100 + "\n\n")
            f.write(content['full_text'])
        
        self.logger.info(f"Extraction complete! Results saved to: {self.output_file}")
        
        # Print summary
        print(f"\n✅ PDF Extraction Complete!")
        print(f"📄 File: {content['filename']}")
        print(f"📊 Parameters found: {len(high_relevance)}")
        print(f"📂 Categories: {len([cat for cat, params in grouped_params.items() if params])}")
        print(f"💾 Output: {self.output_file}")

def extract_single_pdf(pdf_filename: str, output_file: str = "Single_PDF_Extraction.txt"):
    """
    Extract DHF data from a single PDF file
    
    Args:
        pdf_filename: Name of the PDF file to process
        output_file: Output text file name
    """
    
    print("\n" + "#" * 80)
    print("#" + " SINGLE PDF DHF EXTRACTOR".center(78) + "#")
    print("#" * 80)
    print(f"\nPDF File: {pdf_filename}")
    print(f"Output: {output_file}\n")
    
    extractor = SinglePDFExtractor(pdf_filename, output_file)
    extractor.process_pdf()

# === Entry Point ===
if __name__ == "__main__":
    # Configuration - SPECIFY YOUR PDF FILE HERE
    PDF_FILENAME = "Eto revalidation 26.02.2025.pdf" 
    OUTPUT_FILE = "DHF_Single_Extraction.txt"  # Output file name
    
    # Execute single PDF extraction
    extract_single_pdf(PDF_FILENAME, OUTPUT_FILE)