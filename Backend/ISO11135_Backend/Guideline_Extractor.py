# Regulatory_V2/Pdf_Extractor.py

import fitz 
import re
import pandas as pd
from typing import Dict, List, Tuple
import difflib
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import logging
import logging_setup

logger = logging_setup.get_logger(__name__)

def extract_pdf_content(pdf_path: str) -> dict:
    """Enhanced PDF extraction with better structure preservation"""
    try:
        with open(pdf_path, "rb") as f:
            doc = fitz.open(stream=f.read(), filetype="pdf")
    except FileNotFoundError:
        logger.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    except Exception as e:
        logger.exception(f"Error opening PDF: {pdf_path}")
        raise Exception(f"Error opening PDF: {e}")

    content = {
        'text': '',
        'tables': [],
        'metadata': {
            'total_pages': len(doc),
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', '')
        }
    }
    
    full_text = ""
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()
        full_text += f"\n--- PAGE {page_num + 1} ---\n{page_text}"
        
        # Extract tables
        try:
            tables = page.find_tables()
            for table_num, table in enumerate(tables):
                df = table.to_pandas()
                if not df.empty:
                    content['tables'].append({
                        'page': page_num + 1,
                        'table_num': table_num + 1,
                        'data': df,
                        'bbox': table.bbox
                    })
        except:
            pass  # Skip if table extraction fails
    
    content['text'] = full_text
    doc.close()
    return content

TARGET_SECTIONS = {
    "5": "Sterilizing agent characterization",
    "6": "Process and equipment characterization", 
    "7": "Product definition",
    "8": "Process definition",
    "9": "Validation",
    "10": "Routine monitoring and control",
    "A": "Annex A",
    "B": "Annex B", 
    "C": "Annex C",
    "D": "Annex D"
}

PARAMETER_CATEGORIES = {
    'concentration': ['EO concentration', 'ethylene oxide', 'ppm', 'mg/L', 'concentration', 'gas concentration'],
    'humidity': ['relative humidity', 'RH', '%RH', 'moisture', 'water vapor', 'humidity level'],
    'temperature': ['temperature', '°C', '°F', 'celsius', 'fahrenheit', 'thermal', 'heat'],
    'time': ['exposure time', 'dwell time', 'contact time', 'cycle time', 'duration', 'treatment time'],
    'pressure': ['pressure', 'kPa', 'mmHg', 'torr', 'vacuum', 'atmospheric pressure', 'gauge pressure'],
    'biological': ['D-value', 'Z-value', 'SLR', 'spore log reduction', 'biological indicator', 'BI', 'microbial'],
    'validation': ['validation', 'qualification', 'IQ', 'OQ', 'PQ', 'performance qualification'],
    'monitoring': ['monitoring', 'control', 'parametric release', 'process control', 'measurement']
}

# --- Patterns for extracting value-based sentences from guideline ---
PARAMETER_PATTERNS = {
    "temperature": [r"\b\d+[-–to]*\d*\s*°C\b", r"\b\d+\.?\d*\s*(?:Celsius|°C)\b"],
    "humidity": [r"\b\d+[-–to]*\d*\s*%RH\b", r"\b\d+\s*%\s*(?:RH)?"],
    "pressure": [r"\b\d+[-–to]*\d*\s*(?:kPa|mmHg|torr|bar)\b"],
    "exposure time": [r"\b\d+[-–to]*\d*\s*(?:hours?|minutes?|mins?|sec|seconds?)\b"],
    "EO concentration": [r"\b\d+[-–to]*\d*\s*(?:ppm|mg/L|mg per litre)\b"]  
}

def is_complete_sentence(line: str) -> bool:
    return (
        line[0].isupper() and 
        line[-1] in ".;" and 
        len(line.split()) > 6
    )

def guideline_score(line: str) -> int:
    line = line.strip()
    score = 0

    if re.search(r"(temperature|humidity|pressure|concentration|exposure time|EO)", line, re.IGNORECASE):
        score += 1
    if re.search(r"\d+[-–to]*\d*\s*(°C|%RH|hours?|minutes?|ppm|kPa|mg/L)", line):
        score += 2
    if re.search(r"(during|in the|throughout|within|phase|cycle)", line, re.IGNORECASE):
        score += 1
    if line and line[0].isupper() and line[-1] in ".;":
        score += 1
    
    return score

def merge_line_fragments(lines: List[str]) -> List[str]:
    merged = []
    buffer = ""

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        if buffer:
            buffer += " " + line
        else:
            buffer = line

        # If line ends with sentence punctuation, it's complete
        if line.endswith(('.', ';', ':')):
            merged.append(buffer.strip())
            buffer = ""

    if buffer:
        merged.append(buffer.strip())

    return merged

def extract_guideline_sentences_with_parameters(text: str, parameter_patterns: dict) -> dict:
    guideline_lines = defaultdict(list)
    lines = text.split("\n")

    for line in lines:
        clean_line = ' '.join(line.strip().split())  # normalize extra whitespace

        for param, patterns in parameter_patterns.items():
            for pattern in patterns:
                if re.search(pattern, clean_line, re.IGNORECASE):
                    if (
                        re.search(r"(shall|should|must|required|maintain|not exceed|within|range)", clean_line, re.IGNORECASE)
                        and clean_line not in guideline_lines[param]
                    ):
                        guideline_lines[param].append(clean_line)
    return guideline_lines

def find_section_boundaries(text: str) -> List[Tuple[int, str, str]]:
    """Find section boundaries with flexible matching"""
    boundaries = []
    
    for section_num, section_name in TARGET_SECTIONS.items():
        patterns = [
            # Exact match
            rf'\b{section_num}\s+{re.escape(section_name)}\b',
            # With dots/colons
            rf'\b{section_num}\.?\s*{re.escape(section_name)}\b',
            # Case variations
            rf'\b{section_num}\s+{re.escape(section_name.upper())}\b',
            rf'\b{section_num}\s+{re.escape(section_name.lower())}\b',
            # Annex variations
            rf'\bAnnex\s+{section_num}\b' if section_num in ['A', 'B', 'C', 'D'] else None
        ]
        
        for pattern in patterns:
            if pattern:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                for match in matches:
                    boundaries.append((match.start(), section_num, section_name))
    
    return sorted(boundaries, key=lambda x: x[0])

def extract_sections_robust(text: str) -> Dict[str, str]:
    """Extract sections with improved boundary detection"""
    boundaries = find_section_boundaries(text)
    sections = {}
    
    for i, (start_pos, section_num, section_name) in enumerate(boundaries):
        # Find end position
        if i + 1 < len(boundaries):
            end_pos = boundaries[i + 1][0]
        else:
            end_pos = len(text)
        
        section_text = text[start_pos:end_pos].strip()
        sections[f"{section_num} {section_name}"] = section_text
    
    return sections

def extract_parameters_with_context(text: str, context_window: int = 300) -> List[Dict]:
    """Extract parameters with surrounding context and scoring"""
    results = []
    
    for category, keywords in PARAMETER_CATEGORIES.items():
        for keyword in keywords:
            # Find all occurrences
            pattern = rf'\b{re.escape(keyword)}\b'
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            for match in matches:
                start = max(0, match.start() - context_window)
                end = min(len(text), match.end() + context_window)
                context = text[start:end].strip()
                
                # Clean up context
                context = ' '.join(context.split())  # Normalize whitespace
                
                # Score relevance
                relevance_score = calculate_relevance_score(context, keyword, category)
                
                results.append({
                    'category': category,
                    'keyword': keyword,
                    'context': context,
                    'position': match.start(),
                    'relevance_score': relevance_score,
                    'has_numerical': bool(re.search(r'\d+\.?\d*\s*(?:°C|%|ppm|kPa|minutes?|hours?|mg/L)', context))
                })
    
    # Remove duplicates and sort by relevance
    unique_results = []
    seen_contexts = set()
    
    for result in sorted(results, key=lambda x: x['relevance_score'], reverse=True):
        context_key = result['context'][:100]  # Use first 100 chars as key
        if context_key not in seen_contexts:
            seen_contexts.add(context_key)
            unique_results.append(result)
    
    return unique_results

def calculate_relevance_score(context: str, keyword: str, category: str) -> float:
    """Calculate relevance score for extracted content"""
    score = 0.0
    
    # Base score for keyword match
    score += 1.0
    
    # Bonus for numerical values with units
    unit_patterns = {
        'concentration': r'\d+\.?\d*\s*(?:ppm|mg/L|%)',
        'humidity': r'\d+\.?\d*\s*(?:%|%RH)',
        'temperature': r'\d+\.?\d*\s*(?:°C|°F|celsius|fahrenheit)',
        'time': r'\d+\.?\d*\s*(?:min|hour|sec|minutes?|hours?)',
        'pressure': r'\d+\.?\d*\s*(?:kPa|mmHg|torr|bar)',
        'biological': r'\d+\.?\d*\s*(?:log|D-value|Z-value)',
    }
    
    if category in unit_patterns and re.search(unit_patterns[category], context, re.IGNORECASE):
        score += 2.0
    
    # Bonus for requirement language
    if re.search(r'\b(?:shall|must|should|required|specified|minimum|maximum)\b', context, re.IGNORECASE):
        score += 1.0
    
    # Bonus for ranges (e.g., "40-60°C")
    if re.search(r'\d+\.?\d*\s*[-–to]\s*\d+\.?\d*', context):
        score += 1.0
    
    # Bonus for multiple related keywords
    related_count = sum(1 for kw in PARAMETER_CATEGORIES[category] if kw.lower() in context.lower())
    score += related_count * 0.5
    
    return score

def analyze_tables(tables: List[Dict]) -> List[Dict]:
    """Analyze extracted tables for relevant parameters"""
    relevant_tables = []
    
    for table_info in tables:
        df = table_info['data']
        relevance_score = 0
        
        # Check column headers and cell content for keywords
        all_text = ' '.join([str(cell) for cell in df.values.flatten() if pd.notna(cell)])
        all_text += ' '.join([str(col) for col in df.columns])
        
        for category, keywords in PARAMETER_CATEGORIES.items():
            for keyword in keywords:
                if keyword.lower() in all_text.lower():
                    relevance_score += 1
        
        if relevance_score > 0:
            table_info['relevance_score'] = relevance_score
            relevant_tables.append(table_info)
    
    return sorted(relevant_tables, key=lambda x: x['relevance_score'], reverse=True)

def save_results_to_text(parameters: List[Dict], output_path: str):
    """
    Save the extracted parameters to a text file, grouped by category.
    """
    grouped = defaultdict(list)
    for param in parameters:
        grouped[param['category']].append(param)

    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("#" * 80 + "\n")
        f.write("#" + " " * 78 + "#\n")
        f.write("#" + " GUIDELINE EXTRACTION REPORT".center(78) + "#\n")
        f.write("#" + " ISO 11135 Parameters & Requirements".center(78) + "#\n")
        f.write("#" + " " * 78 + "#\n")
        f.write("#" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Parameters Extracted: {len(parameters)}\n\n")
        
        for category, items in grouped.items():
            f.write(f"\nCATEGORY: {category.upper()}\n\n")
            
            for idx, param in enumerate(items, 1):
                score_badge = "[HIGH]" if param['relevance_score'] >= 3.0 else "[MED ]" if param['relevance_score'] >= 2.0 else "[LOW ]"
                numerical_badge = "[NUMERICAL]" if param['has_numerical'] else ""
                
                f.write(f"\u250c─ PARAMETER {idx:02d} {score_badge} {numerical_badge} " + "─" * (80 - 25 - len(score_badge) - len(numerical_badge)) + "\n")
                f.write(f"│\n")
                f.write(f"│  Keyword: {param['keyword']}\n")
                f.write(f"│  Relevance Score: {param['relevance_score']:.1f}/5.0\n")
                f.write(f"│\n")
                f.write(f"│  Context:\n")
                
                # Wrap context text
                context_lines = param['context'].split('\n')
                for line in context_lines:
                    if len(line) <= 70:
                        f.write(f"│    {line}\n")
                    else:
                        words = line.split()
                        current_line = ""
                        for word in words:
                            if len(current_line) + len(word) + 1 <= 70:
                                current_line += word + " "
                            else:
                                f.write(f"│    {current_line.strip()}\n")
                                current_line = word + " "
                        if current_line:
                            f.write(f"│    {current_line.strip()}\n")
                
                f.write(f"│\n")
                f.write("└" + "─" * 79 + "\n\n")

if __name__ == "__main__":
    # Fixed: Added .pdf extension
    PDF_FILENAME = "BS EN ISO 11135-2014+A1-2019.pdf"
    OUTPUT_FILENAME = "guideline_extraction_output.txt"

    # Check if file exists
    if not Path(PDF_FILENAME).exists():
        print(f"❌ Error: PDF file '{PDF_FILENAME}' not found.")
        print("Please ensure the file is in the same directory as this script.")
        exit(1)

    try:
        logger.info(f"Reading guideline PDF: {PDF_FILENAME}")
        content = extract_pdf_content(PDF_FILENAME)
        logger.info("Successfully extracted %d pages", content['metadata']['total_pages'])
        logger.info("Title: %s", content['metadata']['title'])

        logger.info("Extracting parameters from guideline text")
        parameters = extract_parameters_with_context(content['text'])

        # Filter for high-relevance parameters
        high_relevance_params = [p for p in parameters if p['relevance_score'] >= 2.0]

        logger.info("Found %d total parameters, %d high-relevance", len(parameters), len(high_relevance_params))

        logger.info("Saving results to %s", OUTPUT_FILENAME)
        save_results_to_text(high_relevance_params, OUTPUT_FILENAME)

        logger.info("Guideline extraction complete; results saved to %s", OUTPUT_FILENAME)

        # Quick preview of top results (also log)
        logger.debug("Top 3 high-relevance parameters: %s", high_relevance_params[:3])

    except Exception as e:
        logger.exception("Error occurred during guideline extraction")
        print("❌ Error occurred: check logs for details")


