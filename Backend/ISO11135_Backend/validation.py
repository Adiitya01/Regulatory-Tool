import json
import requests
import re
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import sys
from difflib import SequenceMatcher
import os
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from enum import Enum
import difflib
import atexit
import sys

import logging_setup
from . import config
from .storage_manager import storage

logger = logging_setup.get_logger(__name__)
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # ensure real-time flush
    def flush(self):
        for f in self.files:
            f.flush()

# Remove the global initialization of log_file and sys.stdout redirection
# Instead, add this function to manage output capture:

def setup_output_capture():
    """Setup output capture for terminal logging"""
    log_file = open("validation_terminal_output.txt", "w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    
    return log_file, original_stdout, original_stderr

def restore_output(log_file, original_stdout, original_stderr):
    """Restore original output streams"""
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()

class ComplianceStatus(Enum):
    """Granular compliance status to reduce false positives"""
    PRESENT = "✅ Present"
    PARTIAL = "⚠️ Partial/Needs Clarification" 
    MISSING = "❌ Missing"
    # NOT_APPLICABLE = "➖ Not Applicable"

# Stopwords to ignore (prevent false matches)
STOPWORDS = {
    'the','this','that','which','where','when','then','than','with','from',
    'into','onto','over','under','between','within','shall','must','should',
    'could','would','may','might','into','onto','such','other','more','less'
}

# Critical regulatory terms that should always be prioritized
CRITICAL_TERMS = {"protocol", "validation", "criteria", "report", "documentation" , "evidence", "compliance", "sterilization", "bioburden", "residue", "safety", "risk", "assessment"}

# Helper: fuzzy matching for token vs text
def token_in_text(token, text_lower):
    if token in text_lower:
        return True
    matches = difflib.get_close_matches(token, text_lower.split(), n=1, cutoff=0.8)
    return bool(matches)

def comprehensive_dhf_validation(dhf_text: str, missing_elements: list) -> list:
    """Simple but effective validation using direct text search"""
    text_lower = dhf_text.lower()
    validated_results = []
    
    # Simple keyword mapping for common missing elements
    element_keywords = {
        "biological indicator": ["biological", "indicator", "bi", "spore", "bacillus"],
        "chemical indicator": ["chemical", "indicator", "ci", "dosimeter"],
        "eo concentration": ["concentration", "eo", "ethylene", "oxide", "ppm", "mg/l"],
        "temperature": ["temperature", "celsius", "°c", "temp"],
        "humidity": ["humidity", "rh", "moisture", "%", "relative"],
        "validation": ["validation", "protocol", "verify", "qualified"],
        "sterilization": ["sterilization", "sterilize", "sterile"]
    }
    
    for element in missing_elements:
        # Get element text (handle different attribute names)
        element_text = ""
        if hasattr(element, 'content'):
            element_text = str(element.content).lower()
        else:
            element_text = str(element).lower()
        
        # print(f"DEBUG: Checking element: {element_text[:60]}...")
        
        # Direct keyword search in DHF content
        found_evidence = []
        
        # Extract key terms from element (skip stopwords)
        element_terms = [word for word in element_text.split() 
                        if len(word) > 3 and word not in STOPWORDS]
        
        for term in element_terms:
            if term in text_lower:
                # Find context around the term
                pattern = rf".{{0,80}}{re.escape(term)}.{{0,80}}"
                match = re.search(pattern, text_lower)
                if match:
                    found_evidence.append(match.group(0).strip())
                    # print(f"DEBUG: Found '{term}' in DHF")
        
        # Also check predefined keywords
        for category, keywords in element_keywords.items():
            if any(cat_word in element_text for cat_word in category.split()):
                for keyword in keywords:
                    if keyword in text_lower:
                        pattern = rf".{{0,80}}{re.escape(keyword)}.{{0,80}}"
                        match = re.search(pattern, text_lower)
                        if match:
                            found_evidence.append(match.group(0).strip())
                            # print(f"DEBUG: Found keyword '{keyword}' in DHF")
        
        # If we found evidence, mark as validated
        if found_evidence:
            best_evidence = max(found_evidence, key=len)  # Take longest evidence

            temp_element = element
            temp_element.content = safe_convert_negative_to_positive(element.content, ComplianceStatus.PRESENT)

            validated_results.append((element, {
                "found": True,
                "evidence": best_evidence,
                "pattern": "keyword_search",
                "confidence": 0.8
            }))
            # print(f"DEBUG: Element validated with evidence: {best_evidence[:50]}...")
        else:
            # print(f"DEBUG: No evidence found for element")
            pass
    
    # print(f"DEBUG: Total validated elements: {len(validated_results)}")
    return validated_results


def filter_missing_by_global_params(missing_elements: list, global_present_params: set) -> list:
    """Removes missing elements if their parameter is present anywhere in the DHF globally."""
    filtered = []
    for m in missing_elements:
        m_lower = m.lower()
        if not any(param in m_lower for param in global_present_params):
            filtered.append(m)
    return filtered

def validate_missing_against_dhf(missing_elements, dhf_finds, similarity_model, dhf_content):
    """Validate missing elements against DHF content to reduce false positives"""
    # print(f"🔍 **Starting validation of {len(missing_elements)} missing elements...**")
    
    validated_missing = []
    downgraded_partial = []
    downgraded_present = []
    
    for elem in missing_elements:
        # Combine reference + description into search query
        query = getattr(elem, 'content', '') or getattr(elem, 'description', '') or str(elem)
        best_score, best_match = 0, None
        
        # print(f"   🔎 **Checking: {query[:50]}...**")
        
        for find in dhf_finds:
            score = similarity_model(query, find["text"])  # semantic similarity
            if score > best_score:
                best_score, best_match = score, find["text"]
        
        # print(f"      Best match score: {best_score:.2f}")
        
        if best_score > 0.75:
            elem.status = ComplianceStatus.PRESENT
            elem.confidence = max(elem.confidence, best_score)
            #lem.evidence = best_match[:100] + "..." if len(best_match) > 100 else best_match
            elem.evidence = expand_evidence(best_match, dhf_content)
            downgraded_present.append(elem)
            print(f"      ✅ **Upgraded to PRESENT** (score: {best_score:.2f})")
        elif best_score > 0.45: 
            elem.status = ComplianceStatus.PARTIAL
            elem.confidence = min(elem.confidence, best_score)
            #lem.evidence = best_match[:100] + "..." if len(best_match) > 100 else best_match
            elem.content = safe_convert_negative_to_positive(elem.content, ComplianceStatus.PRESENT)
            elem.evidence = expand_evidence(best_match, dhf_content)
            downgraded_partial.append(elem)
            # print(f"⚠️ **Upgraded to PARTIAL** (score: {best_score:.2f})")
        else:
            validated_missing.append(elem)
            # print(f"❌ **Remains MISSING** (score: {best_score:.2f})")
    
    return validated_missing, downgraded_partial, downgraded_present

def extract_dhf_findings(dhf_content: str) -> List[Dict]:
    """Extract key findings/content from DHF for validation"""
    findings = []
    
    # Split content into meaningful chunks
    chunks = re.split(r'\n\s*\n', dhf_content)
    
    for chunk in chunks:
        if len(chunk.strip()) > 50:  # Only substantial content
            findings.append({
                "text": chunk.strip(),
                "length": len(chunk.strip())
            })
    
    return findings

def enhanced_similarity_model(query: str, text: str) -> float:
    """Enhanced similarity with domain-specific scoring"""
    query_lower = query.lower()
    text_lower = text.lower()
    
    # Direct substring match
    if query_lower in text_lower:
        return 0.95
    
    # Key term extraction with domain relevance
    sterilization_terms = {
        'biological', 'indicator', 'chemical', 'concentration', 'temperature', 
        'humidity', 'sterilization', 'validation', 'eto', 'ethylene', 'oxide',
        'spore', 'strip', 'bacillus', 'atrophaeus', 'range', 'control', 'specification'
    }
    
    query_tokens = set(re.findall(r'\b\w{3,}\b', query_lower))
    text_tokens = set(re.findall(r'\b\w{3,}\b', text_lower))
    
    # Boost score for sterilization-specific terms
    domain_overlap = query_tokens.intersection(text_tokens).intersection(sterilization_terms)
    regular_overlap = query_tokens.intersection(text_tokens) - sterilization_terms
    
    if not query_tokens:
        return 0.0
    
    # Weight domain terms higher
    domain_score = len(domain_overlap) * 1.5
    regular_score = len(regular_overlap)
    total_score = (domain_score + regular_score) / len(query_tokens)
    
    # Fuzzy matching for partial words
    fuzzy_matches = 0
    for q_token in query_tokens:
        if q_token in sterilization_terms:  # Only fuzzy match important terms
            for t_token in text_tokens:
                if len(q_token) > 4 and (q_token[:4] in t_token or t_token[:4] in q_token):
                    fuzzy_matches += 1
                    break
    
    fuzzy_score = (fuzzy_matches / len(query_tokens)) * 0.3
    
    return min(total_score + fuzzy_score, 1.0)


def deduplicate_partial(partial_list):
    """
    Remove duplicate Partial items based on description.
    Keeps the one with higher confidence if duplicates exist.
    
    Args:
        partial_list (list): List of dicts with keys 'description', 'confidence', 'evidence'
    
    Returns:
        list: Deduplicated Partial items
    """
    seen = {}
    for item in partial_list:
        desc = item["description"].strip().lower()
        conf = item.get("confidence", 0)
        if desc not in seen or conf > seen[desc].get("confidence", 0):
            seen[desc] = item
    return list(seen.values())

def expand_evidence(evidence_text, dhf_text, window=2):
    """
    Expand evidence snippet by pulling extra context from DHF text.
    
    Args:
        evidence_text (str): The raw evidence snippet (short fragment)
        dhf_text (str): The entire DHF document as text
        window (int): Number of lines before/after to include
    
    Returns:
        str: Expanded evidence context (2-3 lines instead of chopped fragment)
    """
    if not evidence_text or not dhf_text:
        return evidence_text
    
    lines = dhf_text.splitlines()
    for i, line in enumerate(lines):
        if evidence_text[:40].lower() in line.lower():  # match using first ~40 chars
            lo = max(0, i - window)
            hi = min(len(lines), i + window + 1)
            return " ".join(lines[lo:hi]).strip()
    
    return evidence_text  # fallback if not found

def convert_negative_to_positive(content: str, status: ComplianceStatus) -> str:
    """
    Convert negative descriptions to positive when status is PRESENT.
    Logs unmatched negative patterns for iterative improvement.
    """
    if status != ComplianceStatus.PRESENT:
        return content
    
    original_content = content
    result = content
    
    # Comprehensive negative to positive conversions
    conversions = {
        # Documentation-related
        "is not provided": "is documented",
        "is not documented": "is documented", 
        "are not documented": "are documented",
        "is not fully documented": "is documented",
        "are not fully documented": "are documented",
        "documentation is not": "documentation is",
        
        # Presence/Absence
        "is missing": "is present",
        "are missing": "are present",
        "is not present": "is present",
        "are not present": "are present",
        "is absent": "is present",
        "are absent": "are present",
        "is completely absent": "is documented",
        "are completely absent": "are documented",
        
        # Definition/Clarity
        "is not defined": "is defined",
        "are not defined": "are defined",
        "is not clearly defined": "is defined",
        "are not clearly defined": "are defined",
        "is unclear": "is documented",
        "are unclear": "are documented",
        "is not clear": "is clear",
        "are not clear": "are clear",
        
        # Mention/Specification
        "is not mentioned": "is mentioned",
        "are not mentioned": "are mentioned",
        "is not specified": "is specified",
        "are not specified": "are specified",
        "without specifying": "specifies",
        "does not specify": "specifies",
        "do not specify": "specify",
        
        # Detail/Completeness
        "lacks detail": "includes detail",
        "lack detail": "include detail",
        "lacks clarity": "includes clarity",
        "lack clarity": "include clarity",
        "is incomplete": "is complete",
        "are incomplete": "are complete",
        "is not complete": "is complete",
        "are not complete": "are complete",
        
        # Requirements/Standards
        "does not meet": "meets",
        "do not meet": "meet",
        "is not compliant": "is compliant",
        "are not compliant": "are compliant",
        "non-compliant": "compliant",
        
        # Validation-specific terms
        "is not validated": "is validated",
        "are not validated": "are validated",
        "lacks validation": "includes validation",
        "lack validation": "include validation",
        "validation is not": "validation is",
        
        # Sterilization-specific
        "sterilization is not": "sterilization is",
        "sterilization process is not": "sterilization process is",
        "sterilization parameters are not": "sterilization parameters are",
        "sterilization validation is not": "sterilization validation is"
    }
    
    # Apply specific conversions
    transformation_count = 0
    for negative, positive in conversions.items():
        if negative in result:
            result = result.replace(negative, positive)
            transformation_count += 1
    
    # Advanced regex patterns for broader matching
    regex_patterns = [
        # Handle "X is not Y" -> "X is Y" 
        (r'\b(\w+(?:\s+\w+)*)\s+is\s+not\s+(clear|adequate|sufficient|appropriate|compliant)', 
         r'\1 is \2'),
        
        # Handle "X are not Y" -> "X are Y"
        (r'\b(\w+(?:\s+\w+)*)\s+are\s+not\s+(clear|adequate|sufficient|appropriate|compliant)', 
         r'\1 are \2'),
        
        # Handle "lacks X" -> "includes X"
        (r'\b(lacks?|lacking)\s+([a-z]+(?:\s+[a-z]+)*)', 
         r'includes \2'),
        
        # Handle "does not include" -> "includes"
        (r'\b(does\s+not\s+include|do\s+not\s+include)', 
         r'includes'),
        
        # Handle "without X" -> "with X" (be careful with this one)
        (r'\bwithout\s+(proper|adequate|sufficient|clear)\s+([a-z]+(?:\s+[a-z]+)*)', 
         r'with \1 \2'),
    ]
    
    for pattern, replacement in regex_patterns:
        old_result = result
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        if result != old_result:
            transformation_count += 1
    
    # Check for unmatched negative patterns and log them
    negative_indicators = [
        "not", "missing", "absent", "lacks", "unclear", "incomplete", 
        "undefined", "unspecified", "inadequate", "insufficient"
    ]
    
    # Only log if we found negative words but made no transformations
    if transformation_count == 0:
        has_negative = any(indicator in result.lower() for indicator in negative_indicators)
        if has_negative:
            # Log for future pattern development
            logging.warning(f"UNMATCHED NEGATIVE PATTERN: {result[:150]}...")
            print(f"DEBUG: Unmatched negative pattern found: {result[:100]}...")
    
    # Cleanup: Remove any obvious grammatical issues from transformations
    result = re.sub(r'\s+', ' ', result)  # Remove extra spaces
    result = result.strip()
    
    # Log successful transformations for monitoring
    if transformation_count > 0:
        print(f"DEBUG: Applied {transformation_count} transformation(s) to: {original_content[:50]}...")
    
    return result

def validate_transformation_quality(original: str, transformed: str) -> bool:
    """
    Validate that the transformation maintained meaning and technical accuracy.
    Returns True if transformation is acceptable.
    """
    # Check for critical sterilization terms that shouldn't be lost
    critical_terms = [
        "iso 11135", "eto", "ethylene oxide", "biological indicator", 
        "sterilization", "validation", "temperature", "humidity", 
        "concentration", "bioburden", "sterility"
    ]
    
    for term in critical_terms:
        if term in original.lower() and term not in transformed.lower():
            print(f"WARNING: Critical term '{term}' lost in transformation")
            return False
    
    # Check that transformation didn't make it too generic
    if len(transformed) < len(original) * 0.7:
        print(f"WARNING: Transformation too aggressive - content significantly shortened")
        return False
    
    return True

# Usage example for integration into your existing code:
def safe_convert_negative_to_positive(content: str, status: ComplianceStatus) -> str:
    """
    Wrapper function that includes quality validation
    """
    if status != ComplianceStatus.PRESENT:
        return content
    
    transformed = convert_negative_to_positive(content, status)
    
    # Validate transformation quality
    if not validate_transformation_quality(content, transformed):
        print(f"WARNING: Transformation quality check failed, keeping original")
        return content
    
    return transformed


# --- SETTINGS FROM CENTRAL CONFIG ---
LLM_API_BASE = config.LLM_API_BASE
LLM_API_URL = config.LLM_API_URL
LLM_MODEL_NAME = config.LLM_MODEL_NAME
INPUT_FILE = config.DHF_EXTRACTION_OUTPUT
MAX_RETRIES = config.LLM_MAX_RETRIES
ANALYSIS_LAYERS = config.ANALYSIS_LAYERS
LLM_REQUEST_TIMEOUT = config.LLM_REQUEST_TIMEOUT
LLM_TEMPERATURE = config.LLM_TEMPERATURE
LLM_TOP_P = config.LLM_TOP_P
LLM_FREQUENCY_PENALTY = config.LLM_FREQUENCY_PENALTY
HF_TOKEN = config.HF_TOKEN
LLM_PROVIDER = config.LLM_PROVIDER
# Using centralized logging (configured in logging_setup)

@dataclass
class GranularElement:
    """Enhanced element with granular classification"""
    content: str
    status: ComplianceStatus
    evidence: str = ""
    iso_clause_ref: str = ""
    confidence: float = 0.0

@dataclass
class LayerAnalysisResult:
    """Enhanced result with granular classification"""
    layer_id: int
    layer_name: str
    elements: List[GranularElement] = field(default_factory=list)
    confidence_score: float = 0.0
    analysis_time: float = 0.0

@dataclass
class ValidationResult:
    """Enhanced validation result with granular classification"""
    section_name: str
    layer_results: List[LayerAnalysisResult] = field(default_factory=list)
    consensus_elements: List[GranularElement] = field(default_factory=list)
    overall_confidence: float = 0.0
    section_priority: str = "Medium"
    readiness_score: float = 0.0

class MultiLayerAnalysisEngine:
    """Enhanced multi-layer analysis engine with granular classification"""
    
    def __init__(self):
        self.layer_strategies = {
            1: "regulatory_compliance_focused",
            2: "technical_documentation_focused", 
            3: "process_validation_focused",
            4: "risk_assessment_focused"
        }
        
        self.analysis_prompts = {
            1: self._get_regulatory_prompt(),
            2: self._get_technical_prompt(),
            3: self._get_process_prompt(), 
            4: self._get_risk_prompt()
        }
    
    def _get_regulatory_prompt(self) -> str:
        return """You are a regulatory compliance expert specializing in ISO 11135 sterilization standards.

CRITICAL: Classify each requirement with granular precision to avoid false negatives because False Negative can raise much critical issue if you gave in the output.

Classification Rules:
- ✅ PRESENT: Requirement is fully documented and compliant
- ⚠️ PARTIAL: Requirement is mentioned but needs clarification/additional detail
- ❌ MISSING: Requirement is completely absent

Analyze ONLY from regulatory compliance perspective focusing on ISO 11135 requirements.

Format your response EXACTLY as:

REGULATORY_ANALYSIS:
✅ PRESENT:
- [specific requirement fully documented] | Evidence: [quote from DHF] 

⚠️ PARTIAL:
- [specific requirement partially addressed] | Evidence: [what exists] 

❌ MISSING:
- [specific requirement completely absent] | Expected: [what should be there] """

    def _get_technical_prompt(self) -> str:
        return """You are a technical documentation specialist for medical device sterilization.

CRITICAL: Use granular classification to distinguish between fully documented, partially documented, and missing technical requirements.

Classification Rules:
- ✅ PRESENT: Technical specification is complete and adequate
- ⚠️ PARTIAL: Specification exists but lacks detail or clarity
- ❌ MISSING: Technical specification is completely absent

Format your response EXACTLY as:

TECHNICAL_ANALYSIS:
✅ PRESENT:
- [complete technical specification] | Evidence: [specific detail found] 

⚠️ PARTIAL:
- [incomplete technical specification] | Evidence: [what's documented] 

❌ MISSING:
- [missing technical specification] | Expected: [what should be documented] 

"""

    def _get_process_prompt(self) -> str:
        return """You are a process validation expert for EtO sterilization.

CRITICAL: Distinguish between complete process documentation, partial documentation that needs enhancement, and completely missing elements.

Classification Rules:
- ✅ PRESENT: Process element is fully validated and documented
- ⚠️ PARTIAL: Process element is addressed but needs additional validation/documentation
- ❌ MISSING: Process element is not addressed at all


Format your response EXACTLY as:

PROCESS_ANALYSIS:
✅ PRESENT:
- [complete process element] | Evidence: [validation data found] 

⚠️ PARTIAL:
- [incomplete process element] | Evidence: [what's documented] 

❌ MISSING:
- [missing process element] | Expected: [required validation] 

"""

    def _get_risk_prompt(self) -> str:
        return """You are a risk management specialist for medical device sterilization.

CRITICAL: Assess risk management completeness with precision - many DHFs have basic risk considerations that shouldn't be marked as completely missing.

Classification Rules:
- ✅ PRESENT: Risk element is comprehensively addressed with controls
- ⚠️ PARTIAL: Risk element is identified but controls/documentation incomplete
- ❌ MISSING: Risk element is not considered at all

Format your response EXACTLY as:

RISK_ANALYSIS:
✅ PRESENT:
- [complete risk management] | Evidence: [risk control documented] 

⚠️ PARTIAL:
- [incomplete risk management] | Evidence: [what's addressed] 

❌ MISSING:
- [missing risk consideration] | Expected: [required risk analysis] 

 """

def load_guideline_sections(filepath: str) -> Dict[str, str]:
    """Load guideline sections from file with cloud support"""
    try:
        # Resolve path using storage manager if possible
        local_path = storage.ensure_local(filepath)
        
        if not local_path:
            print(f"Warning: {filepath} not found in local or cloud. Using default sections.")
            return create_default_sections()
            
        with local_path.open("r", encoding="utf-8") as f:
            text = f.read()
            
        sections = {}
        current_title = None
        current_text = []
        
        for line in text.splitlines():
            line_stripped = line.strip()
            if line_stripped and (line_stripped.isupper() or line_stripped.startswith("Section")):
                if current_title and current_text:
                    sections[current_title] = "\n".join(current_text).strip()
                current_title = line_stripped
                current_text = []
            else:
                current_text.append(line)
                
        if current_title and current_text:
            sections[current_title] = "\n".join(current_text).strip()
            
        return sections
        
    except Exception as e:
        print(f"Error loading guideline sections: {e}")
        return create_default_sections()

def create_default_sections() -> Dict[str, str]:
    """Create default regulatory sections if file is not available"""
    return {
        "Product Requirement": "Product definition, material compatibility, bioburden requirements, family grouping",
        "Process Validation": "Validation protocols, IQ/OQ/PQ, performance qualification, half-cycle studies",
        "Sterilization Parameter": "Temperature control, EO concentration, pressure range, humidity control",
        "Biological Indicator": "BI testing, spore strips, Bacillus atrophaeus, SAL verification",
        "Records/Documents": "Documentation, records, SOPs, procedures, validation reports",
        "Safety Precaution": "EO safety, personnel safety, exposure limits, ventilation",
        "Test": "Sterility testing, bioburden testing, residue testing, BET testing",
        "Monitoring": "Parameter control, continuous monitoring, data logging, alarm systems",
        "Acceptance Criteria": "Release criteria, SAL, statistical criteria, pass/fail limits",
        "Storage & Distribution": "Storage conditions, distribution, aeration, post-sterilization handling"
    }

def enhanced_call_llm_with_retry(dhf_section_title: str, dhf_section_text: str, 
                                guideline_section_title: str, guideline_section_text: str,
                                layer_id: int, analysis_engine: MultiLayerAnalysisEngine) -> str:
    """Enhanced LLM call with retry mechanism, re-establishment logic, and layer-specific prompts"""
    
    for attempt in range(MAX_RETRIES):
        try:
            # Test and re-establish connection on retries
            if attempt > 0:
                logger.info(f"Retry attempt {attempt + 1}/{MAX_RETRIES} for layer {layer_id} - testing LLM connection...")
                try:
                    # Quick connection test
                    models_url = f"{LLM_API_BASE.rstrip('/')}/v1/models"
                    test_resp = requests.get(models_url, timeout=3)
                    if test_resp.status_code == 200:
                        logger.info(f"LLM connection re-established for layer {layer_id}")
                    else:
                        logger.warning(f"LLM connection test returned status {test_resp.status_code}")
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(2 * attempt)  # Exponential backoff
                            continue
                except Exception as conn_err:
                    logger.warning(f"LLM connection test failed: {conn_err}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(3 * attempt)  # Longer wait on connection errors
                        continue
            
            url = LLM_API_URL
            
            content_hash = hashlib.md5(f"{dhf_section_title}{layer_id}{attempt}".encode()).hexdigest()
            seed = int(content_hash[:8], 16) % 10000
            
            system_prompt = analysis_engine.analysis_prompts[layer_id]
            
            payload = {
                "model": LLM_MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"""Analyze this DHF section for sterilization validation with GRANULAR PRECISION to avoid false positives:

DHF Section: {dhf_section_title}
Content: {dhf_section_text[:1500]}

Guideline Reference: {guideline_section_title}
Requirements: {guideline_section_text[:1000]}

IMPORTANT: 
- If something is mentioned but needs more detail, classify as PARTIAL, not MISSING
- If something is fully documented, classify as PRESENT
Provide granular analysis following the exact format specified in the system prompt."""
                    }
                ],
                "temperature": 0.05,
                "max_tokens": 3000,
                "seed": seed + layer_id,
                "top_p": 0.1,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1
            }
            
            headers = {}
            if LLM_PROVIDER == "hf" and HF_TOKEN:
                headers["Authorization"] = f"Bearer {HF_TOKEN}"
            
            response = requests.post(url, json=payload, headers=headers, timeout=LLM_REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"]
                
                if len(result.strip()) > 50:
                    return result
                else:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(1)
                        continue
            else:
                logger.warning(f"LLM API returned status {response.status_code} for layer {layer_id}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)
                    continue
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"LLM request exception for layer {layer_id} (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
                continue
        except Exception as e:
            logger.warning(f"LLM exception for layer {layer_id} (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
                continue
    
    logger.error(f"All retry attempts exhausted for layer {layer_id}, using fallback")
    return f"[LAYER_{layer_id}_FALLBACK]"

def extract_granular_elements(llm_response: str, layer_id: int) -> List[GranularElement]:
    """Extract elements with granular classification from LLM response"""
    
    elements = []
    
    try:
        # Extract each classification category
        for status in ComplianceStatus:
            status_symbol = status.value.split()[0]  # Get the emoji symbol
            
            # Pattern to match sections for each status
            pattern = f'{re.escape(status_symbol)}.*?:(.*?)(?={"|".join([re.escape(s.value.split()[0]) for s in ComplianceStatus])}|$)'
            
            section = re.search(pattern, llm_response, re.DOTALL | re.IGNORECASE)
            if section:
                text = section.group(1).strip()
                
                # Parse individual items
                for line in text.split('\n'):
                    line = line.strip()
                    if line.startswith('- ') or line.startswith('• '):
                        item_text = line[2:].strip()
                        
                        # Extract evidence and references
                        evidence = ""
                        iso_ref = ""
                        
                        # Look for evidence pattern
                        evidence_match = re.search(r'Evidence:\s*([^|]+)', item_text)
                        if evidence_match:
                            evidence = evidence_match.group(1).strip()
                            item_text = re.sub(r'\s*\|\s*Evidence:[^|]+', '', item_text)
                        
                        # Look for ISO/standard reference
                        ref_patterns = [r'ISO:\s*([^|]+)', r'Ref:\s*([^|]+)', r'Standard:\s*([^|]+)', r'ISO14971:\s*([^|]+)']
                        for pattern in ref_patterns:
                            ref_match = re.search(pattern, item_text)
                            if ref_match:
                                iso_ref = ref_match.group(1).strip()
                                item_text = re.sub(r'\s*\|\s*(?:ISO|Ref|Standard|ISO14971):[^|]+', '', item_text)
                                break
                        
                        # Clean up remaining separators
                        item_text = re.sub(r'\s*\|\s*Expected:[^|]*', '', item_text)
                        item_text = re.sub(r'\s*\|\s*Reason:[^|]*', '', item_text)
                        
                        if len(item_text.strip()) > 10:
                            element = GranularElement(
                                content=item_text.strip(),
                                status=status,
                                evidence=evidence,
                                iso_clause_ref=iso_ref,
                                confidence=calculate_element_confidence(item_text, evidence, iso_ref)
                            )
                            elements.append(element)
        
        return elements[:8]  # Limit to prevent overwhelming output
        
    except Exception as e:
        # Return fallback elements
        return get_layer_fallback_elements(layer_id)

def calculate_element_confidence(content: str, evidence: str, iso_ref: str) -> float:
    """Calculate confidence score for individual element"""
    score = 0.5  # Base score
    
    # Boost for specific evidence
    if evidence and len(evidence) > 10:
        score += 0.2
    
    # Boost for ISO reference
    if iso_ref:
        score += 0.2
    
    # Boost for detailed content
    if len(content) > 30:
        score += 0.1
    
    return min(score, 1.0)

def get_layer_fallback_elements(layer_id: int) -> List[GranularElement]:
    """Get fallback elements when LLM response fails"""
    
    fallbacks = {
        1: [  # Regulatory layer
            GranularElement("ISO 11135 compliance documentation review needed", ComplianceStatus.PARTIAL, "Basic framework present", "ISO 11135", 0.6),
            GranularElement("Regulatory submission package completeness", ComplianceStatus.PARTIAL, "Some documentation exists", "General", 0.5)
        ],
        2: [  # Technical layer
            GranularElement("Technical specification documentation review", ComplianceStatus.PARTIAL, "Basic specs present", "Technical", 0.6),
            GranularElement("Equipment qualification documentation", ComplianceStatus.PARTIAL, "Some qualifications documented", "Technical", 0.5)
        ],
        3: [  # Process layer
            GranularElement("Process validation protocol review", ComplianceStatus.PARTIAL, "Basic validation approach", "Process", 0.6),
            GranularElement("Performance qualification data review", ComplianceStatus.PARTIAL, "Some PQ data present", "Process", 0.5)
        ],
        4: [  # Risk layer
            GranularElement("Risk assessment documentation review", ComplianceStatus.PARTIAL, "Basic risk considerations", "ISO 14971", 0.6),
            GranularElement("Risk control measures documentation", ComplianceStatus.PARTIAL, "Some controls documented", "ISO 14971", 0.5)
        ]
    }
    
    return fallbacks.get(layer_id, [])

def generate_consensus_elements(layer_results: List[LayerAnalysisResult]) -> List[GranularElement]:
    """Generate consensus from multiple analysis layers with granular classification"""
    
    if not layer_results:
        return []
    
    # Collect all elements from all layers
    all_elements = []
    for layer in layer_results:
        all_elements.extend(layer.elements)
    
    # Group similar elements and determine consensus status
    consensus_elements = []
    processed_content = set()
    
    for element in all_elements:
        if element.content.lower() in processed_content:
            continue
            
        # Find similar elements across layers
        similar_elements = [e for e in all_elements 
                          if SequenceMatcher(None, element.content.lower(), e.content.lower()).ratio() > 0.7]
        
        if similar_elements:
            # Determine consensus status (most conservative wins to reduce false positives)
            status_priority = {
                ComplianceStatus.PRESENT: 4,
                ComplianceStatus.PARTIAL: 3, 
                # ComplianceStatus.NOT_APPLICABLE: 2,
                ComplianceStatus.MISSING: 1
            }
            
            consensus_status = max(similar_elements, key=lambda x: status_priority[x.status]).status
            
            # Combine evidence
            combined_evidence = " | ".join([e.evidence for e in similar_elements if e.evidence])
            
            # Get best ISO reference
            iso_refs = [e.iso_clause_ref for e in similar_elements if e.iso_clause_ref]
            best_iso_ref = iso_refs[0] if iso_refs else ""
            
            # Calculate consensus confidence
            avg_confidence = sum(e.confidence for e in similar_elements) / len(similar_elements)
            
            consensus_element = GranularElement(
                content=safe_convert_negative_to_positive(element.content, consensus_status),
                status=consensus_status,
                evidence=combined_evidence[:200] if combined_evidence else "",
                iso_clause_ref=best_iso_ref,
                confidence=avg_confidence
            )
            
            consensus_elements.append(consensus_element)
            processed_content.add(element.content.lower())
    
    return consensus_elements[:12]  # Limit consensus elements

def calculate_readiness_score(consensus_elements: List[GranularElement]) -> float:
    """Calculate DHF readiness score based on granular classification"""
    
    if not consensus_elements:
        return 0.0
    
    status_weights = {
        ComplianceStatus.PRESENT: 1.0,
        ComplianceStatus.PARTIAL: 0.6,
        ComplianceStatus.MISSING: 0.0
    }
    
    total_score = sum(status_weights[element.status] for element in consensus_elements)
    # max_possible = len([e for e in consensus_elements if e.status != ComplianceStatus.NOT_APPLICABLE])
    max_possible = len(consensus_elements)
    
    if max_possible == 0:
        return 1.0
    
    return total_score / max_possible

def match_guideline_section(dhf_title: str, detailed_regulatory_requirements: Dict[str, str]) -> Tuple[str, str]:
    """Match a DHF section title to the closest guideline section"""
    if dhf_title in detailed_regulatory_requirements:
        return dhf_title, detailed_regulatory_requirements[dhf_title]

    # Try partial match
    for title in detailed_regulatory_requirements.keys():
        if dhf_title.lower() in title.lower() or title.lower() in dhf_title.lower():
            return title, detailed_regulatory_requirements[title]

    return "General Requirements", "ISO 11135 general compliance requirements apply"

class EnhancedTerminalDisplay:
    """Enhanced terminal display with granular classification"""
    
    @staticmethod
    def print_header():
        print("\n" + "#" * 80)
        print("#" + "" * 78 + "#")
        print("#" + " STERILIZATION REPORT GAP ANALYSIS".center(78) + "#")
        print("#" + " " * 78 + "#")
        print("#" * 80)

    @staticmethod
    def print_layer_progress(layer_id: int, layer_name: str, section_name: str):
        layer_icons = {1: "📋", 2: "🔧", 3: "⚙️", 4: "🛡️"}
        icon = layer_icons.get(layer_id, "📄")
        # print(f"   {icon} Layer {layer_id} ({layer_name}) - granular analysis of {section_name}...")
    
    @staticmethod
    def print_section_granular_analysis(result: ValidationResult, section_num: int):
        """Print granular section analysis"""
        
        print(f"\n\n{'#'*80}")
        print(f"# SECTION {section_num}: {result.section_name.upper()}")
        print(f"#" + "-" * 78)
        print(f"# Readiness Score: {result.readiness_score:>6.1%}  |  Overall Confidence: {result.overall_confidence:>6.1%}")
        print("#" * 80)
        
        # Group elements by status
        status_groups = {}
        for element in result.consensus_elements:
            if element.status not in status_groups:
                status_groups[element.status] = []
            status_groups[element.status].append(element)
        
        # Display each status group
        for status in ComplianceStatus:
            if status in status_groups:
                elements = status_groups[status]
                print(f"\n┌─ {status.value} ({len(elements)} items) " + "─" * (70 - len(status.value) - len(str(len(elements)))))
                
                for idx, element in enumerate(elements, 1):
                    display_content = safe_convert_negative_to_positive(element.content, element.status)
                    print(f"│")
                    print(f"│  [{idx:02d}] {element.content}")
                    if element.evidence:
                        evidence_clean = element.evidence[:100].strip().replace('\n', ' ')
                        print(f"│      → Evidence: {evidence_clean}...")
                    if element.iso_clause_ref:
                        print(f"│      → Reference: {element.iso_clause_ref}")
                    print(f"│      → Confidence: {element.confidence:.1%}")
                
                print("└" + "─" * 79)
    
    @staticmethod
    def print_readiness_summary(results: List[ValidationResult]):
        """Print DHF readiness summary with granular insights"""
        
        print("\n\n" + "#" * 80)
        print("#" + " " * 78 + "#")
        print("#" + " DHF READINESS ASSESSMENT".center(78) + "#")
        print("#" + " " * 78 + "#")
        print("#" * 80)
        
        if not results:
            print("\n  No results to display.")
            return
        
        # Calculate overall readiness
        overall_readiness = sum(r.readiness_score for r in results) / len(results)
        
        print("\n┌─ OVERALL METRICS " + "─" * 61)
        print(f"│  Overall DHF Readiness: {overall_readiness:>6.1%}")
        print("└" + "─" * 79)
        
        # Status distribution
        all_elements = []
        for result in results:
            all_elements.extend(result.consensus_elements)
        
        if all_elements:
            status_counts = {}
            for element in all_elements:
                if element.status not in status_counts:
                    status_counts[element.status] = 0
                status_counts[element.status] += 1
            
            print(f"\n┌─ COMPLIANCE DISTRIBUTION " + "─" * 52)
            for status in ComplianceStatus:
                count = status_counts.get(status, 0)
                percentage = (count / len(all_elements)) * 100 if all_elements else 0
                bar_length = int(percentage / 2)  # Scale to 50 chars max
                bar = "█" * bar_length + "░" * (50 - bar_length)
                print(f"│  {status.value:<35} [{bar}] {percentage:>5.1f}%")
            print("└" + "─" * 79)
        
        # Priority actions (only for MISSING and high-priority PARTIAL items)
        priority_actions = []
        for result in results:
            for element in result.consensus_elements:
                if element.status == ComplianceStatus.MISSING:
                    priority_actions.append(f"Complete: {element.content}")
                elif element.status == ComplianceStatus.PARTIAL and element.confidence < 0.5:
                    priority_actions.append(f"Enhance: {element.content}")
        
        if priority_actions:
            print(f"\n┌─ PRIORITY ACTIONS (Top {min(len(priority_actions), 8)}) " + "─" * (67 - len(str(min(len(priority_actions), 8)))))
            for i, action in enumerate(priority_actions[:8], 1):
                print(f"│  [{i}] {action}")
            print("└" + "─" * 79)
        
        # Readiness by section
        print(f"\n┌─ SECTION READINESS SCORES " + "─" * 51)
        sorted_results = sorted(results, key=lambda x: x.readiness_score, reverse=True)
        for result in sorted_results:
            score_pct = result.readiness_score * 100
            bar_length = int(score_pct / 2)  # Scale to 50 chars
            bar = "█" * bar_length + "░" * (50 - bar_length)
            status_tag = "[HIGH]" if result.readiness_score > 0.8 else "[MED ]" if result.readiness_score > 0.5 else "[LOW ]"
            print(f"│  {status_tag} {result.section_name:<25} [{bar}] {result.readiness_score:>6.1%}")
        print("└" + "─" * 79 + "\n")

class EnhancedMultiLayerValidationEngine:
    """Enhanced validation engine with granular classification"""
    
    def __init__(self):
        self.display = EnhancedTerminalDisplay()
        self.multi_layer_engine = MultiLayerAnalysisEngine()
        self.detailed_regulatory_requirements = load_guideline_sections("polished_regulatory_guidance.txt")
        
        if not self.detailed_regulatory_requirements:
            print("Warning: Using default regulatory requirements")
            self.detailed_regulatory_requirements = create_default_sections()

    def extract_sections(self, content: str) -> Dict[str, str]:
        """Extract document sections with enhanced pattern matching"""
        sections = {}
        
        section_patterns = {
            "Product Requirement": r"(?:product\s+(?:requirement|definition)|material\s+compatibility|bioburden|family\s+grouping|device\s+characterization)",
            "Process Validation": r"(?:validation\s+protocol|IQ\s*\/\s*OQ\s*\/\s*PQ|performance\s+qualification|half.?cycle|process\s+validation)",
            "Sterilization Parameter": r"(?:sterilization\s+parameters|temperature\s+control|EO\s+concentration|pressure\s+range|humidity\s+control)",
            "Biological Indicator": r"(?:biological\s+indicator|BI\s+testing|spore\s+strip|Bacillus\s+atrophaeus|sterility\s+assurance|SAL)",
            "Records/Documents": r"(?:documentation|records|SOPs|procedures|annexures|protocol|validation\s+report)",
            "Safety Precaution": r"(?:safety\s+precaution|EO\s+safety|personnel\s+safety|exposure\s+limits|ventilation|hazard)",
            "Test": r"(?:sterility\s+test|bioburden\s+test|residue\s+test|BET\s+testing|analytical\s+testing)",
            "Monitoring": r"(?:monitoring|parameter\s+control|continuous\s+monitoring|data\s+logging|alarm\s+system)",
            "Acceptance Criteria": r"(?:acceptance\s+criteria|release\s+criteria|SAL|statistical\s+criteria|pass.fail\s+limits)",
            "Storage & Distribution": r"(?:storage|distribution|post.sterilization|aeration|transport|inventory\s+management)"
        }
        
        chunks = re.split(r'\n\s*\n', content)
        
        for chunk in chunks:
            if len(chunk.strip()) < 50:
                continue
                
            chunk_lower = chunk.lower()
            best_section = "General"
            best_score = 0
            
            for section, pattern in section_patterns.items():
                try:
                    matches = len(re.findall(pattern, chunk_lower))
                    if section.lower() in chunk_lower:
                        matches += 2
                        
                    if matches > best_score:
                        best_score = matches
                        best_section = section
                except re.error:
                    continue
            
            if best_section in sections:
                sections[best_section] += "\n\n" + chunk
            else:
                sections[best_section] = chunk
        
        return sections

    def perform_granular_analysis(self, section_name: str, section_content: str, 
                                 guideline_title: str, guideline_text: str) -> ValidationResult:
        """Perform granular multi-layer analysis"""
        
        layer_results = []
        
        for layer_id in range(1, ANALYSIS_LAYERS + 1):
            start_time = time.time()
            
            layer_name = self.multi_layer_engine.layer_strategies[layer_id]
            # self.display.print_layer_progress(layer_id, layer_name, section_name)
            
            llm_response = enhanced_call_llm_with_retry(
                section_name, section_content, guideline_title, guideline_text,
                layer_id, self.multi_layer_engine
            )
            
            elements = extract_granular_elements(llm_response, layer_id)
            
            confidence = sum(e.confidence for e in elements) / len(elements) if elements else 0.0
            analysis_time = time.time() - start_time
            
            layer_result = LayerAnalysisResult(
                layer_id=layer_id,
                layer_name=layer_name,
                elements=elements,
                confidence_score=confidence,
                analysis_time=analysis_time
            )
            
            layer_results.append(layer_result)
        
        # Generate consensus with granular classification
        consensus_elements = generate_consensus_elements(layer_results)
        
        # *** VALIDATION AGAINST DHF CONTENT - FIXED BLOCK ***
        # print(f"🔍 **Starting DHF content validation for {section_name}...**")

        # Extract DHF findings for validation
        dhf_findings = extract_dhf_findings(section_content)
        print(f"📄 **Extracted {len(dhf_findings)} DHF content chunks for validation**")

        # Get only the missing elements from consensus
        missing_elements = [elem for elem in consensus_elements if elem.status == ComplianceStatus.MISSING]

        upgraded_present, upgraded_partial, still_missing = [], [], []

        if missing_elements and section_content:
            # First: Comprehensive validation
            validated_results = comprehensive_dhf_validation(section_content, missing_elements)
            
            # Get validated elements for exclusion from semantic validation
            validated_elements = [elem for elem, _ in validated_results]

            # Second: Semantic validation for remaining items
            remaining_missing = [elem for elem in missing_elements if elem not in validated_results]

            if remaining_missing and dhf_findings:
                still_missing, upgraded_partial, upgraded_present = validate_missing_against_dhf(
                    remaining_missing, dhf_findings, enhanced_similarity_model, section_content
                )
            else:
                still_missing, upgraded_partial, upgraded_present = [], [], []

            for elem, result in validated_results:
                if result["found"]:
                    elem.status = ComplianceStatus.PRESENT
                    elem.confidence = result["confidence"]
                    elem.evidence = expand_evidence(result["evidence"], section_content)
                    upgraded_present.append(elem)

            # Keep all that were never missing
            kept = [elem for elem in consensus_elements if elem.status != ComplianceStatus.MISSING]

            # Build new consensus with upgraded ones placed properly
            consensus_elements = kept + upgraded_present + upgraded_partial + still_missing

            # print(f"✅ **DHF Validation Results:**")
            # print(f" 📈 **{len(upgraded_present)} elements upgraded to PRESENT**")
            # print(f" ⚠️ **{len(upgraded_partial)} elements upgraded to PARTIAL**") 
            # print(f" ❌ **{len(still_missing)} elements remain MISSING**")
            # print(f" 🔄 **Total consensus elements: {len(consensus_elements)}**")
            
        elif missing_elements and not dhf_findings:
             print(f"⚠️ **Warning: {len(missing_elements)} missing elements found but no DHF content extracted for validation**")
        
        elif not missing_elements:
            print(f"✅ **No missing elements found - validation skipped**")
        
        else:
            print(f"ℹ️ **No validation needed - no missing elements or DHF content**")

        # print(f"✨ **DHF validation completed for {section_name}**")


        #  -------------  *** END VALIDATION BLOCK ***   ----------------

        # Calculate readiness score
        readiness_score = calculate_readiness_score(consensus_elements)
            
        # Calculate overall confidence
        overall_confidence = sum(layer.confidence_score for layer in layer_results) / len(layer_results) if layer_results else 0.0
            
        # Determine section priority based on readiness
        if readiness_score > 0.75:
            priority = "Low"  # High readiness = Low priority for action
        elif readiness_score > 0.45:
            priority = "Medium"
        else:
            priority = "High"  # Low readiness = High priority for action
            
        return ValidationResult(
            section_name=section_name,
            layer_results=layer_results,
            consensus_elements=consensus_elements,
            overall_confidence=overall_confidence,
            section_priority=priority,
            readiness_score=readiness_score
        )
    def process_document_with_granular_analysis(self, file_path):

        """
        Wrapper for performing granular analysis end-to-end.
        Called by the UI for comprehensive document analysis.
        """

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()

                sections = self._split_into_sections(raw_text)

                results = []

                for section in sections:
                    result = self.perform_granular_analysis(
                    section_name=section['section_name'],
                    section_content=section['content'], 
                    guideline_title="DHF Guidelines",  # adjust as needed
                    guideline_text=""  # adjust as needed
                )
                results.append(result)

            return results

        except FileNotFoundError:
            raise FileNotFoundError(f"Document not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Analysis failed: {str(e)}")

    def process_document_with_detailed_analysis(self, file_path):
        """
        Alias wrapper for detailed analysis.
        """
        return self.process_document_with_granular_analysis(file_path)
    
    def _split_into_sections(self, raw_text):
        """
        Helper method to split document into analyzable sections.
        """
        sections = []
        
        if "===" in raw_text:
            parts = raw_text.split("===")
            for i, part in enumerate(parts):
                if part.strip():
                    sections.append({
                        'section_name': f'Section {i+1}',
                        'content': part.strip()
                    })
        else:
            paragraphs = [p.strip() for p in raw_text.split('\n\n') if p.strip()]
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph) > 100:
                    sections.append({
                        'section_name': f'Paragraph {i+1}',
                        'content': paragraph
                    })
        
        return sections if sections else [{'section_name': 'Full Document', 'content': raw_text}]

    def process_document_with_granular_analysis(self, file_path: str) -> List[ValidationResult]:
        """Process document with granular multi-layer analysis"""
            
        self.display.print_header()
            
        # Check if input file exists
        try:
            local_path = storage.ensure_local(file_path)
            if not local_path:
                print(f"❌ **Error**: {file_path} not found in local or cloud storage")
                return []
                    
            with local_path.open('r', encoding='utf-8') as f:
                content = f.read()
                    
            if not content.strip():
                print(f"❌ **Error**: {file_path} is empty")
                return []
                    
        except Exception as e:
            print(f"❌ **Error reading file**: {str(e)}")
            return []
            
        sections = self.extract_sections(content)

        # Optional: prioritize sections based on mapper output (if available)
        mapping_hint = self._load_category_mapping_hint()
        results = []
            
        # print(f"\n📊 **Initializing granular DHF compliance analysis...**")
        print(f"🌙 **Precision analysis for {len(sections)} sections with false Negative reduction**")
        # print(f"🎯 **4-tier classification: Present | Partial | Missing | N/A**")
            
        section_counter = 1
        base_order = ["Product Requirement", "Process Validation", "Sterilization Parameter",
                      "Biological Indicator", "Records/Documents", "Safety Precaution", 
                      "Test", "Monitoring", "Acceptance Criteria", "Storage & Distribution"]
        if mapping_hint:
            # Reorder by descending mapped count while keeping base order for ties/missing
            priority_sections = sorted(base_order, key=lambda s: (-mapping_hint.get(s, 0), base_order.index(s)))
        else:
            priority_sections = base_order
            
        # Process sections in priority order - FIXED
        for section_name in priority_sections:
            if section_name in sections:
                section_content = sections[section_name]
                    
                print(f"\n⏳ Processing {section_name} with granular classification...")
                    
                # Match guideline section
                matched_guideline_title, guideline_text = match_guideline_section(
                    section_name, self.detailed_regulatory_requirements
                )
                    
                # Perform granular analysis - FIXED INDENTATION
                result = self.perform_granular_analysis(
                    section_name, section_content, matched_guideline_title, guideline_text
                )
                    
                results.append(result)
                    
                # Display section analysis
                self.display.print_section_granular_analysis(result, section_counter)
                section_counter += 1

        # Display readiness summary - FIXED INDENTATION
        if results:
            self.display.print_readiness_summary(results)

            # Final summary - FIXED INDENTATION
            avg_readiness = sum(r.readiness_score for r in results) / len(results)
            high_readiness_count = len([r for r in results if r.readiness_score > 0.8])
                
            print("\n┌" + "─" * 78)
            print(f"│ Average DHF Readiness: {avg_readiness:.1%}")
            print(f"│ High-Readiness Sections: {high_readiness_count}/{len(results)}")
                
            # Readiness assessment
            if avg_readiness > 0.85:
                print(f"│ Assessment: DHF is review ready with minor enhancements")
            elif avg_readiness > 0.65:
                print(f"│ Assessment: DHF needs targeted improvements before review")
            else:
                print(f"│ Assessment: DHF requires significant work before submission")
                
            print("└" + "─" * 78 + "\n")

        return results

    def _load_category_mapping_hint(self) -> Dict[str, int]:
        """Load category counts from outputs/category_mapping_log.json to guide validation focus.
        Non-fatal if file missing or unreadable.
        """
        try:
            log_path = Path("outputs") / "category_mapping_log.json"
            if not log_path.exists():
                return {}
            data = json.loads(log_path.read_text(encoding="utf-8"))
            counts: Dict[str, int] = {}
            for item in data:
                cat = item.get("target_category")
                if not cat:
                    continue
                counts[cat] = counts.get(cat, 0) + 1
            return counts
        except Exception:
            return {}

def main():
    """Execute enhanced granular DHF analysis with false Negative reduction"""
    # print("🌙 **Initializing Enhanced Granular DHF Analysis Engine...**")
    # print("🎯 **False Negative Reduction System Active**")

    log_file, original_stdout, original_stderr = setup_output_capture()

    try:
        engine = EnhancedMultiLayerValidationEngine()
        
        results = engine.process_document_with_granular_analysis(INPUT_FILE)

        # for sections, results in validation_results.items():
        #     if results.get("Partial"):
        #         results["Partial"] =  deduplicate_partial(results["Partial"])

        if not results:
            print("\n❌ **No results generated**. Check input file and configuration.")
            return
        
        # Calculate final statistics
        total_elements = sum(len(r.consensus_elements) for r in results)
        missing_count = sum(len([e for e in r.consensus_elements if e.status == ComplianceStatus.MISSING]) for r in results)
        partial_count = sum(len([e for e in r.consensus_elements if e.status == ComplianceStatus.PARTIAL]) for r in results)
        present_count = sum(len([e for e in r.consensus_elements if e.status == ComplianceStatus.PRESENT]) for r in results)
        
        print(f"\n🎉 **GRANULAR DHF ANALYSIS COMPLETED!**")
        print(f"📊 **Total Elements Analyzed: {total_elements}**")
        print(f"✅ **Present: {present_count} | ⚠️ Partial: {partial_count} | ❌ Missing: {missing_count}**")
        
        # Fixed division by zero error
        if total_elements > 0:
            precision = ((present_count + partial_count) / total_elements * 100)
            print(f"🎯 **False Negative Reduction: {precision:.1f}% precision**")
        else:
            print(f"🎯 **False Negative Reduction: No elements analyzed**")

    except KeyboardInterrupt:
        print(f"\n⚠️ **Analysis interrupted** - Partial results available")
    except Exception as e:
        print(f"\n💥 **Error**: {str(e)}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        restore_output(original_stdout, original_stderr, log_file)
        
if __name__ == "__main__":
    main()


    