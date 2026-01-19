"""
Data validation utilities for input/output validation
"""
import re
from typing import Optional, Dict, List
from pathlib import Path

def validate_pdf_file(file_path: str) -> tuple[bool, Optional[str]]:
    """
    Validate PDF file exists and is readable
    
    Returns:
        (is_valid, error_message)
    """
    path = Path(file_path)
    
    if not path.exists():
        return False, f"File does not exist: {file_path}"
    
    if not path.is_file():
        return False, f"Path is not a file: {file_path}"
    
    if path.suffix.lower() != '.pdf':
        return False, f"File is not a PDF: {file_path}"
    
    if path.stat().st_size == 0:
        return False, f"File is empty: {file_path}"
    
    # Check if file is readable
    try:
        with open(path, 'rb') as f:
            # Try to read first few bytes
            header = f.read(4)
            if header != b'%PDF':
                return False, f"File does not appear to be a valid PDF: {file_path}"
    except Exception as e:
        return False, f"Cannot read file: {str(e)}"
    
    return True, None

def validate_extracted_parameters(parameters: List[Dict]) -> tuple[bool, Optional[str]]:
    """
    Validate extracted parameter data structure
    
    Returns:
        (is_valid, error_message)
    """
    if not parameters:
        return False, "No parameters extracted"
    
    required_keys = {'category', 'keyword', 'context', 'relevance_score'}
    
    for idx, param in enumerate(parameters):
        if not isinstance(param, dict):
            return False, f"Parameter {idx} is not a dictionary"
        
        missing_keys = required_keys - set(param.keys())
        if missing_keys:
            return False, f"Parameter {idx} missing keys: {missing_keys}"
        
        if not isinstance(param['relevance_score'], (int, float)):
            return False, f"Parameter {idx} has invalid relevance_score type"
        
        if param['relevance_score'] < 0:
            return False, f"Parameter {idx} has negative relevance_score"
    
    return True, None

def validate_category_mapping(mapping: Dict[str, List]) -> tuple[bool, Optional[str]]:
    """
    Validate category mapping structure
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(mapping, dict):
        return False, "Mapping is not a dictionary"
    
    if not mapping:
        return False, "Mapping is empty"
    
    for category, items in mapping.items():
        if not isinstance(items, list):
            return False, f"Category '{category}' does not contain a list"
        
        if not items:
            continue  # Empty category is OK
        
        for idx, item in enumerate(items):
            if not isinstance(item, str):
                return False, f"Category '{category}' item {idx} is not a string"
            
            if not item.strip():
                return False, f"Category '{category}' item {idx} is empty"
    
    return True, None

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file system operations
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        sanitized = name[:255-len(ext)-1] + (f'.{ext}' if ext else '')
    
    return sanitized or 'unnamed'

def validate_llm_response(response: str) -> tuple[bool, Optional[str]]:
    """
    Validate LLM response is usable
    
    Returns:
        (is_valid, error_message)
    """
    if not response:
        return False, "Empty response"
    
    if len(response.strip()) < 10:
        return False, "Response too short"
    
    # Check for error indicators
    error_indicators = [
        "❌ [LLM ERROR]",
        "[ERROR]",
        "API Error",
        "Connection refused"
    ]
    
    for indicator in error_indicators:
        if indicator in response:
            return False, f"Response contains error: {indicator}"
    
    return True, None

def validate_context_window(window: int) -> tuple[bool, Optional[str]]:
    """
    Validate context window parameter
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(window, int):
        return False, "Context window must be an integer"
    
    if window < 50:
        return False, "Context window too small (minimum 50)"
    
    if window > 2000:
        return False, "Context window too large (maximum 2000)"
    
    return True, None

def validate_relevance_threshold(threshold: float) -> tuple[bool, Optional[str]]:
    """
    Validate relevance score threshold
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(threshold, (int, float)):
        return False, "Threshold must be a number"
    
    if threshold < 0:
        return False, "Threshold cannot be negative"
    
    if threshold > 10:
        return False, "Threshold too high (maximum 10)"
    
    return True, None
