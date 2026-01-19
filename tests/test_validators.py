"""
Unit tests for validation utilities
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from validators import (
    validate_pdf_file,
    validate_extracted_parameters,
    validate_category_mapping,
    sanitize_filename,
    validate_llm_response,
    validate_context_window,
    validate_relevance_threshold
)

class TestPDFValidation:
    def test_validate_nonexistent_pdf(self):
        is_valid, error = validate_pdf_file("nonexistent.pdf")
        assert not is_valid
        assert "does not exist" in error
    
    def test_validate_non_pdf_extension(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        is_valid, error = validate_pdf_file(str(test_file))
        assert not is_valid
        assert "not a PDF" in error
    
    def test_validate_empty_pdf(self, tmp_path):
        test_file = tmp_path / "empty.pdf"
        test_file.touch()
        is_valid, error = validate_pdf_file(str(test_file))
        assert not is_valid
        assert "empty" in error.lower()

class TestParameterValidation:
    def test_validate_empty_parameters(self):
        is_valid, error = validate_extracted_parameters([])
        assert not is_valid
        assert "No parameters" in error
    
    def test_validate_valid_parameters(self):
        params = [
            {
                'category': 'temperature',
                'keyword': 'temperature control',
                'context': 'Temperature shall be maintained',
                'relevance_score': 3.5
            }
        ]
        is_valid, error = validate_extracted_parameters(params)
        assert is_valid
        assert error is None
    
    def test_validate_missing_keys(self):
        params = [{'category': 'test'}]
        is_valid, error = validate_extracted_parameters(params)
        assert not is_valid
        assert "missing keys" in error
    
    def test_validate_negative_score(self):
        params = [
            {
                'category': 'test',
                'keyword': 'test',
                'context': 'test',
                'relevance_score': -1.0
            }
        ]
        is_valid, error = validate_extracted_parameters(params)
        assert not is_valid
        assert "negative" in error

class TestCategoryMapping:
    def test_validate_empty_mapping(self):
        is_valid, error = validate_category_mapping({})
        assert not is_valid
        assert "empty" in error.lower()
    
    def test_validate_valid_mapping(self):
        mapping = {
            'Product Requirement': ['item1', 'item2'],
            'Process Validation': []
        }
        is_valid, error = validate_category_mapping(mapping)
        assert is_valid
        assert error is None
    
    def test_validate_non_list_value(self):
        mapping = {'category': 'not a list'}
        is_valid, error = validate_category_mapping(mapping)
        assert not is_valid
        assert "not contain a list" in error

class TestFilenameSanitization:
    def test_sanitize_invalid_characters(self):
        result = sanitize_filename('test<file>name.pdf')
        assert '<' not in result
        assert '>' not in result
        assert result == 'test_file_name.pdf'
    
    def test_sanitize_long_filename(self):
        long_name = 'a' * 300 + '.pdf'
        result = sanitize_filename(long_name)
        assert len(result) <= 255
    
    def test_sanitize_empty_filename(self):
        result = sanitize_filename('')
        assert result == 'unnamed'

class TestLLMResponseValidation:
    def test_validate_empty_response(self):
        is_valid, error = validate_llm_response('')
        assert not is_valid
        assert "Empty" in error
    
    def test_validate_too_short(self):
        is_valid, error = validate_llm_response('test')
        assert not is_valid
        assert "too short" in error
    
    def test_validate_error_in_response(self):
        is_valid, error = validate_llm_response('❌ [LLM ERROR] Connection failed')
        assert not is_valid
        assert "error" in error.lower()
    
    def test_validate_valid_response(self):
        response = 'This is a valid LLM response with sufficient content.'
        is_valid, error = validate_llm_response(response)
        assert is_valid
        assert error is None

class TestContextWindow:
    def test_validate_non_integer(self):
        is_valid, error = validate_context_window('not an int')
        assert not is_valid
        assert "integer" in error
    
    def test_validate_too_small(self):
        is_valid, error = validate_context_window(10)
        assert not is_valid
        assert "too small" in error
    
    def test_validate_too_large(self):
        is_valid, error = validate_context_window(3000)
        assert not is_valid
        assert "too large" in error
    
    def test_validate_valid_window(self):
        is_valid, error = validate_context_window(400)
        assert is_valid
        assert error is None

class TestRelevanceThreshold:
    def test_validate_non_numeric(self):
        is_valid, error = validate_relevance_threshold('not a number')
        assert not is_valid
        assert "number" in error
    
    def test_validate_negative(self):
        is_valid, error = validate_relevance_threshold(-1.0)
        assert not is_valid
        assert "negative" in error
    
    def test_validate_too_high(self):
        is_valid, error = validate_relevance_threshold(15.0)
        assert not is_valid
        assert "too high" in error
    
    def test_validate_valid_threshold(self):
        is_valid, error = validate_relevance_threshold(2.5)
        assert is_valid
        assert error is None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
