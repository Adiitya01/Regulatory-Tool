import os
import json
from pathlib import Path
import importlib
import pytest

# Import the module under test
MODULE_NAME = 'LLM_Engine'
from LLM_Engine import ISO11135CategoryMapper

@pytest.fixture()
def engine(tmp_path, monkeypatch):
    # Copy module into path or adjust cwd
    monkeypatch.chdir(tmp_path)
    # Create a dummy extracted file path variable
    module = importlib.import_module(MODULE_NAME)
    return module


def write_extracted(tmp_path, content):
    p = tmp_path / 'guideline_extraction_output.txt'
    p.write_text(content, encoding='utf-8')
    return p


def test_simple_mapping(engine, tmp_path, monkeypatch):
    sample = "Sterilization Parameters:\n🎯 Exposure Time: 30 min at 55°C\n🎯 Temperature: 55°C\n"
    extracted = write_extracted(tmp_path, sample)
    monkeypatch.setattr(engine, 'EXTRACTED_TEXT_FILE', str(extracted))
    categories = engine.load_raw_extracted_text()
    assert 'Sterilization Parameter' in categories
    ster_items = categories['Sterilization Parameter']
    assert any('Exposure Time' in it for it in ster_items)
    assert any('55°C' in it for it in ster_items)
    log_path = tmp_path / 'outputs' / 'category_mapping_log.json'
    assert log_path.exists(), 'Mapping log file should be created.'
    data = json.loads(log_path.read_text(encoding='utf-8'))
    assert data, 'Log should not be empty.'
    assert 'original_category' in data[0]
    assert 'score' in data[0]


def test_pattern_priority(engine, tmp_path, monkeypatch):
    sample = "Validation Parameters:\n🎯 IQ: Installation qualification to be performed.\n"
    extracted = write_extracted(tmp_path, sample)
    monkeypatch.setattr(engine, 'EXTRACTED_TEXT_FILE', str(extracted))
    categories = engine.load_raw_extracted_text()
    # IQ should map to Process Validation via pattern \bIQ\b
    assert 'Process Validation' in categories
    assert any(it.startswith('IQ:') for it in categories['Process Validation'])


def test_records_mapping(engine, tmp_path, monkeypatch):
    sample = "Quality Management Parameters:\n📍 Documentation: Maintain sterilization records for each load.\n"
    extracted = write_extracted(tmp_path, sample)
    monkeypatch.setattr(engine, 'EXTRACTED_TEXT_FILE', str(extracted))
    categories = engine.load_raw_extracted_text()
    assert 'Records/Documents' in categories
    assert any('Documentation:' in it for it in categories['Records/Documents'])


def test_empty_file(engine, tmp_path, monkeypatch):
    sample = ''
    extracted = write_extracted(tmp_path, sample)
    monkeypatch.setattr(engine, 'EXTRACTED_TEXT_FILE', str(extracted))
    categories = engine.load_raw_extracted_text()
    # All categories should be present but empty
    assert isinstance(categories, dict)
    assert all(isinstance(v, list) for v in categories.values())
    # No log file expected (could be empty) — we accept either missing or empty file
    log_path = tmp_path / 'outputs' / 'category_mapping_log.json'
    if log_path.exists():
        data = json.loads(log_path.read_text(encoding='utf-8'))
        assert data == [] or data == []


def test_duplicate_dedup(engine, tmp_path, monkeypatch):
    sample = (
        "Sterilization Parameters:\n"
        "🎯 Exposure Time: 30 min at 55°C\n"
        "🎯 Exposure Time: 30 min at 55°C\n"  # duplicate
        "🎯 Temperature: 55°C\n"
    )
    extracted = write_extracted(tmp_path, sample)
    monkeypatch.setattr(engine, 'EXTRACTED_TEXT_FILE', str(extracted))
    categories = engine.load_raw_extracted_text()
    ster_items = categories['Sterilization Parameter']
    # Dedup should remove one of the duplicates
    exp_time_count = sum(1 for it in ster_items if it.startswith('Exposure Time'))
    assert exp_time_count == 1


def test_dhf_category_config_loading(tmp_path):
    """Ensure mapper loads categories from JSON config and preserves behaviour if invalid."""
    # Create a minimal DHF category config in temp
    cfg = {
        "Product Requirement": {"keywords": ["product"], "patterns": ["product\\s+definition"]},
        "Process Validation": {"keywords": ["IQ", "OQ", "PQ"], "patterns": ["\\bIQ\\b"]}
    }
    cfg_path = tmp_path / "dhf_categories.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    mapper = ISO11135CategoryMapper(config_path=str(cfg_path))
    # Should reflect config keys
    assert "Product Requirement" in mapper.target_categories
    assert "Process Validation" in mapper.target_categories
    # Mapping should use config patterns
    cat = mapper.categorize_content("IQ: Installation qualification.", original_category="validation")
    assert cat == "Process Validation"

    # If config invalid, fall back to built-ins
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("[]", encoding="utf-8")
    mapper2 = ISO11135CategoryMapper(config_path=str(bad_path))
    assert "Sterilization Parameter" in mapper2.target_categories
