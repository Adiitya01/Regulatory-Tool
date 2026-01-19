# 🏥 DHF Multi-Document Processor

**Professional regulatory compliance document processing system for ISO 11135 EtO sterilization validation.**

## 📋 Overview

This system automates the extraction, analysis, and validation of Design History File (DHF) documentation against ISO 11135 standards for ethylene oxide sterilization processes.

### Key Features

- **Guideline Extraction**: Automated parameter extraction from ISO 11135 PDFs
- **LLM-Powered Polishing**: AI-enhanced regulatory content refinement
- **DHF Analysis**: Comprehensive extraction of sterilization parameters
- **Multi-Layer Validation**: 4-tier granular compliance assessment with false negative reduction
- **Interactive UI**: Streamlit-based web interface for easy operation

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- [LM Studio](https://lmstudio.ai/) running locally (default: `http://127.0.0.1:1234`)
- ISO 11135 PDF guideline document
- DHF documentation in PDF format

### Installation

```powershell
# Clone/download the project
cd Regulatory_Final

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Set environment variables (optional):

```powershell
$env:LLM_API_BASE = "http://127.0.0.1:1234"
$env:LLM_MODEL_NAME = "meta-llama-3.1-8b-instruct"
```

### Running the Application

**Option 1: Web UI (Recommended)**
```powershell
streamlit run UI.py
```

**Option 2: Command Line**
```powershell
# Extract guideline parameters
python Guideline_Extractor.py

# Polish content with LLM
python LLM_Engine.py

# Extract DHF parameters
python DHF_Extractor.py

# Run validation
python validation.py
```

## 📁 Project Structure

```
Regulatory_Final/
├── UI.py                          # Streamlit web interface
├── Guideline_Extractor.py         # ISO 11135 parameter extraction
├── LLM_Engine.py                  # AI-powered content polishing
├── DHF_Extractor.py               # DHF document analysis
├── validation.py                  # Multi-layer compliance validation
├── config.py                      # Centralized configuration
├── error_handler.py               # Error handling utilities
├── validators.py                  # Input/output validation
├── logging_setup.py               # Logging configuration
├── requirements.txt               # Python dependencies
├── inputs/                        # Input PDF files
├── outputs/                       # Generated outputs
├── reports/                       # Validation reports
└── temp/                          # Temporary files
```

## 🔄 Processing Pipeline

```
1. Guideline Extraction
   ├─ Upload ISO 11135 PDF
   ├─ Extract parameters with context
   └─ Output: guideline_extraction_output.txt

2. LLM Polishing
   ├─ Load extracted parameters
   ├─ Categorize into 10 ISO categories
   ├─ AI enhancement for clarity
   └─ Output: polished_regulatory_guidance.txt

3. DHF Extraction
   ├─ Upload DHF PDF
   ├─ Extract sterilization parameters
   ├─ Map to ISO categories
   └─ Output: DHF_Single_Extraction.txt

4. Validation
   ├─ Compare DHF vs Guidelines
   ├─ Multi-layer analysis (4 perspectives)
   ├─ Granular compliance classification
   └─ Output: validation_report.txt
```

## 📊 10 ISO 11135 Categories

1. **Product Requirement** - Device definition, materials, bioburden
2. **Process Validation** - IQ/OQ/PQ protocols, qualification
3. **Sterilization Parameter** - Temperature, humidity, EO concentration
4. **Biological Indicator** - BI testing, spore strips, SAL
5. **Records/Documents** - SOPs, procedures, validation reports
6. **Safety Precaution** - EO safety, personnel protection
7. **Test** - Sterility, bioburden, residue testing
8. **Monitoring** - Parameter control, data logging
9. **Acceptance Criteria** - Release criteria, statistical limits
10. **Storage & Distribution** - Post-sterilization handling

## 🎯 Validation System

### Multi-Layer Analysis

- **Layer 1**: Regulatory Compliance Focus
- **Layer 2**: Technical Documentation Focus
- **Layer 3**: Process Validation Focus
- **Layer 4**: Risk Assessment Focus

### Granular Classification

- ✅ **PRESENT**: Fully documented and compliant
- ⚠️ **PARTIAL**: Mentioned but needs clarification
- ❌ **MISSING**: Completely absent

### False Negative Reduction

- Semantic similarity matching
- Comprehensive DHF content validation
- Multi-perspective consensus building
- Evidence expansion with context

## 🛠️ Configuration

Edit `config.py` to customize:

```python
# LLM Settings
LLM_API_BASE = "http://127.0.0.1:1234"
LLM_MODEL_NAME = "meta-llama-3.1-8b-instruct"
LLM_TEMPERATURE = 0.3

# Extraction Settings
CONTEXT_WINDOW = 400
MIN_RELEVANCE_SCORE = 2.0

# Validation Settings
ANALYSIS_LAYERS = 4
```

## 📈 Output Files

### guideline_extraction_output.txt
Structured parameter extraction from ISO 11135 with relevance scoring

### polished_regulatory_guidance.txt
Professional, audit-ready regulatory guidance organized by category

### DHF_Single_Extraction.txt
Complete DHF parameter extraction with categorical mapping

### validation_report.txt
Comprehensive compliance gap analysis with readiness scores

### category_mapping_log.json
Detailed mapping audit trail for traceability

## 🔧 DHF Mapper Configuration (New)

This project supports configurable category mapping for DHF content using the existing `ISO11135CategoryMapper` with an optional JSON config.

- Config file: `inputs/dhf_categories.json`
- Default behavior: if the config is missing or invalid, the mapper falls back to the original 10 ISO categories.

### Using DHF categories in extractor

`DHF_Extractor.py` initializes the mapper with the DHF config:

```
mapper = ISO11135CategoryMapper(config_path=str(Path("inputs") / "dhf_categories.json"))
```

Guideline flows in `LLM_Engine.py` remain unchanged.

### Optional scoring enhancements

You can opt-in to enhanced scoring (keyword/pattern weights, numeric boost, original-category prior):

```
mapper = ISO11135CategoryMapper(
   config_path=str(Path("inputs") / "dhf_categories.json"),
   weights={"keyword":1.0, "pattern":2.0, "original":0.3, "numeric":1.2},
   enable_numeric_boost=True
)
```

Defaults preserve existing behavior when not provided.

### Validator integration (optional)

`validation.py` optionally reads `outputs/category_mapping_log.json` to prioritize section analysis by mapped item counts. If the log is absent, it uses the original fixed order.

### Mapping log details

`outputs/category_mapping_log.json` includes:
- `original_category`, `target_category`
- `score`, `matched_keywords`, `pattern_matches`, `confidence`

This supports better auditing and performance insights.

## 🧪 Tests & Run

Run unit tests:

```
python -m pytest -q
python -m pytest tests/test_mapping.py::test_dhf_category_config_loading -q
```

For DHF, run `DHF_Extractor.py` to produce `DHF_Single_Extraction.txt` and mapping logs; validator can then use the log for prioritization.

## 🐛 Troubleshooting

### LM Studio Connection Issues
```powershell
# Verify LM Studio is running
Test-NetConnection -ComputerName localhost -Port 1234

# Check model is loaded in LM Studio
# Ensure model name matches config
```

### PDF Extraction Errors
- Ensure PDFs are not password-protected
- Check file permissions
- Verify PDF is not corrupted

### Memory Issues
- Reduce `CONTEXT_WINDOW` in config
- Process documents in smaller sections
- Close other applications

## 📝 Best Practices

1. **Input Quality**: Use high-quality, text-based PDFs (not scanned images)
2. **Model Selection**: Use instruction-tuned models for best results
3. **Review Output**: Always review AI-generated content for accuracy
4. **Version Control**: Keep backups of validated reports
5. **Iterative Refinement**: Re-run validation after DHF updates

## 🔐 Data Security

- All processing is local (no cloud dependencies)
- PDFs never leave your machine
- LM Studio runs offline
- Sensitive data remains confidential

## 🤝 Support

For issues or questions:
1. Check logs in `outputs/` directory
2. Review error messages in console
3. Verify configuration in `config.py`
4. Ensure all dependencies are installed

## 📄 License

Proprietary - Designed by Ethosh

## 🔄 Version History

- **v2.0.0** - Multi-layer validation with false negative reduction
- **v1.5.0** - 10-category ISO mapping system
- **v1.0.0** - Initial release

---

**Made with ❤️ for Regulatory Excellence**
