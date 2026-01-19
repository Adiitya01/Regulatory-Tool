"""
Centralized configuration for ISO 11135 Pipeline
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# === PROJECT PATHS ===
PROJECT_ROOT = Path(__file__).parent.parent  # Go up one level to Backend/
BACKEND_ROOT = PROJECT_ROOT
INPUTS_DIR = PROJECT_ROOT / "inputs"
# Point to root-level outputs directory (two levels up from Backend/)
OUTPUTS_DIR = PROJECT_ROOT.parent / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
TEMP_DIR = PROJECT_ROOT / "temp"

# Ensure iso11135 output directory exists
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# === LLM CONFIGURATION ===
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")  # 'local' or 'hf'
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Default LM Studio Base
LMSTUDIO_BASE = os.getenv("LMSTUDIO_API_BASE", "http://127.0.0.1:1234")

if LLM_PROVIDER == "hf":
    # Hugging Face Inference API (OpenAI compatible router)
    LLM_API_BASE = "https://router.huggingface.co/v1"
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct")
    LLM_API_URL = f"{LLM_API_BASE}/chat/completions"
else:
    # Local LM Studio
    LLM_API_BASE = os.getenv("LLM_API_BASE", LMSTUDIO_BASE)
    LLM_API_URL = os.getenv("LLM_API_URL", f"{LLM_API_BASE}/v1/chat/completions")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "meta-llama-3.1-8b-instruct")


# LLM Request Settings
LLM_TEMPERATURE = 0.2  # Lowered for higher precision in regulatory tasks
LLM_MAX_TOKENS = 2048  # Increased to prevent truncation of requirements
LLM_TOP_P = 0.9
LLM_FREQUENCY_PENALTY = 0.1
LLM_REQUEST_TIMEOUT = 60
LLM_MAX_RETRIES = 3

# === FILE NAMES ===
GUIDELINE_EXTRACTION_OUTPUT = "guideline_extraction_output.txt"
POLISHED_OUTPUT_FILE = "polished_regulatory_guidance.txt"
DHF_EXTRACTION_OUTPUT = "DHF_Single_Extraction.txt"
VALIDATION_REPORT = "validation_report.txt"
VALIDATION_TERMINAL_OUTPUT = "validation_terminal_output.txt"
CATEGORY_MAPPING_LOG = "category_mapping_log.json"

# === EXTRACTION SETTINGS ===
CONTEXT_WINDOW = 800   # Increased from 400 for better regulatory context
CONTEXT_EXTRA_PERCENT = 0.3
MIN_RELEVANCE_SCORE = 1.8  # Slightly lowered to catch more edge cases
CHUNK_SIZE = 6  # Smaller chunks for more focused LLM polishing

# === VALIDATION SETTINGS ===
ANALYSIS_LAYERS = 4
VALIDATION_SIMILARITY_THRESHOLD_HIGH = 0.75
VALIDATION_SIMILARITY_THRESHOLD_PARTIAL = 0.45

# === UI SETTINGS ===
UI_PAGE_TITLE = "DHF Multi-Document Processor"
UI_PAGE_ICON = "🏥"
UI_LAYOUT = "wide"

# === COLOR SCHEME ===
COLOR_PRIMARY = "#C8242F"
COLOR_SECONDARY = "#0F3041"
COLOR_BACKGROUND = "#f2f2f2"

# === LOGGING ===
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def ensure_directories():
    """Create necessary directories if they don't exist"""
    for directory in [INPUTS_DIR, OUTPUTS_DIR, REPORTS_DIR, TEMP_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

# Create directories on import
ensure_directories()
