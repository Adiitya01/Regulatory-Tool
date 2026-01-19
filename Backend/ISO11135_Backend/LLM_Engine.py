import json
import requests
import re
import logging
import json
import os
from typing import List, Dict, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import logging_setup
from . import config

logger = logging_setup.get_logger(__name__)

# --- SETTINGS FROM CENTRAL CONFIG ---
LLM_API_URL = config.LLM_API_URL
LLM_MODEL_NAME = config.LLM_MODEL_NAME
HF_TOKEN = config.HF_TOKEN
LLM_PROVIDER = config.LLM_PROVIDER

EXTRACTED_TEXT_FILE = config.GUIDELINE_EXTRACTION_OUTPUT
POLISHED_OUTPUT_FILE = config.POLISHED_OUTPUT_FILE

# Performance tuning
DEFAULT_CHUNK_SIZE = config.CHUNK_SIZE


# --- ENHANCED 10-CATEGORY MAPPING SYSTEM ---
@dataclass
class CategoryMatch:
    category: str
    content: str
    confidence: float
    matched_keywords: List[str]

class ISO11135CategoryMapper:
    def __init__(self, config_path: str = None, weights: Dict[str, float] = None, enable_numeric_boost: bool = False):
        """Initialize mapper.

        If a JSON config path is provided, load categories from it; otherwise use built-ins.
        Backward compatible: public APIs unchanged.
        """
        # Order matters for tie-breaker (earlier wins if scores & pattern matches equal)
        built_in = {
            "Product Requirement": {
                "keywords": ["product definition", "microbiological quality", "material compatibility", 
                           "product characteristics", "safety requirements", "cleanliness specifications",
                           "sterilization compatibility", "bioburden", "product safety"],
                "patterns": [r"product\s+(?:definition|requirement|characteristic)", 
                           r"microbiological\s+quality", r"material\s+compatibility"]
            },
            
            "Process Validation": {
                "keywords": ["IQ", "OQ", "PQ", "installation qualification", "operational qualification",
                           "performance qualification", "MPQ", "PPQ", "validation process",
                           "microbiological performance", "physical performance"],
                "patterns": [r"\b(?:IQ|OQ|PQ|MPQ|PPQ)\b", 
                           r"(?:installation|operational|performance)\s+qualification"]
            },
            
            "Sterilization Parameter": {
                "keywords": ["preconditioning", "sterilization phase", "aeration", "temperature",
                           "pressure", "humidity", "EO concentration", "exposure time",
                           "gas circulation", "conditioning time", "chamber humidity"],
                "patterns": [r"(?:preconditioning|sterilization|aeration)\s+(?:phase|time|parameters)",
                           r"(?:temperature|pressure|humidity)\s+(?:control|parameters|requirements)"]
            },
            
            "Biological Indicator": {
                "keywords": ["BI", "biological indicator", "overkill method", "spore strip",
                           "spore population", "D-value", "Z-value", "half-cycle",
                           "sterility testing", "spore resistance"],
                "patterns": [r"\bBI\b|biological\s+indicator", r"(?:overkill|half-cycle)\s+method",
                           r"spore\s+(?:strip|population|resistance)"]
            },
            
            "Records/Documents": {
                "keywords": ["documentation", "records", "quality management", "procedures",
                           "parametric release documentation", "traceability", "record keeping",
                           "management responsibility"],
                "patterns": [r"(?:documentation|records|procedures)\s+(?:control|requirements)",
                           r"quality\s+management", r"parametric\s+release"]
            },
            
            "Safety Precaution": {
                "keywords": ["EO safety", "flammable", "carcinogenic", "sub-atmospheric pressure",
                           "residual EO limits", "personnel safety", "equipment safety",
                           "below 6 psia", "EO exposure"],
                "patterns": [r"(?:EO|ethylene\s+oxide)\s+safety", r"(?:flammable|carcinogenic)",
                           r"sub-atmospheric\s+pressure", r"personnel\s+safety"] 
            },
            
            "Test": {
                "keywords": ["sterility test", "ToS", "sublethal cycle", "calibration testing",
                           "equipment performance testing", "BI testing", "PCD testing",
                           "analytical testing"],
                "patterns": [r"(?:sterility|ToS)\s+test", r"sublethal\s+cycle\s+testing",
                           r"calibration\s+testing", r"(?:BI|PCD)\s+testing"]
            },
            
            "Monitoring": {
                "keywords": ["routine monitoring", "physical parameter monitoring", "temperature monitoring",
                           "pressure monitoring", "humidity monitoring", "EO concentration monitoring",
                           "chamber atmosphere analysis", "continuous monitoring"],
                "patterns": [r"(?:routine|continuous)\s+monitoring", 
                           r"(?:temperature|pressure|humidity)\s+monitoring"]
            },
            
            "Acceptance Criteria": {
                "keywords": ["acceptance criteria", "parametric release criteria", "SAL requirements",
                           "sterility assurance level", "load configuration specifications",
                           "predetermined criteria", "tolerance limits"],
                "patterns": [r"acceptance\s+criteria", r"parametric\s+release\s+criteria",
                           r"SAL\s+requirements", r"tolerance\s+limits"]
            },
            
            "Storage & Distribution": {
                "keywords": ["product release", "post-sterilization handling", "aeration requirements",
                           "sterile product storage", "storage conditions", "traceability procedures",
                           "distribution", "release from sterilization"],
                "patterns": [r"product\s+release", r"post-sterilization\s+handling",
                           r"sterile\s+product\s+storage", r"storage\s+conditions"]
            }
        }

        # Attempt to load config if provided
        self.target_categories = built_in
        # Scoring weights (opt-in). Defaults preserve existing behaviour.
        self.w_kw = 1.0
        self.w_pat = 2.0
        self.w_orig = 0.0
        self.w_num = 0.0
        self.enable_numeric_boost = bool(enable_numeric_boost)
        if isinstance(weights, dict):
            self.w_kw = float(weights.get("keyword", self.w_kw))
            self.w_pat = float(weights.get("pattern", self.w_pat))
            self.w_orig = float(weights.get("original", self.w_orig))
            self.w_num = float(weights.get("numeric", self.w_num))
        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                    if isinstance(data, dict) and data:
                        # Validate minimal schema
                        valid = all(isinstance(v, dict) for v in data.values())
                        if valid:
                            self.target_categories = {k: {
                                "keywords": v.get("keywords", []),
                                "patterns": v.get("patterns", [])
                            } for k, v in data.items()}
                            logger.info("Loaded category config: %s", config_path)
            except Exception as e:
                logger.warning("Failed to load category config %s: %s. Using built-ins.", config_path, e)
    
    def categorize_content(self, content: str, original_category: str) -> str:
        """Backward compatibility wrapper returning just the category"""
        detailed = self.categorize_content_detailed(content, original_category)
        return detailed["category"]

    def categorize_content_detailed(self, content: str, original_category: str) -> Dict[str, object]:
        """Return detailed mapping info including score, matched keywords, pattern match count.
        Tie-breaker: higher score wins. On score tie prefer more pattern matches. If still tie, keep
        earlier category order in self.target_categories.
        """
        best_category = "Product Requirement"
        best_score = -1  # allow 0 score categories to override default if first matched
        best_patterns = 0
        best_keywords: List[str] = []

        content_lower = content.lower()

        for target_category, data in self.target_categories.items():
            score = 0
            matched_keywords = []
            pattern_matches = 0

            # Keyword matching (case-insensitive, simple containment)
            for keyword in data["keywords"]:
                if keyword.lower() in content_lower:
                    score += self.w_kw
                    matched_keywords.append(keyword)

            # Pattern matching (weighted higher)
            for pattern in data["patterns"]:
                if re.search(pattern, content, re.IGNORECASE):
                    score += self.w_pat
                    pattern_matches += 1

            # Optional: original category prior (light boost)
            if self.w_orig and original_category and original_category.lower() in target_category.lower():
                score += self.w_orig

            # Optional: numeric evidence boost
            if self.enable_numeric_boost and self.w_num:
                if re.search(r"\d+\s*(?:%|°C|°F|kPa|Pa|bar|ppm|mg/L|minutes?|hours?|sec)", content, re.IGNORECASE):
                    score += self.w_num

            # Apply tie-breaker logic
            if score > best_score:
                best_score = score
                best_category = target_category
                best_patterns = pattern_matches
                best_keywords = matched_keywords
            elif score == best_score:
                if pattern_matches > best_patterns:
                    best_category = target_category
                    best_patterns = pattern_matches
                    best_keywords = matched_keywords
                # If pattern matches equal, earlier category (existing best) is kept implicitly

        return {
            "category": best_category,
            "score": best_score if best_score >= 0 else 0,
            "matched_keywords": best_keywords,
            "pattern_matches": best_patterns,
            "original_category": original_category,
            "content": content
        }

# --- ENHANCED LOAD AND PARSE WITH 10-CATEGORY MAPPING ---
def load_raw_extracted_text(preserve_history: bool = False) -> Dict[str, List[str]]:
    """Load, parse, and map to 10 target categories.

    Supports BOTH legacy format ("Sterilization Parameters:" headers + emoji/bullet lines) and
    the newer structured format with CATEGORY:/PARAMETER blocks. Returns
    target_categories dict unchanged vs previous versions.
    """
    try:
        extracted_path = Path(EXTRACTED_TEXT_FILE)
        if not extracted_path.exists():
            logging.error(f"❌ Extracted text file not found: {EXTRACTED_TEXT_FILE}")
            return {}

        content = extracted_path.read_text(encoding="utf-8")
        mapper = ISO11135CategoryMapper()
        target_categories = {category: [] for category in mapper.target_categories.keys()}

        lines = content.splitlines()

        # --- First attempt: NEW format parsing ---
        original_categories: Dict[str, List[str]] = {}
        current_category = None
        current_items: List[str] = []

        i = 0
        while i < len(lines):
            raw_line = lines[i]
            line_stripped = raw_line.strip()

            if not line_stripped or line_stripped.startswith("=" * 10) or line_stripped.startswith("#"):
                i += 1
                continue

            if line_stripped.startswith("CATEGORY:"):
                if current_category and current_items:
                    original_categories[current_category] = current_items
                current_category = line_stripped.replace("CATEGORY:", "").strip()
                current_items = []
                i += 1
                continue

            if line_stripped.startswith("┌─ PARAMETER"):
                keyword = None
                context_text = None
                j = i + 1
                while j < len(lines):
                    scan = lines[j].strip()
                    if scan.startswith("└─"):
                        break
                    if scan.startswith("Keyword:"):
                        keyword = scan.replace("Keyword:", "").strip()
                    if scan.startswith("Context:"):
                        ctx_lines = []
                        k = j + 1
                        while k < len(lines):
                            ctx_line = lines[k].strip()
                            if ctx_line.startswith("└─"):
                                break
                            cleaned = ctx_line.replace("│", "").strip()
                            if cleaned:
                                ctx_lines.append(cleaned)
                            k += 1
                        context_text = " ".join(ctx_lines).strip()
                        break
                    j += 1
                if keyword and context_text:
                    current_items.append(f"{keyword}: {context_text}")
                i = j + 1
                continue

            i += 1

        if current_category and current_items:
            original_categories[current_category] = current_items

        # --- If NEW format produced nothing, attempt a more robust regex-based parse ---
        if not original_categories:
            logging.debug("No new-format categories found; attempting enhanced regex fallback parse.")
            try:
                # Capture CATEGORY blocks: (category name, block text)
                cat_blocks = re.findall(r"CATEGORY:\s*(.+?)\n([\s\S]*?)(?=\nCATEGORY:|\Z)", content, flags=re.IGNORECASE)
                if cat_blocks:
                    logging.debug(f"Found {len(cat_blocks)} CATEGORY blocks via regex fallback")
                    for cat_name, block in cat_blocks:
                        items = []
                        # find parameter blocks within the category block
                        for m in re.finditer(r"┌─ PARAMETER[\s\S]*?Keyword:\s*(.+?)\n[\s\S]*?Context:\s*([\s\S]*?)(?=\n└─|\Z)", block, flags=re.IGNORECASE):
                            keyword = m.group(1).strip()
                            raw_ctx = m.group(2)
                            # clean context: remove box-drawing vertical bars and excess whitespace
                            lines_ctx = []
                            for cl in raw_ctx.splitlines():
                                clc = cl.replace('│', '').replace('\r', '').strip()
                                if clc:
                                    lines_ctx.append(clc)
                            context_text = ' '.join(lines_ctx).strip()
                            if keyword and context_text:
                                items.append(f"{keyword}: {context_text}")
                        if items:
                            original_categories[cat_name.strip()] = items
            except Exception as e:
                logging.debug(f"Regex fallback parse failed: {e}")

            # --- If regex fallback still produced nothing, fallback to LEGACY format ---
            if not original_categories:
                logging.debug("Regex fallback produced nothing; attempting legacy format parse.")
                legacy_categories: Dict[str, List[str]] = {}
                legacy_current_cat = None
                legacy_items: List[str] = []
                header_regex = re.compile(r"^(.*? Parameters:)\s*$")  # matches 'XYZ Parameters:'
                emoji_prefixes = ("🎯", "📍", "🔢")

                def flush_legacy():
                    nonlocal legacy_current_cat, legacy_items
                    if legacy_current_cat and legacy_items:
                        legacy_categories[legacy_current_cat.replace(" Parameters:", "")] = legacy_items
                    legacy_items = []

                for line in lines:
                    ls = line.strip()
                    if not ls or set(ls) <= {"-"}:  # skip blank or dash-only lines
                        continue
                    header_match = header_regex.match(ls)
                    if header_match:
                        flush_legacy()
                        legacy_current_cat = header_match.group(1)
                        continue
                    if legacy_current_cat:
                        # Item lines
                        if ls.startswith(emoji_prefixes) or ls.startswith("-"):
                            item_line = ls
                            # Remove leading emoji/bullet marker
                            item_line = re.sub(r"^[🎯📍🔢\-\*]+\s*", "", item_line)
                            legacy_items.append(item_line)
                flush_legacy()
                original_categories = legacy_categories
                logging.debug(f"Legacy format categories parsed: {list(original_categories.keys())}")

        # Deduplicate items per original category (exact match, preserve order)
        for cat, items in list(original_categories.items()):
            seen = set()
            deduped = []
            for it in items:
                if it not in seen:
                    seen.add(it)
                    deduped.append(it)
            original_categories[cat] = deduped

        logging.debug(f"Parsed original categories count: {len(original_categories)}")

        # --- Mapping to target categories with detailed scoring ---
        mapping_log = []
        for orig_category, items in original_categories.items():
            for item in items:
                info = mapper.categorize_content_detailed(item, orig_category)
                target_categories[info["category"]].append(item)
                mapping_log.append({
                    "original_category": orig_category,
                    "target_category": info["category"],
                    "content_preview": item[:100] + ("..." if len(item) > 100 else ""),
                    "score": info["score"],
                    "matched_keywords": info["matched_keywords"],
                    "pattern_matches": info.get("pattern_matches", 0),
                    "confidence": info.get("confidence", 0.0)
                })

        # --- Atomic write of mapping log ---
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        log_file = outputs_dir / "category_mapping_log.json"
        tmp_file = log_file.with_suffix(".tmp")
        try:
            with tmp_file.open("w", encoding="utf-8") as fh:
                json.dump(mapping_log, fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            tmp_file.replace(log_file)
            # Also write a compatibility copy at the repository root (original behaviour)
            try:
                root_log = Path.cwd() / "category_mapping_log.json"
                with (root_log.with_suffix('.tmp')).open('w', encoding='utf-8') as rf:
                    json.dump(mapping_log, rf, ensure_ascii=False, indent=2)
                    rf.flush()
                    os.fsync(rf.fileno())
                (root_log.with_suffix('.tmp')).replace(root_log)
            except Exception:
                # Non-fatal: log but continue
                logging.debug("Failed to write root-level category_mapping_log.json (non-fatal)")
            if preserve_history:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup = outputs_dir / f"category_mapping_log_{timestamp}.json"
                backup.write_text(json.dumps(mapping_log, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:  # log but do not raise
            logging.error(f"Failed to write mapping log atomically: {e}")

        # --- Console summary ---
        print("\n10-Category Mapping Results:")
        total_mapped = 0
        for category, items in target_categories.items():
            if items:
                print(f"[OK] {category}: {len(items)} items")
                total_mapped += len(items)
            else:
                print(f"[MISSING] {category}: No items mapped")

        logging.info(
            f"Mapped {total_mapped} items into {len([c for c,i in target_categories.items() if i])} categories"
        )
        return target_categories
    except Exception as e:
        logging.error(f"Failed to load and map extracted text: {e}")
        return {}

# --- ENHANCED PROMPT BUILDER ---
def build_polishing_prompt(raw_content: List[str], category_name: str) -> str:
    """Build enhanced prompt with 10-category focus"""
    
    prompt = f"""You are an expert regulatory compliance writer specializing in ISO 11135 standard for ethylene oxide sterilization.

Your task is to transform the raw extracted regulatory content into professional, polished regulatory guidance for the category: **{category_name}**

INSTRUCTIONS:
1. Focus specifically on {category_name} requirements from ISO 11135
2. Rewrite content into clear, authoritative regulatory language
3. Structure information logically with proper formatting
4. Preserve all numerical values, ranges, and technical requirements exactly
5. Use formal regulatory terminology (shall, must, required, specified, etc.)
6. Create clear headings and organize related requirements together
7. Add professional context about compliance importance for {category_name}
8. Make it comprehensive and audit-ready

CATEGORY FOCUS: {category_name}

RAW CONTENT TO POLISH:
{chr(10).join(f"• {item}" for item in raw_content)}

Transform this into professional {category_name} regulatory guidance:"""

    return prompt

# --- EMERGENCY BYPASS FUNCTION ---
def emergency_polish_categories(raw_categories: Dict[str, List[str]]) -> str:
    """Emergency processing - NO LLM calls, pure formatting"""
    
    logging.error("🚨 EMERGENCY MODE ACTIVATED - Bypassing LLM entirely")
    
    polished_sections = []
    
    # Professional header
    header = f"""# 📋 **ISO 11135 Professional Regulatory Compliance Guide**
*Emergency Processing Mode - Direct Formatting Applied*

**Document Source**: ISO 11135:2014+A1:2019 - Sterilization of health care products — Ethylene oxide  
**Processing Mode**: Emergency Bypass (LLM Unavailable)  
**Total Parameter Categories**: {len(raw_categories)}  
**Total Requirements**: {sum(len(items) for items in raw_categories.values())}

---

"""
    polished_sections.append(header)
    
    for i, (category, raw_items) in enumerate(raw_categories.items(), 1):
        logging.info(f"⚡ Emergency processing category {i}/{len(raw_categories)}: {category}")
        
        # Create professional section without LLM
        section = f"""
## 🔹 **{category.title()} Compliance Requirements**

### **Regulatory Overview**
The following {len(raw_items)} requirements are **mandatory** per ISO 11135:2014+A1:2019 for {category.lower()} control in ethylene oxide sterilization processes.

### **Specific Requirements**

"""
        
        # Format each item professionally
        for idx, item in enumerate(raw_items, 1):
            # Clean and format the item
            clean_item = item.strip()
            if ':' in clean_item:
                param, description = clean_item.split(':', 1)
                section += f"""**{idx}. {param.strip()}**  
   • **Requirement**: {description.strip()}  
   • **Compliance**: Mandatory per ISO 11135  
   • **Validation**: Must be documented and verified

"""
            else:
                section += f"""**{idx}. {clean_item}**  
   • **Compliance**: Mandatory per ISO 11135  
   • **Validation**: Must be documented and verified

"""
        
        section += f"""
### **Category Summary**
- **Total Parameters**: {len(raw_items)}
- **Compliance Level**: Mandatory
- **Standard Reference**: ISO 11135:2014+A1:2019
- **Audit Requirement**: Full documentation required

---

"""
        polished_sections.append(section)
        
    # Professional footer
    footer = f"""
## 📊 **Regulatory Compliance Summary**

This emergency-processed document contains **{sum(len(items) for items in raw_categories.values())} regulatory requirements** across **{len(raw_categories)} compliance categories**.

### **Critical Compliance Areas**
{chr(10).join(f'• **{category.title()}**: {len(items)} mandatory requirements' for category, items in raw_categories.items())}

### **Emergency Processing Notes**
- All numerical specifications preserved exactly as extracted
- Professional formatting applied using regulatory standards
- Each requirement carries mandatory compliance status
- Full audit trail documentation required for all parameters

### **Next Steps**
1. **Immediate Use**: This document is audit-ready for regulatory submissions
2. **Technical Review**: Verify all numerical values against source standard
3. **Process Integration**: Implement requirements in QMS procedures
4. **Validation**: Complete IQ/OQ/PQ validation per requirements

---

**Document Status**: Emergency Processing Complete  
**Quality Level**: Professional/Audit-Ready  
**Processing Time**: Immediate (LLM Bypass)  
**Regulatory Standard**: ISO 11135:2014+A1:2019

"""
    
    polished_sections.append(footer)
    
    return "".join(polished_sections)

# --- LLM CALL FUNCTION ---
def call_llama(prompt: str, max_retries: int = 3, system_message: str = None) -> str:
    """Call LLM model with retry and reconnection logic"""
    
    if system_message is None:
        system_message = "You are a professional regulatory compliance assistant. You write clear, structured, and authoritative guidance from raw notes."

    for attempt in range(max_retries):
        try:
            # Test connection before attempting call (on retries)
            if attempt > 0:
                logging.info(f"Retry attempt {attempt + 1}/{max_retries} - testing LLM connection...")
                ok, info = test_llm_connection(timeout=3)
                if not ok:
                    logging.warning(f"LLM connection test failed: {info}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(2 * (attempt + 1))  # Exponential backoff
                        continue
                else:
                    logging.info(f"LLM connection re-established: {info}")
            
            payload = {
                "model": LLM_MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": config.LLM_TEMPERATURE,
                "max_tokens": config.LLM_MAX_TOKENS,
                "top_p": config.LLM_TOP_P,
                "frequency_penalty": config.LLM_FREQUENCY_PENALTY
            }
            
            headers = {}
            if HF_TOKEN:
                headers["Authorization"] = f"Bearer {HF_TOKEN}"
     
            response = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=config.LLM_REQUEST_TIMEOUT)
            if response.status_code == 200:
                # Support both chat-style and completions-style responses
                data = response.json()
                # chat-completions: choices[0].message.content
                if "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    if isinstance(choice.get("message"), dict) and "content" in choice["message"]:
                        result = choice["message"]["content"].strip()
                    elif "text" in choice:
                        result = choice["text"].strip()
                    else:
                        result = json.dumps(choice)
                else:
                    result = json.dumps(data)
                result = re.sub(r'\n{3,}', '\n\n', result)
                return result
            else:
                logging.error(f"❌ LLM API returned status code: {response.status_code}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1)
                    continue
                return f"❌ [LLM ERROR] API returned status {response.status_code}"
     
        except requests.exceptions.Timeout:
            logging.error(f"❌ LLM API timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                import time
                time.sleep(2)
                continue
            return "❌ [LLM ERROR] Request timeout - LM Studio may be overloaded"
        except requests.exceptions.ConnectionError:
            provider_name = "Hugging Face" if LLM_PROVIDER == "hf" else "LM Studio"
            logging.error(f"❌ Cannot connect to {provider_name} API (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                import time
                time.sleep(3)
                continue
            return f"❌ [LLM ERROR] Cannot connect to {provider_name}. Please ensure it is active."
        except Exception as e:
            logging.error(f"❌ LLM API Error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(1)
                continue
            return f"❌ [LLM ERROR] {str(e)}"
    
    return "❌ [LLM ERROR] Max retries exceeded"


def test_llm_connection(timeout: int = 5) -> Tuple[bool, str]:
    """Quickly test LLM connectivity (local or HF)"""
    if LLM_PROVIDER == "hf":
        if not HF_TOKEN:
            return False, "Hugging Face Token (HF_TOKEN) is missing in environment."
        return True, f"HF Mode: Model {LLM_MODEL_NAME} will be used on call."

    models_url = f"{config.LLM_API_BASE.rstrip('/')}/v1/models"

    try:
        resp = requests.get(models_url, timeout=timeout)
        if resp.status_code == 200:
            try:
                content = resp.json()
                # Provide basic model listing or acknowledgement
                models = content.get("models") if isinstance(content, dict) else None
                if models:
                    return True, f"LM Studio reachable. Models: {', '.join(m.get('id','<unknown>') for m in models[:5])}"
                return True, "LM Studio reachable (models endpoint responded)."
            except Exception:
                return True, "LM Studio reachable (non-JSON response)."
        else:
            return False, f"Models endpoint returned status {resp.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {e}"

def remove_exact_duplicates(lines: List[str]) -> List[str]:
    seen = set()
    cleaned = []
    for line in lines:
        line_clean = line.strip().lower()  # Normalize case
        if line_clean and line_clean not in seen:
            seen.add(line_clean)
            cleaned.append(line)
    return cleaned

# --- BATCH PROCESSING FUNCTION ---
def polish_all_categories(raw_categories: Dict[str, List[str]]) -> str:
    """Process all categories and create polished output with retry and emergency fallback"""
    
    polished_sections = []
    total_categories = len(raw_categories)
    
    # Test LLM connection before starting
    ok, info = test_llm_connection()
    if not ok:
        logging.warning(f"Initial LLM connection test failed: {info}")
        logging.info("Attempting to re-establish LLM connection...")
        import time
        time.sleep(2)
        ok, info = test_llm_connection()
        if not ok:
            logging.error(f"Failed to re-establish LLM connection: {info}")
            logging.error("Falling back to emergency bypass mode")
            return emergency_polish_categories(raw_categories)
        else:
            logging.info(f"LLM connection re-established successfully: {info}")
    else:
        logging.info(f"LLM connection verified: {info}")
    
    # Add professional header
    header = f"""# 📋 **ISO 11135 Professional Regulatory Compliance Guide**
*Comprehensive Ethylene Oxide Sterilization Requirements & Guidelines*

**Document Source**: ISO 11135:2014+A1:2019 - Sterilization of health care products — Ethylene oxide  
**Total Parameter Categories**: {total_categories}  
**Total Requirements**: {sum(len(items) for items in raw_categories.values())}

---

"""
    polished_sections.append(header)
    
    for i, (category, raw_items) in enumerate(raw_categories.items(), 1):
        if not raw_items:
            continue
            
        logging.info(f"🔄 Processing category {i}/{total_categories}: {category}")
        
        # Adaptive chunk size: larger chunks for simpler content, smaller for complex
        # Use environment-configurable default, or adaptive sizing based on content
        chunk_size = DEFAULT_CHUNK_SIZE
        
        # For categories with many items, use slightly larger chunks for efficiency
        if len(raw_items) > 30:
            chunk_size = min(12, DEFAULT_CHUNK_SIZE + 2)
        
        category_content = []
        llm_failed = False
        
        # Process chunks with simple parallel processing (if many chunks)
        chunks = [(chunk_start, raw_items[chunk_start:chunk_start + chunk_size]) 
                  for chunk_start in range(0, len(raw_items), chunk_size)]
        
        # For small categories (<=2 chunks), process sequentially
        # For larger categories, we could add parallel processing here if needed
        for chunk_idx, (chunk_start, chunk) in enumerate(chunks, 1):
            logging.info(f"   📝 Processing chunk {chunk_idx}/{len(chunks)} ({len(chunk)} items)")
            
            # Build prompt and get polished content
            prompt = build_polishing_prompt(chunk, category)
            polished_chunk = call_llama(prompt)
            
            # Check for errors
            if "❌ [LLM ERROR]" in polished_chunk:
                logging.warning(f"⚠️ LLM failed for {category}. Switching to emergency formatting.")
                llm_failed = True
                break
            else:
                category_content.append(polished_chunk)
        
        # If LLM failed, use emergency formatting for this category
        if llm_failed:
            fallback = emergency_polish_categories({category: raw_items})
            polished_sections.append(fallback)
            logging.info(f"✅ Completed {category} ({len(raw_items)} items) - Emergency Mode")
            continue

        # Combine all chunks for this category (only if LLM succeeded)
        if category_content:
            full_category_content = f"\n## 🔹 **{category.title()} Compliance Requirements**\n\n"
            raw_combined = "\n\n".join(category_content)

            lines = raw_combined.split('\n')
            deduped_lines = remove_exact_duplicates(lines)
            deduped_content = "\n".join(deduped_lines)

            full_category_content += deduped_content
            full_category_content += f"\n\n*({len(raw_items)} requirements processed)*\n\n---\n"
            
            polished_sections.append(full_category_content)
            logging.info(f"✅ Completed {category} ({len(raw_items)} items) - LLM Enhanced")
    
    # Add professional footer
    footer = f"""
## 📊 **Document Summary**

This comprehensive regulatory guidance document has been professionally compiled from ISO 11135:2014+A1:2019 standard requirements. All numerical specifications, ranges, and technical requirements have been preserved while enhancing readability and professional presentation.

**Key Compliance Areas Covered:**
{chr(10).join(f'• {category.title()}' for category in raw_categories.keys())}

**Usage Notes:**
- All requirements marked with specific values must be strictly followed
- Regular validation and monitoring per ISO requirements is mandatory  
- Consult original ISO 11135 standard for complete regulatory context
- This document serves as a professional summary for compliance planning

---
**Generated**: Professional Regulatory Compliance Documentation System  
**Standard Reference**: ISO 11135:2014+A1:2019  
**Quality Level**: Professional/Audit-Ready
"""
    
    polished_sections.append(footer)
    
    return "".join(polished_sections)

# --- SAVE POLISHED OUTPUT ---
def save_polished_output(polished_content: str, file_path: str = None) -> None:
    """Save the polished content to output file"""
    # Determine final output path. Default to `outputs/POLISHED_OUTPUT_FILE` if not provided.
    if file_path:
        output_path = Path(file_path)
    else:
        output_dir = Path.cwd() / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / POLISHED_OUTPUT_FILE

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            # Add professional header
            f.write("#" * 80 + "\n")
            f.write("#" + " " * 78 + "#\n")
            f.write("#" + " POLISHED REGULATORY GUIDANCE".center(78) + "#\n")
            f.write("#" + " ISO 11135 Requirements - LLM Enhanced".center(78) + "#\n")
            f.write("#" + " " * 78 + "#\n")
            f.write("#" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("┌" + "─" * 78 + "\n")
            f.write("│ This document contains polished regulatory requirements extracted from\n")
            f.write("│ ISO 11135 standard and enhanced using AI language models for clarity.\n")
            f.write("└" + "─" * 78 + "\n\n")

            # Write the actual polished content
            f.write(polished_content)

        file_size = Path(output_path).stat().st_size / 1024  # KB
        logger.info("✅ Polished content saved to %s (%.1f KB)", output_path, file_size)

    except Exception as e:
        logger.exception("❌ Failed to save polished output: %s", e)

# --- MAIN EXECUTION ---
def main():
    """Main execution function"""
    print("\n" + "#" * 70)
    print("#" + " ENHANCED REGULATORY TEXT POLISHING ENGINE".center(68) + "#")
    print("#" + " Transform Raw Extractions into Professional Guidance".center(68) + "#")
    print("#" * 70 + "\n")
    
    # Quick check: LM Studio connectivity
    ok, info = test_llm_connection()
    if ok:
        print(f"✅ LM Studio reachable — {info}")
    else:
        print(f"⚠️ LM Studio connectivity warning: {info}")

    # Step 1: Load raw extracted text with 10-category mapping
    print("\n📂 Loading and mapping raw extracted text...")
    raw_categories = load_raw_extracted_text()
    
    if not raw_categories:
        print("❌ No content loaded. Please check your extracted text file.")
        return
    
    print(f"✅ Loaded and mapped to {len([cat for cat, items in raw_categories.items() if items])} categories")
    for category, items in raw_categories.items():
        if items:
            print(f"   • {category}: {len(items)} items")
    
    # Step 2: Process and polish content
    print(f"\n🔄 Starting polishing process...")
    print("⏱️  This may take several minutes depending on content size...")
    
    try:
        polished_content = polish_all_categories(raw_categories)
        
        # Step 3: Save results
        print(f"\n💾 Saving polished content...")
        save_polished_output(polished_content)
        
        # Step 4: Show preview
        print(f"\n✅ **PROCESSING COMPLETE!**")
        print(f"📄 Output saved to: {POLISHED_OUTPUT_FILE}")
        print(f"📊 Total content: {len(polished_content):,} characters")
        
        # Show preview
        print(f"\n📖 **Preview of polished content:**")
        print("-" * 50)
        preview_lines = polished_content.split('\n')[:15]
        print('\n'.join(preview_lines))
        if len(polished_content.split('\n')) > 15:
            print("...")
            print(f"[+{len(polished_content.split('\n')) - 15} more lines in file]")
        print("-" * 50)
        
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        logging.error(f"Main processing error: {e}")

# --- CLI INTERFACE ---
if __name__ == "__main__":
    main()
