import re
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class CategoryMatch:
    category: str
    content: str
    confidence: float
    matched_keywords: List[str]
    pattern_matches: int
    score: float

class CategoryMapper:
    """Configurable category mapper.

    - Accepts a category definition dict with keys:
      { "Category": { "keywords": [...], "patterns": [...], "priority": int }}
    - Provides detailed mapping info with score and confidence.
    - Backward-compatible scoring similar to ISO11135CategoryMapper.
    """

    def __init__(self, category_definitions: Dict[str, Dict[str, List[str]]], default_category: str = "Product Requirement"):
        self.categories = category_definitions
        self.order = list(self.categories.keys())
        self.default_category = default_category if default_category in self.categories else self.order[0]

    def categorize_content_detailed(self, content: str, original_category: Optional[str] = None) -> Dict[str, object]:
        best_category = self.default_category
        best_score = -1
        best_patterns = 0
        best_keywords: List[str] = []

        content_lower = content.lower()

        for target_category, data in self.categories.items():
            score = 0.0
            matched_keywords = []
            pattern_matches = 0

            # Keywords
            for keyword in data.get("keywords", []):
                if keyword.lower() in content_lower:
                    score += 1.0
                    matched_keywords.append(keyword)

            # Patterns (weighted higher)
            for pattern in data.get("patterns", []):
                if re.search(pattern, content, re.IGNORECASE):
                    score += 2.0
                    pattern_matches += 1

            # Original category prior (light boost)
            if original_category and original_category.lower() in target_category.lower():
                score += 0.3

            # Priority tweak
            priority = data.get("priority", 0)
            score += 0.0 if priority is None else 0.0  # placeholder, keep neutral to avoid changing base behaviour

            if score > best_score or (score == best_score and pattern_matches > best_patterns):
                best_score = score
                best_category = target_category
                best_patterns = pattern_matches
                best_keywords = matched_keywords

        confidence = min(max(best_score / 5.0, 0.0), 1.0)  # normalize roughly to [0,1]

        return {
            "category": best_category,
            "score": best_score if best_score >= 0 else 0.0,
            "matched_keywords": best_keywords,
            "pattern_matches": best_patterns,
            "original_category": original_category,
            "content": content,
            "confidence": confidence,
        }

    def categorize_content(self, content: str, original_category: Optional[str] = None) -> str:
        return self.categorize_content_detailed(content, original_category)["category"]
