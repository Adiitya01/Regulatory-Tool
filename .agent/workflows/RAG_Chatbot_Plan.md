---
description: Implementation Plan for RAG-Based Regulatory Chatbot
---

# Implementation Plan: ISO 11135 Regulatory Chatbot (RAG)

## 1. Objective
Create an intelligent "Regulatory Consultant" chatbot capable of:
1.  **Explaining Failures:** Why a specific parameter is marked "missing" in the validation report.
2.  ** prescribing Fixes:** what document or data needs to be added to the DHF.
3.  **Clarifying Standards:** Answering questions about specific ISO 11135 clauses and requirements.

## 2. Technical Architecture

### **The Stack**
*   **Vector Database (Memory):** `ChromaDB` (Local, open-source, file-based).
*   **Embedding Model (Indexer):** `all-MiniLM-L6-v2` (via `sentence-transformers`). fast, local, no API cost.
*   **LLM (Reasoning):** `Qwen/Qwen2.5-72B-Instruct` (via existing Hugging Face API connection).
*   **Backend:** FastAPI (Existing `api.py`).

### **Data Source Strategy**
The RAG system will maintain two distinct "Knowledge Collections":

| Collection Name | Source File | Purpose |
| :--- | :--- | :--- |
| **`iso_standards`** | `polished_regulatory_guidance.txt` | Contains the "Rules". Used to answer "Which clause?" and "What is required?" |
| **`dhf_evidence`** | `DHF_Single_Extraction.txt` + `validation_report.txt` | Contains the "Facts". Used to answer "Why is this missing?" (by seeing what *isn't* there). |

## 3. Detailed Workflow

### **Step 1: Ingestion (Indexing)**
This process runs automatically after the pipeline finishes, or on server startup.

1.  **Clean & Chunk:**
    *   **Standards:** Split `polished_regulatory_guidance.txt` by logical sections (e.g., "Product Requirement", "IQ/OQ/PQ"). Preserve Clause Numbers.
    *   **Reports:** Split `validation_report.txt` by line item (e.g., "❌ [MISSING] Bioburden").
2.  **Embed:** Convert text chunks into vector numbers using `all-MiniLM-L6-v2`.
3.  **Store:** Save vectors into the local ChromaDB folder (`./chroma_db`).

### **Step 2: Retrieval (The "Search")**
When User asks: *"Why is bioburden showing as missing?"*

1.  **Query Generation:** The system searches the **Validation Report** collection for "bioburden".
    *   *Result:* "❌ [MISSING] Bioburden - No matching keywords found."
2.  **Cross-Reference:** The system searches the **ISO Standards** collection for "bioburden requirements".
    *   *Result:* "Clause 7.1: Product shall be designed... documented assessment of bioburden is required."

### **Step 3: Generation (The "Answer")**
The system sends this prompt to Qwen-72B:

```text
CONTEXT FROM VALIDATION REPORT:
"❌ [MISSING] Bioburden - No matching keywords found."

CONTEXT FROM ISO STANDARD:
"Clause 7.1 requires a documented assessment of bioburden consistency..."

USER QUESTION:
"Why is bioburden showing as missing?"

INSTRUCTION:
Explain the failure using the report status, and tell them exactly what document to add based on the standard.
```

## 4. Implementation Steps

### **Phase 1: Setup & Dependencies**
1.  Add `chromadb` and `sentence-transformers` to `requirements.txt`.
2.  Create a new module `Backend/ISO11135_Backend/RAG_Engine.py`.

### **Phase 2: Build the RAG Engine**
1.  **`initialize_db()`**: Sets up the ChromaDB client.
2.  **`ingest_data()`**: reliable function to read `.txt` files in `outputs/` and populate the DB.
3.  **`search()`**: Function to query relevant context.

### **Phase 3: API Integration**
1.  Update `Backend/api.py`.
2.  Add endpoint: `POST /api/chat`.
    *   Accepts: `{ "message": "user question" }`
    *   Returns: `{ "response": "AI answer", "sources": ["Clause 7.1", "Validation Report Line 45"] }`

### **Phase 4: The "Consultant" Persona**
1.  Refine the System Prompt in the LLM Engine to ensure it acts as a helpful, authoritative compliance officer.
2.  Ensure it distinguishes between "I can't find it" (Extraction issue) vs "You didn't do it" (Compliance issue).

## 5. Timeline Estimate
*   **Setup & Ingestion Logic:** ~1 Hour
*   **Retrieval & Chat Endpoint:** ~1 Hour
*   **Testing & Prompt Tuning:** ~1-2 Hours

This plan requires NO new API costs (standard embeddings are local) and leverages your powerful 72B model for the final intelligence.
