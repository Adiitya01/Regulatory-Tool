
import os
import requests
import logging
import chromadb
from typing import List, Dict, Any, Optional
from pathlib import Path
from . import config
from .storage_manager import storage

# Setup logging
logger = logging.getLogger(__name__)

class LightweightRAGEngine:
    """
    Render-Friendly RAG Engine compatible with ChromaDB 0.3.x
    Uses Hugging Face API for embeddings (Serverless) + ChromaDB (In-Memory).
    """


    def __init__(self):
        # 1. Initialize ChromaDB in Ephemeral (Memory) mode
        self.chroma_client = chromadb.EphemeralClient()
        self.embedding_model = None
            
        # 2. Create Collections
        self.standards_collection = self.chroma_client.get_or_create_collection(name="iso_standards")
        self.evidence_collection = self.chroma_client.get_or_create_collection(name="dhf_evidence")
        
        logger.info("✅ RAG Engine Initialized (Lazy Loading Enabled)")

    def _get_embedding(self, text: str) -> List[float]:
        """Lazy-load model and get embedding"""
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("📥 Loading SentenceTransformer (Lazy Load)...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer: {e}")
                return [0.0] * 384
        
        try:
            return self.embedding_model.encode(text).tolist()
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return [0.0] * 384

    def ingest_data(self):
        """Read output files and index them into ChromaDB."""
        logger.info("🔄 Starting RAG Ingestion...")
        
        self._ingest_standard()
        self._ingest_evidence()
        
        logger.info(f"✅ Ingestion Complete. Standards: {self.standards_collection.count()}, Evidence: {self.evidence_collection.count()}")

    def _ingest_standard(self):
        """Ingest the Polished Regulatory Guidance (The Rules)"""
        file_path = storage.ensure_local(config.POLISHED_OUTPUT_FILE)
        if not file_path:
            logger.warning(f"Standard file not found in local or cloud: {config.POLISHED_OUTPUT_FILE}")
            return

        content = file_path.read_text(encoding="utf-8")
        
        # Simple splitting by "Category" headers (Markdown ##)
        chunks = content.split("## 🔹")
        
        documents = []
        metadatas = []
        ids = []
        embeddings = []

        for i, chunk in enumerate(chunks):
            if not chunk.strip(): continue
            
            # Clean up the chunk
            text = "## " + chunk if i > 0 else chunk
            text = text[:2000]  # Limit chunk size
            
            # Extract title for metadata
            lines = text.strip().split('\n')
            title = lines[0].replace('#', '').strip()
            
            documents.append(text)
            metadatas.append({"source": "ISO 11135 Guidance", "section": title})
            ids.append(f"iso_{i}")
            
            # Get embedding explicitly
            embedding = self._get_embedding(text)
            embeddings.append(embedding)

        if documents:
            self.standards_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )

    def _ingest_evidence(self):
        """Ingest extracted DHF data and Validation Reports (The Facts)"""
        report_path = storage.ensure_local(config.VALIDATION_REPORT)
        if not report_path:
            logger.warning(f"Validation report not found in local or cloud: {config.VALIDATION_REPORT}")
            return
            
        content = report_path.read_text(encoding="utf-8")
        lines = content.split('\n')
        
        documents = []
        metadatas = []
        ids = []
        embeddings = []
        
        for i, line in enumerate(lines):
            if "❌" in line or "⚠️" in line or "Missing" in line or "MISSING" in line:
                # Capture context (line itself + surrounding lines)
                start = max(0, i-1)
                end = min(len(lines), i+2)
                context_chunk = "\n".join(lines[start:end])
                
                documents.append(context_chunk)
                metadatas.append({"source": "Validation Report", "type": "Failure"})
                ids.append(f"val_{i}")
                
                # Get embedding explicitly
                embedding = self._get_embedding(context_chunk)
                embeddings.append(embedding)
        
        if documents:
            self.evidence_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )

    def retrieve_context(self, query: str) -> str:
        """Finds the most relevant facts and rules for the user's query."""
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # 1. Search Evidence (What happened in the user's files?)
        try:
            evidence_results = self.evidence_collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )
        except Exception as e:
            logger.error(f"Evidence query error: {e}")
            evidence_results = {'documents': [[]]}
        
        # 2. Search Standards (What rules apply?)
        try:
            rules_results = self.standards_collection.query(
                query_embeddings=[query_embedding],
                n_results=2
            )
        except Exception as e:
            logger.error(f"Standards query error: {e}")
            rules_results = {'documents': [[]]}
        
        # 3. Format Context String for the LLM
        context = "### DHF EVIDENCE (USER'S FILE STATUS):\n"
        if evidence_results['documents'] and len(evidence_results['documents'][0]) > 0:
            for doc in evidence_results['documents'][0]:
                context += f"- {doc}\n"
        else:
            context += "- No specific evidence found\n"
            
        context += "\n### ISO 11135 STANDARDS (REGULATORY RULES):\n"
        if rules_results['documents'] and len(rules_results['documents'][0]) > 0:
            for doc in rules_results['documents'][0]:
                context += f"- {doc}\n"
        else:
            context += "- No specific standards found\n"
            
        return context

# Singleton
_rag_engine = None

def get_rag_engine():
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = LightweightRAGEngine()
    return _rag_engine
