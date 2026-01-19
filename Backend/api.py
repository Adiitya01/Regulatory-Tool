"""
FastAPI Backend for DHF Multi-Document Processor
Provides REST API endpoints for all processing functions
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import sys
from pathlib import Path
import shutil
import json
import asyncio
from datetime import datetime
import logging

# Add Backend directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import backend modules
from ISO11135_Backend.Guideline_Extractor import extract_pdf_content, extract_parameters_with_context, save_results_to_text
from ISO11135_Backend.LLM_Engine import load_raw_extracted_text, polish_all_categories, save_polished_output, test_llm_connection
from ISO11135_Backend.DHF_Extractor import extract_single_pdf
from ISO11135_Backend.validation import EnhancedMultiLayerValidationEngine
from ISO11135_Backend.pipeline import DHFPipeline, PipelineResult, PipelineStep
from ISO11135_Backend.RAG_Engine import get_rag_engine
import ISO11135_Backend.config as iso11135_config
from ISO11135_Backend.storage_manager import storage
import logging_setup

logger = logging_setup.get_logger(__name__)

# Load guidelines configuration
def load_guidelines_config():
    """Load guidelines configuration from JSON"""
    config_path = Path(__file__).parent / "guidelines_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {"guidelines": [], "default_guideline": "iso11135"}

guidelines_config = load_guidelines_config()

class ChatRequest(BaseModel):
    message: str

# Initialize FastAPI app
app = FastAPI(
    title="DHF Document Validator API",
    description="Multi-guideline regulatory compliance document processing system",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
iso11135_config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ==================== Pydantic Models ====================

class StatusResponse(BaseModel):
    status: str
    message: str
    timestamp: str

class ProcessingStatus(BaseModel):
    status: str  # "pending", "processing", "completed", "error"
    progress: int  # 0-100
    message: str
    result: Optional[Dict] = None

class FileInfo(BaseModel):
    filename: str
    exists: bool
    size: Optional[int] = None
    modified: Optional[str] = None

class LLMConnectionStatus(BaseModel):
    connected: bool
    message: str
    model: Optional[str] = None

class PipelineCompletionStatus(BaseModel):
    step: str
    name: str
    status: str  # "completed", "pending"
    file_exists: bool
    filename: str
    description: str
    prerequisites_met: bool
    message: str
    file_info: Optional[Dict] = None

# ==================== Helper Functions ====================

def get_file_info(filename: str, guideline_id: str = "iso11135") -> FileInfo:
    """Get information about a file - check local and cloud storage"""
    # 1. Check if file exists (Local or Cloud)
    if storage.exists(filename):
        # If cloud, we don't necessarily have local stat
        local_path = iso11135_config.OUTPUTS_DIR / filename
        if local_path.exists():
            stat = local_path.stat()
            return FileInfo(
                filename=filename,
                exists=True,
                size=stat.st_size,
                modified=datetime.fromtimestamp(stat.st_mtime).isoformat()
            )
        else:
            # Cloud exists, but no local details
            return FileInfo(
                filename=filename,
                exists=True,
                size=0,
                modified=datetime.now().isoformat()
            )
    return FileInfo(filename=filename, exists=False)

# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "DHF Document Validator API",
        "version": "2.0.0",
        "status": "running"
    }

@app.get("/api/guidelines")
async def get_available_guidelines():
    """Get list of available guidelines"""
    return guidelines_config

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return StatusResponse(
        status="healthy",
        message="API is running",
        timestamp=datetime.now().isoformat()
    )

@app.get("/api/llm/status", response_model=LLMConnectionStatus)
async def check_llm_status():
    """Check LLM (LM Studio) connection status"""
    try:
        connected, message = test_llm_connection()
        return LLMConnectionStatus(
            connected=connected,
            message=message,
            model=iso11135_config.LLM_MODEL_NAME if connected else None
        )
    except Exception as e:
        logger.exception("Error checking LLM status")
        return LLMConnectionStatus(
            connected=False,
            message=f"Error: {str(e)}"
        )

@app.get("/api/storage/test")
async def test_storage_connection():
    """Diagnostic endpoint for storage"""
    try:
        if storage.provider == "local":
            return {"status": "local", "path": str(iso11135_config.OUTPUTS_DIR)}
        
        if not storage.client:
             return {"status": "error", "message": "Supabase client not initialized"}

        # Test Supabase
        files = storage.client.storage.from_(storage.bucket_name).list()
        return {
            "status": "supabase",
            "bucket": storage.bucket_name,
            "file_count": len(files),
            "files": [f['name'] for f in files]
        }
    except Exception as e:
        logger.error(f"Storage test failed: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/files/status")
async def get_files_status(guideline_id: str = "iso11135"):
    """Get status of all output files for a specific guideline"""
    files = [
        iso11135_config.GUIDELINE_EXTRACTION_OUTPUT,
        iso11135_config.POLISHED_OUTPUT_FILE,
        iso11135_config.DHF_EXTRACTION_OUTPUT,
        iso11135_config.VALIDATION_REPORT,
        iso11135_config.VALIDATION_TERMINAL_OUTPUT
    ]
    
    return {
        "files": {filename: get_file_info(filename, guideline_id).dict() for filename in files}
    }

@app.get("/api/pipeline/completion")
async def get_pipeline_completion_status(guideline_id: str = "iso11135"):
    """Get completion status for each pipeline step based on output files"""
    # Define pipeline steps and their corresponding output files
    pipeline_steps = [
        {
            "step": "guideline_extraction",
            "name": "Guideline Extraction",
            "filename": iso11135_config.GUIDELINE_EXTRACTION_OUTPUT,
            "description": "Extract parameters from guideline PDF"
        },
        {
            "step": "polishing",
            "name": "LLM Polishing",
            "filename": iso11135_config.POLISHED_OUTPUT_FILE,
            "description": "Polish extracted content using LLM",
            "prerequisite": iso11135_config.GUIDELINE_EXTRACTION_OUTPUT
        },
        {
            "step": "dhf_extraction",
            "name": "DHF Extraction",
            "filename": iso11135_config.DHF_EXTRACTION_OUTPUT,
            "description": "Extract content from DHF PDF"
        },
        {
            "step": "validation",
            "name": "Validation",
            "filename": iso11135_config.VALIDATION_REPORT,
            "description": "Run validation analysis",
            "prerequisites": [
                iso11135_config.POLISHED_OUTPUT_FILE,
                iso11135_config.DHF_EXTRACTION_OUTPUT
            ]
        }
    ]
    
    completion_status = []
    
    for step_info in pipeline_steps:
        file_info = get_file_info(step_info["filename"], guideline_id)
        file_exists = file_info.exists
        
        # Check prerequisites if they exist
        prerequisites_met = True
        if "prerequisite" in step_info:
            prereq_info = get_file_info(step_info["prerequisite"], guideline_id)
            prerequisites_met = prereq_info.exists
        elif "prerequisites" in step_info:
            prerequisites_met = all(
                get_file_info(prereq, guideline_id).exists 
                for prereq in step_info["prerequisites"]
            )
        
        # Determine status
        if file_exists:
            status = "completed"
            message = f"{step_info['name']} completed successfully"
        elif not prerequisites_met:
            status = "pending"
            message = f"Prerequisites not met for {step_info['name']}"
        else:
            status = "pending"
            message = f"{step_info['name']} not yet completed"
        
        completion_status.append({
            "step": step_info["step"],
            "name": step_info["name"],
            "status": status,
            "file_exists": file_exists,
            "filename": step_info["filename"],
            "description": step_info["description"],
            "prerequisites_met": prerequisites_met,
            "message": message,
            "file_info": file_info.dict() if file_exists else None
        })
    
    # Calculate overall completion
    completed_steps = sum(1 for step in completion_status if step["status"] == "completed")
    total_steps = len(completion_status)
    overall_progress = (completed_steps / total_steps * 100) if total_steps > 0 else 0
    
    return {
        "overall_progress": round(overall_progress, 1),
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "steps": completion_status
    }

@app.post("/api/guideline/upload")
async def upload_guideline(file: UploadFile = File(...)):
    """Upload and process ISO 11135 guideline PDF"""
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Save uploaded file temporarily
        temp_path = iso11135_config.TEMP_DIR / file.filename
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing guideline PDF: {file.filename}")
        
        # Extract PDF content
        content = extract_pdf_content(str(temp_path))
        
        # Extract parameters with context
        parameters = extract_parameters_with_context(content['text'])
        
        # Filter by relevance score
        high_relevance_params = [p for p in parameters if p['relevance_score'] >= iso11135_config.MIN_RELEVANCE_SCORE]
        
        # Save results locally
        output_path = iso11135_config.OUTPUTS_DIR / iso11135_config.GUIDELINE_EXTRACTION_OUTPUT
        save_results_to_text(high_relevance_params, str(output_path))
        
        # Persist to Storage (Cloud/Local)
        storage.save_file(output_path, iso11135_config.GUIDELINE_EXTRACTION_OUTPUT)
        
        # Clean up temp file
        temp_path.unlink()
        
        return {
            "status": "success",
            "message": f"Extracted {len(high_relevance_params)} parameters",
            "parameters_count": len(high_relevance_params),
            "output_file": iso11135_config.GUIDELINE_EXTRACTION_OUTPUT
        }
        
    except Exception as e:
        logger.exception(f"Error processing guideline PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing guideline: {str(e)}")

@app.post("/api/guideline/polish")
async def polish_guideline():
    """Polish extracted guideline content using LLM"""
    try:
        # Check if guideline extraction exists (Local or Cloud)
        guideline_file = storage.ensure_local(iso11135_config.GUIDELINE_EXTRACTION_OUTPUT)
        if not guideline_file:
            raise HTTPException(
                status_code=400, 
                detail="Guideline extraction not found. Please upload and extract guideline first."
            )
        
        logger.info("Starting LLM polishing process")
        
        # Temporarily change to outputs directory for file loading
        import os
        original_cwd = os.getcwd()
        original_file = os.getenv("EXTRACTED_TEXT_FILE")
        
        try:
            os.chdir(str(iso11135_config.OUTPUTS_DIR))
            os.environ["EXTRACTED_TEXT_FILE"] = str(iso11135_config.OUTPUTS_DIR / iso11135_config.GUIDELINE_EXTRACTION_OUTPUT)
            
            # Load raw extracted text
            raw_categories = load_raw_extracted_text()
            if not raw_categories:
                raise HTTPException(
                    status_code=400,
                    detail="No categories found in guideline extraction"
                )
            
            # Polish all categories
            polished_content = polish_all_categories(raw_categories)
            
            # Save polished output
            output_path = iso11135_config.OUTPUTS_DIR / iso11135_config.POLISHED_OUTPUT_FILE
            save_polished_output(polished_content, str(output_path))
            
            # Persist to Storage
            storage.save_file(output_path, iso11135_config.POLISHED_OUTPUT_FILE)
            
        finally:
            os.chdir(original_cwd)
            if original_file:
                os.environ["EXTRACTED_TEXT_FILE"] = original_file
            elif "EXTRACTED_TEXT_FILE" in os.environ:
                del os.environ["EXTRACTED_TEXT_FILE"]
        
        return {
            "status": "success",
            "message": "Guideline content polished successfully",
            "output_file": iso11135_config.POLISHED_OUTPUT_FILE
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error polishing guideline: {e}")
        raise HTTPException(status_code=500, detail=f"Error polishing guideline: {str(e)}")

@app.post("/api/dhf/upload")
async def upload_dhf(file: UploadFile = File(...)):
    """Upload and process DHF PDF"""
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Save uploaded file temporarily
        temp_path = iso11135_config.TEMP_DIR / file.filename
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing DHF PDF: {file.filename}")
        
        # Extract DHF content
        output_file = str(iso11135_config.OUTPUTS_DIR / iso11135_config.DHF_EXTRACTION_OUTPUT)
        extract_single_pdf(str(temp_path), output_file)
        
        # Persist to Storage
        storage.save_file(output_file, iso11135_config.DHF_EXTRACTION_OUTPUT)
        
        # Clean up temp file
        temp_path.unlink()
        
        return {
            "status": "success",
            "message": "DHF extraction completed successfully",
            "output_file": iso11135_config.DHF_EXTRACTION_OUTPUT
        }
        
    except Exception as e:
        logger.exception(f"Error processing DHF PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing DHF: {str(e)}")

@app.post("/api/validation/run")
async def run_validation():
    """Run validation analysis"""
    try:
        # Check prerequisites (Local or Cloud)
        polished_file = storage.ensure_local(iso11135_config.POLISHED_OUTPUT_FILE)
        dhf_file = storage.ensure_local(iso11135_config.DHF_EXTRACTION_OUTPUT)
        
        if not polished_file:
            raise HTTPException(
                status_code=400,
                detail="Polished guideline not found. Please complete guideline polishing first."
            )
        
        if not dhf_file:
            raise HTTPException(
                status_code=400,
                detail="DHF extraction not found. Please upload and extract DHF first."
            )
        
        logger.info("Starting validation process")
        
        # Run validation
        # Note: EnhancedMultiLayerValidationEngine expects polished_regulatory_guidance.txt in current directory
        # We need to ensure it can find the file
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(str(iso11135_config.OUTPUTS_DIR))
            # Create engine - it will look for polished_regulatory_guidance.txt in current dir
            engine = EnhancedMultiLayerValidationEngine()
            # Process DHF file (use filename only since we're in outputs directory)
            dhf_filename = Path(dhf_file).name
            results = engine.process_document_with_granular_analysis(dhf_filename)
        finally:
            os.chdir(original_cwd)
        
        # Save validation report
        report_path = iso11135_config.OUTPUTS_DIR / iso11135_config.VALIDATION_REPORT
        save_validation_report(results, str(report_path))
        
        # Persist to Storage
        storage.save_file(report_path, iso11135_config.VALIDATION_REPORT)
        
        # Calculate overall readiness
        overall_readiness = sum(r.readiness_score for r in results) / len(results) if results else 0.0
        
        return {
            "status": "success",
            "message": "Validation completed successfully",
            "sections_processed": len(results),
            "overall_readiness": overall_readiness,
            "output_file": iso11135_config.VALIDATION_REPORT
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error running validation: {e}")
        raise HTTPException(status_code=500, detail=f"Error running validation: {str(e)}")

def save_validation_report(results, report_path: str):
    """Save validation results to report file"""
    report_content = []
    report_content.append("#" * 80)
    report_content.append("#" + " " * 78 + "#")
    report_content.append("#" + " DHF VALIDATION REPORT".center(78) + "#")
    report_content.append("#" + " " * 78 + "#")
    report_content.append("#" * 80)
    report_content.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append("")

    if results and len(results) > 0:
        overall_readiness = sum(r.readiness_score for r in results) / len(results)
        report_content.append("\nâ”Œâ”€ OVERALL READINESS " + "â”€" * 58)
        report_content.append(f"â”‚  DHF Readiness Score: {overall_readiness:>6.1%}")
        bar_length = int(overall_readiness * 50)
        bar = "â–ˆ" * bar_length + "â–‘" * (50 - bar_length)
        report_content.append(f"â”‚  [{bar}]")
        report_content.append("â””" + "â”€" * 79)
        report_content.append("")

        for i, result in enumerate(results, 1):
            report_content.append(f"\n\n{'#'*80}")
            report_content.append(f"# SECTION {i}: {result.section_name.upper()}")
            report_content.append(f"#" + "-" * 78)
            report_content.append(f"# Readiness Score: {result.readiness_score:>6.1%}")
            report_content.append("#" * 80)

            status_groups = {}
            for element in result.consensus_elements:
                if element.status not in status_groups:
                    status_groups[element.status] = []
                status_groups[element.status].append(element)

            for status in status_groups:
                elements = status_groups[status]
                report_content.append(f"\nâ”Œâ”€ {status.value} ({len(elements)} items) " + "â”€" * (70 - len(status.value) - len(str(len(elements)))))

                for idx, element in enumerate(elements, 1):
                    report_content.append(f"â”‚")
                    report_content.append(f"â”‚  [{idx:02d}] {element.content}")
                    if element.evidence:
                        snippet = element.evidence
                        if len(snippet) > 100: 
                            snippet = snippet[:100].strip() + "..."
                        snippet = snippet.replace('\n', ' ')
                        report_content.append(f"â”‚      â†’ Evidence: {snippet}")
                    report_content.append("â”‚")
                
                report_content.append("â””" + "â”€" * 79)
    else:
        report_content.append("No validation results were processed.")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))

@app.get("/api/files/{filename}")
async def download_file(filename: str):
    """Download a generated file (Local or Cloud)"""
    # Security: Only allow downloading from outputs directory
    allowed_files = [
        iso11135_config.GUIDELINE_EXTRACTION_OUTPUT,
        iso11135_config.POLISHED_OUTPUT_FILE,
        iso11135_config.DHF_EXTRACTION_OUTPUT,
        iso11135_config.VALIDATION_REPORT,
        iso11135_config.VALIDATION_TERMINAL_OUTPUT
    ]
    
    if filename not in allowed_files:
        raise HTTPException(status_code=403, detail="File not allowed")
    
    # 1. Check Cloud Storage First
    if storage.provider == "supabase":
        file_url = storage.get_file_url(filename)
        if file_url:
            from fastapi.responses import RedirectResponse
            return RedirectResponse(file_url)

    # 2. Check Local Storage (Fallback)
    # Check Backend/outputs first
    file_path = iso11135_config.OUTPUTS_DIR / filename
    
    # If not found, check root/outputs (legacy)
    if not file_path.exists():
        root_outputs = iso11135_config.PROJECT_ROOT.parent / "outputs" / filename
        if root_outputs.exists():
            file_path = root_outputs
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='text/plain'
    )

@app.get("/api/files/{filename}/content")
async def get_file_content(filename: str):
    """Get file content as JSON"""
    allowed_files = [
        iso11135_config.GUIDELINE_EXTRACTION_OUTPUT,
        iso11135_config.POLISHED_OUTPUT_FILE,
        iso11135_config.DHF_EXTRACTION_OUTPUT,
        iso11135_config.VALIDATION_REPORT,
        iso11135_config.VALIDATION_TERMINAL_OUTPUT
    ]
    
    if filename not in allowed_files:
        raise HTTPException(status_code=403, detail="File not allowed")
    
    content = storage.fetch_file_content(filename)
    
    if content is None:
        raise HTTPException(status_code=404, detail="File not found")
    
    return {
        "filename": filename,
        "content": content,
        "size": len(content)
    }

@app.post("/api/pipeline/run")
async def run_full_pipeline(
    guideline_file: UploadFile = File(...),
    dhf_file: UploadFile = File(...)
):
    """
    Run the complete pipeline: guideline extraction â†’ polishing â†’ DHF extraction â†’ validation
    This is the main endpoint that runs all processing steps automatically.
    """
    try:
        # Validate file types
        if not guideline_file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Guideline file must be a PDF")
        if not dhf_file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="DHF file must be a PDF")
        
        # Save uploaded files temporarily
        guideline_temp = iso11135_config.TEMP_DIR / guideline_file.filename
        dhf_temp = iso11135_config.TEMP_DIR / dhf_file.filename
        
        with open(guideline_temp, "wb") as buffer:
            shutil.copyfileobj(guideline_file.file, buffer)
        
        with open(dhf_temp, "wb") as buffer:
            shutil.copyfileobj(dhf_file.file, buffer)
        
        logger.info(f"Starting full pipeline: guideline={guideline_file.filename}, dhf={dhf_file.filename}")
        
        # Run the complete pipeline
        pipeline = DHFPipeline()
        result = pipeline.run_full_pipeline(str(guideline_temp), str(dhf_temp))
        
        # Clean up temp files
        try:
            guideline_temp.unlink()
            dhf_temp.unlink()
        except Exception as e:
            logger.warning(f"Error cleaning up temp files: {e}")
        
        # Convert PipelineResult to dict for JSON response
        response_data = {
            "success": result.success,
            "message": result.message,
            "timestamp": result.timestamp,
            "overall_progress": result.overall_progress,
            "steps": [
                {
                    "name": step.name,
                    "status": step.status,
                    "message": step.message,
                    "output_file": step.output_file,
                    "error": step.error
                }
                for step in result.steps
            ]
        }
        
        if result.success:
            return response_data
        else:
            raise HTTPException(status_code=500, detail=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error running pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Error running pipeline: {str(e)}")


# --- RAG ENDPOINTS ---

@app.post("/api/chat")
async def chat_with_consultant(request: ChatRequest):
    """
    Chat with the ISO 11135 Regulatory Consultant (RAG-Enabled).
    """
    try:
        rag = get_rag_engine()
        
        # 1. Retrieve Context
        context = rag.retrieve_context(request.message)
        
        # 2. Build Prompt
        system_prompt = (
            "You are an expert ISO 11135 Regulatory Consultant. "
            "Your goal is to help the user understand their compliance status.\n"
            "You have access to the user's Validation Report (Evidence) and the ISO Standards (Rules).\n"
            "Use the provided context to explain WHY failures occurred and exactly WHAT to do.\n"
            "If the evidence says something is missing, assume it is true and advise on how to generate that data.\n"
            "Be professional, authoritative, and helpful.\n"
            "IMPORTANT: Format your response using clear Markdown.\n"
            "- Use bold headings (###) for sections.\n"
            "- Use bullet points for lists.\n"
            "- ALWAYS put a blank line before a header or a list.\n"
            "- Ensure the output is easy to read."
        )
        
        full_prompt = (
            f"CONTEXT:\n{context}\n\n"
            f"USER QUESTION: {request.message}\n\n"
            "INSTRUCTION: Answer the user's question based strictly on the provided context. Use Markdown formatting with clear spacing between sections."
        )

        from ISO11135_Backend.LLM_Engine import call_llama
        
        # 3. Call LLM with Custom System Message
        response_text = call_llama(full_prompt, system_message=system_prompt)
        
        return {
            "response": response_text,
            "sources": context
        }
            
    except Exception as e:
        logger.exception(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag/ingest")
async def trigger_rag_ingestion(background_tasks: BackgroundTasks):
    """Trigger background ingestion of RAG data."""
    try:
        rag = get_rag_engine()
        background_tasks.add_task(rag.ingest_data)
        return {"status": "Ingestion started in background"}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# trigger reload rag
