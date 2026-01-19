"""
Unified Pipeline Orchestrator for DHF Multi-Document Processor
Runs all processing steps sequentially in a single pipeline
"""
import os
import sys
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging

# Add Backend directory to path
sys.path.insert(0, str(Path(__file__).parent))

from .Guideline_Extractor import extract_pdf_content, extract_parameters_with_context, save_results_to_text
from .LLM_Engine import load_raw_extracted_text, polish_all_categories, save_polished_output
from .DHF_Extractor import extract_single_pdf
from .validation import EnhancedMultiLayerValidationEngine
from . import config
import logging_setup

logger = logging_setup.get_logger(__name__)


@dataclass
class PipelineStep:
    """Represents a single step in the pipeline"""
    name: str
    status: str  # "pending", "running", "completed", "error"
    message: str
    output_file: Optional[str] = None
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Result of running the complete pipeline"""
    success: bool
    steps: List[PipelineStep]
    overall_progress: int  # 0-100
    message: str
    timestamp: str


class DHFPipeline:
    """Unified pipeline orchestrator for DHF processing"""
    
    def __init__(self):
        self.steps: List[PipelineStep] = []
        self.current_step_index = 0
        
    def add_step(self, name: str, message: str = ""):
        """Add a step to the pipeline"""
        step = PipelineStep(
            name=name,
            status="pending",
            message=message
        )
        self.steps.append(step)
        return step
    
    def update_step(self, step_index: int, status: str, message: str = "", 
                   output_file: str = None, error: str = None):
        """Update a step's status"""
        if 0 <= step_index < len(self.steps):
            step = self.steps[step_index]
            step.status = status
            if message:
                step.message = message
            if output_file:
                step.output_file = output_file
            if error:
                step.error = error
    
    def get_progress(self) -> int:
        """Calculate overall progress percentage"""
        if not self.steps:
            return 0
        
        completed = sum(1 for s in self.steps if s.status == "completed")
        total = len(self.steps)
        return int((completed / total) * 100)
    
    def run_full_pipeline(self, guideline_pdf_path: str, dhf_pdf_path: str) -> PipelineResult:
        """
        Run the complete pipeline from start to finish
        
        Args:
            guideline_pdf_path: Path to ISO 11135 guideline PDF
            dhf_pdf_path: Path to DHF PDF
            
        Returns:
            PipelineResult with all steps and final status
        """
        logger.info("Starting full pipeline execution")
        logger.info(f"Guideline PDF: {guideline_pdf_path}")
        logger.info(f"DHF PDF: {dhf_pdf_path}")
        
        # Initialize pipeline steps
        self.steps = []
        self.add_step("guideline_extraction", "Extracting parameters from guideline PDF")
        self.add_step("guideline_polishing", "Polishing guideline content with LLM")
        self.add_step("dhf_extraction", "Extracting parameters from DHF PDF")
        self.add_step("validation", "Running validation analysis")
        
        try:
            # Step 1: Guideline Extraction
            logger.info("Step 1/4: Guideline Extraction")
            self.update_step(0, "running", "Processing guideline PDF...")
            
            try:
                content = extract_pdf_content(guideline_pdf_path)
                parameters = extract_parameters_with_context(content['text'])
                high_relevance_params = [
                    p for p in parameters 
                    if p['relevance_score'] >= config.MIN_RELEVANCE_SCORE
                ]
                
                output_path = config.OUTPUTS_DIR / config.GUIDELINE_EXTRACTION_OUTPUT
                save_results_to_text(high_relevance_params, str(output_path))
                
                self.update_step(
                    0, "completed",
                    f"Extracted {len(high_relevance_params)} parameters",
                    config.GUIDELINE_EXTRACTION_OUTPUT
                )
                logger.info(f"Guideline extraction completed: {len(high_relevance_params)} parameters")
                
            except Exception as e:
                error_msg = f"Error in guideline extraction: {str(e)}"
                logger.exception(error_msg)
                self.update_step(0, "error", error_msg, error=str(e))
                return self._create_result(False, "Pipeline failed at guideline extraction")
            
            # Step 2: LLM Polishing
            logger.info("Step 2/4: LLM Polishing")
            self.update_step(1, "running", "Polishing content with LLM...")
            
            try:
                # Temporarily change to outputs directory for file loading
                original_cwd = os.getcwd()
                original_file = os.getenv("EXTRACTED_TEXT_FILE")
                
                try:
                    os.chdir(str(config.OUTPUTS_DIR))
                    os.environ["EXTRACTED_TEXT_FILE"] = str(config.OUTPUTS_DIR / config.GUIDELINE_EXTRACTION_OUTPUT)
                    
                    raw_categories = load_raw_extracted_text()
                    if not raw_categories:
                        raise Exception("No categories found in guideline extraction")
                    
                    polished_content = polish_all_categories(raw_categories)
                    
                    output_path = config.OUTPUTS_DIR / config.POLISHED_OUTPUT_FILE
                    save_polished_output(polished_content, str(output_path))
                    
                finally:
                    os.chdir(original_cwd)
                    if original_file:
                        os.environ["EXTRACTED_TEXT_FILE"] = original_file
                    elif "EXTRACTED_TEXT_FILE" in os.environ:
                        del os.environ["EXTRACTED_TEXT_FILE"]
                
                self.update_step(
                    1, "completed",
                    "Guideline content polished successfully",
                    config.POLISHED_OUTPUT_FILE
                )
                logger.info("LLM polishing completed")
                
            except Exception as e:
                error_msg = f"Error in LLM polishing: {str(e)}"
                logger.exception(error_msg)
                self.update_step(1, "error", error_msg, error=str(e))
                return self._create_result(False, "Pipeline failed at LLM polishing")
            
            # Step 3: DHF Extraction
            logger.info("Step 3/4: DHF Extraction")
            self.update_step(2, "running", "Extracting DHF parameters...")
            
            try:
                output_file = str(config.OUTPUTS_DIR / config.DHF_EXTRACTION_OUTPUT)
                extract_single_pdf(dhf_pdf_path, output_file)
                
                self.update_step(
                    2, "completed",
                    "DHF extraction completed successfully",
                    config.DHF_EXTRACTION_OUTPUT
                )
                logger.info("DHF extraction completed")
                
            except Exception as e:
                error_msg = f"Error in DHF extraction: {str(e)}"
                logger.exception(error_msg)
                self.update_step(2, "error", error_msg, error=str(e))
                return self._create_result(False, "Pipeline failed at DHF extraction")
            
            # Step 4: Validation
            logger.info("Step 4/4: Validation")
            self.update_step(3, "running", "Running validation analysis...")
            
            try:
                # Change to outputs directory for validation
                original_cwd = os.getcwd()
                try:
                    os.chdir(str(config.OUTPUTS_DIR))
                    
                    engine = EnhancedMultiLayerValidationEngine()
                    dhf_filename = Path(dhf_pdf_path).name
                    results = engine.process_document_with_granular_analysis(dhf_filename)
                    
                    # Save validation report
                    report_path = config.OUTPUTS_DIR / config.VALIDATION_REPORT
                    self._save_validation_report(results, str(report_path))
                    
                finally:
                    os.chdir(original_cwd)
                
                overall_readiness = sum(r.readiness_score for r in results) / len(results) if results else 0.0
                
                self.update_step(
                    3, "completed",
                    f"Validation completed: {len(results)} sections processed, {overall_readiness:.1%} readiness",
                    config.VALIDATION_REPORT
                )
                logger.info(f"Validation completed: {len(results)} sections, {overall_readiness:.1%} readiness")
                
            except Exception as e:
                error_msg = f"Error in validation: {str(e)}"
                logger.exception(error_msg)
                self.update_step(3, "error", error_msg, error=str(e))
                return self._create_result(False, "Pipeline failed at validation")
            
            # All steps completed successfully
            logger.info("Pipeline completed successfully")
            return self._create_result(True, "Pipeline completed successfully")
            
        except Exception as e:
            logger.exception(f"Unexpected error in pipeline: {e}")
            return self._create_result(False, f"Unexpected pipeline error: {str(e)}")
    
    def _save_validation_report(self, results, report_path: str):
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
            report_content.append("\n┌─ OVERALL READINESS " + "─" * 58)
            report_content.append(f"│  DHF Readiness Score: {overall_readiness:>6.1%}")
            bar_length = int(overall_readiness * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            report_content.append(f"│  [{bar}]")
            report_content.append("└" + "─" * 79)
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
                    report_content.append(f"\n┌─ {status.value} ({len(elements)} items) " + "─" * (70 - len(status.value) - len(str(len(elements)))))

                    for idx, element in enumerate(elements, 1):
                        report_content.append(f"│")
                        report_content.append(f"│  [{idx:02d}] {element.content}")
                        if element.evidence:
                            snippet = element.evidence
                            if len(snippet) > 100: 
                                snippet = snippet[:100].strip() + "..."
                            snippet = snippet.replace('\n', ' ')
                            report_content.append(f"│      → Evidence: {snippet}")
                        report_content.append("│")
                    
                    report_content.append("└" + "─" * 79)
        else:
            report_content.append("No validation results were processed.")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
    
    def _create_result(self, success: bool, message: str) -> PipelineResult:
        """Create a PipelineResult from current state"""
        return PipelineResult(
            success=success,
            steps=self.steps,
            overall_progress=self.get_progress(),
            message=message,
            timestamp=datetime.now().isoformat()
        )


def run_pipeline(guideline_pdf_path: str, dhf_pdf_path: str) -> PipelineResult:
    """
    Convenience function to run the complete pipeline
    
    Args:
        guideline_pdf_path: Path to ISO 11135 guideline PDF
        dhf_pdf_path: Path to DHF PDF
        
    Returns:
        PipelineResult with all steps and final status
    """
    pipeline = DHFPipeline()
    return pipeline.run_full_pipeline(guideline_pdf_path, dhf_pdf_path)

