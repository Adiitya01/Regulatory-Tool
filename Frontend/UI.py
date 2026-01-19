import streamlit as st
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd
import io
import zipfile
import json
import contextlib
import traceback

import logging_setup

# Module-level logger
logger = logging_setup.get_logger(__name__)

# Import backend modules with error handling
try:
    from Guideline_Extractor import extract_pdf_content, extract_parameters_with_context, save_results_to_text
    from LLM_Engine import load_raw_extracted_text, polish_all_categories, save_polished_output, test_llm_connection
    from DHF_Extractor import extract_single_pdf
    from validation import EnhancedMultiLayerValidationEngine
    backend_available = True
except ImportError as e:
    backend_available = False
    import logging_setup
    logger = logging_setup.get_logger(__name__)
    logger.exception("Backend import error: %s", e)

import requests

# Configure Streamlit page
st.set_page_config(
    page_title="DHF Multi-Document Processor",
    layout="wide",
    initial_sidebar_state="expanded"
)

def _apply_theme():
    """Apply strict color palette: #0F3041 (dark blue), #C8242F (red), #f2f2f2 (light gray)"""
    css = r"""
    :root{
        --primary: #C8242F;
        --secondary: #0F3041;
        --bg: #f2f2f2;
        --card-bg: #ffffff;
        --text: #0F3041;
        --sidebar-bg: #0F3041;
        --sidebar-text: #f2f2f2;
        --button-bg: #C8242F;
        --button-text: #ffffff;
    }

    .stApp, .reportview-container, .main, body {
        background: var(--bg) !important;
        color: var(--text) !important;
    }

    .status-card{
        background: var(--card-bg) !important;
        color: var(--text) !important;
        padding: 16px !important;
        border-radius: 8px !important;
        border: 2px solid var(--secondary) !important;
        box-shadow: 0 2px 8px rgba(15,48,65,0.1) !important;
        transition: all 0.2s ease-in-out;
    }
    .status-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(200,36,47,0.15) !important;
        border-color: var(--primary) !important;
    }

    .stButton>button, .stDownloadButton>button {
        background: var(--button-bg) !important;
        color: var(--button-text) !important;
        border: 1px solid var(--secondary) !important;
        padding: 8px 12px !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 10px rgba(200,36,47,0.15) !important;
    }

    .stButton>button:hover:not(:disabled) {
        background: #a01d26 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 14px rgba(200,36,47,0.3) !important;
    }

    .stButton>button:disabled {
        opacity: 0.5 !important;
        background: #999999 !important;
    }

    [data-testid="stSidebar"] {
        background: var(--sidebar-bg) !important;
        color: var(--sidebar-text) !important;
        box-shadow: 2px 0 12px rgba(15,48,65,0.15) !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] strong,
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] label {
        color: var(--sidebar-text) !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(242,242,242,0.2) !important;
    }

    [data-testid="stSidebar"] input, 
    [data-testid="stSidebar"] select, 
    [data-testid="stSidebar"] textarea {
        background: rgba(255,255,255,0.95) !important;
        color: var(--text) !important;
        border: 1px solid rgba(242,242,242,0.3) !important;
    }

    h1, h2, h3, h4 {
        color: var(--text) !important;
        font-weight: 700 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        border-bottom: 2px solid var(--secondary) !important;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: var(--text) !important;
        border-radius: 6px 6px 0 0 !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        border: 1px solid transparent !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--primary) !important;
        color: #ffffff !important;
        border-color: var(--primary) !important;
    }

    .stTextInput>div>div>input, 
    .stNumberInput>div>div>input, 
    textarea, select {
        background: #ffffff !important;
        color: var(--text) !important;
        border: 2px solid var(--secondary) !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
    }

    .stTextInput>div>div>input:focus, 
    .stNumberInput>div>div>input:focus, 
    textarea:focus, select:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(200,36,47,0.1) !important;
    }

    .stButton>button:focus {
        outline: 3px solid rgba(200,36,47,0.3) !important;
        outline-offset: 2px !important;
    }

    .file-content-viewer {
        background-color: #ffffff !important;
        border: 2px solid var(--secondary) !important;
        padding: 1rem;
        border-radius: 6px;
        max-height: 500px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
        font-family: 'Courier New', monospace;
        color: var(--text);
    }

    .block-container {
        max-width: 1200px !important;
        padding-top: 22px !important;
    }
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

logger.info("UI module loaded")


class RegulatoryComplianceUI:
    def __init__(self):
        self.project_dir = Path.cwd()
        import logging_setup
        self.logger = logging_setup.get_logger(__name__)
        self.ensure_directories()
        self.progress_bar = None
        self.progress_text = None
        self._progress_start_time = None
    
    def _start_progress(self, label: str):
        """Initialize a progress bar with ETA text."""
        self.progress_bar = st.progress(0)
        self.progress_text = st.empty()
        self._progress_start_time = datetime.now()
        self.progress_text.write(f"{label} — 0% | ETA: calculating…")

    def _update_progress(self, percent: int, label: str, current: int = None, total: int = None):
        percent = max(0, min(100, int(percent)))
        self.progress_bar.progress(percent / 100.0)  # Streamlit expects 0.0 to 1.0
        eta_str = "calculating…"

        if self._progress_start_time:
            elapsed = (datetime.now() - self._progress_start_time).total_seconds()

            # Only show ETA after we have meaningful progress (at least 2% and 2 seconds)
            if elapsed > 2 and percent >= 2:
                progress_ratio = percent / 100.0
                
                # Calculate remaining time based on current progress
                if progress_ratio > 0.01:  # Avoid division by very small numbers
                    total_estimated_time = elapsed / progress_ratio
                    remaining = total_estimated_time - elapsed

                    if remaining > 0:
                        if remaining < 60:
                            eta_str = f"{int(remaining)}s"
                        elif remaining < 3600:
                            mm = int(remaining // 60)
                            ss = int(remaining % 60)
                            eta_str = f"{mm:02d}:{ss:02d}"
                        else:
                            hh = int(remaining // 3600)
                            mm = int((remaining % 3600) // 60)
                            eta_str = f"{hh}h{mm:02d}m"
                    elif percent >= 95:
                        eta_str = "finishing..."

        self.progress_text.write(f"{label} — {percent}% | ETA: {eta_str}")

    def _finish_progress(self, label_success: str):
        if self.progress_bar:
            self.progress_bar.progress(1.0)  # 1.0 = 100%
        if self.progress_text:
            self.progress_text.write(f"{label_success} — 100% | ETA: done")
        
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        directories = ['inputs', 'outputs', 'reports', 'temp']
        for dir_name in directories:
            (self.project_dir / dir_name).mkdir(exist_ok=True)
        try:
            self.logger.debug("Ensured project directories exist: %s", directories)
        except Exception:
            pass
    
    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <h1>DHF Multi-Document Processor</h1>
        </div>
        """, unsafe_allow_html=True)

    def check_backend_status(self):
        """Check if backend + LM Studio are connected"""
        backend_ok = backend_available
        # Use the dedicated test helper from LLM_Engine which queries the supported
        # `/v1/models` endpoint instead of the unsupported `/health` endpoint.
        try:
            ok, info = test_llm_connection()
            if not ok:
                self.logger.debug("LM Studio connectivity check failed: %s", info)
            return backend_ok and ok
        except Exception:
            self.logger.exception("Error checking LM Studio connectivity")
            return False

    def render_sidebar(self):
        """Render the sidebar with controls"""
        self.logger.debug("Rendering sidebar; project_dir=%s", self.project_dir)
        with st.sidebar:
            st.markdown("### Control Panel")
            
            st.markdown("### System Status")
            if self.check_backend_status():
                st.success("Systems Online") 
            else:
                st.error("Backend Offline")
            
            st.markdown(f"**Project:** `{self.project_dir.name}`")
            st.divider()
            
            st.markdown("### Quick Actions")
            
            if st.button("Refresh", width='stretch'):
                self.logger.info("User requested refresh")
                st.rerun()
                
            if st.button("Open Folder", width='stretch'):
                try:
                    self.logger.info("User requested to open project folder: %s", self.project_dir)
                    if os.name == 'nt':
                        os.startfile(str(self.project_dir))
                    elif os.name == 'posix':
                        subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', str(self.project_dir)])
                    st.success("Folder opened!")
                except Exception as e:
                    self.logger.exception("Error opening project folder: %s", e)
                    st.error(f"Error: {e}")
    
    def render_pipeline_overview(self):
        """Render the processing pipeline overview"""
        st.markdown("### Processing Pipeline")
        
        steps = [
            {"name": "Guideline Extraction", "file": "guideline_extraction_output.txt"},
            {"name": "LLM Polishing", "file": "polished_regulatory_guidance.txt"},
            {"name": "DHF Extraction", "file": "DHF_Single_Extraction.txt"},
            {"name": "Validation Report", "file": "validation_report.txt"}
        ]
        
        cols = st.columns(4)
        
        for i, step in enumerate(steps):
            with cols[i]:
                file_exists = (self.project_dir / step["file"]).exists()
                status = "Complete" if file_exists else "Pending"
                
                st.markdown(f"""
                <div class="status-card">
                    <div style="text-align: center;">
                        <div style="font-weight: 600;">{step["name"]}</div>
                        <div style="margin-top: 0.5rem; font-size: 1.2rem;">{status}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    def process_guideline_extraction(self, pdf_file):
        """Process guideline extraction"""
        if not backend_available:
            st.error("Backend systems not available")
            return False
            
        try:
            try:
                file_name = getattr(pdf_file, 'name', None)
                file_size = len(pdf_file.getvalue()) if pdf_file else 0
            except Exception:
                file_name = None
                file_size = 0
            self.logger.info("Starting guideline extraction for uploaded file: %s (size=%d bytes)", file_name, file_size)
            self._start_progress("Extracting parameters from guideline PDF")
            temp_path = self.project_dir / "temp" / pdf_file.name
            with open(temp_path, "wb") as f:
                f.write(pdf_file.getvalue())
            self._update_progress(10, "Extracting parameters from guideline PDF", current=1, total=10)

            content = extract_pdf_content(str(temp_path))
            self._update_progress(35, "Parsing guideline content", current=3, total=10)
            parameters = extract_parameters_with_context(content['text'])
            self._update_progress(65, "Scoring and filtering parameters", current=6, total=10)
            high_relevance_params = [p for p in parameters if p['relevance_score'] >= 2.0]

            output_path = self.project_dir / "guideline_extraction_output.txt"
            save_results_to_text(high_relevance_params, str(output_path))
            self._update_progress(90, "Writing results", current=9, total=10)

            temp_path.unlink()
            self._finish_progress(f"Extracted {len(high_relevance_params)} parameters")
            self.logger.info("Guideline extraction finished: %d parameters", len(high_relevance_params))
            return True
                
        except Exception as e:
            self.logger.exception("Error during guideline extraction: %s", e)
            st.error(f"Error: {str(e)}")
            if st.session_state.get("debug_mode"):
                st.code(traceback.format_exc())
            return False
    
    def process_llm_polishing(self):
        """Process LLM polishing"""
        if not backend_available:
            st.error("Backend systems not available")
            return False
            
        try:
            self.logger.info("Starting LLM polishing process")
            self._start_progress("Polishing regulatory guidance with LLM")
            raw_categories = load_raw_extracted_text()
            if not raw_categories:
                st.error("No guideline extraction found. Run guideline extraction first.")
                return False
            self._update_progress(25, "Loaded categories", current=2, total=8)

            polished_content = polish_all_categories(raw_categories)
            self._update_progress(75, "LLM polishing in progress", current=6, total=8)
            save_polished_output(polished_content, file_path=str(self.project_dir / "polished_regulatory_guidance.txt"))
            self._finish_progress("Polishing completed")
            self.logger.info("LLM polishing completed and saved")
            return True
                
        except Exception as e:
            self.logger.exception("Error during LLM polishing: %s", e)
            st.error(f"Error: {str(e)}")
            if st.session_state.get("debug_mode"):
                st.code(traceback.format_exc())
            return False
    
    def process_dhf_extraction(self, pdf_file):
        """Process DHF extraction"""
        if not backend_available:
            st.error("Backend systems not available")
            return False
            
        try:
            try:
                file_name = getattr(pdf_file, 'name', None)
                file_size = len(pdf_file.getvalue()) if pdf_file else 0
            except Exception:
                file_name = None
                file_size = 0
            self.logger.info("Starting DHF extraction for uploaded file: %s (size=%d bytes)", file_name, file_size)
            self._start_progress("Extracting DHF parameters")
            temp_path = self.project_dir / "temp" / pdf_file.name
            with open(temp_path, "wb") as f:
                f.write(pdf_file.getvalue())
            self._update_progress(10, "Preparing DHF file", current=1, total=10)

            output_file = str(self.project_dir / "DHF_Single_Extraction.txt")
            extract_single_pdf(str(temp_path), output_file)
            self._update_progress(90, "Mapping and writing output", current=9, total=10)

            temp_path.unlink()
            self._finish_progress("DHF extraction completed")
            self.logger.info("DHF extraction completed and output saved to %s", output_file)
            return True
                
        except Exception as e:
            self.logger.exception("Error during DHF extraction: %s", e)
            st.error(f"Error: {str(e)}")
            if st.session_state.get("debug_mode"):
                st.code(traceback.format_exc())
            return False
    
    def process_validation(self):
        """Process validation report"""
        if not backend_available:
            st.error("Backend systems not available")
            return False
            
        try:
            self.logger.info("Starting validation process")
            self._start_progress("Generating validation report")
            engine = EnhancedMultiLayerValidationEngine()
            dhf_file = str(self.project_dir / "DHF_Single_Extraction.txt")

            if not Path(dhf_file).exists():
                st.error("DHF extraction file not found. Run DHF extraction first.")
                return False
            self._update_progress(15, "Loading DHF content", current=1, total=6)

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                results = engine.process_document_with_granular_analysis(dhf_file)
            self._update_progress(65, "Performing granular analysis", current=4, total=6)

            terminal_log_path = self.project_dir / "validation_terminal_output.txt"
            with open(terminal_log_path, 'w', encoding='utf-8') as f:
                f.write(buffer.getvalue())
            self._update_progress(80, "Saving terminal output", current=5, total=6)

            if results:
                self.save_validation_report(results)
                self._finish_progress("Validation report generated")
                self.logger.info("Validation completed: %d sections processed", len(results))
                return True
            else:
                st.error("No validation results generated")
                return False
                    
        except Exception as e:
            self.logger.exception("Error during validation: %s", e)
            st.error(f"Error: {str(e)}")
            if st.session_state.get("debug_mode"):
                st.code(traceback.format_exc())
            return False

    def save_validation_report(self, results):
        """Save validation results to report file"""
        try:
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

            report_path = self.project_dir / "validation_report.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_content))
            try:
                self.logger.info("Validation report saved: %s", report_path)
            except Exception:
                pass

        except Exception as e:
            self.logger.exception("Error saving validation report: %s", e)
            st.error(f"Error saving validation report: {e}")
    
    def render_file_processor(self):
        """Render the main file processing interface"""
        self.logger.debug("Rendering file processor")
        st.markdown("### Processing Interface")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Guideline", 
            "Polishing", 
            "DHF", 
            "Validation"
        ])
        
        with tab1:
            st.markdown("#### Upload ISO 11135 Guideline PDF")
            guideline_file = st.file_uploader(
                "Choose guideline PDF", 
                type=['pdf'],
                key="guideline_upload"
            )
            
            col1, col2, col3 = st.columns([2, 2, 2])
            with col1:
                if guideline_file is not None:
                    try:
                        self.logger.debug("Guideline uploaded: %s (size=%d)", getattr(guideline_file, 'name', None), len(guideline_file.getvalue()))
                    except Exception:
                        self.logger.debug("Guideline uploaded (could not get size)")
                if st.button("Extract Parameters", key="extract_params_btn", disabled=guideline_file is None, width='stretch'):
                    self.logger.info("User clicked Extract Parameters")
                    if self.process_guideline_extraction(guideline_file):
                        st.rerun()
            
            output_file = "guideline_extraction_output.txt"
            if (self.project_dir / output_file).exists():
                with col2:
                    if st.button("View Results", key="view_guideline_results", width='content'):
                        st.session_state['file_to_view'] = output_file
                        st.session_state['show_file_content'] = True
                with col3:
                    with open(self.project_dir / output_file, 'r', encoding='utf-8') as f:
                            st.download_button("Download", data=f.read(), file_name=output_file, width='stretch')
        
        with tab2:
            st.markdown("#### LLM-Powered Content Polishing")
            
            guideline_exists = (self.project_dir / "guideline_extraction_output.txt").exists()
            
            if not guideline_exists:
                st.warning("Complete Guideline Processing first")
            
            col1, col2, col3 = st.columns([2, 2, 2])
            with col1:
                if st.button("Polish Content", key="polish_content_btn", disabled=not guideline_exists, width='stretch'):
                    self.logger.info("User clicked Polish Content")
                    if self.process_llm_polishing():
                        st.rerun()
            
            output_file = "polished_regulatory_guidance.txt"
            if (self.project_dir / output_file).exists():
                with col2:
                    if st.button("View Results", key="view_polished_results", width='content'):
                        st.session_state['file_to_view'] = output_file
                        st.session_state['show_file_content'] = True
                with col3:
                    with open(self.project_dir / output_file, 'r', encoding='utf-8') as f:
                            st.download_button("Download", data=f.read(), file_name=output_file, width='stretch')
        
        with tab3:
            st.markdown("#### Upload DHF Document")
            dhf_file = st.file_uploader(
                "Choose DHF PDF", 
                type=['pdf'],
                key="dhf_upload"
            )
            
            col1, col2, col3 = st.columns([2, 2, 2])
            with col1:
                if dhf_file is not None:
                    try:
                        self.logger.debug("DHF uploaded: %s (size=%d)", getattr(dhf_file, 'name', None), len(dhf_file.getvalue()))
                    except Exception:
                        self.logger.debug("DHF uploaded (could not get size)")
                if st.button("Extract DHF", key="extract_dhf_btn", disabled=dhf_file is None, width='stretch'):
                    self.logger.info("User clicked Extract DHF")
                    if self.process_dhf_extraction(dhf_file):
                        st.rerun()
            
            output_file = "DHF_Single_Extraction.txt"
            if (self.project_dir / output_file).exists():
                with col2:
                    if st.button("View Results", key="view_dhf_results", width='content'):
                        st.session_state['file_to_view'] = output_file
                        st.session_state['show_file_content'] = True
                with col3:
                    with open(self.project_dir / output_file, 'r', encoding='utf-8') as f:
                            st.download_button("Download", data=f.read(), file_name=output_file, width='stretch')
        
        with tab4:
            st.markdown("#### Generate Compliance Report")
            
            polished_exists = (self.project_dir / "polished_regulatory_guidance.txt").exists()
            dhf_exists = (self.project_dir / "DHF_Single_Extraction.txt").exists()
            prerequisites = polished_exists and dhf_exists

            if not prerequisites:
                st.warning("Need Polished Guidelines and DHF Extraction")
            
            validation_report_exists = (self.project_dir / "validation_report.txt").exists()
            terminal_output_exists = (self.project_dir / "validation_terminal_output.txt").exists()
    
            col1, col2, col3, col4 = st.columns(4)
    
            with col1:
                if st.button("Generate", key="generate_report_btn", disabled=not prerequisites, width='stretch'):
                    if self.process_validation():
                        st.rerun()
    
            if validation_report_exists:
                with col2:
                    if st.button("View Report", key="view_validation_report", width='content'):
                        st.session_state['file_to_view'] = "validation_report.txt"
                        st.session_state['show_file_content'] = True
                with col3:
                    with open(self.project_dir / "validation_report.txt", 'r', encoding='utf-8') as f:
                            st.download_button("Download", data=f.read(), file_name="validation_report.txt", width='stretch')
            
            if terminal_output_exists:
                with col4:
                    if st.button("Terminal", key="view_terminal_output", width='content'):
                        st.session_state['file_to_view'] = "validation_terminal_output.txt"
                        st.session_state['show_file_content'] = True
        
        if st.session_state.get('show_file_content'):
            self.display_file_content_modal(st.session_state['file_to_view'])

    def display_file_content_modal(self, filename):
        """Display file content viewer"""
        if not filename:
            return
            
        file_path = self.project_dir / filename
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                try:
                    self.logger.info("Displaying file content: %s (size=%d bytes)", filename, len(content))
                except Exception:
                    self.logger.info("Displaying file content: %s", filename)
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
            
            st.markdown("---")
            st.markdown(f"#### {filename}")
            
            col1, col2 = st.columns([1, 5])
            with col1:
                st.download_button("Download", data=content, file_name=filename, mime="text/plain", width='stretch')
            with col2:
                if st.button("Close", key="close_file_viewer", width='content'):
                    st.session_state['show_file_content'] = False
                    st.rerun()
            
            st.markdown(f'<div class="file-content-viewer">{content}</div>', unsafe_allow_html=True)
        else:
            st.error(f"File not found")
            st.session_state['show_file_content'] = False
    
    def render_results_viewer(self):
        """Render the results viewing interface"""
        st.markdown("### Generated Files")
        
        output_files = []
        for pattern in ["*.txt", "*.json"]:
            output_files.extend(self.project_dir.glob(pattern))
        
        output_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        if output_files:
            file_data = []
            for file_path in output_files:
                stat = file_path.stat()
                file_data.append({
                    "File": file_path.name,
                    "Size": f"{stat.st_size / 1024:.1f} KB",
                    "Modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                })
            
            df = pd.DataFrame(file_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            if st.button("Download All as ZIP", width='content'):
                self.create_results_package()
        else:
            st.info("No files generated yet")
    
    def create_results_package(self):
        """Create a ZIP package of all results"""
        try:
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for pattern in ["*.txt", "*.json"]:
                    for file_path in self.project_dir.glob(pattern):
                        zip_file.write(file_path, file_path.name)
                
                metadata = {
                    "generated_at": datetime.now().isoformat(),
                    "system": "DHF Multi-Document Processor",
                    "version": "2.0.0"
                }
                
                zip_file.writestr("metadata.json", json.dumps(metadata, indent=2))
            
            zip_buffer.seek(0)
            
            data_bytes = zip_buffer.getvalue()
            st.download_button(
                label="Download Results Package",
                data=data_bytes,
                file_name=f"dhf_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )
            try:
                self.logger.info("Results package created and presented for download (size=%d bytes)", len(data_bytes))
            except Exception:
                pass
            st.success("Results package created!")
            
        except Exception as e:
            st.error(f"Error creating results package: {e}")
    
    def render_chatbot(self):
        """Render the RAG-powered chatbot interface"""
        st.markdown("### 🤖 Regulatory Consultant Chatbot")
        st.markdown("Ask questions about your compliance status, missing requirements, or ISO 11135 clauses.")
        
        # Initialize chat history in session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'rag_ingested' not in st.session_state:
            st.session_state.rag_ingested = False
        
        # Check if data exists
        polished_exists = (self.project_dir / "polished_regulatory_guidance.txt").exists()
        validation_exists = (self.project_dir / "validation_report.txt").exists()
        
        if not (polished_exists or validation_exists):
            st.warning("⚠️ No data available. Please run the pipeline first (Guideline → Polishing → DHF → Validation)")
            return
        
        # Ingest data button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("💡 Click 'Load Knowledge Base' to enable the chatbot")
        with col2:
            if st.button("🔄 Load Knowledge Base", key="ingest_rag"):
                with st.spinner("Indexing documents..."):
                    try:
                        response = requests.post("http://127.0.0.1:8000/api/rag/ingest", timeout=30)
                        if response.status_code == 200:
                            st.session_state.rag_ingested = True
                            st.success("✅ Knowledge base loaded!")
                        else:
                            st.error(f"Failed to load: {response.status_code}")
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.info("Make sure the backend server is running on port 8000")
        
        if not st.session_state.rag_ingested:
            return
        
        st.divider()
        
        # Chat interface
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div style="background: var(--secondary); color: white; padding: 12px; border-radius: 8px; margin: 8px 0;">
                        <strong>You:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: var(--card-bg); border: 2px solid var(--primary); padding: 12px; border-radius: 8px; margin: 8px 0;">
                        <strong>🤖 Consultant:</strong><br>{message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Input area
        st.markdown("---")
        user_input = st.text_input(
            "Ask a question:",
            key="chat_input",
            placeholder="e.g., Why is bioburden marked as missing?"
        )
        
        col1, col2, col3 = st.columns([2, 2, 6])
        with col1:
            send_button = st.button("📤 Send", key="send_chat")
        with col2:
            if st.button("🗑️ Clear Chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Handle send
        if send_button and user_input:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Call API
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/api/chat",
                        json={"message": user_input},
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        bot_response = data.get("response", "No response")
                        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
                    else:
                        error_msg = f"Error {response.status_code}: {response.text}"
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                        
                except Exception as e:
                    error_msg = f"Connection error: {str(e)}"
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            
            st.rerun()
        
        # Example questions
        st.markdown("---")
        st.markdown("**💡 Example Questions:**")
        examples = [
            "Why is bioburden marked as missing?",
            "What is clause 9.4 in ISO 11135?",
            "Which documents do I need to add for IQ/OQ/PQ?",
            "Explain the validation requirements"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(f"📝 {example}", key=f"example_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": example})
                    
                    with st.spinner("Thinking..."):
                        try:
                            response = requests.post(
                                "http://127.0.0.1:8000/api/chat",
                                json={"message": example},
                                timeout=60
                            )
                            
                            if response.status_code == 200:
                                data = response.json()
                                bot_response = data.get("response", "No response")
                                st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
                            else:
                                error_msg = f"Error {response.status_code}"
                                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                        except Exception as e:
                            error_msg = f"Error: {str(e)}"
                            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    
                    st.rerun()
    
    def run(self):
        """Main application runner"""
        self.render_header()
        _apply_theme()
        self.render_sidebar()
        
        main_tabs = st.tabs(["Processing", "Files", "Chatbot"])
        
        with main_tabs[0]:
            self.render_pipeline_overview()
            st.divider()
            self.render_file_processor()
        
        with main_tabs[1]:
            self.render_results_viewer()
        
        with main_tabs[2]:
            self.render_chatbot()
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: var(--text); padding: 0.5rem; opacity: 0.6;">
            <p>DHF Multi-Document Processor | Designed by Ethosh</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application entry point"""
    if 'show_file_content' not in st.session_state:
        st.session_state['show_file_content'] = False
    if 'file_to_view' not in st.session_state:
        st.session_state['file_to_view'] = None

    try:
        logger.info("Starting RegulatoryComplianceUI application")
        app = RegulatoryComplianceUI()
        app.run()
    except Exception as e:
        logger.exception("Unhandled exception in main: %s", e)
        st.error(f"Application Error: {e}")
        st.info("Check console and verify LM Studio is running")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
