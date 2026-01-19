"""
Centralized error handling and user feedback utilities
"""
import logging
from typing import Optional, Any, Callable
from functools import wraps
import traceback

import logging_setup
logger = logging_setup.get_logger(__name__)

class ProcessingError(Exception):
    """Base exception for processing errors"""
    pass

class PDFExtractionError(ProcessingError):
    """PDF extraction failed"""
    pass

class LLMConnectionError(ProcessingError):
    """LLM connection failed"""
    pass

class ValidationError(ProcessingError):
    """Validation processing failed"""
    pass

def handle_errors(error_type: type = Exception, 
                 default_return: Any = None,
                 user_message: str = "An error occurred"):
    """
    Decorator for consistent error handling across modules
    
    Args:
        error_type: Type of exception to catch
        default_return: Value to return on error
        user_message: User-friendly error message
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_type as e:
                logger.exception(f"{user_message}: {str(e)}")
                print(f"❌ {user_message}: {str(e)}")
                return default_return
            except Exception as e:
                logger.exception(f"Unexpected error in {func.__name__}: {str(e)}")
                print(f"❌ Unexpected error: {str(e)}")
                return default_return
        return wrapper
    return decorator

def safe_file_operation(operation: Callable, 
                       file_path: str, 
                       operation_name: str = "file operation") -> Optional[Any]:
    """
    Safely perform file operations with consistent error handling
    
    Args:
        operation: Function to execute
        file_path: Path to file being operated on
        operation_name: Description of operation for error messages
        
    Returns:
        Result of operation or None on error
    """
    try:
        return operation()
    except FileNotFoundError:
        logger.error(f"File not found during {operation_name}: {file_path}")
        print(f"❌ File not found: {file_path}")
        return None
    except PermissionError:
        logger.error(f"Permission denied during {operation_name}: {file_path}")
        print(f"❌ Permission denied: {file_path}")
        return None
    except Exception as e:
        logger.exception(f"Error during {operation_name} on {file_path}: {str(e)}")
        print(f"❌ Error during {operation_name}: {str(e)}")
        return None

def get_detailed_error_info() -> str:
    """Get detailed error information for debugging"""
    return traceback.format_exc()

class ProgressTracker:
    """Simple progress tracking with user feedback"""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        
    def update(self, step_name: str = ""):
        """Update progress"""
        self.current_step += 1
        percentage = (self.current_step / self.total_steps) * 100
        bar_length = 40
        filled = int(bar_length * self.current_step / self.total_steps)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        print(f"\r{self.description}: [{bar}] {percentage:.1f}% - {step_name}", end="", flush=True)
        
        if self.current_step >= self.total_steps:
            print()  # New line when complete
            
    def complete(self, message: str = "Complete!"):
        """Mark as complete"""
        self.current_step = self.total_steps
        self.update(message)
