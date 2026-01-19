"""
Performance monitoring and profiling utilities
"""
import time
import functools
from typing import Callable, Dict, Any
from datetime import datetime
import json
from pathlib import Path

import logging_setup
logger = logging_setup.get_logger(__name__)

class PerformanceMonitor:
    """Track and log performance metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.start_time = None
        
    def start(self):
        """Start monitoring"""
        self.start_time = time.time()
        
    def record(self, operation: str, duration: float, metadata: Dict = None):
        """Record an operation's performance"""
        if operation not in self.metrics:
            self.metrics[operation] = {
                'count': 0,
                'total_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0,
                'executions': []
            }
        
        stats = self.metrics[operation]
        stats['count'] += 1
        stats['total_time'] += duration
        stats['min_time'] = min(stats['min_time'], duration)
        stats['max_time'] = max(stats['max_time'], duration)
        
        execution = {
            'timestamp': datetime.now().isoformat(),
            'duration': round(duration, 3),
        }
        
        if metadata:
            execution['metadata'] = metadata
            
        stats['executions'].append(execution)
        
        # Keep only last 10 executions
        if len(stats['executions']) > 10:
            stats['executions'] = stats['executions'][-10:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {}
        
        for operation, stats in self.metrics.items():
            avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
            
            summary[operation] = {
                'executions': stats['count'],
                'total_time': round(stats['total_time'], 3),
                'avg_time': round(avg_time, 3),
                'min_time': round(stats['min_time'], 3),
                'max_time': round(stats['max_time'], 3)
            }
        
        if self.start_time:
            summary['total_runtime'] = round(time.time() - self.start_time, 3)
        
        return summary
    
    def save_report(self, output_dir: str = "reports"):
        """Save performance report to file"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path(output_dir) / f"performance_report_{timestamp}.json"
        
        summary = self.get_summary()
        summary['report_generated'] = datetime.now().isoformat()
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Performance report saved: {report_file}")
        return report_file
    
    def print_summary(self):
        """Print performance summary to console"""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY".center(60))
        print("=" * 60)
        
        if 'total_runtime' in summary:
            print(f"\nTotal Runtime: {summary['total_runtime']:.3f} seconds")
        
        print("\nOperation Metrics:")
        print("-" * 60)
        
        for operation, metrics in summary.items():
            if operation == 'total_runtime':
                continue
                
            print(f"\n{operation}:")
            print(f"  Executions: {metrics['executions']}")
            print(f"  Total Time: {metrics['total_time']:.3f}s")
            print(f"  Avg Time:   {metrics['avg_time']:.3f}s")
            print(f"  Min Time:   {metrics['min_time']:.3f}s")
            print(f"  Max Time:   {metrics['max_time']:.3f}s")
        
        print("\n" + "=" * 60 + "\n")

# Global monitor instance
_monitor = PerformanceMonitor()

def get_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    return _monitor

def timed_operation(operation_name: str = None):
    """
    Decorator to time function execution
    
    Usage:
        @timed_operation("pdf_extraction")
        def extract_pdf(path):
            ...
    """
    def decorator(func: Callable):
        nonlocal operation_name
        if operation_name is None:
            operation_name = func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                
                # Record successful execution
                _monitor.record(operation_name, duration, {'status': 'success'})
                
                logger.debug(f"{operation_name} completed in {duration:.3f}s")
                return result
                
            except Exception as e:
                duration = time.time() - start
                
                # Record failed execution
                _monitor.record(operation_name, duration, {
                    'status': 'error',
                    'error': str(e)
                })
                
                logger.error(f"{operation_name} failed after {duration:.3f}s: {str(e)}")
                raise
        
        return wrapper
    return decorator

class TimedContext:
    """Context manager for timing code blocks"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        metadata = {'status': 'success' if exc_type is None else 'error'}
        if exc_type:
            metadata['error'] = str(exc_val)
        
        _monitor.record(self.operation_name, duration, metadata)
        
        return False  # Don't suppress exceptions

# Example usage:
# with TimedContext("data_processing"):
#     process_data()
