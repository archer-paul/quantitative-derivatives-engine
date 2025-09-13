"""
Logging configuration module for the derivatives pricing engine.

This module provides centralized logging configuration with different
levels, formatters, and handlers for various components.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import os


class PricingEngineLogger:
    """Centralized logging configuration for the pricing engine."""
    
    _loggers = {}
    _configured = False
    
    @classmethod
    def get_logger(cls, name: str, level: Optional[str] = None) -> logging.Logger:
        """Get or create a logger with the specified name."""
        if name not in cls._loggers:
            cls._loggers[name] = cls._create_logger(name, level)
        return cls._loggers[name]
    
    @classmethod
    def _create_logger(cls, name: str, level: Optional[str] = None) -> logging.Logger:
        """Create a new logger with appropriate configuration."""
        logger = logging.getLogger(name)
        
        if not cls._configured:
            cls._configure_root_logger()
            cls._configured = True
        
        # Set level if specified
        if level:
            logger.setLevel(getattr(logging, level.upper()))
        
        return logger
    
    @classmethod
    def _configure_root_logger(cls):
        """Configure the root logger with appropriate handlers and formatters."""
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler with UTF-8 encoding
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler for general logs with UTF-8 encoding
        file_handler = logging.FileHandler(logs_dir / "pricing_engine.log", encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Error file handler with UTF-8 encoding
        error_handler = logging.FileHandler(logs_dir / "errors.log", encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
    
    @classmethod
    def set_level(cls, logger_name: str, level: str):
        """Set the logging level for a specific logger."""
        if logger_name in cls._loggers:
            cls._loggers[logger_name].setLevel(getattr(logging, level.upper()))
    
    @classmethod
    def enable_debug_mode(cls):
        """Enable debug mode for all loggers."""
        logging.getLogger().setLevel(logging.DEBUG)
        for logger in cls._loggers.values():
            logger.setLevel(logging.DEBUG)


def log_function_call(func):
    """Decorator to log function calls with parameters and execution time."""
    def wrapper(*args, **kwargs):
        logger = PricingEngineLogger.get_logger(func.__module__)
        func_name = f"{func.__qualname__}"
        
        # Log function entry
        logger.debug(f"Entering {func_name} with args={args}, kwargs={kwargs}")
        
        try:
            import time
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            # Log successful execution
            logger.debug(f"Completed {func_name} in {end_time - start_time:.6f} seconds")
            return result
            
        except Exception as e:
            # Log the error
            logger.error(f"Error in {func_name}: {str(e)}", exc_info=True)
            raise
    
    return wrapper


def log_performance(operation_name: str):
    """Decorator to log performance metrics for specific operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = PricingEngineLogger.get_logger("performance")
            
            import time
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                
                logger.info(f"{operation_name} completed in {execution_time:.6f} seconds")
                return result
                
            except Exception as e:
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                logger.error(f"{operation_name} failed after {execution_time:.6f} seconds: {str(e)}")
                raise
        
        return wrapper
    return decorator


class ContextualLogger:
    """Context manager for logging with additional context information."""
    
    def __init__(self, logger_name: str, operation: str, **context):
        self.logger = PricingEngineLogger.get_logger(logger_name)
        self.operation = operation
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
        self.logger.info(f"Starting {self.operation} with context: {context_str}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        end_time = time.perf_counter()
        duration = end_time - self.start_time
        
        if exc_type is None:
            self.logger.info(f"Completed {self.operation} successfully in {duration:.6f} seconds")
        else:
            self.logger.error(f"Failed {self.operation} after {duration:.6f} seconds: {exc_val}")
        
        return False  # Don't suppress exceptions


# Module-level convenience functions
def get_logger(name: str = None) -> logging.Logger:
    """Get a logger for the calling module."""
    if name is None:
        # Get the caller's module name
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'derivatives_engine')
    
    return PricingEngineLogger.get_logger(name)