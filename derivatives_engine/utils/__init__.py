# derivatives_engine/utils/__init__.py  
"""
Utility modules for the derivatives engine.

This module contains helper functions for visualization, performance analysis,
risk management calculations, logging, validation, configuration, and error handling.
"""

# Core robustness utilities (always available)
from .exceptions import (
    PricingEngineError, ValidationError, ModelError, CalibrationError,
    DataError, ConvergenceError, ParameterError, NumericalError,
    ConfigurationError, DependencyError
)

from .logging_config import PricingEngineLogger, get_logger, log_function_call, log_performance

from .config import (
    PricingEngineConfig, NumericalConfig, CacheConfig, LoggingConfig,
    DataConfig, PerformanceConfig, ConfigManager, get_config
)

from .validation import InputValidator, MarketDataValidator, validate_inputs, validate_market_data_input

# Optional performance utilities
try:
    from .performance import PerformanceBenchmark, PerformanceMetrics
    _PERFORMANCE_AVAILABLE = True
except ImportError:
    PerformanceBenchmark = None
    PerformanceMetrics = None
    _PERFORMANCE_AVAILABLE = False

# Optional visualization utilities
try:
    from .visualization import VisualizationTools
    _VISUALIZATION_AVAILABLE = True
except ImportError:
    VisualizationTools = None
    _VISUALIZATION_AVAILABLE = False

# Optional risk analysis utilities
try:
    from .risk_analysis import RiskAnalyzer, VaRResult, StressTestResult
    _RISK_ANALYSIS_AVAILABLE = True
except ImportError:
    RiskAnalyzer = None
    VaRResult = None
    StressTestResult = None
    _RISK_ANALYSIS_AVAILABLE = False

# Optional optimization utilities
try:
    from .optimizations import (
        OptimizedPricingEngine, FastMonteCarlo, CacheManager as OptimizedCacheManager,
        BenchmarkSuite
    )
    _OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    OptimizedPricingEngine = None
    FastMonteCarlo = None
    OptimizedCacheManager = None
    BenchmarkSuite = None
    _OPTIMIZATIONS_AVAILABLE = False

__all__ = [
    # Core robustness
    'PricingEngineError', 'ValidationError', 'ModelError', 'CalibrationError',
    'DataError', 'ConvergenceError', 'ParameterError', 'NumericalError',
    'ConfigurationError', 'DependencyError',
    'PricingEngineLogger', 'get_logger', 'log_function_call', 'log_performance',
    'PricingEngineConfig', 'NumericalConfig', 'CacheConfig', 'LoggingConfig',
    'DataConfig', 'PerformanceConfig', 'ConfigManager', 'get_config',
    'InputValidator', 'MarketDataValidator', 'validate_inputs', 'validate_market_data_input',
    
    # Optional utilities
    'PerformanceBenchmark', 'PerformanceMetrics', 
    'VisualizationTools',
    'RiskAnalyzer', 'VaRResult', 'StressTestResult',
    'OptimizedPricingEngine', 'FastMonteCarlo', 'OptimizedCacheManager', 'BenchmarkSuite'
]