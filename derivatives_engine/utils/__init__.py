# derivatives_engine/utils/__init__.py  
"""
Utility modules for the derivatives engine.

This module contains helper functions for visualization, performance analysis,
and risk management calculations.
"""

try:
    from .performance import PerformanceBenchmark, PerformanceMetrics
    _PERFORMANCE_AVAILABLE = True
except ImportError:
    PerformanceBenchmark = None
    PerformanceMetrics = None
    _PERFORMANCE_AVAILABLE = False

try:
    from .visualization import VisualizationTools
    _VISUALIZATION_AVAILABLE = True
except ImportError:
    VisualizationTools = None
    _VISUALIZATION_AVAILABLE = False

try:
    from .risk_analysis import RiskAnalyzer, VaRResult, StressTestResult
    _RISK_ANALYSIS_AVAILABLE = True
except ImportError:
    RiskAnalyzer = None
    VaRResult = None
    StressTestResult = None
    _RISK_ANALYSIS_AVAILABLE = False

__all__ = [
    'PerformanceBenchmark',
    'PerformanceMetrics', 
    'VisualizationTools',
    'RiskAnalyzer',
    'VaRResult', 
    'StressTestResult'
]