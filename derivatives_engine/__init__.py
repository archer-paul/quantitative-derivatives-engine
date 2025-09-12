"""
Quantitative Derivatives Engine

A comprehensive derivatives pricing engine implementing advanced mathematical models
for vanilla and exotic options pricing, with automatic differentiation for Greeks
calculation and comprehensive risk analysis capabilities.

Main Components:
- Black-Scholes model with automatic differentiation
- Heston stochastic volatility model
- Merton jump-diffusion model
- Exotic options pricing (barrier, Asian, lookback)
- Portfolio risk analysis and Greeks aggregation
- Finite difference methods for PDE solving

Author: [Your Name]
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core imports
from .core.market_data import MarketData, OptionType
from .core.auto_diff import DualNumber
from .core.pricing_engine import DerivativesPricingEngine

# Model imports
from .models.black_scholes import BlackScholesModel
from .models.heston import HestonModel, HestonParameters
from .models.jump_diffusion import MertonJumpDiffusionModel, JumpDiffusionParameters
from .models.finite_difference import FiniteDifferencePricer

# Exotic options imports
from .exotic.exotic_engine import ExoticOptions

# Utility imports (optional dependencies)
try:
    from .utils.visualization import VisualizationTools
except ImportError:
    VisualizationTools = None

try:
    from .utils.performance import PerformanceBenchmark
except ImportError:
    PerformanceBenchmark = None
    
try:
    from .utils.risk_analysis import RiskAnalyzer
except ImportError:
    RiskAnalyzer = None

# Make key classes available at package level
__all__ = [
    # Core classes
    "DerivativesPricingEngine",
    "MarketData",
    "OptionType",
    "DualNumber",
    
    # Pricing models
    "BlackScholesModel",
    "HestonModel",
    "MertonJumpDiffusionModel",
    "FiniteDifferencePricer",
    
    # Model parameters
    "HestonParameters",
    "JumpDiffusionParameters",
    
    # Exotic options
    "ExoticOptions",
    
    # Utilities (may be None if dependencies not installed)
    "VisualizationTools",
    "PerformanceBenchmark", 
    "RiskAnalyzer",
]

# Package metadata
__title__ = "quantitative-derivatives-engine"
__description__ = "Advanced derivatives pricing engine with multiple models and automatic differentiation"
__url__ = "https://github.com/yourusername/quantitative-derivatives-engine"
__license__ = "MIT"
__copyright__ = "Copyright 2024, [Your Name]"