# derivatives_engine/exotic/__init__.py
"""
Exotic options pricing module.

This module contains implementations for pricing various exotic option types
using Monte Carlo simulation and other numerical methods.
"""

from .exotic_engine import ExoticOptions

__all__ = [
    "ExoticOptions",
]