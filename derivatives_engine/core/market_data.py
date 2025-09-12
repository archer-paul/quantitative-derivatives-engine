"""
Market data structures and option type definitions.

This module defines the core data structures used throughout the pricing engine
to represent market conditions and option specifications.
"""

from dataclasses import dataclass
from typing import Union


class OptionType:
    """Option type constants."""
    CALL = "call"
    PUT = "put"


@dataclass
class MarketData:
    """
    Container for market data and option specifications.
    
    This class holds all the market parameters required for option pricing,
    including the current asset price, strike price, time to expiration,
    risk-free rate, dividend yield, and volatility.
    
    Attributes:
        S0 (float): Current stock price
        K (float): Strike price  
        T (float): Time to expiration (in years)
        r (float): Risk-free interest rate (annualized)
        q (float): Dividend yield (annualized, default: 0.0)
        sigma (float): Volatility (annualized, default: 0.2)
    
    Example:
        >>> market = MarketData(S0=100, K=105, T=0.25, r=0.05, q=0.02, sigma=0.20)
        >>> print(f"Stock price: ${market.S0}")
        Stock price: $100
    """
    
    S0: float          # Current stock price
    K: float           # Strike price
    T: float           # Time to expiration (years)
    r: float           # Risk-free rate (annualized)
    q: float = 0.0     # Dividend yield (annualized)
    sigma: float = 0.2 # Volatility (annualized)
    
    def __post_init__(self):
        """Validate market data inputs after initialization."""
        from .auto_diff import DualNumber
        
        # Extract real values for validation if DualNumber objects are used
        S0_val = self.S0.real if isinstance(self.S0, DualNumber) else self.S0
        K_val = self.K.real if isinstance(self.K, DualNumber) else self.K
        T_val = self.T.real if isinstance(self.T, DualNumber) else self.T
        sigma_val = self.sigma.real if isinstance(self.sigma, DualNumber) else self.sigma
        q_val = self.q.real if isinstance(self.q, DualNumber) else self.q
        
        if S0_val <= 0:
            raise ValueError("Stock price S0 must be positive")
        if K_val <= 0:
            raise ValueError("Strike price K must be positive")
        if T_val < 0:
            raise ValueError("Time to expiration T must be non-negative")
        if sigma_val < 0:
            raise ValueError("Volatility sigma must be non-negative")
        if q_val < 0:
            raise ValueError("Dividend yield q must be non-negative")
    
    @property
    def moneyness(self) -> float:
        """Calculate the moneyness (S0/K) of the option."""
        from .auto_diff import DualNumber
        S0_val = self.S0.real if isinstance(self.S0, DualNumber) else self.S0
        K_val = self.K.real if isinstance(self.K, DualNumber) else self.K
        return S0_val / K_val
    
    @property 
    def forward_price(self) -> float:
        """Calculate the forward price of the underlying asset."""
        import math
        from .auto_diff import DualNumber
        S0_val = self.S0.real if isinstance(self.S0, DualNumber) else self.S0
        r_val = self.r.real if isinstance(self.r, DualNumber) else self.r
        q_val = self.q.real if isinstance(self.q, DualNumber) else self.q
        T_val = self.T.real if isinstance(self.T, DualNumber) else self.T
        return S0_val * math.exp((r_val - q_val) * T_val)
    
    def is_itm(self, option_type: str) -> bool:
        """
        Check if the option is in-the-money.
        
        Args:
            option_type: Either OptionType.CALL or OptionType.PUT
            
        Returns:
            bool: True if option is in-the-money
        """
        from .auto_diff import DualNumber
        S0_val = self.S0.real if isinstance(self.S0, DualNumber) else self.S0
        K_val = self.K.real if isinstance(self.K, DualNumber) else self.K
        
        if option_type == OptionType.CALL:
            return S0_val > K_val
        elif option_type == OptionType.PUT:
            return S0_val < K_val
        else:
            raise ValueError(f"Invalid option type: {option_type}")
    
    def copy(self, **kwargs) -> 'MarketData':
        """
        Create a copy of the market data with optional parameter updates.
        
        Args:
            **kwargs: Parameters to update in the copy
            
        Returns:
            MarketData: New instance with updated parameters
        """
        params = {
            'S0': self.S0,
            'K': self.K, 
            'T': self.T,
            'r': self.r,
            'q': self.q,
            'sigma': self.sigma
        }
        params.update(kwargs)
        return MarketData(**params)
    
    def __repr__(self) -> str:
        """String representation of market data."""
        return (f"MarketData(S0={self.S0}, K={self.K}, T={self.T:.4f}, "
                f"r={self.r:.4f}, q={self.q:.4f}, sigma={self.sigma:.4f})")


def validate_option_type(option_type: str) -> str:
    """
    Validate and normalize option type string.
    
    Args:
        option_type: Option type string to validate
        
    Returns:
        str: Normalized option type
        
    Raises:
        ValueError: If option type is not valid
    """
    option_type = option_type.lower().strip()
    if option_type not in [OptionType.CALL, OptionType.PUT]:
        raise ValueError(f"Invalid option type: {option_type}. Must be 'call' or 'put'")
    return option_type