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
        if self.S0 <= 0:
            raise ValueError("Stock price S0 must be positive")
        if self.K <= 0:
            raise ValueError("Strike price K must be positive")
        if self.T < 0:
            raise ValueError("Time to expiration T must be non-negative")
        if self.sigma < 0:
            raise ValueError("Volatility sigma must be non-negative")
        if self.q < 0:
            raise ValueError("Dividend yield q must be non-negative")
    
    @property
    def moneyness(self) -> float:
        """Calculate the moneyness (S0/K) of the option."""
        return self.S0 / self.K
    
    @property 
    def forward_price(self) -> float:
        """Calculate the forward price of the underlying asset."""
        import math
        return self.S0 * math.exp((self.r - self.q) * self.T)
    
    def is_itm(self, option_type: str) -> bool:
        """
        Check if the option is in-the-money.
        
        Args:
            option_type: Either OptionType.CALL or OptionType.PUT
            
        Returns:
            bool: True if option is in-the-money
        """
        if option_type == OptionType.CALL:
            return self.S0 > self.K
        elif option_type == OptionType.PUT:
            return self.S0 < self.K
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