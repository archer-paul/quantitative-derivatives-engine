"""
Merton jump-diffusion model implementation.

This module implements the Merton jump-diffusion model for option pricing,
which extends the Black-Scholes model to include jump processes in asset prices.
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple

from ..core.market_data import MarketData, OptionType, validate_option_type
from .black_scholes import BlackScholesModel


@dataclass
class JumpDiffusionParameters:
    """Parameters for the Merton jump-diffusion model."""
    lambda_j: float  # Jump intensity (jumps per year)
    mu_j: float      # Mean jump size (log-normal)
    sigma_j: float   # Jump size volatility


class MertonJumpDiffusionModel:
    """
    Merton jump-diffusion model implementation.
    
    The model assumes the asset follows:
    dS_t = (r - q - λm)S_t dt + σS_t dW_t + S_t ∫(e^J - 1) N(dt, dJ)
    
    where N(dt, dJ) is a Poisson random measure and J ~ N(μ_J, σ_J²).
    """
    
    def __init__(self, jump_params: JumpDiffusionParameters):
        """
        Initialize jump-diffusion model with parameters.
        
        Args:
            jump_params: Jump process parameters
        """
        self.params = jump_params
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate jump-diffusion parameters."""
        p = self.params
        
        if p.lambda_j < 0:
            raise ValueError("Jump intensity lambda_j must be non-negative")
        if p.sigma_j < 0:
            raise ValueError("Jump volatility sigma_j must be non-negative")
    
    def price_analytical(
        self, 
        market_data: MarketData, 
        option_type: str, 
        max_jumps: int = 50
    ) -> float:
        """
        Analytical pricing using infinite series expansion.
        
        The price is computed as a weighted sum of Black-Scholes prices
        with different adjusted parameters for each possible number of jumps.
        
        Args:
            market_data: Market conditions
            option_type: 'call' or 'put'
            max_jumps: Maximum number of jumps to consider
            
        Returns:
            Option price
        """
        option_type = validate_option_type(option_type)
        S0, K, T, r, q, sigma = (
            market_data.S0, market_data.K, market_data.T,
            market_data.r, market_data.q, market_data.sigma
        )
        lambda_j, mu_j, sigma_j = self.params.lambda_j, self.params.mu_j, self.params.sigma_j
        
        # Expected jump size
        m = np.exp(mu_j + 0.5 * sigma_j**2) - 1
        
        total_price = 0.0
        
        for n in range(max_jumps):
            # Poisson probability for exactly n jumps
            poisson_prob = np.exp(-lambda_j * T) * (lambda_j * T)**n / math.factorial(n)
            
            # Early termination for small probabilities
            if poisson_prob < 1e-10:
                break
            
            # Adjusted parameters for n jumps
            sigma_n = np.sqrt(sigma**2 + n * sigma_j**2 / T)
            r_n = r - lambda_j * m + n * (mu_j + 0.5 * sigma_j**2) / T
            
            # Black-Scholes price with adjusted parameters
            market_n = MarketData(S0, K, T, r_n, q, sigma_n)
            bs_price = BlackScholesModel.price(market_n, option_type)
            
            total_price += poisson_prob * bs_price
        
        return total_price
    
    def monte_carlo_price(
        self,
        market_data: MarketData,
        option_type: str,
        n_paths: int = 100000,
        n_steps: int = 252
    ) -> Tuple[float, float]:
        """
        Monte Carlo pricing with jump processes.
        
        Args:
            market_data: Market conditions
            option_type: 'call' or 'put'
            n_paths: Number of simulation paths
            n_steps: Number of time steps per path
            
        Returns:
            Tuple of (option_price, confidence_interval)
        """
        option_type = validate_option_type(option_type)
        S0, K, T, r, q, sigma = (
            market_data.S0, market_data.K, market_data.T,
            market_data.r, market_data.q, market_data.sigma
        )
        lambda_j, mu_j, sigma_j = self.params.lambda_j, self.params.mu_j, self.params.sigma_j
        
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Expected jump size
        m = np.exp(mu_j + 0.5 * sigma_j**2) - 1
        
        # Drift adjustment for jumps
        drift = (r - q - lambda_j * m - 0.5 * sigma**2) * dt
        
        np.random.seed(42)  # For reproducibility
        
        # Initialize paths
        S = np.ones((n_paths, n_steps + 1)) * S0
        
        # Generate paths with jumps
        for path in range(n_paths):
            for step in range(n_steps):
                # Diffusion component
                dW = np.random.standard_normal() * sqrt_dt
                diffusion = sigma * dW
                
                # Jump component
                n_jumps = np.random.poisson(lambda_j * dt)
                
                if n_jumps > 0:
                    # Generate jump sizes
                    jump_sizes = np.random.normal(mu_j, sigma_j, n_jumps)
                    total_jump = np.sum(np.exp(jump_sizes) - 1)
                else:
                    total_jump = 0
                
                # Update stock price
                S[path, step + 1] = S[path, step] * np.exp(drift + diffusion) * (1 + total_jump)
        
        # Calculate payoffs
        if option_type == OptionType.CALL:
            payoffs = np.maximum(S[:, -1] - K, 0)
        else:
            payoffs = np.maximum(K - S[:, -1], 0)
        
        # Discount to present value
        option_prices = np.exp(-r * T) * payoffs
        price = np.mean(option_prices)
        std_error = np.std(option_prices) / np.sqrt(n_paths)
        confidence_interval = 1.96 * std_error
        
        return price, confidence_interval