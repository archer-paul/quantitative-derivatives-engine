"""
Heston stochastic volatility model implementation.

This module implements the Heston model for options pricing with stochastic volatility,
including both Monte Carlo simulation and characteristic function methods.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
from scipy import integrate

from ..core.market_data import MarketData, OptionType, validate_option_type


@dataclass
class HestonParameters:
    """Parameters for the Heston stochastic volatility model."""
    v0: float       # Initial variance
    theta: float    # Long-term variance
    kappa: float    # Mean reversion speed
    sigma_v: float  # Volatility of variance
    rho: float      # Correlation between asset and variance processes


class HestonModel:
    """
    Heston stochastic volatility model implementation.
    
    The Heston model assumes the following dynamics:
    dS_t = (r - q)S_t dt + √v_t S_t dW_t^S
    dv_t = κ(θ - v_t)dt + σ_v √v_t dW_t^v
    
    where dW_t^S and dW_t^v have correlation ρ.
    """
    
    def __init__(self, heston_params: HestonParameters):
        """
        Initialize Heston model with parameters.
        
        Args:
            heston_params: Heston model parameters
        """
        self.params = heston_params
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate Heston model parameters."""
        p = self.params
        
        if p.v0 <= 0:
            raise ValueError("Initial variance v0 must be positive")
        if p.theta <= 0:
            raise ValueError("Long-term variance theta must be positive")
        if p.kappa <= 0:
            raise ValueError("Mean reversion speed kappa must be positive")
        if p.sigma_v <= 0:
            raise ValueError("Volatility of variance sigma_v must be positive")
        if abs(p.rho) >= 1:
            raise ValueError("Correlation rho must be in (-1, 1)")
        
        # Feller condition for non-negative variance
        if 2 * p.kappa * p.theta <= p.sigma_v**2:
            import warnings
            warnings.warn("Feller condition violated: variance may become negative")
    
    def characteristic_function(self, u: complex, market_data: MarketData) -> complex:
        """
        Heston characteristic function for Fourier-based pricing.
        
        Args:
            u: Frequency parameter (complex)
            market_data: Market conditions
            
        Returns:
            Characteristic function value
        """
        S0, K, T, r, q = market_data.S0, market_data.K, market_data.T, market_data.r, market_data.q
        v0, theta, kappa, sigma_v, rho = (
            self.params.v0, self.params.theta, self.params.kappa,
            self.params.sigma_v, self.params.rho
        )
        
        # Complex calculations for characteristic function
        i = 1j
        
        # Auxiliary variables
        xi = kappa - rho * sigma_v * u * i
        d = np.sqrt(xi**2 + sigma_v**2 * (u * i + u**2))
        
        # Avoid division by zero
        if abs(d) < 1e-10:
            return 0.0
        
        g = (xi - d) / (xi + d)
        
        # Calculate components
        A = u * i * (np.log(S0) + (r - q) * T)
        
        exp_dT = np.exp(-d * T)
        B = (theta * kappa / (sigma_v**2)) * (
            (xi - d) * T - 2 * np.log((1 - g * exp_dT) / (1 - g))
        )
        
        C = (v0 / (sigma_v**2)) * (xi - d) * (1 - exp_dT) / (1 - g * exp_dT)
        
        return np.exp(A + B + C)
    
    def price_fft(self, market_data: MarketData, option_type: str, N: int = 2**12) -> float:
        """
        Price options using FFT method (Carr-Madan approach).
        
        Args:
            market_data: Market conditions
            option_type: 'call' or 'put'
            N: Number of FFT points
            
        Returns:
            Option price
        """
        option_type = validate_option_type(option_type)
        S0, K, T, r = market_data.S0, market_data.K, market_data.T, market_data.r
        
        # FFT parameters
        alpha = 1.5  # Damping parameter for calls
        eta = 0.15   # Grid spacing in frequency domain
        lambda_val = 2 * np.pi / (N * eta)
        
        # Frequency and log-strike grids
        v = np.arange(N) * eta
        k = -lambda_val * N / 2 + lambda_val * np.arange(N)
        
        # Modified characteristic function for FFT
        def psi(u):
            try:
                char_func = self.characteristic_function(u - (alpha + 1) * 1j, market_data)
                denominator = alpha**2 + alpha - u**2 + 1j * (2 * alpha + 1) * u
                if abs(denominator) < 1e-10:
                    return 0.0
                return char_func / denominator
            except:
                return 0.0
        
        # Prepare for FFT
        x = np.zeros(N, dtype=complex)
        for i in range(N):
            x[i] = np.exp(1j * v[i] * (-lambda_val * N / 2)) * psi(v[i]) * eta
        
        # Apply FFT
        y = np.fft.fft(x)
        
        # Extract option prices
        strikes = S0 * np.exp(k)
        call_prices = np.real(np.exp(-alpha * k - r * T) * y / np.pi)
        
        # Interpolate to get price for specific strike
        if K < strikes[0] or K > strikes[-1]:
            # Fallback to Monte Carlo if strike is outside FFT range
            return self.monte_carlo_price(market_data, option_type, n_paths=50000)[0]
        
        call_price = np.interp(K, strikes, call_prices)
        
        if option_type == OptionType.PUT:
            # Put-call parity: P = C - S*e^(-qT) + K*e^(-rT)
            call_price = call_price - S0 * np.exp(-market_data.q * T) + K * np.exp(-r * T)
        
        return max(call_price, 0)
    
    def monte_carlo_price(
        self, 
        market_data: MarketData, 
        option_type: str,
        n_paths: int = 100000, 
        n_steps: int = 252
    ) -> Tuple[float, float]:
        """
        Price options using Monte Carlo simulation with Euler scheme.
        
        Args:
            market_data: Market conditions
            option_type: 'call' or 'put'
            n_paths: Number of simulation paths
            n_steps: Number of time steps per path
            
        Returns:
            Tuple of (option_price, confidence_interval)
        """
        option_type = validate_option_type(option_type)
        S0, K, T, r, q = market_data.S0, market_data.K, market_data.T, market_data.r, market_data.q
        v0, theta, kappa, sigma_v, rho = (
            self.params.v0, self.params.theta, self.params.kappa,
            self.params.sigma_v, self.params.rho
        )
        
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Pre-generate correlated random numbers
        np.random.seed(42)  # For reproducibility
        Z1 = np.random.standard_normal((n_paths, n_steps))
        Z2 = np.random.standard_normal((n_paths, n_steps))
        W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2  # Correlated Brownian motion
        
        # Initialize paths
        S = np.ones((n_paths, n_steps + 1)) * S0
        v = np.ones((n_paths, n_steps + 1)) * v0
        
        # Euler discretization with full truncation scheme
        for t in range(n_steps):
            v_pos = np.maximum(v[:, t], 0)  # Full truncation scheme
            sqrt_v = np.sqrt(v_pos)
            
            # Asset price evolution
            S[:, t + 1] = S[:, t] * np.exp(
                (r - q - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * Z1[:, t]
            )
            
            # Variance evolution with full truncation
            v[:, t + 1] = v[:, t] + kappa * (theta - v_pos) * dt + \
                         sigma_v * sqrt_v * sqrt_dt * W2[:, t]
        
        # Calculate payoffs
        if option_type == OptionType.CALL:
            payoffs = np.maximum(S[:, -1] - K, 0)
        else:
            payoffs = np.maximum(K - S[:, -1], 0)
        
        # Discount to present value
        option_prices = np.exp(-r * T) * payoffs
        price = np.mean(option_prices)
        std_error = np.std(option_prices) / np.sqrt(n_paths)
        confidence_interval = 1.96 * std_error  # 95% CI
        
        return price, confidence_interval