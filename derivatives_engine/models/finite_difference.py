"""
Finite difference methods for option pricing PDEs.

This module implements numerical methods for solving the Black-Scholes PDE
using finite difference schemes.
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from typing import Tuple

from ..core.market_data import MarketData, OptionType, validate_option_type


class FiniteDifferencePricer:
    """Finite difference methods for option pricing."""
    
    @staticmethod
    def implicit_fd_vanilla(
        market_data: MarketData,
        option_type: str,
        S_max: float = 300,
        M: int = 100,
        N: int = 1000
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Implicit finite difference method for vanilla European options.
        
        Solves the Black-Scholes PDE:
        ∂V/∂t + (1/2)σ²S²∂²V/∂S² + (r-q)S∂V/∂S - rV = 0
        
        Args:
            market_data: Market conditions
            option_type: 'call' or 'put'
            S_max: Maximum stock price in grid
            M: Number of stock price steps
            N: Number of time steps
            
        Returns:
            Tuple of (option_price, price_grid, stock_prices)
        """
        option_type = validate_option_type(option_type)
        S0, K, T, r, q, sigma = (
            market_data.S0, market_data.K, market_data.T,
            market_data.r, market_data.q, market_data.sigma
        )
        
        # Grid setup
        dS = S_max / M
        dt = T / N
        S = np.linspace(0, S_max, M + 1)
        
        # Initialize option values at expiration (boundary condition in time)
        if option_type == OptionType.CALL:
            V = np.maximum(S - K, 0)
        else:
            V = np.maximum(K - S, 0)
        
        # Store the grid for visualization
        V_grid = np.zeros((N + 1, M + 1))
        V_grid[N, :] = V.copy()
        
        # Build tridiagonal matrix for implicit scheme
        # The PDE discretized becomes: (I - dt*A)V^{n+1} = V^n
        
        # Coefficients for the tridiagonal matrix
        j = np.arange(1, M)  # Interior points (excluding boundaries)
        
        # Lower diagonal (coefficient of V_{j-1})
        alpha = 0.5 * dt * (sigma**2 * j**2 - (r - q) * j)
        
        # Main diagonal (coefficient of V_j)
        beta = 1 + dt * (sigma**2 * j**2 + r)
        
        # Upper diagonal (coefficient of V_{j+1})
        gamma = -0.5 * dt * (sigma**2 * j**2 + (r - q) * j)
        
        # Build the tridiagonal matrix
        A = diags([alpha[1:], beta, gamma[:-1]], [-1, 0, 1], 
                 shape=(M-1, M-1), format='csc')
        
        # Time stepping backwards from expiration to present
        for n in range(N-1, -1, -1):
            # Set boundary conditions
            if option_type == OptionType.CALL:
                V[0] = 0  # Call worthless at S=0
                V[M] = S_max - K * np.exp(-r * (N-n) * dt)  # Call at high S
            else:
                V[0] = K * np.exp(-r * (N-n) * dt)  # Put at S=0
                V[M] = 0  # Put worthless at high S
            
            # Right-hand side vector
            b = V[1:M].copy()
            
            # Adjust for boundary conditions
            b[0] -= alpha[0] * V[0]
            b[-1] -= gamma[-1] * V[M]
            
            # Solve the linear system
            V[1:M] = spsolve(A, b)
            
            # Store the solution
            V_grid[n, :] = V.copy()
        
        # Interpolate to get option price at S0
        option_price = np.interp(S0, S, V)
        
        return option_price, V_grid, S
    
    @staticmethod
    def explicit_fd_vanilla(
        market_data: MarketData,
        option_type: str,
        S_max: float = 300,
        M: int = 100,
        N: int = 10000  # More steps needed for stability
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Explicit finite difference method for vanilla European options.
        
        Note: Requires small time steps for stability (dt ≤ dS²/(σ²S_max²))
        
        Args:
            market_data: Market conditions
            option_type: 'call' or 'put'
            S_max: Maximum stock price in grid
            M: Number of stock price steps
            N: Number of time steps (should be large for stability)
            
        Returns:
            Tuple of (option_price, price_grid, stock_prices)
        """
        option_type = validate_option_type(option_type)
        S0, K, T, r, q, sigma = (
            market_data.S0, market_data.K, market_data.T,
            market_data.r, market_data.q, market_data.sigma
        )
        
        # Grid setup
        dS = S_max / M
        dt = T / N
        S = np.linspace(0, S_max, M + 1)
        
        # Stability check
        max_dt = 0.5 * dS**2 / (sigma**2 * S_max**2)
        if dt > max_dt:
            import warnings
            warnings.warn(f"Time step dt={dt:.6f} may be too large for stability. "
                         f"Consider dt ≤ {max_dt:.6f}")
        
        # Initialize option values at expiration
        if option_type == OptionType.CALL:
            V = np.maximum(S - K, 0)
        else:
            V = np.maximum(K - S, 0)
        
        # Store the grid
        V_grid = np.zeros((N + 1, M + 1))
        V_grid[N, :] = V.copy()
        
        # Time stepping backwards
        for n in range(N-1, -1, -1):
            V_new = V.copy()
            
            # Update interior points
            for j in range(1, M):
                # Finite difference coefficients
                dV_dS = (V[j+1] - V[j-1]) / (2 * dS)
                d2V_dS2 = (V[j+1] - 2*V[j] + V[j-1]) / (dS**2)
                
                # PDE terms
                drift_term = (r - q) * S[j] * dV_dS
                diffusion_term = 0.5 * sigma**2 * S[j]**2 * d2V_dS2
                discount_term = r * V[j]
                
                # Explicit update
                V_new[j] = V[j] + dt * (drift_term + diffusion_term - discount_term)
            
            # Boundary conditions
            if option_type == OptionType.CALL:
                V_new[0] = 0
                V_new[M] = S_max - K * np.exp(-r * (N-n) * dt)
            else:
                V_new[0] = K * np.exp(-r * (N-n) * dt)
                V_new[M] = 0
            
            V = V_new
            V_grid[n, :] = V.copy()
        
        # Interpolate to get option price at S0
        option_price = np.interp(S0, S, V)
        
        return option_price, V_grid, S
    
    @staticmethod
    def crank_nicolson_vanilla(
        market_data: MarketData,
        option_type: str,
        S_max: float = 300,
        M: int = 100,
        N: int = 1000
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Crank-Nicolson method for vanilla European options.
        
        This method is second-order accurate in both time and space,
        and is unconditionally stable.
        
        Args:
            market_data: Market conditions
            option_type: 'call' or 'put'
            S_max: Maximum stock price in grid
            M: Number of stock price steps
            N: Number of time steps
            
        Returns:
            Tuple of (option_price, price_grid, stock_prices)
        """
        option_type = validate_option_type(option_type)
        S0, K, T, r, q, sigma = (
            market_data.S0, market_data.K, market_data.T,
            market_data.r, market_data.q, market_data.sigma
        )
        
        # Grid setup
        dS = S_max / M
        dt = T / N
        S = np.linspace(0, S_max, M + 1)
        
        # Initialize option values at expiration
        if option_type == OptionType.CALL:
            V = np.maximum(S - K, 0)
        else:
            V = np.maximum(K - S, 0)
        
        V_grid = np.zeros((N + 1, M + 1))
        V_grid[N, :] = V.copy()
        
        # Build matrices for Crank-Nicolson scheme
        j = np.arange(1, M)
        
        # Coefficients
        alpha = 0.25 * dt * (sigma**2 * j**2 - (r - q) * j)
        beta = 0.5 * dt * (sigma**2 * j**2 + r)
        gamma = 0.25 * dt * (sigma**2 * j**2 + (r - q) * j)
        
        # Left-hand side matrix (implicit part)
        A = diags([alpha[1:], 1 + beta, -gamma[:-1]], [-1, 0, 1], 
                 shape=(M-1, M-1), format='csc')
        
        # Right-hand side matrix (explicit part)
        B = diags([-alpha[1:], 1 - beta, gamma[:-1]], [-1, 0, 1], 
                 shape=(M-1, M-1), format='csc')
        
        # Time stepping
        for n in range(N-1, -1, -1):
            # Boundary conditions
            if option_type == OptionType.CALL:
                V[0] = 0
                V[M] = S_max - K * np.exp(-r * (N-n) * dt)
            else:
                V[0] = K * np.exp(-r * (N-n) * dt)
                V[M] = 0
            
            # Right-hand side
            b = B @ V[1:M]
            
            # Adjust for boundary conditions
            b[0] += alpha[0] * (V[0] + V[0])  # V[0] at both time levels
            b[-1] += gamma[-1] * (V[M] + V[M])  # V[M] at both time levels
            
            # Solve linear system
            V[1:M] = spsolve(A, b)
            
            V_grid[n, :] = V.copy()
        
        # Interpolate to get option price at S0
        option_price = np.interp(S0, S, V)
        
        return option_price, V_grid, S