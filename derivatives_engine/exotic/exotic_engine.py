"""
Exotic options pricing engine using Monte Carlo simulation.

This module provides pricing for various exotic option types including
barrier options, Asian options, and lookback options.
"""

import numpy as np
from typing import Tuple
from scipy import stats

from ..core.market_data import MarketData, OptionType, validate_option_type


class ExoticOptions:
    """Exotic options pricing using Monte Carlo simulation."""
    
    @staticmethod
    def barrier_option_mc(
        market_data: MarketData,
        option_type: str,
        barrier_type: str,
        barrier_level: float,
        n_paths: int = 100000,
        n_steps: int = 1000
    ) -> Tuple[float, float]:
        """
        Price barrier options using Monte Carlo simulation.
        
        Args:
            market_data: Market conditions
            option_type: 'call' or 'put'
            barrier_type: 'up_in', 'up_out', 'down_in', 'down_out'
            barrier_level: Barrier price level
            n_paths: Number of Monte Carlo paths
            n_steps: Number of time steps per path
            
        Returns:
            Tuple of (option_price, confidence_interval)
        """
        option_type = validate_option_type(option_type)
        S0, K, T, r, q, sigma = (
            market_data.S0, market_data.K, market_data.T,
            market_data.r, market_data.q, market_data.sigma
        )
        
        valid_barriers = ['up_in', 'up_out', 'down_in', 'down_out']
        if barrier_type not in valid_barriers:
            raise ValueError(f"Invalid barrier type. Must be one of {valid_barriers}")
        
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        drift = (r - q - 0.5 * sigma**2) * dt
        
        # Pre-generate random numbers for efficiency
        np.random.seed(42)  # For reproducibility
        random_matrix = np.random.standard_normal((n_paths, n_steps))
        
        payoffs = np.zeros(n_paths)
        
        for path in range(n_paths):
            S_path = [S0]
            S_current = S0
            barrier_hit = False
            
            # Generate price path
            for step in range(n_steps):
                dW = random_matrix[path, step]
                S_current = S_current * np.exp(drift + sigma * sqrt_dt * dW)
                S_path.append(S_current)
                
                # Check barrier condition
                if barrier_type in ['up_in', 'up_out'] and S_current >= barrier_level:
                    barrier_hit = True
                elif barrier_type in ['down_in', 'down_out'] and S_current <= barrier_level:
                    barrier_hit = True
            
            # Calculate payoff based on barrier type and option type
            if option_type == OptionType.CALL:
                intrinsic_value = max(S_current - K, 0)
            else:
                intrinsic_value = max(K - S_current, 0)
            
            if barrier_type in ['up_in', 'down_in']:
                # Knock-in: option becomes active if barrier is hit
                payoffs[path] = intrinsic_value if barrier_hit else 0
            else:
                # Knock-out: option becomes worthless if barrier is hit
                payoffs[path] = intrinsic_value if not barrier_hit else 0
        
        # Discount to present value
        option_prices = np.exp(-r * T) * payoffs
        price = np.mean(option_prices)
        std_error = np.std(option_prices) / np.sqrt(n_paths)
        confidence_interval = 1.96 * std_error  # 95% CI
        
        return price, confidence_interval
    
    @staticmethod
    def asian_option_mc(
        market_data: MarketData,
        option_type: str,
        avg_type: str = 'arithmetic',
        n_paths: int = 100000,
        n_steps: int = 252
    ) -> Tuple[float, float]:
        """
        Price Asian options using Monte Carlo simulation.
        
        Args:
            market_data: Market conditions
            option_type: 'call' or 'put'
            avg_type: 'arithmetic' or 'geometric'
            n_paths: Number of Monte Carlo paths
            n_steps: Number of averaging points
            
        Returns:
            Tuple of (option_price, confidence_interval)
        """
        option_type = validate_option_type(option_type)
        S0, K, T, r, q, sigma = (
            market_data.S0, market_data.K, market_data.T,
            market_data.r, market_data.q, market_data.sigma
        )
        
        if avg_type not in ['arithmetic', 'geometric']:
            raise ValueError("avg_type must be 'arithmetic' or 'geometric'")
        
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        drift = (r - q - 0.5 * sigma**2) * dt
        
        np.random.seed(42)
        payoffs = np.zeros(n_paths)
        
        for path in range(n_paths):
            S_path = np.zeros(n_steps + 1)
            S_path[0] = S0
            
            # Generate price path
            for step in range(n_steps):
                dW = np.random.standard_normal()
                S_path[step + 1] = S_path[step] * np.exp(drift + sigma * sqrt_dt * dW)
            
            # Calculate average price (excluding initial price)
            price_samples = S_path[1:]
            
            if avg_type == 'arithmetic':
                avg_price = np.mean(price_samples)
            else:  # geometric
                avg_price = np.exp(np.mean(np.log(price_samples)))
            
            # Calculate payoff
            if option_type == OptionType.CALL:
                payoffs[path] = max(avg_price - K, 0)
            else:
                payoffs[path] = max(K - avg_price, 0)
        
        # Discount to present value
        option_prices = np.exp(-r * T) * payoffs
        price = np.mean(option_prices)
        std_error = np.std(option_prices) / np.sqrt(n_paths)
        confidence_interval = 1.96 * std_error
        
        return price, confidence_interval
    
    @staticmethod
    def lookback_option_mc(
        market_data: MarketData,
        option_type: str,
        n_paths: int = 100000,
        n_steps: int = 252
    ) -> Tuple[float, float]:
        """
        Price lookback options using Monte Carlo simulation.
        
        Args:
            market_data: Market conditions
            option_type: 'lookback_call' or 'lookback_put'
            n_paths: Number of Monte Carlo paths
            n_steps: Number of time steps
            
        Returns:
            Tuple of (option_price, confidence_interval)
        """
        S0, K, T, r, q, sigma = (
            market_data.S0, market_data.K, market_data.T,
            market_data.r, market_data.q, market_data.sigma
        )
        
        valid_types = ['lookback_call', 'lookback_put']
        if option_type not in valid_types:
            raise ValueError(f"option_type must be one of {valid_types}")
        
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        drift = (r - q - 0.5 * sigma**2) * dt
        
        np.random.seed(42)
        payoffs = np.zeros(n_paths)
        
        for path in range(n_paths):
            S_path = np.zeros(n_steps + 1)
            S_path[0] = S0
            
            # Generate price path
            for step in range(n_steps):
                dW = np.random.standard_normal()
                S_path[step + 1] = S_path[step] * np.exp(drift + sigma * sqrt_dt * dW)
            
            # Calculate payoff based on lookback type
            if option_type == "lookback_call":
                # Payoff = S_max - K (floating strike) or max(S_T - S_min, 0) (fixed strike)
                payoffs[path] = max(np.max(S_path) - K, 0)
            elif option_type == "lookback_put":
                # Payoff = K - S_min (floating strike) or max(S_max - S_T, 0) (fixed strike)
                payoffs[path] = max(K - np.min(S_path), 0)
        
        # Discount to present value
        option_prices = np.exp(-r * T) * payoffs
        price = np.mean(option_prices)
        std_error = np.std(option_prices) / np.sqrt(n_paths)
        confidence_interval = 1.96 * std_error
        
        return price, confidence_interval
    
    @staticmethod
    def rainbow_option_mc(
        market_data_list: list,
        option_type: str,
        correlation_matrix: np.ndarray,
        n_paths: int = 100000,
        n_steps: int = 252
    ) -> Tuple[float, float]:
        """
        Price rainbow (multi-asset) options using Monte Carlo simulation.
        
        Args:
            market_data_list: List of MarketData objects for each asset
            option_type: 'max_call', 'min_call', 'max_put', 'min_put'
            correlation_matrix: Correlation matrix between assets
            n_paths: Number of Monte Carlo paths
            n_steps: Number of time steps
            
        Returns:
            Tuple of (option_price, confidence_interval)
        """
        n_assets = len(market_data_list)
        
        if correlation_matrix.shape != (n_assets, n_assets):
            raise ValueError("Correlation matrix dimensions must match number of assets")
        
        valid_types = ['max_call', 'min_call', 'max_put', 'min_put']
        if option_type not in valid_types:
            raise ValueError(f"option_type must be one of {valid_types}")
        
        # Extract parameters (assuming same T, r, q for all assets)
        T = market_data_list[0].T
        r = market_data_list[0].r
        q = market_data_list[0].q
        K = market_data_list[0].K
        
        S0_list = [md.S0 for md in market_data_list]
        sigma_list = [md.sigma for md in market_data_list]
        
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Cholesky decomposition for correlated random numbers
        L = np.linalg.cholesky(correlation_matrix)
        
        np.random.seed(42)
        payoffs = np.zeros(n_paths)
        
        for path in range(n_paths):
            # Initialize asset prices
            S_paths = np.zeros((n_assets, n_steps + 1))
            for i in range(n_assets):
                S_paths[i, 0] = S0_list[i]
            
            # Generate correlated paths
            for step in range(n_steps):
                # Generate independent random numbers
                Z = np.random.standard_normal(n_assets)
                # Apply correlation
                Z_corr = L @ Z
                
                # Update each asset price
                for i in range(n_assets):
                    drift = (r - q - 0.5 * sigma_list[i]**2) * dt
                    diffusion = sigma_list[i] * sqrt_dt * Z_corr[i]
                    S_paths[i, step + 1] = S_paths[i, step] * np.exp(drift + diffusion)
            
            # Calculate payoff
            final_prices = S_paths[:, -1]
            
            if option_type == 'max_call':
                payoffs[path] = max(np.max(final_prices) - K, 0)
            elif option_type == 'min_call':
                payoffs[path] = max(np.min(final_prices) - K, 0)
            elif option_type == 'max_put':
                payoffs[path] = max(K - np.max(final_prices), 0)
            elif option_type == 'min_put':
                payoffs[path] = max(K - np.min(final_prices), 0)
        
        # Discount to present value
        option_prices = np.exp(-r * T) * payoffs
        price = np.mean(option_prices)
        std_error = np.std(option_prices) / np.sqrt(n_paths)
        confidence_interval = 1.96 * std_error
        
        return price, confidence_interval
    
    @staticmethod
    def american_option_binomial(
        market_data: MarketData,
        option_type: str,
        n_steps: int = 1000
    ) -> float:
        """
        Price American options using binomial tree method.
        
        Args:
            market_data: Market conditions
            option_type: 'call' or 'put'
            n_steps: Number of binomial steps
            
        Returns:
            American option price
        """
        option_type = validate_option_type(option_type)
        S0, K, T, r, q, sigma = (
            market_data.S0, market_data.K, market_data.T,
            market_data.r, market_data.q, market_data.sigma
        )
        
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (np.exp((r - q) * dt) - d) / (u - d)  # Risk-neutral probability
        discount = np.exp(-r * dt)
        
        # Initialize asset prices at maturity
        S = np.zeros(n_steps + 1)
        for i in range(n_steps + 1):
            S[i] = S0 * (u ** (n_steps - i)) * (d ** i)
        
        # Initialize option values at maturity
        V = np.zeros(n_steps + 1)
        for i in range(n_steps + 1):
            if option_type == OptionType.CALL:
                V[i] = max(S[i] - K, 0)
            else:
                V[i] = max(K - S[i], 0)
        
        # Backward induction
        for step in range(n_steps - 1, -1, -1):
            for i in range(step + 1):
                # European continuation value
                continuation = discount * (p * V[i] + (1 - p) * V[i + 1])
                
                # Current stock price
                S_current = S0 * (u ** (step - i)) * (d ** i)
                
                # Exercise value
                if option_type == OptionType.CALL:
                    exercise = max(S_current - K, 0)
                else:
                    exercise = max(K - S_current, 0)
                
                # American option value (max of continuation and exercise)
                V[i] = max(continuation, exercise)
        
        return V[0]
    
    @staticmethod
    def quanto_option_mc(
        market_data: MarketData,
        option_type: str,
        fx_volatility: float,
        fx_correlation: float,
        foreign_rate: float,
        n_paths: int = 100000,
        n_steps: int = 252
    ) -> Tuple[float, float]:
        """
        Price quanto options using Monte Carlo simulation.
        
        Args:
            market_data: Market conditions for underlying asset
            option_type: 'call' or 'put'
            fx_volatility: FX rate volatility
            fx_correlation: Correlation between asset and FX rate
            foreign_rate: Foreign risk-free rate
            n_paths: Number of Monte Carlo paths
            n_steps: Number of time steps
            
        Returns:
            Tuple of (option_price, confidence_interval)
        """
        option_type = validate_option_type(option_type)
        S0, K, T, r, q, sigma = (
            market_data.S0, market_data.K, market_data.T,
            market_data.r, market_data.q, market_data.sigma
        )
        
        # Quanto adjustment to drift
        quanto_adjustment = fx_correlation * sigma * fx_volatility
        adjusted_drift = r - q - quanto_adjustment - 0.5 * sigma**2
        
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        np.random.seed(42)
        payoffs = np.zeros(n_paths)
        
        for path in range(n_paths):
            S_current = S0
            
            # Generate asset price path with quanto adjustment
            for step in range(n_steps):
                dW = np.random.standard_normal()
                S_current = S_current * np.exp(adjusted_drift * dt + sigma * sqrt_dt * dW)
            
            # Calculate payoff
            if option_type == OptionType.CALL:
                payoffs[path] = max(S_current - K, 0)
            else:
                payoffs[path] = max(K - S_current, 0)
        
        # Discount at domestic rate
        option_prices = np.exp(-r * T) * payoffs
        price = np.mean(option_prices)
        std_error = np.std(option_prices) / np.sqrt(n_paths)
        confidence_interval = 1.96 * std_error
        
        return price, confidence_interval