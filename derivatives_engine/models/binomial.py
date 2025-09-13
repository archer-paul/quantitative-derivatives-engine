"""
Binomial tree model for American option pricing.

This module implements the Cox-Ross-Rubinstein binomial tree model
for pricing American options with early exercise features.
"""

import numpy as np
from typing import Tuple, Optional
import logging

from ..core.market_data import MarketData, OptionType, validate_option_type

logger = logging.getLogger(__name__)


class BinomialModel:
    """
    Cox-Ross-Rubinstein binomial tree model for American option pricing.
    
    This model can price American options that allow early exercise,
    as well as European options for comparison purposes.
    """
    
    @staticmethod
    def price_american(
        market_data: MarketData,
        option_type: str,
        n_steps: int = 1000,
        early_exercise: bool = True
    ) -> float:
        """
        Price American option using binomial tree method.
        
        Args:
            market_data: Market conditions and option parameters
            option_type: 'call' or 'put'
            n_steps: Number of time steps in the binomial tree
            early_exercise: Allow early exercise (American) or not (European)
            
        Returns:
            Option price
        """
        option_type = validate_option_type(option_type)
        S0, K, T, r, q, sigma = (
            market_data.S0, market_data.K, market_data.T,
            market_data.r, market_data.q, market_data.sigma
        )
        
        logger.debug(f"Pricing {option_type} option with {n_steps} steps")
        
        # Time step
        dt = T / n_steps
        
        # Binomial parameters
        u = np.exp(sigma * np.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (np.exp((r - q) * dt) - d) / (u - d)  # Risk-neutral probability
        discount = np.exp(-r * dt)  # Discount factor
        
        # Validate parameters
        if not (0 < p < 1):
            raise ValueError(f"Invalid risk-neutral probability: {p}. Check model parameters.")
        
        # Initialize asset prices at maturity
        S = np.zeros(n_steps + 1)
        for i in range(n_steps + 1):
            S[i] = S0 * (u ** (n_steps - i)) * (d ** i)
        
        # Initialize option values at maturity (payoff)
        V = np.zeros(n_steps + 1)
        for i in range(n_steps + 1):
            if option_type == OptionType.CALL:
                V[i] = max(S[i] - K, 0)
            else:  # PUT
                V[i] = max(K - S[i], 0)
        
        # Backward induction through the tree
        for step in range(n_steps - 1, -1, -1):
            for i in range(step + 1):
                # European continuation value
                continuation = discount * (p * V[i] + (1 - p) * V[i + 1])
                
                if early_exercise:
                    # Current stock price at this node
                    S_current = S0 * (u ** (step - i)) * (d ** i)
                    
                    # Exercise value
                    if option_type == OptionType.CALL:
                        exercise = max(S_current - K, 0)
                    else:  # PUT
                        exercise = max(K - S_current, 0)
                    
                    # American option value (max of continuation and exercise)
                    V[i] = max(continuation, exercise)
                else:
                    # European option (no early exercise)
                    V[i] = continuation
        
        return V[0]
    
    @staticmethod
    def price_european(
        market_data: MarketData,
        option_type: str,
        n_steps: int = 1000
    ) -> float:
        """
        Price European option using binomial tree method.
        
        Args:
            market_data: Market conditions and option parameters
            option_type: 'call' or 'put'
            n_steps: Number of time steps in the binomial tree
            
        Returns:
            European option price
        """
        return BinomialModel.price_american(
            market_data, option_type, n_steps, early_exercise=False
        )
    
    @staticmethod
    def early_exercise_boundary(
        market_data: MarketData,
        option_type: str,
        n_steps: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate early exercise boundary for American options.
        
        Args:
            market_data: Market conditions and option parameters
            option_type: 'call' or 'put'
            n_steps: Number of time steps
            
        Returns:
            Tuple of (time_points, exercise_boundary_prices)
        """
        option_type = validate_option_type(option_type)
        S0, K, T, r, q, sigma = (
            market_data.S0, market_data.K, market_data.T,
            market_data.r, market_data.q, market_data.sigma
        )
        
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)
        discount = np.exp(-r * dt)
        
        # Build full tree of stock prices
        stock_tree = np.zeros((n_steps + 1, n_steps + 1))
        option_tree = np.zeros((n_steps + 1, n_steps + 1))
        exercise_tree = np.zeros((n_steps + 1, n_steps + 1), dtype=bool)
        
        # Initialize stock prices
        for step in range(n_steps + 1):
            for i in range(step + 1):
                stock_tree[step, i] = S0 * (u ** (step - i)) * (d ** i)
        
        # Initialize option values at maturity
        for i in range(n_steps + 1):
            if option_type == OptionType.CALL:
                option_tree[n_steps, i] = max(stock_tree[n_steps, i] - K, 0)
            else:
                option_tree[n_steps, i] = max(K - stock_tree[n_steps, i], 0)
        
        # Backward induction with exercise decision tracking
        for step in range(n_steps - 1, -1, -1):
            for i in range(step + 1):
                # Continuation value
                continuation = discount * (
                    p * option_tree[step + 1, i] + 
                    (1 - p) * option_tree[step + 1, i + 1]
                )
                
                # Exercise value
                if option_type == OptionType.CALL:
                    exercise_value = max(stock_tree[step, i] - K, 0)
                else:
                    exercise_value = max(K - stock_tree[step, i], 0)
                
                # Optimal decision
                if exercise_value > continuation:
                    option_tree[step, i] = exercise_value
                    exercise_tree[step, i] = True
                else:
                    option_tree[step, i] = continuation
                    exercise_tree[step, i] = False
        
        # Extract exercise boundary
        time_points = []
        boundary_prices = []
        
        for step in range(n_steps + 1):
            times = step * dt
            exercise_prices = []
            
            for i in range(step + 1):
                if exercise_tree[step, i]:
                    exercise_prices.append(stock_tree[step, i])
            
            if exercise_prices:
                time_points.append(times)
                if option_type == OptionType.PUT:
                    # For puts, boundary is typically the highest price where exercise occurs
                    boundary_prices.append(max(exercise_prices))
                else:
                    # For calls, boundary is typically the lowest price where exercise occurs
                    boundary_prices.append(min(exercise_prices))
        
        return np.array(time_points), np.array(boundary_prices)
    
    @staticmethod
    def compare_american_european(
        market_data: MarketData,
        option_type: str,
        n_steps: int = 1000
    ) -> dict:
        """
        Compare American and European option prices.
        
        Args:
            market_data: Market conditions
            option_type: 'call' or 'put'
            n_steps: Number of tree steps
            
        Returns:
            Dictionary with American price, European price, and early exercise premium
        """
        american_price = BinomialModel.price_american(
            market_data, option_type, n_steps, early_exercise=True
        )
        european_price = BinomialModel.price_european(
            market_data, option_type, n_steps
        )
        
        early_exercise_premium = american_price - european_price
        premium_percentage = (early_exercise_premium / european_price) * 100 if european_price > 0 else 0
        
        return {
            'american_price': american_price,
            'european_price': european_price,
            'early_exercise_premium': early_exercise_premium,
            'premium_percentage': premium_percentage,
            'worth_exercising_early': early_exercise_premium > 1e-6
        }
    
    @staticmethod
    def sensitivity_analysis(
        market_data: MarketData,
        option_type: str,
        n_steps: int = 500
    ) -> dict:
        """
        Perform sensitivity analysis on binomial tree parameters.
        
        Args:
            market_data: Base market conditions
            option_type: 'call' or 'put'  
            n_steps: Number of tree steps
            
        Returns:
            Dictionary with sensitivity results
        """
        base_price = BinomialModel.price_american(market_data, option_type, n_steps)
        
        sensitivities = {
            'base_price': base_price,
            'step_sensitivity': [],
            'convergence_analysis': []
        }
        
        # Test different number of steps for convergence
        step_counts = [50, 100, 200, 500, 1000, 2000]
        for steps in step_counts:
            price = BinomialModel.price_american(market_data, option_type, steps)
            sensitivities['step_sensitivity'].append({
                'steps': steps,
                'price': price,
                'difference_from_base': price - base_price
            })
        
        # Convergence analysis
        if len(sensitivities['step_sensitivity']) >= 2:
            prices = [s['price'] for s in sensitivities['step_sensitivity']]
            for i in range(1, len(prices)):
                convergence = abs(prices[i] - prices[i-1])
                sensitivities['convergence_analysis'].append({
                    'from_steps': step_counts[i-1],
                    'to_steps': step_counts[i],
                    'price_change': convergence,
                    'converged': convergence < 0.001
                })
        
        return sensitivities


class BinomialGreeks:
    """Calculate Greeks using binomial tree method."""
    
    @staticmethod
    def calculate_greeks(
        market_data: MarketData,
        option_type: str,
        n_steps: int = 1000,
        bump_size: float = 0.01
    ) -> dict:
        """
        Calculate Greeks using finite difference on binomial prices.
        
        Args:
            market_data: Market conditions
            option_type: 'call' or 'put'
            n_steps: Number of tree steps
            bump_size: Size of parameter bumps for finite difference
            
        Returns:
            Dictionary containing all Greeks
        """
        base_price = BinomialModel.price_american(market_data, option_type, n_steps)
        
        # Delta: ∂V/∂S
        S_up = market_data.copy(S0=market_data.S0 * (1 + bump_size))
        S_down = market_data.copy(S0=market_data.S0 * (1 - bump_size))
        price_up = BinomialModel.price_american(S_up, option_type, n_steps)
        price_down = BinomialModel.price_american(S_down, option_type, n_steps)
        delta = (price_up - price_down) / (2 * market_data.S0 * bump_size)
        
        # Gamma: ∂²V/∂S²
        gamma = (price_up - 2 * base_price + price_down) / (market_data.S0 * bump_size) ** 2
        
        # Theta: ∂V/∂T
        if market_data.T > bump_size:
            T_down = market_data.copy(T=market_data.T - bump_size)
            price_T_down = BinomialModel.price_american(T_down, option_type, n_steps)
            theta = -(base_price - price_T_down) / bump_size  # Negative for time decay
        else:
            theta = 0
        
        # Vega: ∂V/∂σ
        vol_up = market_data.copy(sigma=market_data.sigma + bump_size)
        vol_down = market_data.copy(sigma=market_data.sigma - bump_size)
        price_vol_up = BinomialModel.price_american(vol_up, option_type, n_steps)
        price_vol_down = BinomialModel.price_american(vol_down, option_type, n_steps)
        vega = (price_vol_up - price_vol_down) / (2 * bump_size)
        
        # Rho: ∂V/∂r
        r_up = market_data.copy(r=market_data.r + bump_size)
        r_down = market_data.copy(r=market_data.r - bump_size)
        price_r_up = BinomialModel.price_american(r_up, option_type, n_steps)
        price_r_down = BinomialModel.price_american(r_down, option_type, n_steps)
        rho = (price_r_up - price_r_down) / (2 * bump_size)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho,
            'price': base_price
        }