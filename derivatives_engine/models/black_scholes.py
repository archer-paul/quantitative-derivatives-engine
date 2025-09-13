"""
Black-Scholes option pricing model with automatic differentiation.

This module implements the classical Black-Scholes model for European option
pricing with automatic differentiation for efficient Greeks calculation.
"""

import numpy as np
from typing import Dict
from scipy import stats

from ..core.market_data import MarketData, OptionType, validate_option_type
from ..core.auto_diff import DualNumber, exp, log, sqrt, norm_cdf
from ..utils.logging_config import get_logger, log_performance
from ..utils.exceptions import ModelError, handle_numerical_result
from ..utils.validation import validate_market_data_input


class BlackScholesModel:
    """
    Black-Scholes option pricing model with automatic differentiation for Greeks.
    
    This implementation uses dual numbers for exact Greek calculation without
    numerical differentiation errors.
    """
    
    @staticmethod
    def _d1_d2(S, K, T, r, q, sigma):
        """
        Calculate d1 and d2 parameters for Black-Scholes formula.
        
        Args:
            S: Stock price (can be DualNumber)
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            q: Dividend yield
            sigma: Volatility
            
        Returns:
            Tuple of (d1, d2)
        """
        from ..core.auto_diff import DualNumber
        T_val = T.real if isinstance(T, DualNumber) else T
        if T_val <= 0:
            # Handle expiration case
            if isinstance(S, DualNumber):
                return DualNumber(0.0, 0.0), DualNumber(0.0, 0.0)
            else:
                return 0.0, 0.0
        
        sqrt_T = sqrt(T)
        log_SK = log(S / K)
        
        d1 = (log_SK + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        return d1, d2
    
    @staticmethod
    @validate_market_data_input
    @log_performance("Black-Scholes pricing")
    def price(market_data: MarketData, option_type: str) -> float:
        """
        Calculate Black-Scholes option price.
        
        Uses the classical Black-Scholes formula:
        C = S*e^(-qT)*N(d1) - K*e^(-rT)*N(d2)
        P = K*e^(-rT)*N(-d2) - S*e^(-qT)*N(-d1)
        
        Args:
            market_data: Market conditions and option parameters
            option_type: 'call' or 'put'
            
        Returns:
            Option price as float
            
        Raises:
            ModelError: If pricing calculation fails
            ValidationError: If inputs are invalid
        """
        logger = get_logger(__name__)
        
        try:
            option_type = validate_option_type(option_type)
            S, K, T, r, q, sigma = (
                market_data.S0, market_data.K, market_data.T,
                market_data.r, market_data.q, market_data.sigma
            )
        
            # Handle expiration case
            from ..core.auto_diff import DualNumber
            T_val = T.real if isinstance(T, DualNumber) else T
            if T_val <= 0:
                S_val = S.real if isinstance(S, DualNumber) else S
                K_val = K.real if isinstance(K, DualNumber) else K
                if option_type == OptionType.CALL:
                    result = max(S_val - K_val, 0)
                else:
                    result = max(K_val - S_val, 0)
                logger.debug(f"Option at expiration, intrinsic value: {result}")
                return handle_numerical_result(result, "Black-Scholes intrinsic value")
            
            # Calculate d1 and d2
            d1, d2 = BlackScholesModel._d1_d2(S, K, T, r, q, sigma)
            
            # Calculate option price
            if option_type == OptionType.CALL:
                price = S * exp(-q * T) * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)
            else:  # PUT
                price = K * exp(-r * T) * norm_cdf(-d2) - S * exp(-q * T) * norm_cdf(-d1)
            
            # Extract real value if DualNumber
            if isinstance(price, DualNumber):
                result = price.real
            else:
                result = float(price)
            
            # Validate result
            result = handle_numerical_result(result, "Black-Scholes price")
            
            logger.debug(f"Black-Scholes {option_type} price calculated: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Black-Scholes pricing failed: {e}")
            if isinstance(e, (ModelError, ValueError)):
                raise
            raise ModelError(f"Black-Scholes pricing calculation failed: {e}", "BlackScholesModel")
    
    @staticmethod
    @validate_market_data_input
    @log_performance("Black-Scholes Greeks calculation")
    def greeks(market_data: MarketData, option_type: str) -> Dict[str, float]:
        """
        Calculate all Greeks using automatic differentiation.
        
        The Greeks are calculated by applying automatic differentiation to the
        Black-Scholes formula with respect to each parameter.
        
        Args:
            market_data: Market conditions and option parameters
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary containing all Greeks:
            - delta: ∂V/∂S (price sensitivity to underlying)
            - gamma: ∂²V/∂S² (delta sensitivity to underlying)  
            - theta: ∂V/∂T (time decay)
            - vega: ∂V/∂σ (volatility sensitivity)
            - rho: ∂V/∂r (interest rate sensitivity)
            
        Raises:
            ModelError: If Greeks calculation fails
            ValidationError: If inputs are invalid
        """
        logger = get_logger(__name__)
        
        try:
            option_type = validate_option_type(option_type)
            S, K, T, r, q, sigma = (
                market_data.S0, market_data.K, market_data.T,
                market_data.r, market_data.q, market_data.sigma
            )
            
            # Handle expiration case
            from ..core.auto_diff import DualNumber
            T_val = T.real if isinstance(T, DualNumber) else T
            if T_val <= 0:
                S_val = S.real if isinstance(S, DualNumber) else S
                K_val = K.real if isinstance(K, DualNumber) else K
                if option_type == OptionType.CALL:
                    delta = 1.0 if S_val > K_val else (0.5 if S_val == K_val else 0.0)
                else:  # PUT
                    delta = -1.0 if S_val < K_val else (-0.5 if S_val == K_val else 0.0)
                
                return {
                    'delta': delta,
                    'gamma': 0.0,
                    'theta': 0.0,
                    'vega': 0.0,
                    'rho': 0.0
                }
            
            # Delta: sensitivity to spot price
            S_dual = DualNumber(S, 1.0)
            market_delta = MarketData(S_dual, K, T, r, q, sigma)
            delta = BlackScholesModel.price(market_delta, option_type)
            if hasattr(delta, 'dual'):
                delta = delta.dual
            else:
                # Fallback to analytical formula
                d1, _ = BlackScholesModel._d1_d2(S, K, T, r, q, sigma)
                if option_type == OptionType.CALL:
                    delta = np.exp(-q * T) * stats.norm.cdf(d1)
                else:
                    delta = -np.exp(-q * T) * stats.norm.cdf(-d1)
            
            # Gamma: second derivative w.r.t. spot price (using finite difference)
            eps = max(0.01, S * 0.0001)  # Adaptive step size
            S_up = S + eps
            S_down = S - eps
            
            market_up = MarketData(S_up, K, T, r, q, sigma)
            market_down = MarketData(S_down, K, T, r, q, sigma)
            market_center = MarketData(S, K, T, r, q, sigma)
            
            price_up = BlackScholesModel.price(market_up, option_type)
            price_down = BlackScholesModel.price(market_down, option_type)
            price_center = BlackScholesModel.price(market_center, option_type)
            
            gamma = (price_up - 2 * price_center + price_down) / (eps ** 2)
            
            # Theta: sensitivity to time (negative for time decay)
            T_dual = DualNumber(T, 1.0)
            market_theta = MarketData(S, K, T_dual, r, q, sigma)
            theta_result = BlackScholesModel.price(market_theta, option_type)
            if hasattr(theta_result, 'dual'):
                theta = -theta_result.dual  # Negative for time decay
            else:
                # Fallback: small finite difference
                T_eps = max(T * 0.001, 1/365)  # At least 1 day
                market_theta_fd = MarketData(S, K, T - T_eps, r, q, sigma)
                price_theta = BlackScholesModel.price(market_theta_fd, option_type)
                theta = -(price_center - price_theta) / T_eps
            
            # Vega: sensitivity to volatility
            sigma_dual = DualNumber(sigma, 1.0)
            market_vega = MarketData(S, K, T, r, q, sigma_dual)
            vega_result = BlackScholesModel.price(market_vega, option_type)
            if hasattr(vega_result, 'dual'):
                vega = vega_result.dual
            else:
                # Fallback to analytical formula
                d1, _ = BlackScholesModel._d1_d2(S, K, T, r, q, sigma)
                vega = S * np.exp(-q * T) * stats.norm.pdf(d1) * np.sqrt(T)
            
            # Rho: sensitivity to interest rate
            r_dual = DualNumber(r, 1.0)
            market_rho = MarketData(S, K, T, r_dual, q, sigma)
            rho_result = BlackScholesModel.price(market_rho, option_type)
            if hasattr(rho_result, 'dual'):
                rho = rho_result.dual
            else:
                # Fallback to analytical formula
                _, d2 = BlackScholesModel._d1_d2(S, K, T, r, q, sigma)
                if option_type == OptionType.CALL:
                    rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2)
                else:
                    rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2)
        
            # Validate all Greeks
            greeks_dict = {
                'delta': handle_numerical_result(float(delta), "Delta"),
                'gamma': handle_numerical_result(float(gamma), "Gamma"),
                'theta': handle_numerical_result(float(theta), "Theta", allow_negative=True),
                'vega': handle_numerical_result(float(vega), "Vega"),
                'rho': handle_numerical_result(float(rho), "Rho", allow_negative=True)
            }
            
            logger.debug(f"Black-Scholes Greeks calculated: {greeks_dict}")
            return greeks_dict
            
        except Exception as e:
            logger.error(f"Black-Scholes Greeks calculation failed: {e}")
            if isinstance(e, (ModelError, ValueError)):
                raise
            raise ModelError(f"Greeks calculation failed: {e}", "BlackScholesModel")
    
    @staticmethod
    def implied_volatility(
        market_data: MarketData,
        option_type: str,
        market_price: float,
        tolerance: float = 1e-6,
        max_iterations: int = 100
    ) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            market_data: Market conditions (volatility will be ignored)
            option_type: 'call' or 'put'
            market_price: Observed market price
            tolerance: Convergence tolerance
            max_iterations: Maximum number of iterations
            
        Returns:
            Implied volatility
            
        Raises:
            ValueError: If convergence fails or invalid inputs
        """
        option_type = validate_option_type(option_type)
        
        if market_price <= 0:
            raise ValueError("Market price must be positive")
        
        # Initial guess for volatility
        vol = 0.2
        
        for i in range(max_iterations):
            # Create market data with current volatility guess
            test_market = market_data.copy(sigma=vol)
            
            # Calculate theoretical price and vega
            theo_price = BlackScholesModel.price(test_market, option_type)
            greeks = BlackScholesModel.greeks(test_market, option_type)
            vega = greeks['vega']
            
            # Price difference
            price_diff = theo_price - market_price
            
            # Check convergence
            if abs(price_diff) < tolerance:
                return vol
            
            # Newton-Raphson update
            if abs(vega) < 1e-10:
                raise ValueError("Vega too small, cannot converge")
            
            vol_new = vol - price_diff / vega
            
            # Ensure volatility stays positive
            vol_new = max(vol_new, 0.001)
            
            # Check for convergence in volatility space
            if abs(vol_new - vol) < tolerance:
                return vol_new
            
            vol = vol_new
        
        raise ValueError(f"Failed to converge after {max_iterations} iterations")
    
    @staticmethod
    def binary_option_price(
        market_data: MarketData,
        option_type: str,
        barrier_type: str = 'cash_or_nothing',
        payout: float = 1.0
    ) -> float:
        """
        Price binary (digital) options.
        
        Args:
            market_data: Market conditions
            option_type: 'call' or 'put'
            barrier_type: 'cash_or_nothing' or 'asset_or_nothing'
            payout: Cash payout for cash-or-nothing options
            
        Returns:
            Binary option price
        """
        option_type = validate_option_type(option_type)
        S, K, T, r, q, sigma = (
            market_data.S0, market_data.K, market_data.T,
            market_data.r, market_data.q, market_data.sigma
        )
        
        if T <= 0:
            # At expiration
            if option_type == OptionType.CALL:
                condition = S > K
            else:
                condition = S < K
            
            if barrier_type == 'cash_or_nothing':
                return payout if condition else 0.0
            else:  # asset_or_nothing
                return S if condition else 0.0
        
        # Calculate d2
        _, d2 = BlackScholesModel._d1_d2(S, K, T, r, q, sigma)
        
        if barrier_type == 'cash_or_nothing':
            if option_type == OptionType.CALL:
                price = payout * np.exp(-r * T) * stats.norm.cdf(d2)
            else:
                price = payout * np.exp(-r * T) * stats.norm.cdf(-d2)
        else:  # asset_or_nothing
            d1, _ = BlackScholesModel._d1_d2(S, K, T, r, q, sigma)
            if option_type == OptionType.CALL:
                price = S * np.exp(-q * T) * stats.norm.cdf(d1)
            else:
                price = S * np.exp(-q * T) * stats.norm.cdf(-d1)
        
        return float(price)