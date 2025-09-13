"""
Input validation utilities for the derivatives pricing engine.

This module provides comprehensive validation functions for market data,
model parameters, and other inputs to ensure data integrity and prevent
computational errors.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional, Any
from datetime import datetime, date
from .exceptions import ValidationError, ParameterError
from .logging_config import get_logger

logger = get_logger(__name__)


class InputValidator:
    """Comprehensive input validation for pricing engine components."""
    
    @staticmethod
    def validate_price(price: float, name: str = "price", allow_zero: bool = False) -> float:
        """Validate price input."""
        # Handle DualNumber objects from automatic differentiation
        from ..core.auto_diff import DualNumber
        if isinstance(price, DualNumber):
            # For DualNumber, validate the real part and return the DualNumber unchanged
            price_val = price.real
            if not isinstance(price_val, (int, float, np.number)):
                raise ValidationError(f"{name} must be a number", name, price)

            if np.isnan(price_val) or np.isinf(price_val):
                raise ValidationError(f"{name} must be finite", name, price)

            if allow_zero and price_val < 0:
                raise ValidationError(f"{name} must be non-negative", name, price)
            elif not allow_zero and price_val <= 0:
                raise ValidationError(f"{name} must be positive", name, price)

            return price  # Return DualNumber unchanged
        else:
            # Regular number validation
            if not isinstance(price, (int, float, np.number)):
                raise ValidationError(f"{name} must be a number", name, price)

            if np.isnan(price) or np.isinf(price):
                raise ValidationError(f"{name} must be finite", name, price)

            if allow_zero and price < 0:
                raise ValidationError(f"{name} must be non-negative", name, price)
            elif not allow_zero and price <= 0:
                raise ValidationError(f"{name} must be positive", name, price)

            return float(price)
    
    @staticmethod
    def validate_strike(strike: float, name: str = "strike") -> float:
        """Validate strike price."""
        return InputValidator.validate_price(strike, name, allow_zero=False)
    
    @staticmethod
    def validate_spot(spot: float, name: str = "spot") -> float:
        """Validate spot price."""
        return InputValidator.validate_price(spot, name, allow_zero=False)
    
    @staticmethod
    def validate_time_to_expiry(T: float, name: str = "time_to_expiry") -> float:
        """Validate time to expiry."""
        if not isinstance(T, (int, float, np.number)):
            raise ValidationError(f"{name} must be a number", name, T)
        
        if np.isnan(T) or np.isinf(T):
            raise ValidationError(f"{name} must be finite", name, T)
        
        if T < 0:
            raise ValidationError(f"{name} must be non-negative", name, T)
        
        if T > 50:  # Reasonable upper bound
            logger.warning(f"Very long time to expiry: {T} years")
        
        return float(T)
    
    @staticmethod
    def validate_rate(rate: float, name: str = "rate") -> float:
        """Validate interest rate."""
        if not isinstance(rate, (int, float, np.number)):
            raise ValidationError(f"{name} must be a number", name, rate)
        
        if np.isnan(rate) or np.isinf(rate):
            raise ValidationError(f"{name} must be finite", name, rate)
        
        if abs(rate) > 1.0:  # Rates higher than 100% should be suspicious
            logger.warning(f"Unusually high {name}: {rate:.2%}")
        
        return float(rate)
    
    @staticmethod
    def validate_volatility(vol: float, name: str = "volatility") -> float:
        """Validate volatility."""
        if not isinstance(vol, (int, float, np.number)):
            raise ValidationError(f"{name} must be a number", name, vol)
        
        if np.isnan(vol) or np.isinf(vol):
            raise ValidationError(f"{name} must be finite", name, vol)
        
        if vol < 0:
            raise ValidationError(f"{name} must be non-negative", name, vol)
        
        if vol > 5.0:  # 500% volatility is extreme
            logger.warning(f"Very high {name}: {vol:.2%}")
        
        return float(vol)
    
    @staticmethod
    def validate_option_type(option_type: str, valid_types: List[str] = None) -> str:
        """Validate option type."""
        if not isinstance(option_type, str):
            raise ValidationError("Option type must be a string", "option_type", option_type)
        
        if valid_types is None:
            valid_types = ["call", "put"]
        
        option_type_lower = option_type.lower().strip()
        
        if option_type_lower not in [t.lower() for t in valid_types]:
            raise ValidationError(f"Option type must be one of {valid_types}", "option_type", option_type)
        
        return option_type_lower
    
    @staticmethod
    def validate_correlation(correlation: float, name: str = "correlation") -> float:
        """Validate correlation coefficient."""
        if not isinstance(correlation, (int, float, np.number)):
            raise ValidationError(f"{name} must be a number", name, correlation)
        
        if np.isnan(correlation) or np.isinf(correlation):
            raise ValidationError(f"{name} must be finite", name, correlation)
        
        if not (-1 <= correlation <= 1):
            raise ValidationError(f"{name} must be between -1 and 1", name, correlation)
        
        return float(correlation)
    
    @staticmethod
    def validate_probability(prob: float, name: str = "probability") -> float:
        """Validate probability."""
        if not isinstance(prob, (int, float, np.number)):
            raise ValidationError(f"{name} must be a number", name, prob)
        
        if np.isnan(prob) or np.isinf(prob):
            raise ValidationError(f"{name} must be finite", name, prob)
        
        if not (0 <= prob <= 1):
            raise ValidationError(f"{name} must be between 0 and 1", name, prob)
        
        return float(prob)
    
    @staticmethod
    def validate_positive_integer(value: int, name: str = "value") -> int:
        """Validate positive integer."""
        if not isinstance(value, (int, np.integer)):
            raise ValidationError(f"{name} must be an integer", name, value)
        
        if value <= 0:
            raise ValidationError(f"{name} must be positive", name, value)
        
        return int(value)
    
    @staticmethod
    def validate_non_negative_integer(value: int, name: str = "value") -> int:
        """Validate non-negative integer."""
        if not isinstance(value, (int, np.integer)):
            raise ValidationError(f"{name} must be an integer", name, value)
        
        if value < 0:
            raise ValidationError(f"{name} must be non-negative", name, value)
        
        return int(value)
    
    @staticmethod
    def validate_array(arr: np.ndarray, name: str = "array", 
                      shape: Tuple[int, ...] = None,
                      min_length: int = None,
                      dtype: np.dtype = None) -> np.ndarray:
        """Validate numpy array."""
        if not isinstance(arr, np.ndarray):
            try:
                arr = np.array(arr)
            except Exception as e:
                raise ValidationError(f"{name} cannot be converted to array: {e}", name, arr)
        
        if shape is not None and arr.shape != shape:
            raise ValidationError(f"{name} must have shape {shape}, got {arr.shape}", name, arr)
        
        if min_length is not None and len(arr) < min_length:
            raise ValidationError(f"{name} must have at least {min_length} elements", name, arr)
        
        if dtype is not None and arr.dtype != dtype:
            try:
                arr = arr.astype(dtype)
            except Exception as e:
                raise ValidationError(f"{name} cannot be converted to {dtype}: {e}", name, arr)
        
        if np.any(np.isnan(arr)):
            raise ValidationError(f"{name} contains NaN values", name, arr)
        
        if np.any(np.isinf(arr)):
            raise ValidationError(f"{name} contains infinite values", name, arr)
        
        return arr
    
    @staticmethod
    def validate_correlation_matrix(corr_matrix: np.ndarray, name: str = "correlation_matrix") -> np.ndarray:
        """Validate correlation matrix."""
        # Basic array validation
        corr_matrix = InputValidator.validate_array(corr_matrix, name)
        
        # Must be square
        if corr_matrix.ndim != 2 or corr_matrix.shape[0] != corr_matrix.shape[1]:
            raise ValidationError(f"{name} must be a square matrix", name, corr_matrix)
        
        # Must be symmetric
        if not np.allclose(corr_matrix, corr_matrix.T, rtol=1e-10):
            raise ValidationError(f"{name} must be symmetric", name, corr_matrix)
        
        # Diagonal must be 1
        if not np.allclose(np.diag(corr_matrix), 1.0, rtol=1e-10):
            raise ValidationError(f"{name} must have diagonal elements equal to 1", name, corr_matrix)
        
        # All elements must be between -1 and 1
        if not np.all((-1 <= corr_matrix) & (corr_matrix <= 1)):
            raise ValidationError(f"{name} elements must be between -1 and 1", name, corr_matrix)
        
        # Must be positive semidefinite
        eigenvals = np.linalg.eigvals(corr_matrix)
        if np.any(eigenvals < -1e-10):  # Allow small numerical errors
            raise ValidationError(f"{name} must be positive semidefinite", name, corr_matrix)
        
        return corr_matrix
    
    @staticmethod
    def validate_heston_parameters(v0: float, theta: float, kappa: float, 
                                 sigma_v: float, rho: float) -> Tuple[float, ...]:
        """Validate Heston model parameters."""
        v0 = InputValidator.validate_price(v0, "v0 (initial variance)", allow_zero=True)
        theta = InputValidator.validate_price(theta, "theta (long-term variance)", allow_zero=True)
        kappa = InputValidator.validate_price(kappa, "kappa (mean reversion speed)", allow_zero=False)
        sigma_v = InputValidator.validate_price(sigma_v, "sigma_v (volatility of variance)", allow_zero=True)
        rho = InputValidator.validate_correlation(rho, "rho (correlation)")
        
        # Check Feller condition: 2*kappa*theta >= sigma_v^2
        feller_lhs = 2 * kappa * theta
        feller_rhs = sigma_v ** 2
        
        if feller_lhs < feller_rhs:
            logger.warning(f"Feller condition violated: 2*kappa*theta ({feller_lhs:.6f}) < sigma_v^2 ({feller_rhs:.6f}). "
                         "Variance process may become negative.")
        
        return v0, theta, kappa, sigma_v, rho
    
    @staticmethod
    def validate_jump_diffusion_parameters(jump_intensity: float, jump_mean: float, 
                                         jump_std: float) -> Tuple[float, ...]:
        """Validate jump diffusion parameters."""
        jump_intensity = InputValidator.validate_price(jump_intensity, "jump_intensity", allow_zero=True)
        
        if not isinstance(jump_mean, (int, float, np.number)):
            raise ValidationError("jump_mean must be a number", "jump_mean", jump_mean)
        
        if np.isnan(jump_mean) or np.isinf(jump_mean):
            raise ValidationError("jump_mean must be finite", "jump_mean", jump_mean)
        
        jump_std = InputValidator.validate_price(jump_std, "jump_std", allow_zero=True)
        
        return float(jump_intensity), float(jump_mean), float(jump_std)
    
    @staticmethod
    def validate_monte_carlo_parameters(n_paths: int, n_steps: int = None) -> Tuple[int, ...]:
        """Validate Monte Carlo simulation parameters."""
        n_paths = InputValidator.validate_positive_integer(n_paths, "n_paths")
        
        if n_paths < 1000:
            logger.warning(f"Low number of Monte Carlo paths ({n_paths}). Consider using more for accuracy.")
        
        if n_paths > 10000000:  # 10 million
            logger.warning(f"Very high number of Monte Carlo paths ({n_paths}). This may consume significant memory.")
        
        if n_steps is not None:
            n_steps = InputValidator.validate_positive_integer(n_steps, "n_steps")
            
            if n_steps > 10000:
                logger.warning(f"Very high number of time steps ({n_steps}). This may be computationally expensive.")
            
            return n_paths, n_steps
        
        return (n_paths,)
    
    @staticmethod
    def validate_finite_difference_parameters(S_max: float, M: int, N: int) -> Tuple[float, int, int]:
        """Validate finite difference grid parameters."""
        S_max = InputValidator.validate_price(S_max, "S_max", allow_zero=False)
        M = InputValidator.validate_positive_integer(M, "M (space grid points)")
        N = InputValidator.validate_positive_integer(N, "N (time steps)")
        
        if M * N > 10000000:  # 10 million grid points
            logger.warning(f"Very large finite difference grid ({M}x{N}). This may consume significant memory.")
        
        return S_max, M, N


class MarketDataValidator:
    """Specialized validator for market data."""
    
    @staticmethod
    def validate_market_data_consistency(S0: float, K: float, T: float, r: float, 
                                       q: float = 0.0, sigma: float = None):
        """Validate consistency of market data parameters."""
        # Individual validation
        S0 = InputValidator.validate_spot(S0, "S0")
        K = InputValidator.validate_strike(K, "K")
        T = InputValidator.validate_time_to_expiry(T, "T")
        r = InputValidator.validate_rate(r, "r")
        q = InputValidator.validate_rate(q, "q")
        
        if sigma is not None:
            sigma = InputValidator.validate_volatility(sigma, "sigma")
        
        # Consistency checks
        if T == 0 and S0 != K:
            # At expiration, option value should be intrinsic value
            logger.info("Option at expiration - pricing will use intrinsic value")
        
        if abs(r) > 0.5:  # More than 50% interest rate
            logger.warning(f"Unusually high interest rate: {r:.2%}")
        
        if abs(q) > 0.2:  # More than 20% dividend yield
            logger.warning(f"Unusually high dividend yield: {q:.2%}")
        
        # Forward price check
        forward_price = S0 * np.exp((r - q) * T)
        if T > 0 and (forward_price / K > 10 or K / forward_price > 10):
            logger.warning("Option is very far from at-the-money - pricing may be less accurate")
        
        return S0, K, T, r, q, sigma
    
    @staticmethod
    def validate_option_chain_data(df: pd.DataFrame) -> pd.DataFrame:
        """Validate option chain data from external sources."""
        if not isinstance(df, pd.DataFrame):
            raise ValidationError("Option chain data must be a pandas DataFrame", "option_chain", df)
        
        required_columns = ['strike', 'bid', 'ask', 'option_type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}", "option_chain", df)
        
        # Validate individual fields
        for idx, row in df.iterrows():
            try:
                InputValidator.validate_strike(row['strike'])
                InputValidator.validate_price(row['bid'], "bid", allow_zero=True)
                InputValidator.validate_price(row['ask'], "ask", allow_zero=True)
                InputValidator.validate_option_type(row['option_type'])
                
                # Bid-ask spread validation
                if row['ask'] < row['bid']:
                    logger.warning(f"Invalid bid-ask spread at row {idx}: bid={row['bid']}, ask={row['ask']}")
                
            except ValidationError as e:
                raise ValidationError(f"Invalid data at row {idx}: {e}", "option_chain", row)
        
        return df


# Validation decorators
def validate_inputs(**validators):
    """Decorator to validate function inputs using specified validators."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Apply validators
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    validated_value = validator(value)
                    bound_args.arguments[param_name] = validated_value
            
            return func(*bound_args.args, **bound_args.kwargs)
        return wrapper
    return decorator


def validate_market_data_input(func):
    """Decorator specifically for market data validation."""
    def wrapper(*args, **kwargs):
        # Extract market data from arguments (assuming first arg is MarketData object)
        if args and hasattr(args[0], 'S0'):
            market_data = args[0]

            # Skip validation if any parameter is a DualNumber (for Greeks calculation)
            from ..core.auto_diff import DualNumber
            params = [market_data.S0, market_data.K, market_data.T, market_data.r,
                     getattr(market_data, 'q', 0.0), getattr(market_data, 'sigma', None)]

            has_dual = any(isinstance(p, DualNumber) for p in params if p is not None)

            if not has_dual:
                MarketDataValidator.validate_market_data_consistency(
                    market_data.S0, market_data.K, market_data.T,
                    market_data.r, getattr(market_data, 'q', 0.0),
                    getattr(market_data, 'sigma', None)
                )

        return func(*args, **kwargs)
    return wrapper