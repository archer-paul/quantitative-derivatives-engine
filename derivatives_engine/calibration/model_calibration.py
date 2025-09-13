"""
Advanced model calibration tools for option pricing models.

This module provides comprehensive calibration capabilities for various
option pricing models using market data and optimization techniques.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

from ..core.market_data import MarketData
from ..models.black_scholes import BlackScholesModel
from ..models.heston import HestonModel, HestonParameters
from ..models.jump_diffusion import MertonJumpDiffusionModel, JumpDiffusionParameters

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Container for calibration results."""
    success: bool
    optimal_parameters: Dict[str, float]
    final_error: float
    iterations: int
    execution_time: float
    market_prices: np.ndarray
    model_prices: np.ndarray
    relative_errors: np.ndarray
    rmse: float
    mae: float
    r_squared: float
    message: str
    

class ModelCalibrator(ABC):
    """Abstract base class for model calibrators."""
    
    def __init__(self, 
                 error_function: str = 'rmse',
                 optimization_method: str = 'differential_evolution'):
        """
        Initialize calibrator.
        
        Args:
            error_function: Error function to minimize ('rmse', 'mae', 'mape')
            optimization_method: Optimization method ('minimize', 'differential_evolution')
        """
        self.error_function = error_function
        self.optimization_method = optimization_method
        self.calibration_history = []
        
    @abstractmethod
    def objective_function(self, parameters: np.ndarray, market_data: List[Dict]) -> float:
        """Define the objective function to minimize."""
        pass
    
    @abstractmethod
    def parameter_bounds(self) -> List[Tuple[float, float]]:
        """Define parameter bounds for optimization."""
        pass
    
    @abstractmethod
    def parameters_to_dict(self, parameters: np.ndarray) -> Dict[str, float]:
        """Convert parameter array to dictionary."""
        pass
    
    def calculate_error(self, market_prices: np.ndarray, model_prices: np.ndarray) -> float:
        """
        Calculate error between market and model prices.
        
        Args:
            market_prices: Array of market prices
            model_prices: Array of model prices
            
        Returns:
            Error value based on selected error function
        """
        if self.error_function == 'rmse':
            return np.sqrt(np.mean((market_prices - model_prices) ** 2))
        elif self.error_function == 'mae':
            return np.mean(np.abs(market_prices - model_prices))
        elif self.error_function == 'mape':
            return np.mean(np.abs((market_prices - model_prices) / market_prices)) * 100
        else:
            raise ValueError(f"Unknown error function: {self.error_function}")
    
    def calibrate(self, 
                  market_data: List[Dict],
                  initial_guess: Optional[np.ndarray] = None) -> CalibrationResult:
        """
        Perform model calibration.
        
        Args:
            market_data: List of market data dictionaries with prices
            initial_guess: Optional initial parameter guess
            
        Returns:
            CalibrationResult object
        """
        import time
        start_time = time.time()
        
        bounds = self.parameter_bounds()
        
        if initial_guess is None:
            # Use midpoint of bounds as initial guess
            initial_guess = np.array([(b[0] + b[1]) / 2 for b in bounds])
        
        logger.info(f"Starting calibration with {self.optimization_method}")
        
        try:
            if self.optimization_method == 'minimize':
                result = minimize(
                    self.objective_function,
                    initial_guess,
                    args=(market_data,),
                    bounds=bounds,
                    method='L-BFGS-B',
                    options={'maxiter': 1000}
                )
                success = result.success
                optimal_params = result.x
                final_error = result.fun
                iterations = result.nit
                message = result.message
                
            elif self.optimization_method == 'differential_evolution':
                result = differential_evolution(
                    self.objective_function,
                    bounds,
                    args=(market_data,),
                    maxiter=300,
                    popsize=15,
                    seed=42
                )
                success = result.success
                optimal_params = result.x
                final_error = result.fun
                iterations = result.nit
                message = result.message
                
            else:
                raise ValueError(f"Unknown optimization method: {self.optimization_method}")
            
            # Calculate detailed results
            market_prices = np.array([md['market_price'] for md in market_data])
            model_prices = self._calculate_model_prices(optimal_params, market_data)
            
            relative_errors = (model_prices - market_prices) / market_prices * 100
            rmse = self.calculate_error(market_prices, model_prices)
            mae = np.mean(np.abs(market_prices - model_prices))
            
            # R-squared
            ss_res = np.sum((market_prices - model_prices) ** 2)
            ss_tot = np.sum((market_prices - np.mean(market_prices)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            execution_time = time.time() - start_time
            
            calibration_result = CalibrationResult(
                success=success,
                optimal_parameters=self.parameters_to_dict(optimal_params),
                final_error=final_error,
                iterations=iterations,
                execution_time=execution_time,
                market_prices=market_prices,
                model_prices=model_prices,
                relative_errors=relative_errors,
                rmse=rmse,
                mae=mae,
                r_squared=r_squared,
                message=message
            )
            
            self.calibration_history.append(calibration_result)
            
            logger.info(f"Calibration completed in {execution_time:.2f}s with RMSE: {rmse:.6f}")
            
            return calibration_result
            
        except Exception as e:
            logger.error(f"Calibration failed: {str(e)}")
            execution_time = time.time() - start_time
            
            return CalibrationResult(
                success=False,
                optimal_parameters={},
                final_error=float('inf'),
                iterations=0,
                execution_time=execution_time,
                market_prices=np.array([]),
                model_prices=np.array([]),
                relative_errors=np.array([]),
                rmse=float('inf'),
                mae=float('inf'),
                r_squared=0.0,
                message=str(e)
            )
    
    @abstractmethod
    def _calculate_model_prices(self, parameters: np.ndarray, market_data: List[Dict]) -> np.ndarray:
        """Calculate model prices for given parameters."""
        pass


class HestonCalibrator(ModelCalibrator):
    """Calibrator for Heston stochastic volatility model."""
    
    def parameter_bounds(self) -> List[Tuple[float, float]]:
        """Parameter bounds for Heston model: [v0, theta, kappa, sigma_v, rho]"""
        return [
            (0.001, 1.0),    # v0: initial variance
            (0.001, 1.0),    # theta: long-term variance
            (0.1, 10.0),     # kappa: mean reversion speed
            (0.01, 2.0),     # sigma_v: volatility of variance
            (-0.99, 0.99)    # rho: correlation
        ]
    
    def parameters_to_dict(self, parameters: np.ndarray) -> Dict[str, float]:
        """Convert parameter array to dictionary."""
        return {
            'v0': parameters[0],
            'theta': parameters[1], 
            'kappa': parameters[2],
            'sigma_v': parameters[3],
            'rho': parameters[4]
        }
    
    def objective_function(self, parameters: np.ndarray, market_data: List[Dict]) -> float:
        """Objective function for Heston calibration."""
        try:
            heston_params = HestonParameters(*parameters)
            heston_model = HestonModel(heston_params)
            
            model_prices = []
            market_prices = []
            
            for md in market_data:
                market_condition = md['market_data']
                option_type = md['option_type']
                market_price = md['market_price']
                
                try:
                    # Use Monte Carlo pricing for robustness
                    model_price, _ = heston_model.monte_carlo_price(
                        market_condition, option_type, n_paths=10000
                    )
                    model_prices.append(model_price)
                    market_prices.append(market_price)
                except:
                    # If Monte Carlo fails, penalize heavily
                    return 1e10
            
            return self.calculate_error(np.array(market_prices), np.array(model_prices))
            
        except:
            # Return large error if model parameters are invalid
            return 1e10
    
    def _calculate_model_prices(self, parameters: np.ndarray, market_data: List[Dict]) -> np.ndarray:
        """Calculate Heston model prices."""
        heston_params = HestonParameters(*parameters)
        heston_model = HestonModel(heston_params)
        
        model_prices = []
        for md in market_data:
            market_condition = md['market_data']
            option_type = md['option_type']
            
            model_price, _ = heston_model.monte_carlo_price(
                market_condition, option_type, n_paths=10000
            )
            model_prices.append(model_price)
        
        return np.array(model_prices)


class JumpDiffusionCalibrator(ModelCalibrator):
    """Calibrator for Merton jump-diffusion model."""
    
    def parameter_bounds(self) -> List[Tuple[float, float]]:
        """Parameter bounds: [lambda_j, mu_j, sigma_j]"""
        return [
            (0.0, 10.0),     # lambda_j: jump intensity
            (-0.5, 0.5),     # mu_j: mean jump size
            (0.01, 1.0)      # sigma_j: jump volatility
        ]
    
    def parameters_to_dict(self, parameters: np.ndarray) -> Dict[str, float]:
        """Convert parameter array to dictionary."""
        return {
            'lambda_j': parameters[0],
            'mu_j': parameters[1],
            'sigma_j': parameters[2]
        }
    
    def objective_function(self, parameters: np.ndarray, market_data: List[Dict]) -> float:
        """Objective function for jump-diffusion calibration."""
        try:
            jump_params = JumpDiffusionParameters(*parameters)
            jd_model = MertonJumpDiffusionModel(jump_params)
            
            model_prices = []
            market_prices = []
            
            for md in market_data:
                market_condition = md['market_data']
                option_type = md['option_type']
                market_price = md['market_price']
                
                try:
                    model_price = jd_model.price_analytical(market_condition, option_type)
                    model_prices.append(model_price)
                    market_prices.append(market_price)
                except:
                    return 1e10
            
            return self.calculate_error(np.array(market_prices), np.array(model_prices))
            
        except:
            return 1e10
    
    def _calculate_model_prices(self, parameters: np.ndarray, market_data: List[Dict]) -> np.ndarray:
        """Calculate jump-diffusion model prices."""
        jump_params = JumpDiffusionParameters(*parameters)
        jd_model = MertonJumpDiffusionModel(jump_params)
        
        model_prices = []
        for md in market_data:
            market_condition = md['market_data']
            option_type = md['option_type']
            
            model_price = jd_model.price_analytical(market_condition, option_type)
            model_prices.append(model_price)
        
        return np.array(model_prices)


class ImpliedVolatilityCalibrator:
    """Calibrator for implied volatility surface."""
    
    def __init__(self):
        self.calibration_results = []
    
    def calibrate_vol_surface(self, market_data: List[Dict]) -> pd.DataFrame:
        """
        Calibrate implied volatility surface from market prices.
        
        Args:
            market_data: List of option market data with strikes, expiries, prices
            
        Returns:
            DataFrame with implied volatilities
        """
        results = []
        
        for md in market_data:
            market_condition = md['market_data'] 
            option_type = md['option_type']
            market_price = md['market_price']
            
            try:
                implied_vol = BlackScholesModel.implied_volatility(
                    market_condition, option_type, market_price
                )
                
                results.append({
                    'strike': market_condition.K,
                    'expiry': market_condition.T,
                    'option_type': option_type,
                    'market_price': market_price,
                    'implied_vol': implied_vol,
                    'moneyness': market_condition.moneyness
                })
                
            except Exception as e:
                logger.warning(f"Failed to calculate implied vol: {e}")
                results.append({
                    'strike': market_condition.K,
                    'expiry': market_condition.T,
                    'option_type': option_type,
                    'market_price': market_price,
                    'implied_vol': np.nan,
                    'moneyness': market_condition.moneyness
                })
        
        df = pd.DataFrame(results)
        self.calibration_results.append(df)
        return df
    
    def analyze_vol_smile(self, vol_surface: pd.DataFrame, expiry: float) -> Dict[str, Any]:
        """
        Analyze volatility smile for a specific expiry.
        
        Args:
            vol_surface: DataFrame with volatility surface
            expiry: Time to expiry to analyze
            
        Returns:
            Dictionary with smile analysis
        """
        # Filter for specific expiry (with tolerance)
        expiry_data = vol_surface[abs(vol_surface['expiry'] - expiry) < 0.01]
        
        if expiry_data.empty:
            return {'error': f'No data found for expiry {expiry}'}
        
        # Sort by moneyness
        expiry_data = expiry_data.sort_values('moneyness').copy()
        
        # Calculate smile metrics
        atm_vol = expiry_data.loc[expiry_data['moneyness'].sub(1.0).abs().idxmin(), 'implied_vol']
        min_vol = expiry_data['implied_vol'].min()
        max_vol = expiry_data['implied_vol'].max()
        vol_range = max_vol - min_vol
        
        # Skew (difference between OTM put and call vols)
        otm_puts = expiry_data[(expiry_data['moneyness'] < 0.95) & (expiry_data['option_type'] == 'put')]
        otm_calls = expiry_data[(expiry_data['moneyness'] > 1.05) & (expiry_data['option_type'] == 'call')]
        
        if not otm_puts.empty and not otm_calls.empty:
            put_vol = otm_puts.iloc[0]['implied_vol']
            call_vol = otm_calls.iloc[0]['implied_vol']
            skew = put_vol - call_vol
        else:
            skew = np.nan
        
        return {
            'expiry': expiry,
            'atm_vol': atm_vol,
            'min_vol': min_vol,
            'max_vol': max_vol,
            'vol_range': vol_range,
            'skew': skew,
            'data_points': len(expiry_data)
        }


def create_sample_market_data(n_options: int = 20) -> List[Dict]:
    """
    Create sample market data for calibration testing.
    
    Args:
        n_options: Number of options to generate
        
    Returns:
        List of market data dictionaries
    """
    np.random.seed(42)
    market_data = []
    
    # Base market conditions
    S0 = 100.0
    r = 0.05
    q = 0.02
    
    for _ in range(n_options):
        # Random strikes around ATM
        K = np.random.uniform(80, 120)
        
        # Random expiries
        T = np.random.uniform(0.1, 1.0)
        
        # Random volatility for price generation
        vol = np.random.uniform(0.15, 0.35)
        
        # Random option type
        option_type = np.random.choice(['call', 'put'])
        
        # Create market data
        md = MarketData(S0=S0, K=K, T=T, r=r, q=q, sigma=vol)
        
        # Generate "market" price with some noise
        bs_price = BlackScholesModel.price(md, option_type)
        market_price = bs_price * (1 + np.random.normal(0, 0.02))  # 2% noise
        
        market_data.append({
            'market_data': md,
            'option_type': option_type,
            'market_price': market_price
        })
    
    return market_data