"""
Risk analysis and portfolio management utilities.

This module provides comprehensive risk analysis tools for derivatives portfolios,
including VaR calculation, Greeks aggregation, and stress testing.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import warnings

from ..core.market_data import MarketData, OptionType


@dataclass
class VaRResult:
    """Container for Value at Risk calculation results."""
    var_95: float
    var_99: float
    var_99_9: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    confidence_intervals: Dict[str, float]
    methodology: str
    holding_period_days: int


@dataclass
class StressTestResult:
    """Container for stress test results."""
    scenario_name: str
    base_value: float
    stressed_value: float
    pnl: float
    pnl_percentage: float
    risk_factors: Dict[str, float]


class RiskAnalyzer:
    """Comprehensive risk analysis tools for derivatives portfolios."""
    
    def __init__(self):
        """Initialize risk analyzer."""
        self.stress_scenarios = self._define_standard_scenarios()
    
    def calculate_portfolio_var(
        self,
        portfolio_positions: List[Dict[str, Any]],
        confidence_levels: List[float] = [0.95, 0.99, 0.999],
        holding_period_days: int = 1,
        method: str = "delta_normal"
    ) -> VaRResult:
        """
        Calculate portfolio Value at Risk using different methodologies.
        
        Args:
            portfolio_positions: List of position dictionaries
            confidence_levels: Confidence levels for VaR calculation
            holding_period_days: Holding period in days
            method: VaR methodology ('delta_normal', 'historical', 'monte_carlo')
            
        Returns:
            VaR calculation results
        """
        if method == "delta_normal":
            return self._delta_normal_var(portfolio_positions, confidence_levels, holding_period_days)
        elif method == "historical":
            return self._historical_var(portfolio_positions, confidence_levels, holding_period_days)
        elif method == "monte_carlo":
            return self._monte_carlo_var(portfolio_positions, confidence_levels, holding_period_days)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    def _delta_normal_var(
        self,
        positions: List[Dict[str, Any]],
        confidence_levels: List[float],
        holding_period_days: int
    ) -> VaRResult:
        """Calculate VaR using delta-normal method."""
        from ..models.black_scholes import BlackScholesModel
        
        # Calculate portfolio delta and other sensitivities
        total_delta = 0
        total_gamma = 0
        total_vega = 0
        portfolio_value = 0
        
        for position in positions:
            market_data = position['market_data']
            option_type = position['option_type']
            quantity = position['quantity']
            
            # Calculate Greeks
            price = BlackScholesModel.price(market_data, option_type)
            greeks = BlackScholesModel.greeks(market_data, option_type)
            
            portfolio_value += price * quantity
            total_delta += greeks['delta'] * quantity
            total_gamma += greeks['gamma'] * quantity
            total_vega += greeks['vega'] * quantity
        
        # Assume market parameters for volatility calculation
        underlying_vol = 0.20  # 20% annual volatility assumption
        vol_of_vol = 0.15      # 15% volatility of volatility
        
        # Calculate portfolio volatility
        holding_period_factor = np.sqrt(holding_period_days / 252)
        
        # First-order (delta) risk
        delta_risk = abs(total_delta) * underlying_vol * holding_period_factor
        
        # Second-order (gamma) risk - approximate
        gamma_risk = 0.5 * abs(total_gamma) * (underlying_vol * holding_period_factor) ** 2
        
        # Volatility risk (vega)
        vega_risk = abs(total_vega) * vol_of_vol * holding_period_factor
        
        # Total portfolio volatility (simplified approach)
        portfolio_vol = np.sqrt(delta_risk**2 + gamma_risk**2 + vega_risk**2)
        
        # Calculate VaR for different confidence levels
        var_results = {}
        es_results = {}
        
        for conf_level in confidence_levels:
            z_score = stats.norm.ppf(conf_level)
            var_value = z_score * portfolio_vol
            var_results[f'var_{int(conf_level*100)}'] = var_value
            
            # Expected Shortfall (Conditional VaR)
            es_value = portfolio_vol * stats.norm.pdf(z_score) / (1 - conf_level)
            es_results[f'es_{int(conf_level*100)}'] = es_value
        
        return VaRResult(
            var_95=var_results.get('var_95', 0),
            var_99=var_results.get('var_99', 0),
            var_99_9=var_results.get('var_999', 0),
            expected_shortfall_95=es_results.get('es_95', 0),
            expected_shortfall_99=es_results.get('es_99', 0),
            confidence_intervals=var_results,
            methodology="delta_normal",
            holding_period_days=holding_period_days
        )
    
    def _historical_var(
        self,
        positions: List[Dict[str, Any]],
        confidence_levels: List[float],
        holding_period_days: int
    ) -> VaRResult:
        """Calculate VaR using historical simulation method."""
        # This would require historical price data
        # For now, return a placeholder implementation
        warnings.warn("Historical VaR not fully implemented - using delta-normal as fallback")
        return self._delta_normal_var(positions, confidence_levels, holding_period_days)
    
    def _monte_carlo_var(
        self,
        positions: List[Dict[str, Any]],
        confidence_levels: List[float],
        holding_period_days: int,
        n_simulations: int = 10000
    ) -> VaRResult:
        """Calculate VaR using Monte Carlo simulation."""
        from ..models.black_scholes import BlackScholesModel
        
        # Generate random market scenarios
        np.random.seed(42)  # For reproducibility
        
        # Market parameters
        underlying_vol = 0.20
        dt = holding_period_days / 252
        
        portfolio_pnl = []
        
        for _ in range(n_simulations):
            # Generate random market moves
            stock_return = np.random.normal(0, underlying_vol * np.sqrt(dt))
            vol_shock = np.random.normal(0, 0.05)  # 5% vol shock
            
            scenario_pnl = 0
            
            for position in positions:
                market_data = position['market_data']
                option_type = position['option_type']
                quantity = position['quantity']
                
                # Current price
                current_price = BlackScholesModel.price(market_data, option_type)
                
                # Shocked market data
                new_spot = market_data.S0 * (1 + stock_return)
                new_vol = max(0.01, market_data.sigma + vol_shock)
                
                shocked_market = market_data.copy(S0=new_spot, sigma=new_vol)
                shocked_price = BlackScholesModel.price(shocked_market, option_type)
                
                position_pnl = (shocked_price - current_price) * quantity
                scenario_pnl += position_pnl
            
            portfolio_pnl.append(scenario_pnl)
        
        portfolio_pnl = np.array(portfolio_pnl)
        
        # Calculate VaR and Expected Shortfall
        var_results = {}
        es_results = {}
        
        for conf_level in confidence_levels:
            var_percentile = (1 - conf_level) * 100
            var_value = -np.percentile(portfolio_pnl, var_percentile)
            var_results[f'var_{int(conf_level*100)}'] = var_value
            
            # Expected Shortfall
            tail_losses = portfolio_pnl[portfolio_pnl <= -var_value]
            if len(tail_losses) > 0:
                es_value = -np.mean(tail_losses)
            else:
                es_value = var_value
            es_results[f'es_{int(conf_level*100)}'] = es_value
        
        return VaRResult(
            var_95=var_results.get('var_95', 0),
            var_99=var_results.get('var_99', 0),
            var_99_9=var_results.get('var_999', 0),
            expected_shortfall_95=es_results.get('es_95', 0),
            expected_shortfall_99=es_results.get('es_99', 0),
            confidence_intervals=var_results,
            methodology="monte_carlo",
            holding_period_days=holding_period_days
        )
    
    def stress_test_portfolio(
        self,
        portfolio_positions: List[Dict[str, Any]],
        custom_scenarios: Optional[Dict[str, Dict[str, float]]] = None
    ) -> List[StressTestResult]:
        """
        Perform stress testing on portfolio.
        
        Args:
            portfolio_positions: List of position dictionaries
            custom_scenarios: Optional custom stress scenarios
            
        Returns:
            List of stress test results
        """
        from ..models.black_scholes import BlackScholesModel
        
        scenarios = self.stress_scenarios.copy()
        if custom_scenarios:
            scenarios.update(custom_scenarios)
        
        # Calculate base portfolio value
        base_value = 0
        for position in portfolio_positions:
            market_data = position['market_data']
            option_type = position['option_type']
            quantity = position['quantity']
            price = BlackScholesModel.price(market_data, option_type)
            base_value += price * quantity
        
        results = []
        
        for scenario_name, risk_factors in scenarios.items():
            stressed_value = 0
            
            for position in portfolio_positions:
                market_data = position['market_data']
                option_type = position['option_type']
                quantity = position['quantity']
                
                # Apply stress scenario
                stressed_market = self._apply_stress_scenario(market_data, risk_factors)
                stressed_price = BlackScholesModel.price(stressed_market, option_type)
                stressed_value += stressed_price * quantity
            
            pnl = stressed_value - base_value
            pnl_pct = (pnl / base_value * 100) if base_value != 0 else 0
            
            results.append(StressTestResult(
                scenario_name=scenario_name,
                base_value=base_value,
                stressed_value=stressed_value,
                pnl=pnl,
                pnl_percentage=pnl_pct,
                risk_factors=risk_factors
            ))
        
        return results
    
    def _apply_stress_scenario(
        self, 
        market_data: MarketData, 
        risk_factors: Dict[str, float]
    ) -> MarketData:
        """Apply stress scenario to market data."""
        kwargs = {}
        
        if 'stock_shock' in risk_factors:
            kwargs['S0'] = market_data.S0 * (1 + risk_factors['stock_shock'])
        
        if 'vol_shock' in risk_factors:
            kwargs['sigma'] = max(0.01, market_data.sigma + risk_factors['vol_shock'])
        
        if 'rate_shock' in risk_factors:
            kwargs['r'] = max(0, market_data.r + risk_factors['rate_shock'])
        
        if 'time_decay' in risk_factors:
            kwargs['T'] = max(0.001, market_data.T + risk_factors['time_decay'])
        
        return market_data.copy(**kwargs)
    
    def _define_standard_scenarios(self) -> Dict[str, Dict[str, float]]:
        """Define standard stress test scenarios."""
        return {
            "Market_Crash_-20%": {
                "stock_shock": -0.20,
                "vol_shock": 0.10
            },
            "Market_Rally_+15%": {
                "stock_shock": 0.15,
                "vol_shock": -0.05
            },
            "Vol_Spike_+50%": {
                "vol_shock": 0.10,  # 10 percentage points
                "stock_shock": -0.05
            },
            "Vol_Crush_-50%": {
                "vol_shock": -0.05,  # 5 percentage points
                "stock_shock": 0.02
            },
            "Rate_Spike_+200bp": {
                "rate_shock": 0.02,
                "stock_shock": -0.03
            },
            "Rate_Cut_-100bp": {
                "rate_shock": -0.01,
                "stock_shock": 0.02
            },
            "Time_Decay_1_Week": {
                "time_decay": -7/365,
                "stock_shock": 0
            },
            "Black_Monday_1987": {
                "stock_shock": -0.22,
                "vol_shock": 0.15
            },
            "Flash_Crash_2010": {
                "stock_shock": -0.09,
                "vol_shock": 0.08
            },
            "COVID_Crash_2020": {
                "stock_shock": -0.35,
                "vol_shock": 0.20
            }
        }
    
    def calculate_maximum_drawdown(
        self,
        portfolio_values: List[float],
        dates: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Calculate maximum drawdown from portfolio value series.
        
        Args:
            portfolio_values: Time series of portfolio values
            dates: Optional corresponding dates
            
        Returns:
            Dictionary with drawdown statistics
        """
        values = np.array(portfolio_values)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(values)
        
        # Calculate drawdowns
        drawdowns = (values - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown = np.min(drawdowns)
        max_dd_idx = np.argmin(drawdowns)
        
        # Find the peak before max drawdown
        peak_idx = np.argmax(running_max[:max_dd_idx + 1])
        
        # Calculate recovery information
        recovery_idx = None
        if max_dd_idx < len(values) - 1:
            post_dd_values = values[max_dd_idx + 1:]
            recovery_threshold = running_max[max_dd_idx]
            recovery_indices = np.where(post_dd_values >= recovery_threshold)[0]
            if len(recovery_indices) > 0:
                recovery_idx = max_dd_idx + 1 + recovery_indices[0]
        
        result = {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'peak_value': values[peak_idx] if peak_idx is not None else None,
            'trough_value': values[max_dd_idx],
            'peak_date': dates[peak_idx] if dates and peak_idx is not None else peak_idx,
            'trough_date': dates[max_dd_idx] if dates else max_dd_idx,
            'recovery_date': dates[recovery_idx] if dates and recovery_idx is not None else recovery_idx,
            'drawdown_duration': max_dd_idx - peak_idx if peak_idx is not None else None,
            'recovery_duration': recovery_idx - max_dd_idx if recovery_idx is not None else None,
            'current_drawdown': drawdowns[-1],
            'current_drawdown_pct': drawdowns[-1] * 100
        }
        
        return result
    
    def calculate_risk_adjusted_returns(
        self,
        returns: List[float],
        risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """
        Calculate risk-adjusted return metrics.
        
        Args:
            returns: List of portfolio returns
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            
        Returns:
            Dictionary with risk-adjusted metrics
        """
        returns_array = np.array(returns)
        
        # Basic statistics
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)
        
        # Risk-adjusted metrics
        sharpe_ratio = (mean_return - risk_free_rate) / std_return if std_return > 0 else 0
        
        # Sortino ratio (using downside deviation)
        negative_returns = returns_array[returns_array < 0]
        downside_std = np.std(negative_returns, ddof=1) if len(negative_returns) > 1 else 0
        sortino_ratio = (mean_return - risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Calmar ratio (annual return / max drawdown)
        portfolio_values = np.cumprod(1 + returns_array)
        dd_info = self.calculate_maximum_drawdown(portfolio_values.tolist())
        calmar_ratio = (mean_return * 252) / abs(dd_info['max_drawdown']) if dd_info['max_drawdown'] < 0 else 0
        
        return {
            'mean_return': mean_return,
            'volatility': std_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': dd_info['max_drawdown'],
            'skewness': stats.skew(returns_array),
            'kurtosis': stats.kurtosis(returns_array)
        }