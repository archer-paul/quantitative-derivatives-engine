"""
Advanced visualization tools for option pricing and risk analysis.

This module provides comprehensive plotting and visualization capabilities
for option prices, Greeks profiles, risk metrics, and model comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Tuple, Any, Optional
import seaborn as sns

from ..core.market_data import MarketData, OptionType
from ..models.black_scholes import BlackScholesModel


class VisualizationTools:
    """Advanced visualization for option pricing and risk analysis."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualization tools.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        plt.style.use(style)
        self.default_figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_option_surface(
        self, 
        market_data: MarketData, 
        option_type: str,
        S_range: Optional[Tuple[float, float]] = None,
        T_range: Optional[Tuple[float, float]] = None,
        n_points: int = 50
    ) -> plt.Figure:
        """
        Create 3D surface plot of option prices vs spot and time.
        
        Args:
            market_data: Base market conditions
            option_type: 'call' or 'put'
            S_range: (min, max) spot price range
            T_range: (min, max) time to expiry range
            n_points: Number of grid points per dimension
            
        Returns:
            Matplotlib figure object
        """
        if S_range is None:
            S_range = (market_data.S0 * 0.5, market_data.S0 * 1.5)
        if T_range is None:
            T_range = (0.01, market_data.T)
        
        S_values = np.linspace(S_range[0], S_range[1], n_points)
        T_values = np.linspace(T_range[0], T_range[1], n_points)
        S_mesh, T_mesh = np.meshgrid(S_values, T_values)
        
        prices = np.zeros_like(S_mesh)
        for i in range(len(T_values)):
            for j in range(len(S_values)):
                md_temp = market_data.copy(S0=S_mesh[i,j], T=T_mesh[i,j])
                prices[i,j] = BlackScholesModel.price(md_temp, option_type)
        
        fig = plt.figure(figsize=self.default_figsize)
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(S_mesh, T_mesh, prices, cmap='viridis', alpha=0.8)
        
        ax.set_xlabel('Spot Price ($)')
        ax.set_ylabel('Time to Expiry (years)')
        ax.set_zlabel('Option Price ($)')
        ax.set_title(f'{option_type.capitalize()} Option Price Surface (K=${market_data.K})')
        
        # Add colorbar
        plt.colorbar(surf, ax=ax, shrink=0.5)
        
        return fig
    
    def plot_greeks_profile(
        self, 
        market_data: MarketData, 
        option_type: str,
        S_range: Optional[Tuple[float, float]] = None,
        n_points: int = 100
    ) -> plt.Figure:
        """
        Plot Greeks profiles across spot price range.
        
        Args:
            market_data: Market conditions
            option_type: 'call' or 'put'
            S_range: (min, max) spot price range
            n_points: Number of points to calculate
            
        Returns:
            Matplotlib figure with subplots for each Greek
        """
        if S_range is None:
            S_range = (market_data.S0 * 0.7, market_data.S0 * 1.3)
        
        S_values = np.linspace(S_range[0], S_range[1], n_points)
        greeks_data = {'delta': [], 'gamma': [], 'theta': [], 'vega': [], 'rho': []}
        prices = []
        
        for S in S_values:
            md_temp = market_data.copy(S0=S)
            price = BlackScholesModel.price(md_temp, option_type)
            greeks = BlackScholesModel.greeks(md_temp, option_type)
            
            prices.append(price)
            for greek_name in greeks_data:
                greeks_data[greek_name].append(greeks[greek_name])
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot price
        axes[0].plot(S_values, prices, linewidth=2, color=self.colors[0])
        axes[0].set_xlabel('Spot Price ($)')
        axes[0].set_ylabel('Option Price ($)')
        axes[0].set_title('Option Price')
        axes[0].grid(True, alpha=0.3)
        axes[0].axvline(market_data.S0, color='red', linestyle='--', alpha=0.7, label='Current Spot')
        axes[0].axvline(market_data.K, color='orange', linestyle='--', alpha=0.7, label='Strike')
        axes[0].legend()
        
        # Plot Greeks
        for i, (greek_name, values) in enumerate(greeks_data.items(), 1):
            axes[i].plot(S_values, values, linewidth=2, color=self.colors[i % len(self.colors)])
            axes[i].set_xlabel('Spot Price ($)')
            axes[i].set_ylabel(greek_name.capitalize())
            axes[i].set_title(f'{greek_name.capitalize()} Profile')
            axes[i].grid(True, alpha=0.3)
            axes[i].axvline(market_data.S0, color='red', linestyle='--', alpha=0.7, label='Current Spot')
            axes[i].axvline(market_data.K, color='orange', linestyle='--', alpha=0.7, label='Strike')
            axes[i].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self, results: Dict[str, Any]) -> plt.Figure:
        """
        Compare prices across different pricing models.
        
        Args:
            results: Results dictionary from comprehensive pricing
            
        Returns:
            Bar chart comparing model prices
        """
        models = []
        prices = []
        errors = []
        
        # Extract prices from results
        if 'black_scholes' in results:
            models.append('Black-Scholes')
            prices.append(results['black_scholes']['price'])
            errors.append(0)  # No error for analytical solution
        
        if 'heston' in results:
            models.append('Heston (MC)')
            prices.append(results['heston']['monte_carlo_price'])
            errors.append(results['heston']['monte_carlo_ci'])
        
        if 'jump_diffusion' in results:
            models.append('Jump-Diffusion')
            prices.append(results['jump_diffusion']['analytical_price'])
            errors.append(0)  # Analytical solution
        
        if 'finite_difference' in results:
            models.append('Finite Difference')
            prices.append(results['finite_difference']['price'])
            errors.append(0)  # No error estimate
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=self.default_figsize)
        bars = ax.bar(models, prices, yerr=errors, capsize=5,
                     color=self.colors[:len(models)], alpha=0.7)
        
        ax.set_ylabel('Option Price ($)')
        ax.set_title('Model Price Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, price, error in zip(bars, prices, errors):
            height = bar.get_height()
            label = f'${price:.4f}'
            if error > 0:
                label += f'\n±{error:.4f}'
            ax.text(bar.get_x() + bar.get_width()/2, height + max(errors)*0.1,
                   label, ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def plot_volatility_smile(
        self, 
        market_data: MarketData, 
        option_type: str,
        strike_range: Optional[Tuple[float, float]] = None,
        market_prices: Optional[Dict[float, float]] = None
    ) -> plt.Figure:
        """
        Plot implied volatility smile/skew.
        
        Args:
            market_data: Base market conditions
            option_type: 'call' or 'put'
            strike_range: Range of strikes to plot
            market_prices: Optional dictionary of {strike: market_price}
            
        Returns:
            Volatility smile plot
        """
        if strike_range is None:
            strike_range = (market_data.S0 * 0.8, market_data.S0 * 1.2)
        
        strikes = np.linspace(strike_range[0], strike_range[1], 20)
        implied_vols = []
        
        for K in strikes:
            md_temp = market_data.copy(K=K)
            
            if market_prices and K in market_prices:
                # Calculate implied volatility from market price
                try:
                    iv = BlackScholesModel.implied_volatility(
                        md_temp, option_type, market_prices[K]
                    )
                except:
                    iv = market_data.sigma  # Fallback to base volatility
            else:
                # Use base volatility (flat smile)
                iv = market_data.sigma
            
            implied_vols.append(iv)
        
        # Calculate moneyness
        moneyness = strikes / market_data.S0
        
        fig, ax = plt.subplots(figsize=self.default_figsize)
        ax.plot(moneyness, np.array(implied_vols) * 100, 'o-', 
                linewidth=2, markersize=6, color=self.colors[0])
        
        ax.set_xlabel('Moneyness (K/S0)')
        ax.set_ylabel('Implied Volatility (%)')
        ax.set_title(f'Implied Volatility Smile/Skew ({option_type.capitalize()})')
        ax.grid(True, alpha=0.3)
        ax.axvline(1.0, color='red', linestyle='--', alpha=0.7, label='ATM')
        ax.legend()
        
        return fig
    
    def plot_risk_concentration(self, portfolio_analysis: Dict[str, Any]) -> plt.Figure:
        """
        Plot portfolio risk concentration analysis.
        
        Args:
            portfolio_analysis: Results from portfolio risk analysis
            
        Returns:
            Risk concentration visualization
        """
        risk_conc = portfolio_analysis['risk_concentration']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # By option type
        option_types = list(risk_conc['by_option_type'].keys())
        type_values = list(risk_conc['by_option_type'].values())
        
        axes[0, 0].pie(type_values, labels=[t.replace('_percentage', '').title() for t in option_types],
                      autopct='%1.1f%%', colors=self.colors[:len(option_types)])
        axes[0, 0].set_title('Concentration by Option Type')
        
        # By expiry
        expiries = list(risk_conc['by_expiry'].keys())
        expiry_values = list(risk_conc['by_expiry'].values())
        
        if expiries:
            axes[0, 1].bar(range(len(expiries)), expiry_values, color=self.colors[1])
            axes[0, 1].set_xlabel('Time to Expiry (years)')
            axes[0, 1].set_ylabel('Portfolio Weight (%)')
            axes[0, 1].set_title('Concentration by Expiry')
            axes[0, 1].set_xticks(range(len(expiries)))
            axes[0, 1].set_xticklabels([f'{e:.2f}' for e in expiries], rotation=45)
        
        # By moneyness
        moneyness_buckets = list(risk_conc['by_moneyness'].keys())
        moneyness_values = list(risk_conc['by_moneyness'].values())
        
        axes[1, 0].bar(moneyness_buckets, moneyness_values, color=self.colors[2])
        axes[1, 0].set_xlabel('Moneyness Bucket')
        axes[1, 0].set_ylabel('Portfolio Weight (%)')
        axes[1, 0].set_title('Concentration by Moneyness')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Portfolio Greeks
        greeks = portfolio_analysis['portfolio_greeks']
        greek_names = list(greeks.keys())
        greek_values = list(greeks.values())
        
        axes[1, 1].bar(greek_names, greek_values, color=self.colors[3])
        axes[1, 1].set_xlabel('Greeks')
        axes[1, 1].set_ylabel('Portfolio Greek Value')
        axes[1, 1].set_title('Portfolio Greeks')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig