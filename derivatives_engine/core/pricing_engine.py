"""
Main pricing engine that orchestrates all pricing models and analysis.

This module provides the central DerivativesPricingEngine class that coordinates
pricing across different models and provides portfolio-level analysis.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict

from .market_data import MarketData, OptionType, validate_option_type
from ..models.black_scholes import BlackScholesModel
from ..models.heston import HestonModel, HestonParameters
from ..models.jump_diffusion import MertonJumpDiffusionModel, JumpDiffusionParameters
from ..models.finite_difference import FiniteDifferencePricer
from ..exotic.exotic_engine import ExoticOptions


class DerivativesPricingEngine:
    """
    Main pricing engine orchestrating all models and analysis.
    
    This class provides a unified interface for pricing options using multiple
    models, calculating Greeks, performing portfolio analysis, and running
    sensitivity studies.
    """
    
    def __init__(self):
        """Initialize the pricing engine with all models."""
        self.bs_model = BlackScholesModel()
        self.fd_pricer = FiniteDifferencePricer()
        self.exotic_engine = ExoticOptions()
        self.results_history: List[Dict] = []
        
        # Performance tracking
        self._pricing_times: Dict[str, List[float]] = {
            'black_scholes': [],
            'heston_fft': [],
            'heston_mc': [],
            'jump_diffusion': [],
            'finite_difference': []
        }
    
    def comprehensive_pricing(
        self,
        market_data: MarketData,
        option_type: str,
        heston_params: Optional[HestonParameters] = None,
        jump_params: Optional[JumpDiffusionParameters] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive pricing using all available models.
        
        Args:
            market_data: Market conditions and option parameters
            option_type: 'call' or 'put'
            heston_params: Optional Heston model parameters
            jump_params: Optional jump-diffusion parameters
            
        Returns:
            Dictionary containing results from all models
        """
        option_type = validate_option_type(option_type)
        
        results = {
            'market_data': market_data,
            'option_type': option_type,
            'timestamp': pd.Timestamp.now(),
            'pricing_times': {}
        }
        
        # Black-Scholes pricing and Greeks
        print("Computing Black-Scholes pricing...")
        start_time = time.perf_counter()
        
        bs_price = self.bs_model.price(market_data, option_type)
        bs_greeks = self.bs_model.greeks(market_data, option_type)
        
        bs_time = time.perf_counter() - start_time
        self._pricing_times['black_scholes'].append(bs_time)
        
        results['black_scholes'] = {
            'price': bs_price,
            'greeks': bs_greeks
        }
        results['pricing_times']['black_scholes'] = bs_time
        
        # Heston model pricing (if parameters provided)
        if heston_params:
            print("Computing Heston stochastic volatility pricing...")
            heston_model = HestonModel(heston_params)
            
            # FFT pricing
            start_time = time.perf_counter()
            try:
                heston_fft_price = heston_model.price_fft(market_data, option_type)
                fft_time = time.perf_counter() - start_time
                self._pricing_times['heston_fft'].append(fft_time)
                results['pricing_times']['heston_fft'] = fft_time
            except Exception as e:
                print(f"FFT pricing failed: {e}")
                heston_fft_price = None
                fft_time = None
            
            # Monte Carlo pricing
            start_time = time.perf_counter()
            heston_mc_price, heston_mc_ci = heston_model.monte_carlo_price(
                market_data, option_type, n_paths=50000
            )
            mc_time = time.perf_counter() - start_time
            self._pricing_times['heston_mc'].append(mc_time)
            results['pricing_times']['heston_mc'] = mc_time
            
            results['heston'] = {
                'fft_price': heston_fft_price,
                'monte_carlo_price': heston_mc_price,
                'monte_carlo_ci': heston_mc_ci,
                'parameters': asdict(heston_params)
            }
        
        # Jump-diffusion model pricing (if parameters provided)
        if jump_params:
            print("Computing Merton jump-diffusion pricing...")
            jd_model = MertonJumpDiffusionModel(jump_params)
            
            # Analytical pricing
            start_time = time.perf_counter()
            jd_analytical_price = jd_model.price_analytical(market_data, option_type)
            analytical_time = time.perf_counter() - start_time
            self._pricing_times['jump_diffusion'].append(analytical_time)
            results['pricing_times']['jump_diffusion'] = analytical_time
            
            # Monte Carlo pricing
            jd_mc_price, jd_mc_ci = jd_model.monte_carlo_price(
                market_data, option_type, n_paths=50000
            )
            
            results['jump_diffusion'] = {
                'analytical_price': jd_analytical_price,
                'monte_carlo_price': jd_mc_price,
                'monte_carlo_ci': jd_mc_ci,
                'parameters': asdict(jump_params)
            }
        
        # Finite difference pricing
        print("Computing finite difference pricing...")
        start_time = time.perf_counter()
        fd_price, _, _ = self.fd_pricer.implicit_fd_vanilla(
            market_data, option_type, S_max=2*market_data.S0, M=100, N=500
        )
        fd_time = time.perf_counter() - start_time
        self._pricing_times['finite_difference'].append(fd_time)
        results['pricing_times']['finite_difference'] = fd_time
        
        results['finite_difference'] = {'price': fd_price}
        
        # Store results
        self.results_history.append(results)
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate comprehensive pricing report.
        
        Args:
            results: Results dictionary from comprehensive_pricing
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("ADVANCED DERIVATIVES PRICING ENGINE - COMPREHENSIVE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {results['timestamp']}")
        report.append(f"Option Type: {results['option_type'].upper()}")
        report.append("")
        
        # Market data
        md = results['market_data']
        report.append("MARKET DATA:")
        report.append(f"  Spot Price (S0):      ${md.S0:,.2f}")
        report.append(f"  Strike Price (K):     ${md.K:,.2f}")
        report.append(f"  Time to Expiry (T):   {md.T:.4f} years")
        report.append(f"  Risk-free Rate (r):   {md.r:.4%}")
        report.append(f"  Dividend Yield (q):   {md.q:.4%}")
        report.append(f"  Volatility (σ):       {md.sigma:.4%}")
        report.append(f"  Moneyness (S0/K):     {md.moneyness:.3f}")
        report.append(f"  Forward Price:        ${md.forward_price:,.2f}")
        report.append("")
        
        # Black-Scholes results
        bs = results['black_scholes']
        report.append("BLACK-SCHOLES MODEL:")
        report.append("-" * 30)
        report.append(f"  Option Price:         ${bs['price']:.4f}")
        report.append("  Greeks:")
        for greek_name, greek_value in bs['greeks'].items():
            report.append(f"    {greek_name.capitalize():>8}: {greek_value:>10.6f}")
        
        if 'pricing_times' in results and 'black_scholes' in results['pricing_times']:
            report.append(f"  Computation Time:     {results['pricing_times']['black_scholes']*1000:.3f} ms")
        report.append("")
        
        # Heston model results
        if 'heston' in results:
            heston = results['heston']
            report.append("HESTON STOCHASTIC VOLATILITY MODEL:")
            report.append("-" * 40)
            if heston['fft_price'] is not None:
                report.append(f"  FFT Price:            ${heston['fft_price']:.4f}")
            report.append(f"  Monte Carlo Price:    ${heston['monte_carlo_price']:.4f} ± {heston['monte_carlo_ci']:.4f}")
            
            # Model parameters
            params = heston['parameters']
            report.append("  Model Parameters:")
            report.append(f"    v0 (initial var):   {params['v0']:.6f}")
            report.append(f"    theta (long-term):  {params['theta']:.6f}")
            report.append(f"    kappa (mean rev):   {params['kappa']:.4f}")
            report.append(f"    sigma_v (vol vol):  {params['sigma_v']:.4f}")
            report.append(f"    rho (correlation):  {params['rho']:.4f}")
            report.append("")
        
        # Jump-diffusion results
        if 'jump_diffusion' in results:
            jd = results['jump_diffusion']
            report.append("MERTON JUMP-DIFFUSION MODEL:")
            report.append("-" * 35)
            report.append(f"  Analytical Price:     ${jd['analytical_price']:.4f}")
            report.append(f"  Monte Carlo Price:    ${jd['monte_carlo_price']:.4f} ± {jd['monte_carlo_ci']:.4f}")
            
            # Model parameters
            params = jd['parameters']
            report.append("  Model Parameters:")
            report.append(f"    lambda (intensity): {params['lambda_j']:.4f}")
            report.append(f"    mu_j (jump mean):   {params['mu_j']:.4f}")
            report.append(f"    sigma_j (jump vol): {params['sigma_j']:.4f}")
            report.append("")
        
        # Finite difference results
        if 'finite_difference' in results:
            fd = results['finite_difference']
            report.append("FINITE DIFFERENCE METHOD:")
            report.append("-" * 30)
            report.append(f"  Numerical Price:      ${fd['price']:.4f}")
            if 'pricing_times' in results and 'finite_difference' in results['pricing_times']:
                report.append(f"  Computation Time:     {results['pricing_times']['finite_difference']*1000:.1f} ms")
            report.append("")
        
        # Model comparison
        report.append("MODEL COMPARISON:")
        report.append("-" * 20)
        bs_price = bs['price']
        report.append(f"  Black-Scholes:        ${bs_price:.4f} (baseline)")
        
        if 'heston' in results:
            heston_diff = results['heston']['monte_carlo_price'] - bs_price
            heston_pct = (heston_diff / bs_price) * 100
            report.append(f"  Heston Difference:    ${heston_diff:+.4f} ({heston_pct:+.2f}%)")
        
        if 'jump_diffusion' in results:
            jd_diff = results['jump_diffusion']['analytical_price'] - bs_price
            jd_pct = (jd_diff / bs_price) * 100
            report.append(f"  Jump-Diff Difference: ${jd_diff:+.4f} ({jd_pct:+.2f}%)")
        
        if 'finite_difference' in results:
            fd_diff = results['finite_difference']['price'] - bs_price
            fd_pct = (fd_diff / bs_price) * 100
            report.append(f"  Finite Diff Diff:     ${fd_diff:+.4f} ({fd_pct:+.2f}%)")
        
        # Performance summary
        if 'pricing_times' in results:
            report.append("")
            report.append("PERFORMANCE SUMMARY:")
            report.append("-" * 22)
            for method, time_val in results['pricing_times'].items():
                if time_val is not None:
                    report.append(f"  {method.replace('_', ' ').title():20}: {time_val*1000:6.3f} ms")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of historical performance metrics.
        
        Returns:
            Performance statistics across all historical pricing calls
        """
        summary = {}
        
        for method, times in self._pricing_times.items():
            if times:
                times_array = np.array(times) * 1000  # Convert to ms
                summary[method] = {
                    'count': len(times),
                    'mean_ms': np.mean(times_array),
                    'std_ms': np.std(times_array),
                    'min_ms': np.min(times_array),
                    'max_ms': np.max(times_array),
                    'median_ms': np.median(times_array),
                    'p95_ms': np.percentile(times_array, 95),
                    'ops_per_second': 1000 / np.mean(times_array) if np.mean(times_array) > 0 else float('inf')
                }
        
        return summary
    
    def clear_performance_history(self):
        """Clear performance timing history."""
        for method in self._pricing_times:
            self._pricing_times[method].clear()
    
    def validate_models(self, market_data: MarketData) -> Dict[str, bool]:
        """
        Validate that all models produce reasonable results.
        
        Args:
            market_data: Market conditions for validation
            
        Returns:
            Dictionary indicating which models passed validation
        """
        validation_results = {}
        
        # Test Black-Scholes
        try:
            call_price = self.bs_model.price(market_data, OptionType.CALL)
            put_price = self.bs_model.price(market_data, OptionType.PUT)
            greeks = self.bs_model.greeks(market_data, OptionType.CALL)
            
            # Basic validation checks
            validation_results['black_scholes'] = (
                call_price > 0 and put_price > 0 and
                0 <= greeks['delta'] <= 1 and
                greeks['gamma'] >= 0 and
                greeks['vega'] >= 0
            )
        except Exception:
            validation_results['black_scholes'] = False
        
        # Test finite difference
        try:
            fd_price, _, _ = self.fd_pricer.implicit_fd_vanilla(market_data, OptionType.CALL)
            validation_results['finite_difference'] = fd_price > 0
        except Exception:
            validation_results['finite_difference'] = False
        
        # Test exotic options
        try:
            barrier_price, _ = self.exotic_engine.barrier_option_mc(
                market_data, OptionType.CALL, 'up_out', market_data.S0 * 1.2, n_paths=1000
            )
            validation_results['exotic_options'] = barrier_price >= 0
        except Exception:
            validation_results['exotic_options'] = False
        
        return validation_results
    
    def benchmark_performance(
        self,
        market_data: MarketData,
        option_type: str,
        n_iterations: int = 1000
    ) -> Dict[str, Any]:
        """
        Benchmark pricing performance across different methods.
        
        Args:
            market_data: Market conditions for benchmarking
            option_type: 'call' or 'put'
            n_iterations: Number of iterations for timing
            
        Returns:
            Performance benchmark results
        """
        print(f"Benchmarking performance over {n_iterations} iterations...")
        option_type = validate_option_type(option_type)
        
        benchmark_results = {}
        
        # Black-Scholes pricing benchmark
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            _ = self.bs_model.price(market_data, option_type)
        bs_time = (time.perf_counter() - start_time) / n_iterations * 1000  # ms
        
        benchmark_results['black_scholes_pricing'] = {
            'avg_time_ms': bs_time,
            'iterations': n_iterations,
            'ops_per_second': 1000 / bs_time if bs_time > 0 else float('inf')
        }
        
        # Greeks calculation benchmark
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            _ = self.bs_model.greeks(market_data, option_type)
        greeks_time = (time.perf_counter() - start_time) / n_iterations * 1000  # ms
        
        benchmark_results['greeks_calculation'] = {
            'avg_time_ms': greeks_time,
            'iterations': n_iterations,
            'ops_per_second': 1000 / greeks_time if greeks_time > 0 else float('inf')
        }
        
        # Finite difference benchmark (fewer iterations due to computational cost)
        fd_iterations = min(100, n_iterations)
        start_time = time.perf_counter()
        for _ in range(fd_iterations):
            _ = self.fd_pricer.implicit_fd_vanilla(
                market_data, option_type, S_max=2*market_data.S0, M=50, N=100
            )
        fd_time = (time.perf_counter() - start_time) / fd_iterations * 1000  # ms
        
        benchmark_results['finite_difference'] = {
            'avg_time_ms': fd_time,
            'iterations': fd_iterations,
            'ops_per_second': 1000 / fd_time if fd_time > 0 else float('inf')
        }
        
        return benchmark_results
