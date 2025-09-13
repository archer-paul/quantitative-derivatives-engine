"""
Performance optimizations for the derivatives pricing engine.

This module provides various optimization techniques including:
- Numba JIT compilation
- Vectorized operations
- Caching mechanisms
- Memory-efficient algorithms
"""

import numpy as np
import functools
import time
from typing import Callable, Any, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import numba for JIT compilation
try:
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
    logger.info("Numba available for JIT compilation")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available - falling back to pure Python")
    
    # Define dummy decorators if numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args else decorator
    
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args else decorator
    
    prange = range


class OptimizedCache:
    """High-performance caching with automatic cleanup."""
    
    def __init__(self, max_size: int = 1000, ttl: Optional[float] = None):
        """
        Initialize optimized cache.
        
        Args:
            max_size: Maximum number of items to cache
            ttl: Time-to-live in seconds (None for no expiration)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self.insertion_order = []
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key not in self.cache:
            return None
        
        # Check TTL
        if self.ttl is not None:
            if time.time() - self.access_times[key] > self.ttl:
                self._remove(key)
                return None
        
        # Update access time
        self.access_times[key] = time.time()
        return self.cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache."""
        # Remove if already exists
        if key in self.cache:
            self._remove(key)
        
        # Check size limit
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            oldest_key = self.insertion_order[0]
            self._remove(oldest_key)
        
        # Add new item
        self.cache[key] = value
        self.access_times[key] = time.time()
        self.insertion_order.append(key)
    
    def _remove(self, key: str) -> None:
        """Remove item from cache."""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            self.insertion_order.remove(key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.access_times.clear()
        self.insertion_order.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


# Global cache instance
_global_cache = OptimizedCache(max_size=10000, ttl=3600)  # 1 hour TTL


def cached_result(cache_key_func: Optional[Callable] = None, ttl: Optional[float] = None):
    """
    Decorator for caching function results.
    
    Args:
        cache_key_func: Function to generate cache key from arguments
        ttl: Time-to-live override for this function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                # Default: use function name + str of all arguments
                cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            result = _global_cache.get(cache_key)
            if result is not None:
                return result
            
            # Calculate result and cache it
            result = func(*args, **kwargs)
            _global_cache.set(cache_key, result)
            return result
        
        return wrapper
    return decorator


# JIT-optimized mathematical functions
@njit
def fast_exp(x: float) -> float:
    """Fast exponential function."""
    return np.exp(x)


@njit
def fast_log(x: float) -> float:
    """Fast natural logarithm."""
    return np.log(x)


@njit
def fast_sqrt(x: float) -> float:
    """Fast square root."""
    return np.sqrt(x)


@njit
def norm_cdf_approx(x: float) -> float:
    """
    Fast approximation of normal CDF using Abramowitz and Stegun formula.
    
    Maximum error < 7.5e-8 for all x.
    """
    if x < -6:
        return 0.0
    if x > 6:
        return 1.0
    
    # Constants for the approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    
    # Save the sign of x
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    
    # A&S formula
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    
    return 0.5 * (1.0 + sign * y)


@njit
def norm_pdf_fast(x: float) -> float:
    """Fast normal PDF calculation."""
    return np.exp(-0.5 * x * x) / 2.5066282746310002  # sqrt(2*pi) = 2.5066...


@njit
def black_scholes_call_price_fast(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    JIT-optimized Black-Scholes call price calculation.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        
    Returns:
        Call option price
    """
    if T <= 0:
        return max(S - K, 0)
    
    sqrt_T = fast_sqrt(T)
    d1 = (fast_log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    Nd1 = norm_cdf_approx(d1)
    Nd2 = norm_cdf_approx(d2)
    
    call_price = S * fast_exp(-q * T) * Nd1 - K * fast_exp(-r * T) * Nd2
    
    return call_price


@njit
def black_scholes_put_price_fast(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """JIT-optimized Black-Scholes put price calculation."""
    if T <= 0:
        return max(K - S, 0)
    
    sqrt_T = fast_sqrt(T)
    d1 = (fast_log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    Nd1_neg = norm_cdf_approx(-d1)
    Nd2_neg = norm_cdf_approx(-d2)
    
    put_price = K * fast_exp(-r * T) * Nd2_neg - S * fast_exp(-q * T) * Nd1_neg
    
    return put_price


@njit
def black_scholes_delta_fast(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool) -> float:
    """JIT-optimized Black-Scholes delta calculation."""
    if T <= 0:
        if is_call:
            return 1.0 if S > K else (0.5 if S == K else 0.0)
        else:
            return -1.0 if S < K else (-0.5 if S == K else 0.0)
    
    sqrt_T = fast_sqrt(T)
    d1 = (fast_log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    
    if is_call:
        return fast_exp(-q * T) * norm_cdf_approx(d1)
    else:
        return -fast_exp(-q * T) * norm_cdf_approx(-d1)


@njit
def black_scholes_gamma_fast(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """JIT-optimized Black-Scholes gamma calculation."""
    if T <= 0:
        return 0.0
    
    sqrt_T = fast_sqrt(T)
    d1 = (fast_log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    
    return fast_exp(-q * T) * norm_pdf_fast(d1) / (S * sigma * sqrt_T)


@njit
def black_scholes_vega_fast(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """JIT-optimized Black-Scholes vega calculation."""
    if T <= 0:
        return 0.0
    
    sqrt_T = fast_sqrt(T)
    d1 = (fast_log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    
    return S * fast_exp(-q * T) * norm_pdf_fast(d1) * sqrt_T


@njit
def monte_carlo_european_option_fast(S0: float, K: float, T: float, r: float, q: float, 
                                   sigma: float, n_paths: int, is_call: bool) -> Tuple[float, float]:
    """
    JIT-optimized Monte Carlo pricing for European options.
    
    Returns:
        Tuple of (price, standard_error)
    """
    dt = T
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol_term = sigma * np.sqrt(dt)
    
    payoffs = np.zeros(n_paths)
    
    for i in prange(n_paths):
        # Generate random number (note: numba doesn't support np.random.seed in parallel)
        z = np.random.standard_normal()
        S_T = S0 * np.exp(drift + vol_term * z)
        
        if is_call:
            payoffs[i] = max(S_T - K, 0)
        else:
            payoffs[i] = max(K - S_T, 0)
    
    discount_factor = np.exp(-r * T)
    discounted_payoffs = payoffs * discount_factor
    
    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(n_paths)
    
    return price, std_error


@njit
def vectorized_black_scholes_prices(S_array: np.ndarray, K_array: np.ndarray, T_array: np.ndarray,
                                  r_array: np.ndarray, q_array: np.ndarray, sigma_array: np.ndarray,
                                  is_call_array: np.ndarray) -> np.ndarray:
    """
    Vectorized Black-Scholes pricing for multiple options.
    
    All input arrays must have the same length.
    """
    n = len(S_array)
    prices = np.zeros(n)
    
    for i in prange(n):
        if is_call_array[i]:
            prices[i] = black_scholes_call_price_fast(
                S_array[i], K_array[i], T_array[i], 
                r_array[i], q_array[i], sigma_array[i]
            )
        else:
            prices[i] = black_scholes_put_price_fast(
                S_array[i], K_array[i], T_array[i],
                r_array[i], q_array[i], sigma_array[i]
            )
    
    return prices


class VectorizedPricer:
    """Vectorized option pricing for portfolios."""
    
    @staticmethod
    def price_portfolio(portfolio_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Price entire portfolio using vectorized operations.
        
        Args:
            portfolio_data: Dictionary with keys:
                'S0', 'K', 'T', 'r', 'q', 'sigma', 'is_call'
                All values should be numpy arrays of the same length.
        
        Returns:
            Array of option prices
        """
        return vectorized_black_scholes_prices(
            portfolio_data['S0'],
            portfolio_data['K'], 
            portfolio_data['T'],
            portfolio_data['r'],
            portfolio_data['q'],
            portfolio_data['sigma'],
            portfolio_data['is_call']
        )
    
    @staticmethod
    def calculate_portfolio_greeks(portfolio_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calculate Greeks for entire portfolio using vectorized operations.
        
        Returns:
            Dictionary with 'delta', 'gamma', 'vega' arrays
        """
        n = len(portfolio_data['S0'])
        
        deltas = np.zeros(n)
        gammas = np.zeros(n)
        vegas = np.zeros(n)
        
        for i in range(n):
            is_call = portfolio_data['is_call'][i]
            
            deltas[i] = black_scholes_delta_fast(
                portfolio_data['S0'][i], portfolio_data['K'][i], portfolio_data['T'][i],
                portfolio_data['r'][i], portfolio_data['q'][i], portfolio_data['sigma'][i],
                is_call
            )
            
            gammas[i] = black_scholes_gamma_fast(
                portfolio_data['S0'][i], portfolio_data['K'][i], portfolio_data['T'][i],
                portfolio_data['r'][i], portfolio_data['q'][i], portfolio_data['sigma'][i]
            )
            
            vegas[i] = black_scholes_vega_fast(
                portfolio_data['S0'][i], portfolio_data['K'][i], portfolio_data['T'][i],
                portfolio_data['r'][i], portfolio_data['q'][i], portfolio_data['sigma'][i]
            )
        
        return {
            'delta': deltas,
            'gamma': gammas,  
            'vega': vegas
        }


class MemoryOptimizedMonteCarlo:
    """Memory-efficient Monte Carlo implementation for large simulations."""
    
    @staticmethod
    def european_option_chunked(S0: float, K: float, T: float, r: float, q: float, sigma: float,
                              n_paths: int, is_call: bool, chunk_size: int = 100000) -> Tuple[float, float]:
        """
        Monte Carlo pricing with chunked processing to reduce memory usage.
        
        Args:
            chunk_size: Number of paths to process at once
        
        Returns:
            Tuple of (price, standard_error)
        """
        total_payoff = 0.0
        total_payoff_squared = 0.0
        
        drift = (r - q - 0.5 * sigma * sigma) * T
        vol_term = sigma * np.sqrt(T)
        discount_factor = np.exp(-r * T)
        
        n_chunks = (n_paths + chunk_size - 1) // chunk_size
        paths_processed = 0
        
        for chunk in range(n_chunks):
            current_chunk_size = min(chunk_size, n_paths - paths_processed)
            
            # Generate random numbers for this chunk
            z = np.random.standard_normal(current_chunk_size)
            S_T = S0 * np.exp(drift + vol_term * z)
            
            # Calculate payoffs
            if is_call:
                payoffs = np.maximum(S_T - K, 0)
            else:
                payoffs = np.maximum(K - S_T, 0)
            
            # Apply discount factor
            discounted_payoffs = payoffs * discount_factor
            
            # Accumulate statistics
            total_payoff += np.sum(discounted_payoffs)
            total_payoff_squared += np.sum(discounted_payoffs ** 2)
            
            paths_processed += current_chunk_size
        
        # Calculate final statistics
        mean_payoff = total_payoff / n_paths
        mean_payoff_squared = total_payoff_squared / n_paths
        variance = mean_payoff_squared - mean_payoff ** 2
        std_error = np.sqrt(variance / n_paths) if variance > 0 else 0.0
        
        return mean_payoff, std_error


def benchmark_optimizations():
    """
    Benchmark different optimization techniques.
    """
    # Test parameters
    S0, K, T, r, q, sigma = 100.0, 100.0, 0.25, 0.05, 0.02, 0.20
    n_iterations = 10000
    
    print("Benchmarking optimization techniques...")
    print(f"Test parameters: S0={S0}, K={K}, T={T}, r={r}, q={q}, sigma={sigma}")
    print(f"Iterations: {n_iterations}")
    print()
    
    # Test regular vs optimized Black-Scholes
    from ..models.black_scholes import BlackScholesModel
    from ..core.market_data import MarketData
    
    market_data = MarketData(S0, K, T, r, q, sigma)
    
    # Regular Black-Scholes
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        price = BlackScholesModel.price(market_data, "call")
    regular_time = time.perf_counter() - start_time
    
    # Optimized Black-Scholes
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        price_opt = black_scholes_call_price_fast(S0, K, T, r, q, sigma)
    optimized_time = time.perf_counter() - start_time
    
    speedup = regular_time / optimized_time
    
    print(f"Black-Scholes Pricing:")
    print(f"  Regular:    {regular_time:.4f}s ({n_iterations/regular_time:.0f} ops/sec)")
    print(f"  Optimized:  {optimized_time:.4f}s ({n_iterations/optimized_time:.0f} ops/sec)")
    print(f"  Speedup:    {speedup:.1f}x")
    print(f"  Price diff: {abs(price - price_opt):.2e}")
    print()
    
    # Test vectorized pricing
    n_options = 1000
    portfolio_data = {
        'S0': np.full(n_options, S0),
        'K': np.random.uniform(80, 120, n_options),
        'T': np.random.uniform(0.1, 1.0, n_options),
        'r': np.full(n_options, r),
        'q': np.full(n_options, q),
        'sigma': np.random.uniform(0.15, 0.35, n_options),
        'is_call': np.random.choice([True, False], n_options)
    }
    
    # Individual pricing
    start_time = time.perf_counter()
    individual_prices = []
    for i in range(n_options):
        if portfolio_data['is_call'][i]:
            price = black_scholes_call_price_fast(
                portfolio_data['S0'][i], portfolio_data['K'][i], portfolio_data['T'][i],
                portfolio_data['r'][i], portfolio_data['q'][i], portfolio_data['sigma'][i]
            )
        else:
            price = black_scholes_put_price_fast(
                portfolio_data['S0'][i], portfolio_data['K'][i], portfolio_data['T'][i],
                portfolio_data['r'][i], portfolio_data['q'][i], portfolio_data['sigma'][i]
            )
        individual_prices.append(price)
    individual_time = time.perf_counter() - start_time
    
    # Vectorized pricing
    pricer = VectorizedPricer()
    start_time = time.perf_counter()
    vectorized_prices = pricer.price_portfolio(portfolio_data)
    vectorized_time = time.perf_counter() - start_time
    
    vector_speedup = individual_time / vectorized_time
    price_diff = np.max(np.abs(np.array(individual_prices) - vectorized_prices))
    
    print(f"Portfolio Pricing ({n_options} options):")
    print(f"  Individual: {individual_time:.4f}s")
    print(f"  Vectorized: {vectorized_time:.4f}s")
    print(f"  Speedup:    {vector_speedup:.1f}x")
    print(f"  Max diff:   {price_diff:.2e}")
    print()
    
    # Test caching
    cache_test_func = cached_result()(lambda x: black_scholes_call_price_fast(x, K, T, r, q, sigma))
    
    # First run (no cache)
    test_values = np.random.uniform(80, 120, 1000)
    start_time = time.perf_counter()
    for val in test_values:
        cache_test_func(val)
    no_cache_time = time.perf_counter() - start_time
    
    # Second run (with cache)
    start_time = time.perf_counter()
    for val in test_values:
        cache_test_func(val)
    with_cache_time = time.perf_counter() - start_time
    
    cache_speedup = no_cache_time / with_cache_time
    
    print(f"Caching Test (1000 repeated calculations):")
    print(f"  No cache:   {no_cache_time:.4f}s")
    print(f"  With cache: {with_cache_time:.4f}s")
    print(f"  Speedup:    {cache_speedup:.1f}x")
    print()
    
    return {
        'black_scholes_speedup': speedup,
        'vectorization_speedup': vector_speedup,
        'caching_speedup': cache_speedup
    }


if __name__ == "__main__":
    results = benchmark_optimizations()
    print("Optimization benchmark completed!")
    print(f"Overall performance improvements:")
    for opt_type, speedup in results.items():
        print(f"  {opt_type.replace('_', ' ').title()}: {speedup:.1f}x")