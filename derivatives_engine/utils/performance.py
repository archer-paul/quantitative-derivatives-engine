"""
Performance monitoring and benchmarking tools.

This module provides utilities for measuring and analyzing the performance
of different pricing methods and models.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, asdict
import psutil
import sys

from ..core.market_data import MarketData


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    median_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    operations_per_second: float
    total_iterations: int
    memory_usage_mb: Optional[float] = None


class PerformanceBenchmark:
    """Performance benchmarking and monitoring utilities."""
    
    def __init__(self):
        """Initialize performance benchmark."""
        self.results_history: List[Dict[str, Any]] = []
        self.active_timers: Dict[str, float] = {}
    
    def benchmark_function(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        iterations: int = 1000,
        warmup_iterations: int = 100,
        name: str = "benchmark"
    ) -> PerformanceMetrics:
        """
        Benchmark a function's performance.
        
        Args:
            func: Function to benchmark
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            iterations: Number of timing iterations
            warmup_iterations: Number of warmup iterations (not timed)
            name: Name for the benchmark
            
        Returns:
            Performance metrics
        """
        if kwargs is None:
            kwargs = {}
        
        # Warmup phase
        for _ in range(warmup_iterations):
            func(*args, **kwargs)
        
        # Timing phase
        times = []
        initial_memory = self._get_memory_usage()
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        final_memory = self._get_memory_usage()
        memory_delta = final_memory - initial_memory if initial_memory and final_memory else None
        
        # Calculate statistics
        times_array = np.array(times)
        metrics = PerformanceMetrics(
            mean_time_ms=np.mean(times_array),
            std_time_ms=np.std(times_array),
            min_time_ms=np.min(times_array),
            max_time_ms=np.max(times_array),
            median_time_ms=np.median(times_array),
            p95_time_ms=np.percentile(times_array, 95),
            p99_time_ms=np.percentile(times_array, 99),
            operations_per_second=1000 / np.mean(times_array) if np.mean(times_array) > 0 else float('inf'),
            total_iterations=iterations,
            memory_usage_mb=memory_delta
        )
        
        # Store results
        self.results_history.append({
            'name': name,
            'timestamp': pd.Timestamp.now(),
            'metrics': metrics,
            'function_name': func.__name__ if hasattr(func, '__name__') else str(func)
        })
        
        return metrics
    
    def compare_functions(
        self,
        functions: Dict[str, Callable],
        args: tuple = (),
        kwargs: dict = None,
        iterations: int = 1000
    ) -> pd.DataFrame:
        """
        Compare performance of multiple functions.
        
        Args:
            functions: Dictionary mapping names to functions
            args: Common positional arguments
            kwargs: Common keyword arguments
            iterations: Number of iterations per function
            
        Returns:
            DataFrame with comparison results
        """
        if kwargs is None:
            kwargs = {}
        
        results = []
        for name, func in functions.items():
            metrics = self.benchmark_function(func, args, kwargs, iterations, name=name)
            result = asdict(metrics)
            result['function_name'] = name
            results.append(result)
        
        df = pd.DataFrame(results)
        df = df.sort_values('mean_time_ms')
        
        # Add relative performance columns
        baseline_time = df['mean_time_ms'].iloc[0]
        df['relative_speed'] = baseline_time / df['mean_time_ms']
        df['speed_ratio'] = df['mean_time_ms'] / baseline_time
        
        return df
    
    def profile_pricing_models(
        self,
        market_data: MarketData,
        option_type: str,
        iterations: int = 1000
    ) -> pd.DataFrame:
        """
        Profile different pricing model implementations.
        
        Args:
            market_data: Market conditions for testing
            option_type: Option type to price
            iterations: Number of iterations per model
            
        Returns:
            DataFrame with model performance comparison
        """
        from ..models.black_scholes import BlackScholesModel
        from ..models.finite_difference import FiniteDifferencePricer
        
        # Define functions to benchmark
        functions = {
            'Black-Scholes Price': lambda: BlackScholesModel.price(market_data, option_type),
            'Black-Scholes Greeks': lambda: BlackScholesModel.greeks(market_data, option_type),
            'Finite Difference': lambda: FiniteDifferencePricer.implicit_fd_vanilla(
                market_data, option_type, S_max=2*market_data.S0, M=50, N=100
            )[0]  # Only return the price
        }
        
        return self.compare_functions(functions, iterations=iterations)
    
    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self.active_timers[name] = time.perf_counter()
    
    def stop_timer(self, name: str) -> float:
        """
        Stop a named timer and return elapsed time.
        
        Args:
            name: Timer name
            
        Returns:
            Elapsed time in seconds
            
        Raises:
            KeyError: If timer was not started
        """
        if name not in self.active_timers:
            raise KeyError(f"Timer '{name}' was not started")
        
        elapsed = time.perf_counter() - self.active_timers[name]
        del self.active_timers[name]
        return elapsed
    
    def time_context(self, name: str):
        """
        Context manager for timing code blocks.
        
        Args:
            name: Name for the timing measurement
            
        Example:
            with benchmark.time_context("my_calculation"):
                result = expensive_calculation()
        """
        return TimingContext(self, name)
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except:
            return None
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for all benchmarks.
        
        Returns:
            DataFrame with summary statistics
        """
        if not self.results_history:
            return pd.DataFrame()
        
        summary_data = []
        for result in self.results_history:
            metrics = result['metrics']
            summary_data.append({
                'name': result['name'],
                'timestamp': result['timestamp'],
                'mean_time_ms': metrics.mean_time_ms,
                'ops_per_second': metrics.operations_per_second,
                'iterations': metrics.total_iterations,
                'memory_mb': metrics.memory_usage_mb
            })
        
        return pd.DataFrame(summary_data)
    
    def clear_history(self) -> None:
        """Clear all stored benchmark results."""
        self.results_history.clear()
        self.active_timers.clear()


class TimingContext:
    """Context manager for timing code execution."""
    
    def __init__(self, benchmark: PerformanceBenchmark, name: str):
        self.benchmark = benchmark
        self.name = name
        self.elapsed_time = None
    
    def __enter__(self):
        self.benchmark.start_timer(self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time = self.benchmark.stop_timer(self.name)
    
    def get_elapsed_time(self) -> Optional[float]:
        """Get elapsed time in seconds."""
        return self.elapsed_time


def memory_usage_decorator(func):
    """
    Decorator to monitor memory usage of a function.
    
    Args:
        func: Function to decorate
        
    Returns:
        Wrapped function that prints memory usage
    """
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_delta = mem_after - mem_before
        
        print(f"Function '{func.__name__}' memory usage: {mem_delta:.2f} MB")
        return result
    
    return wrapper


def time_function_decorator(iterations: int = 1):
    """
    Decorator to time function execution.
    
    Args:
        iterations: Number of times to run the function for averaging
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            times = []
            result = None
            
            for _ in range(iterations):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # ms
            
            avg_time = np.mean(times)
            print(f"Function '{func.__name__}' average execution time: {avg_time:.3f} ms")
            
            return result
        return wrapper
    return decorator