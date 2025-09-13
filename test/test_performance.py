"""
Performance tests for the derivatives pricing engine.

This module contains performance benchmarks and tests for various
pricing methods and optimization checks.
"""

import pytest
import time
import numpy as np
from derivatives_engine import (
    DerivativesPricingEngine,
    MarketData,
    BlackScholesModel,
    PerformanceBenchmark
)


class TestPricingPerformance:
    """Performance tests for pricing methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = DerivativesPricingEngine()
        self.market_data = MarketData(
            S0=100.0, K=100.0, T=0.25, r=0.05, q=0.02, sigma=0.20
        )
        self.benchmark = PerformanceBenchmark()
    
    def test_black_scholes_pricing_speed(self):
        """Test Black-Scholes pricing performance."""
        n_iterations = 1000
        
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            BlackScholesModel.price(self.market_data, "call")
        end_time = time.perf_counter()
        
        avg_time_ms = (end_time - start_time) / n_iterations * 1000
        
        # Should price at least 1000 options per second (< 1ms each)
        assert avg_time_ms < 1.0, f"Black-Scholes pricing too slow: {avg_time_ms:.3f} ms"
        
        print(f"Black-Scholes pricing: {avg_time_ms:.3f} ms per option")
    
    def test_greeks_calculation_speed(self):
        """Test Greeks calculation performance."""
        n_iterations = 500
        
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            BlackScholesModel.greeks(self.market_data, "call")
        end_time = time.perf_counter()
        
        avg_time_ms = (end_time - start_time) / n_iterations * 1000
        
        # Greeks should be calculated reasonably fast (< 5ms each)
        assert avg_time_ms < 5.0, f"Greeks calculation too slow: {avg_time_ms:.3f} ms"
        
        print(f"Greeks calculation: {avg_time_ms:.3f} ms per option")
    
    def test_monte_carlo_performance(self):
        """Test Monte Carlo performance for exotic options."""
        from derivatives_engine import ExoticOptions
        
        exotic_engine = ExoticOptions()
        n_paths = 10000
        
        start_time = time.perf_counter()
        price, _ = exotic_engine.barrier_option_mc(
            self.market_data, "call", "up_out", 110.0,
            n_paths=n_paths, n_steps=100
        )
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        paths_per_second = n_paths / execution_time
        
        # Should handle at least 1000 paths per second
        assert paths_per_second > 1000, f"Monte Carlo too slow: {paths_per_second:.0f} paths/sec"
        
        print(f"Monte Carlo performance: {paths_per_second:.0f} paths per second")
    
    def test_performance_benchmark_tool(self):
        """Test the performance benchmark utility."""
        # Benchmark Black-Scholes pricing
        metrics = self.benchmark.benchmark_function(
            lambda: BlackScholesModel.price(self.market_data, "call"),
            iterations=1000,
            warmup_iterations=100,
            name="black_scholes_price"
        )
        
        assert metrics.total_iterations == 1000
        assert metrics.mean_time_ms > 0
        assert metrics.operations_per_second > 0
        assert metrics.min_time_ms <= metrics.mean_time_ms <= metrics.max_time_ms
        
        print(f"Benchmark results: {metrics.mean_time_ms:.3f} ms avg, "
              f"{metrics.operations_per_second:.0f} ops/sec")
    
    def test_model_comparison_performance(self):
        """Test comparative performance of different models."""
        functions = {
            'Black-Scholes': lambda: BlackScholesModel.price(self.market_data, "call"),
            'Black-Scholes Greeks': lambda: BlackScholesModel.greeks(self.market_data, "call"),
        }
        
        comparison_df = self.benchmark.compare_functions(
            functions, iterations=500
        )
        
        assert len(comparison_df) == 2
        assert 'mean_time_ms' in comparison_df.columns
        assert 'operations_per_second' in comparison_df.columns
        assert 'relative_speed' in comparison_df.columns
        
        # Print results for visibility
        print("\nModel Performance Comparison:")
        print(comparison_df[['function_name', 'mean_time_ms', 'operations_per_second']])
    
    def test_finite_difference_performance(self):
        """Test finite difference method performance."""
        from derivatives_engine import FiniteDifferencePricer
        
        fd_pricer = FiniteDifferencePricer()
        
        start_time = time.perf_counter()
        price, _, _ = fd_pricer.implicit_fd_vanilla(
            self.market_data, "call", S_max=200, M=100, N=1000
        )
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (< 5 seconds for this grid)
        assert execution_time < 5.0, f"Finite difference too slow: {execution_time:.2f} seconds"
        
        print(f"Finite difference (100x1000 grid): {execution_time:.3f} seconds")
    
    def test_comprehensive_pricing_performance(self):
        """Test performance of comprehensive pricing across multiple models."""
        start_time = time.perf_counter()
        
        # Run comprehensive pricing (excluding slow models for performance test)
        results = self.engine.comprehensive_pricing(
            self.market_data, "call"
            # Don't include Heston/Jump-diffusion for speed
        )
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time
        assert execution_time < 2.0, f"Comprehensive pricing too slow: {execution_time:.2f} seconds"
        
        print(f"Comprehensive pricing: {execution_time:.3f} seconds")
        
        # Check that timing information is recorded
        assert 'pricing_times' in results
        assert 'black_scholes' in results['pricing_times']


class TestMemoryUsage:
    """Memory usage tests."""
    
    def test_memory_efficiency(self):
        """Test memory usage doesn't grow excessively."""
        try:
            import psutil
            process = psutil.Process()
            
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Perform many pricing operations
            market_data = MarketData(100.0, 100.0, 0.25, 0.05, 0.02, 0.20)
            
            for _ in range(1000):
                BlackScholesModel.price(market_data, "call")
                BlackScholesModel.greeks(market_data, "call")
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be minimal (< 50MB for this test)
            assert memory_increase < 50, f"Excessive memory usage: {memory_increase:.1f} MB increase"
            
            print(f"Memory usage: {memory_increase:.1f} MB increase over 1000 operations")
            
        except ImportError:
            pytest.skip("psutil not available for memory testing")
    
    def test_large_monte_carlo_memory(self):
        """Test memory usage for large Monte Carlo simulations."""
        try:
            import psutil
            process = psutil.Process()
            
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            from derivatives_engine import ExoticOptions
            exotic_engine = ExoticOptions()
            market_data = MarketData(100.0, 100.0, 0.25, 0.05, 0.02, 0.20)
            
            # Large Monte Carlo simulation
            price, _ = exotic_engine.barrier_option_mc(
                market_data, "call", "up_out", 110.0,
                n_paths=100000, n_steps=252
            )
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Should handle large simulations without excessive memory
            assert memory_increase < 500, f"Excessive memory for large MC: {memory_increase:.1f} MB"
            
            print(f"Large Monte Carlo memory: {memory_increase:.1f} MB for 100k paths")
            
        except ImportError:
            pytest.skip("psutil not available for memory testing")


class TestScalability:
    """Test scalability with increasing problem sizes."""
    
    def test_monte_carlo_scaling(self):
        """Test Monte Carlo scaling with number of paths."""
        from derivatives_engine import ExoticOptions
        
        exotic_engine = ExoticOptions()
        market_data = MarketData(100.0, 100.0, 0.25, 0.05, 0.02, 0.20)
        
        path_counts = [1000, 5000, 10000, 20000]
        times = []
        
        for n_paths in path_counts:
            start_time = time.perf_counter()
            price, _ = exotic_engine.barrier_option_mc(
                market_data, "call", "up_out", 110.0,
                n_paths=n_paths, n_steps=50
            )
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            paths_per_second = n_paths / (end_time - start_time)
            
            print(f"{n_paths:5d} paths: {end_time - start_time:.3f}s "
                  f"({paths_per_second:.0f} paths/sec)")
        
        # Scaling should be roughly linear
        # Check that doubling paths doesn't more than triple time
        for i in range(1, len(times)):
            path_ratio = path_counts[i] / path_counts[i-1]
            time_ratio = times[i] / times[i-1]
            
            assert time_ratio < path_ratio * 1.5, f"Poor scaling at {path_counts[i]} paths"
    
    def test_finite_difference_grid_scaling(self):
        """Test finite difference scaling with grid size."""
        from derivatives_engine import FiniteDifferencePricer
        
        fd_pricer = FiniteDifferencePricer()
        market_data = MarketData(100.0, 100.0, 0.25, 0.05, 0.02, 0.20)
        
        grid_sizes = [(50, 250), (75, 500), (100, 750)]
        times = []
        
        for M, N in grid_sizes:
            start_time = time.perf_counter()
            price, _, _ = fd_pricer.implicit_fd_vanilla(
                market_data, "call", S_max=200, M=M, N=N
            )
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            grid_points = M * N
            
            print(f"{M:3d}x{N:3d} grid ({grid_points:6d} points): {times[-1]:.3f}s")
        
        # Time should scale reasonably with grid size
        for i in range(1, len(times)):
            prev_points = grid_sizes[i-1][0] * grid_sizes[i-1][1]
            curr_points = grid_sizes[i][0] * grid_sizes[i][1]
            
            point_ratio = curr_points / prev_points
            time_ratio = times[i] / times[i-1]
            
            # Should not scale worse than quadratic
            assert time_ratio < point_ratio * 2, f"Poor FD scaling at grid {grid_sizes[i]}"


class TestNumericalStability:
    """Test numerical stability under various conditions."""
    
    def test_extreme_parameters(self):
        """Test pricing with extreme but valid parameters."""
        # Very high volatility
        high_vol_data = MarketData(100.0, 100.0, 0.25, 0.05, 0.02, 2.0)  # 200% vol
        
        start_time = time.perf_counter()
        price = BlackScholesModel.price(high_vol_data, "call")
        end_time = time.perf_counter()
        
        assert not np.isnan(price)
        assert not np.isinf(price)
        assert price > 0
        assert end_time - start_time < 0.1  # Should still be fast
        
        # Very low volatility
        low_vol_data = MarketData(100.0, 100.0, 0.25, 0.05, 0.02, 0.001)  # 0.1% vol
        price = BlackScholesModel.price(low_vol_data, "call")
        assert not np.isnan(price)
        assert not np.isinf(price)
        
        # Very short expiry
        short_exp_data = MarketData(100.0, 100.0, 0.001, 0.05, 0.02, 0.20)  # ~8 hours
        price = BlackScholesModel.price(short_exp_data, "call")
        assert not np.isnan(price)
        assert not np.isinf(price)
    
    def test_greeks_stability(self):
        """Test Greeks calculation stability."""
        # Test with various market conditions
        test_cases = [
            MarketData(50.0, 100.0, 0.1, 0.01, 0.0, 0.15),   # Deep OTM
            MarketData(150.0, 100.0, 0.1, 0.10, 0.05, 0.40),  # Deep ITM, high rates
            MarketData(100.0, 100.0, 2.0, 0.02, 0.01, 0.60),  # Long expiry, high vol
        ]
        
        for market_data in test_cases:
            greeks = BlackScholesModel.greeks(market_data, "call")
            
            # Check all Greeks are finite
            for greek_name, greek_value in greeks.items():
                assert not np.isnan(greek_value), f"{greek_name} is NaN"
                assert not np.isinf(greek_value), f"{greek_name} is infinite"
    
    def test_monte_carlo_convergence_stability(self):
        """Test Monte Carlo convergence is stable."""
        from derivatives_engine import ExoticOptions
        
        exotic_engine = ExoticOptions()
        market_data = MarketData(100.0, 100.0, 0.25, 0.05, 0.02, 0.20)
        
        # Run same pricing multiple times with different seeds
        prices = []
        for seed in range(5):
            np.random.seed(seed)
            price, _ = exotic_engine.barrier_option_mc(
                market_data, "call", "up_out", 110.0,
                n_paths=20000, n_steps=100
            )
            prices.append(price)
        
        # Standard deviation should be reasonable (< 5% of mean)
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        cv = std_price / mean_price
        
        assert cv < 0.05, f"Monte Carlo not stable: CV = {cv:.3f}"
        
        print(f"Monte Carlo stability: mean={mean_price:.4f}, std={std_price:.4f}, CV={cv:.3f}")


@pytest.mark.slow
class TestLongRunningPerformance:
    """Long-running performance tests (marked as slow)."""
    
    def test_extended_monte_carlo(self):
        """Test very large Monte Carlo simulation."""
        from derivatives_engine import ExoticOptions
        
        exotic_engine = ExoticOptions()
        market_data = MarketData(100.0, 100.0, 1.0, 0.05, 0.02, 0.20)
        
        start_time = time.perf_counter()
        price, ci = exotic_engine.barrier_option_mc(
            market_data, "call", "up_out", 120.0,
            n_paths=500000, n_steps=252  # 500k paths, daily steps
        )
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        assert price > 0
        assert ci > 0
        assert execution_time < 60.0, f"Extended MC too slow: {execution_time:.1f} seconds"
        
        print(f"Extended Monte Carlo (500k paths): {execution_time:.1f} seconds")
    
    def test_high_precision_finite_difference(self):
        """Test high-precision finite difference grid."""
        from derivatives_engine import FiniteDifferencePricer
        
        fd_pricer = FiniteDifferencePricer()
        market_data = MarketData(100.0, 100.0, 0.25, 0.05, 0.02, 0.20)
        
        start_time = time.perf_counter()
        price, _, _ = fd_pricer.implicit_fd_vanilla(
            market_data, "call", S_max=300, M=200, N=2000  # High resolution
        )
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        assert price > 0
        assert execution_time < 30.0, f"High precision FD too slow: {execution_time:.1f} seconds"
        
        print(f"High precision FD (200x2000): {execution_time:.1f} seconds")