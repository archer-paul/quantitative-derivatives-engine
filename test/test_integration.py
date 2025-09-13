"""
Integration tests for the derivatives pricing engine.

This module contains end-to-end integration tests that verify the complete
workflows and interactions between different components of the system.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from derivatives_engine import (
    DerivativesPricingEngine,
    MarketData,
    HestonParameters,
    JumpDiffusionParameters,
    ExoticOptions,
    PerformanceBenchmark,
    RiskAnalyzer
)


class TestCompleteWorkflows:
    """Test complete end-to-end workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = DerivativesPricingEngine()
        self.market_data = MarketData(
            S0=100.0, K=100.0, T=0.25, r=0.05, q=0.02, sigma=0.20
        )
    
    def test_comprehensive_pricing_workflow(self):
        """Test complete pricing workflow across all models."""
        # Define advanced model parameters
        heston_params = HestonParameters(
            v0=0.04, theta=0.04, kappa=2.0, sigma_v=0.3, rho=-0.7
        )
        
        jump_params = JumpDiffusionParameters(
            lambda_j=0.1, mu_j=-0.05, sigma_j=0.15
        )
        
        # Run comprehensive pricing
        results = self.engine.comprehensive_pricing(
            self.market_data, "call", heston_params, jump_params
        )
        
        # Validate all components are present
        assert 'black_scholes' in results
        assert 'heston' in results
        assert 'jump_diffusion' in results
        assert 'finite_difference' in results
        assert 'pricing_times' in results
        assert 'market_data' in results
        assert 'option_type' in results
        
        # Validate Black-Scholes results
        bs_results = results['black_scholes']
        assert 'price' in bs_results
        assert 'greeks' in bs_results
        assert bs_results['price'] > 0
        assert len(bs_results['greeks']) == 5  # All Greeks present
        
        # Validate Heston results
        heston_results = results['heston']
        assert 'monte_carlo_price' in heston_results
        assert 'monte_carlo_ci' in heston_results
        assert 'parameters' in heston_results
        assert heston_results['monte_carlo_price'] > 0
        
        # Validate Jump-Diffusion results
        jd_results = results['jump_diffusion']
        assert 'analytical_price' in jd_results
        assert 'monte_carlo_price' in jd_results
        assert 'parameters' in jd_results
        assert jd_results['analytical_price'] > 0
        
        # Validate Finite Difference results
        fd_results = results['finite_difference']
        assert 'price' in fd_results
        assert fd_results['price'] > 0
        
        # Validate timing information
        timing = results['pricing_times']
        assert 'black_scholes' in timing
        assert 'finite_difference' in timing
        assert all(t > 0 for t in timing.values() if t is not None)
    
    def test_pricing_report_generation(self):
        """Test complete pricing report generation."""
        results = self.engine.comprehensive_pricing(self.market_data, "put")
        
        # Generate report
        report = self.engine.generate_report(results)
        
        assert isinstance(report, str)
        assert len(report) > 1000  # Should be substantial
        
        # Check key sections are present
        assert "ADVANCED DERIVATIVES PRICING ENGINE" in report
        assert "MARKET DATA:" in report
        assert "BLACK-SCHOLES MODEL:" in report
        assert "FINITE DIFFERENCE METHOD:" in report
        assert "MODEL COMPARISON:" in report
        assert "PERFORMANCE SUMMARY:" in report
        
        # Check numerical values are formatted correctly
        assert "$" in report  # Prices should be formatted as currency
        assert "%" in report  # Percentages should be present
        assert "ms" in report  # Timing should be in milliseconds
    
    def test_portfolio_pricing_workflow(self):
        """Test pricing multiple options as a portfolio."""
        # Create multiple market conditions
        options = [
            {'market_data': MarketData(100, 95, 0.25, 0.05, 0.02, 0.20), 'type': 'call'},
            {'market_data': MarketData(100, 100, 0.25, 0.05, 0.02, 0.20), 'type': 'call'},
            {'market_data': MarketData(100, 105, 0.25, 0.05, 0.02, 0.20), 'type': 'call'},
            {'market_data': MarketData(100, 100, 0.5, 0.05, 0.02, 0.20), 'type': 'put'},
        ]
        
        portfolio_results = []
        total_value = 0
        
        for option in options:
            result = self.engine.comprehensive_pricing(
                option['market_data'], option['type']
            )
            portfolio_results.append(result)
            total_value += result['black_scholes']['price']
        
        assert len(portfolio_results) == 4
        assert total_value > 0
        
        # Validate each option was priced successfully
        for result in portfolio_results:
            assert result['black_scholes']['price'] > 0
            assert 'greeks' in result['black_scholes']


class TestExoticOptionsIntegration:
    """Integration tests for exotic options with various models."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exotic_engine = ExoticOptions()
        self.market_data = MarketData(
            S0=100.0, K=100.0, T=0.25, r=0.05, q=0.02, sigma=0.20
        )
    
    def test_exotic_options_suite(self):
        """Test complete suite of exotic options."""
        # Barrier option
        barrier_price, barrier_ci = self.exotic_engine.barrier_option_mc(
            self.market_data, "call", "up_out", 120.0, n_paths=10000
        )
        
        # Asian option
        asian_price, asian_ci = self.exotic_engine.asian_option_mc(
            self.market_data, "call", "arithmetic", n_paths=10000
        )
        
        # Lookback option
        lookback_price, lookback_ci = self.exotic_engine.lookback_option_mc(
            self.market_data, "lookback_call", n_paths=10000
        )
        
        # American option
        american_price = self.exotic_engine.american_option_binomial(
            self.market_data, "put", n_steps=500
        )
        
        # Validate all prices
        assert barrier_price > 0
        assert asian_price > 0
        assert lookback_price > 0
        assert american_price > 0
        
        # Validate confidence intervals
        assert barrier_ci > 0
        assert asian_ci > 0
        assert lookback_ci > 0
        
        # Validate relative pricing relationships
        from derivatives_engine import BlackScholesModel
        vanilla_call = BlackScholesModel.price(self.market_data, "call")
        vanilla_put = BlackScholesModel.price(self.market_data, "put")
        
        # Barrier should be cheaper than vanilla
        assert barrier_price <= vanilla_call
        
        # Asian should be cheaper than vanilla (averaging effect)
        assert asian_price <= vanilla_call
        
        # Lookback should be more expensive than vanilla
        assert lookback_price >= vanilla_call
        
        # American put should be at least as expensive as European put
        assert american_price >= vanilla_put


class TestModelValidationWorkflow:
    """Test model validation and comparison workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = DerivativesPricingEngine()
        self.market_data = MarketData(
            S0=100.0, K=100.0, T=0.25, r=0.05, q=0.02, sigma=0.20
        )
    
    def test_model_validation_suite(self):
        """Test complete model validation workflow."""
        # Run validation on all models
        validation_results = self.engine.validate_models(self.market_data)
        
        assert isinstance(validation_results, dict)
        
        # Check expected models are validated
        expected_models = ['black_scholes', 'finite_difference', 'exotic_options']
        
        for model in expected_models:
            assert model in validation_results
            # Most models should pass validation with reasonable market data
            if model == 'black_scholes':  # This should definitely work
                assert validation_results[model] == True
    
    def test_performance_benchmarking_workflow(self):
        """Test performance benchmarking across models."""
        # Run performance benchmarks
        benchmark_results = self.engine.benchmark_performance(
            self.market_data, "call", n_iterations=500
        )
        
        assert isinstance(benchmark_results, dict)
        
        # Should have benchmarks for different methods
        expected_benchmarks = ['black_scholes_pricing', 'greeks_calculation', 'finite_difference']
        
        for benchmark in expected_benchmarks:
            if benchmark in benchmark_results:
                result = benchmark_results[benchmark]
                assert 'avg_time_ms' in result
                assert 'iterations' in result
                assert 'ops_per_second' in result
                assert result['avg_time_ms'] > 0
                assert result['ops_per_second'] > 0
    
    def test_convergence_analysis(self):
        """Test convergence analysis across different numerical methods."""
        # Test Monte Carlo convergence
        path_counts = [1000, 5000, 10000, 20000]
        mc_prices = []
        
        from derivatives_engine import ExoticOptions
        exotic_engine = ExoticOptions()
        
        for n_paths in path_counts:
            price, _ = exotic_engine.barrier_option_mc(
                self.market_data, "call", "up_out", 110.0, n_paths=n_paths
            )
            mc_prices.append(price)
        
        # Test Finite Difference convergence
        from derivatives_engine import FiniteDifferencePricer
        fd_pricer = FiniteDifferencePricer()
        
        grid_sizes = [(50, 250), (75, 500), (100, 750)]
        fd_prices = []
        
        for M, N in grid_sizes:
            price, _, _ = fd_pricer.implicit_fd_vanilla(
                self.market_data, "call", S_max=200, M=M, N=N
            )
            fd_prices.append(price)
        
        # Test Binomial convergence
        step_counts = [100, 250, 500, 1000]
        binomial_prices = []
        
        for n_steps in step_counts:
            price = exotic_engine.american_option_binomial(
                self.market_data, "call", n_steps=n_steps
            )
            binomial_prices.append(price)
        
        # Validate convergence (prices should stabilize)
        def check_convergence(prices, method_name):
            if len(prices) >= 2:
                final_change = abs(prices[-1] - prices[-2])
                relative_change = final_change / prices[-1] if prices[-1] > 0 else 0
                assert relative_change < 0.05, f"{method_name} not converging: {relative_change:.3f}"
        
        check_convergence(mc_prices, "Monte Carlo")
        check_convergence(fd_prices, "Finite Difference")
        check_convergence(binomial_prices, "Binomial")


class TestDataIntegrationWorkflow:
    """Test data loading and integration workflows."""
    
    def test_market_data_workflow(self):
        """Test complete market data loading and pricing workflow."""
        from derivatives_engine.data import YahooFinanceLoader
        
        # Create data loader
        loader = YahooFinanceLoader()
        
        # Load stock price (using mock data)
        stock_price = loader.load_stock_price("AAPL")
        assert stock_price > 0
        
        # Create market data from loaded price
        market_data = MarketData(
            S0=stock_price, K=stock_price, T=0.25, r=0.05, q=0.02, sigma=0.20
        )
        
        # Price options with loaded data
        from derivatives_engine import BlackScholesModel
        call_price = BlackScholesModel.price(market_data, "call")
        put_price = BlackScholesModel.price(market_data, "put")
        
        assert call_price > 0
        assert put_price > 0
        
        # Validate put-call parity
        pcp_left = call_price - put_price
        pcp_right = (market_data.S0 * np.exp(-market_data.q * market_data.T) - 
                    market_data.K * np.exp(-market_data.r * market_data.T))
        
        assert abs(pcp_left - pcp_right) < 1e-10
    
    def test_option_chain_integration(self):
        """Test option chain loading and analysis workflow."""
        from derivatives_engine.data import YahooFinanceLoader
        from derivatives_engine.calibration import ImpliedVolatilityCalibrator
        
        # Load option chain (mock data)
        loader = YahooFinanceLoader()
        expiry = datetime.now() + timedelta(days=30)
        
        chain_data = loader.load_option_chain("AAPL", expiry)
        
        assert chain_data.underlying_symbol == "AAPL"
        assert chain_data.underlying_price > 0
        assert len(chain_data.options) > 0
        
        # Convert to calibration format
        market_data_list = []
        for option in chain_data.options[:10]:  # Use first 10 options
            md = MarketData(
                S0=chain_data.underlying_price,
                K=option['strike'],
                T=(chain_data.expiry_date - datetime.now()).days / 365.0,
                r=chain_data.risk_free_rate,
                q=chain_data.dividend_yield,
                sigma=0.20  # Initial guess
            )
            
            market_data_list.append({
                'market_data': md,
                'option_type': option['option_type'],
                'market_price': option['price']
            })
        
        # Calibrate implied volatilities
        iv_calibrator = ImpliedVolatilityCalibrator()
        vol_surface = iv_calibrator.calibrate_vol_surface(market_data_list)
        
        assert isinstance(vol_surface, pd.DataFrame)
        assert len(vol_surface) == len(market_data_list)
        assert 'implied_vol' in vol_surface.columns
        
        # Analyze volatility smile
        smile_analysis = iv_calibrator.analyze_vol_smile(
            vol_surface, market_data_list[0]['market_data'].T
        )
        
        assert 'atm_vol' in smile_analysis
        assert 'vol_range' in smile_analysis


class TestRiskManagementIntegration:
    """Test risk management workflow integration."""
    
    def test_portfolio_risk_analysis(self):
        """Test complete portfolio risk analysis workflow."""
        # Create sample portfolio
        portfolio_positions = [
            {
                'market_data': MarketData(100, 95, 0.25, 0.05, 0.02, 0.20),
                'option_type': 'call',
                'quantity': 100
            },
            {
                'market_data': MarketData(100, 105, 0.25, 0.05, 0.02, 0.20),
                'option_type': 'put',
                'quantity': 50
            },
            {
                'market_data': MarketData(100, 100, 0.5, 0.05, 0.02, 0.25),
                'option_type': 'call',
                'quantity': 75
            }
        ]
        
        # Initialize risk analyzer
        try:
            from derivatives_engine import RiskAnalyzer
            risk_analyzer = RiskAnalyzer()
            
            # Calculate VaR
            var_result = risk_analyzer.calculate_portfolio_var(
                portfolio_positions, method="delta_normal"
            )
            
            assert var_result.success == True
            assert var_result.var_95 > 0
            assert var_result.var_99 > var_result.var_95
            assert var_result.expected_shortfall_95 > var_result.var_95
            
            # Stress testing
            stress_results = risk_analyzer.stress_test_portfolio(portfolio_positions)
            
            assert len(stress_results) > 0
            
            for stress_result in stress_results:
                assert hasattr(stress_result, 'scenario_name')
                assert hasattr(stress_result, 'pnl')
                assert hasattr(stress_result, 'base_value')
                assert hasattr(stress_result, 'stressed_value')
                
        except ImportError:
            pytest.skip("Risk analysis module not available")


class TestEndToEndScenarios:
    """Test realistic end-to-end usage scenarios."""
    
    def test_trading_desk_scenario(self):
        """Test typical trading desk workflow."""
        # 1. Load market data
        engine = DerivativesPricingEngine()
        
        # 2. Price various options
        options_to_price = [
            MarketData(100, 95, 0.25, 0.05, 0.02, 0.20),   # ITM Call
            MarketData(100, 100, 0.25, 0.05, 0.02, 0.20),  # ATM Call
            MarketData(100, 105, 0.25, 0.05, 0.02, 0.20),  # OTM Call
            MarketData(100, 100, 0.08, 0.05, 0.02, 0.20),  # Short-term ATM
            MarketData(100, 100, 1.0, 0.05, 0.02, 0.20),   # Long-term ATM
        ]
        
        pricing_results = []
        
        for market_data in options_to_price:
            # Price with multiple models for comparison
            result = engine.comprehensive_pricing(market_data, "call")
            pricing_results.append(result)
            
            # Generate report for each option
            report = engine.generate_report(result)
            assert len(report) > 500  # Substantial report
        
        # 3. Performance analysis
        performance_summary = engine.get_performance_summary()
        assert isinstance(performance_summary, dict)
        
        # 4. Validate all pricing succeeded
        for result in pricing_results:
            assert result['black_scholes']['price'] > 0
            assert 'greeks' in result['black_scholes']
            
        print(f"Successfully priced {len(pricing_results)} options")
    
    def test_risk_manager_scenario(self):
        """Test typical risk management workflow."""
        try:
            from derivatives_engine import RiskAnalyzer
            
            # Create complex portfolio
            portfolio = []
            np.random.seed(42)
            
            for i in range(20):
                strike = 90 + i * 2  # Strikes from 90 to 128
                expiry = 0.1 + (i / 20) * 0.9  # Expiries from 1 month to 1 year
                vol = 0.15 + np.random.random() * 0.20  # Vols 15-35%
                option_type = 'call' if i % 2 == 0 else 'put'
                quantity = np.random.randint(-200, 200)  # Long/short positions
                
                portfolio.append({
                    'market_data': MarketData(100, strike, expiry, 0.05, 0.02, vol),
                    'option_type': option_type,
                    'quantity': quantity
                })
            
            risk_analyzer = RiskAnalyzer()
            
            # Comprehensive risk analysis
            var_result = risk_analyzer.calculate_portfolio_var(
                portfolio, method="monte_carlo"
            )
            
            stress_results = risk_analyzer.stress_test_portfolio(portfolio)
            
            # Validate risk analysis
            assert var_result.success
            assert var_result.var_95 > 0
            assert len(stress_results) > 5  # Multiple stress scenarios
            
            # Find worst stress scenario
            worst_scenario = min(stress_results, key=lambda x: x.pnl)
            assert hasattr(worst_scenario, 'scenario_name')
            
            print(f"Portfolio VaR (95%): ${var_result.var_95:,.2f}")
            print(f"Worst stress scenario: {worst_scenario.scenario_name} "
                  f"(${worst_scenario.pnl:,.2f})")
            
        except ImportError:
            pytest.skip("Risk analysis module not available")
    
    def test_quant_researcher_scenario(self):
        """Test quantitative research workflow."""
        # Model comparison study
        market_data = MarketData(100, 100, 0.25, 0.05, 0.02, 0.20)
        
        # Black-Scholes baseline
        from derivatives_engine import BlackScholesModel
        bs_price = BlackScholesModel.price(market_data, "call")
        bs_greeks = BlackScholesModel.greeks(market_data, "call")
        
        # Heston model
        from derivatives_engine import HestonModel, HestonParameters
        heston_params = HestonParameters(0.04, 0.04, 2.0, 0.3, -0.7)
        heston_model = HestonModel(heston_params)
        heston_price, heston_ci = heston_model.monte_carlo_price(
            market_data, "call", n_paths=50000
        )
        
        # American vs European comparison
        from derivatives_engine import ExoticOptions
        exotic_engine = ExoticOptions()
        american_price = exotic_engine.american_option_binomial(
            market_data, "call", n_steps=1000
        )
        
        # Binomial model for Greeks
        from derivatives_engine.models.binomial import BinomialModel, BinomialGreeks
        binomial_price = BinomialModel.price_american(
            market_data, "call", n_steps=1000, early_exercise=False  # European
        )
        binomial_greeks = BinomialGreeks.calculate_greeks(
            market_data, "call", n_steps=500
        )
        
        # Comparative analysis
        model_comparison = {
            'Black-Scholes': {'price': bs_price, 'greeks': bs_greeks},
            'Heston': {'price': heston_price, 'ci': heston_ci},
            'American': {'price': american_price},
            'Binomial': {'price': binomial_price, 'greeks': binomial_greeks}
        }
        
        # Validate all models produced reasonable results
        for model_name, results in model_comparison.items():
            assert results['price'] > 0, f"{model_name} failed"
            assert results['price'] < 50, f"{model_name} price unreasonable"
        
        # Check American >= European
        assert american_price >= bs_price * 0.99  # Allow small numerical differences
        
        # Check Greeks are similar between methods
        bs_delta = bs_greeks['delta']
        bin_delta = binomial_greeks['delta']
        delta_diff = abs(bs_delta - bin_delta)
        assert delta_diff < 0.05, f"Delta difference too large: {delta_diff}"
        
        print("Model comparison completed successfully:")
        for model_name, results in model_comparison.items():
            price = results['price']
            print(f"  {model_name:12s}: ${price:7.4f}")


if __name__ == "__main__":
    # Run a quick integration test
    test = TestCompleteWorkflows()
    test.setup_method()
    test.test_comprehensive_pricing_workflow()
    print("Basic integration test passed!")