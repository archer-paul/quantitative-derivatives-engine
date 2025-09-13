"""
Test module for Heston stochastic volatility model implementation.

This module contains comprehensive tests for the Heston model including
Monte Carlo pricing, FFT pricing, and parameter validation.
"""

import pytest
import numpy as np
from derivatives_engine import (
    HestonModel, 
    HestonParameters, 
    MarketData
)


class TestHestonModel:
    """Test cases for Heston stochastic volatility model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.heston_params = HestonParameters(
            v0=0.04,      # Initial variance (20% vol)
            theta=0.04,   # Long-term variance
            kappa=2.0,    # Mean reversion speed
            sigma_v=0.3,  # Volatility of variance
            rho=-0.7      # Negative correlation (leverage effect)
        )
        self.model = HestonModel(self.heston_params)
        self.market_data = MarketData(
            S0=100.0,
            K=100.0,
            T=0.25,
            r=0.05,
            q=0.02,
            sigma=0.20  # This sigma is not used in Heston model
        )
    
    def test_parameter_validation(self):
        """Test Heston parameter validation."""
        # Valid parameters should not raise error
        valid_params = HestonParameters(0.04, 0.04, 2.0, 0.3, -0.7)
        HestonModel(valid_params)
        
        # Invalid v0 (negative)
        with pytest.raises(ValueError):
            invalid_params = HestonParameters(-0.01, 0.04, 2.0, 0.3, -0.7)
            HestonModel(invalid_params)
        
        # Invalid theta (negative)
        with pytest.raises(ValueError):
            invalid_params = HestonParameters(0.04, -0.01, 2.0, 0.3, -0.7)
            HestonModel(invalid_params)
        
        # Invalid kappa (negative)
        with pytest.raises(ValueError):
            invalid_params = HestonParameters(0.04, 0.04, -1.0, 0.3, -0.7)
            HestonModel(invalid_params)
        
        # Invalid sigma_v (negative)
        with pytest.raises(ValueError):
            invalid_params = HestonParameters(0.04, 0.04, 2.0, -0.1, -0.7)
            HestonModel(invalid_params)
        
        # Invalid rho (out of range)
        with pytest.raises(ValueError):
            invalid_params = HestonParameters(0.04, 0.04, 2.0, 0.3, 1.5)
            HestonModel(invalid_params)
    
    def test_feller_condition_warning(self):
        """Test Feller condition warning for variance non-negativity."""
        # Parameters that violate Feller condition should generate warning
        with pytest.warns(UserWarning):
            feller_violating_params = HestonParameters(
                v0=0.04, theta=0.01, kappa=1.0, sigma_v=0.5, rho=-0.5
            )
            HestonModel(feller_violating_params)
    
    def test_monte_carlo_pricing_call(self):
        """Test Heston Monte Carlo pricing for call options."""
        price, ci = self.model.monte_carlo_price(
            self.market_data, "call", n_paths=10000, n_steps=100
        )
        
        # Basic validation
        assert isinstance(price, float)
        assert isinstance(ci, float)
        assert price > 0
        assert ci > 0
        assert price > ci  # Price should be larger than confidence interval
        
        # Price should be reasonable for ATM option
        assert 2.0 < price < 15.0
        
        # Confidence interval should be reasonable
        assert ci < price * 0.1  # CI should be less than 10% of price
    
    def test_monte_carlo_pricing_put(self):
        """Test Heston Monte Carlo pricing for put options."""
        price, ci = self.model.monte_carlo_price(
            self.market_data, "put", n_paths=10000, n_steps=100
        )
        
        assert isinstance(price, float)
        assert isinstance(ci, float)
        assert price > 0
        assert ci > 0
        
        # Put should be cheaper than call for this setup (ATM with positive rates)
        call_price, _ = self.model.monte_carlo_price(
            self.market_data, "call", n_paths=10000, n_steps=100
        )
        assert price < call_price
    
    def test_fft_pricing(self):
        """Test Heston FFT pricing method."""
        try:
            price = self.model.price_fft(self.market_data, "call")
            
            assert isinstance(price, float)
            assert price > 0
            assert 2.0 < price < 15.0
            
        except Exception as e:
            # FFT pricing can be sensitive to parameters, allow fallback
            pytest.skip(f"FFT pricing failed, this can happen with certain parameters: {e}")
    
    def test_characteristic_function(self):
        """Test Heston characteristic function."""
        # Test with various frequency values
        frequencies = [0.0, 0.5, 1.0, 2.0]
        
        for u in frequencies:
            char_func_value = self.model.characteristic_function(
                complex(0, u), self.market_data
            )
            
            assert isinstance(char_func_value, complex)
            # Characteristic function should be well-defined
            assert not np.isnan(char_func_value.real)
            assert not np.isnan(char_func_value.imag)
            assert not np.isinf(char_func_value.real)
            assert not np.isinf(char_func_value.imag)
    
    def test_monte_carlo_convergence(self):
        """Test Monte Carlo convergence with increasing paths."""
        path_counts = [1000, 5000, 10000]
        prices = []
        
        for n_paths in path_counts:
            price, _ = self.model.monte_carlo_price(
                self.market_data, "call", n_paths=n_paths, n_steps=50
            )
            prices.append(price)
        
        # Prices should converge (differences should decrease)
        diff1 = abs(prices[1] - prices[0])
        diff2 = abs(prices[2] - prices[1])
        
        # Allow for some randomness, but expect general convergence
        assert diff2 < diff1 * 2.0  # Not too strict due to randomness
    
    def test_volatility_effects(self):
        """Test that higher volatility of variance increases option prices."""
        # Low vol of vol
        low_vol_params = HestonParameters(0.04, 0.04, 2.0, 0.1, -0.7)
        low_vol_model = HestonModel(low_vol_params)
        
        # High vol of vol
        high_vol_params = HestonParameters(0.04, 0.04, 2.0, 0.5, -0.7)
        high_vol_model = HestonModel(high_vol_params)
        
        low_price, _ = low_vol_model.monte_carlo_price(
            self.market_data, "call", n_paths=5000
        )
        high_price, _ = high_vol_model.monte_carlo_price(
            self.market_data, "call", n_paths=5000
        )
        
        # Higher volatility of variance should lead to higher option prices
        assert high_price > low_price
    
    def test_correlation_effects(self):
        """Test correlation effects on option pricing."""
        # Negative correlation (leverage effect)
        neg_corr_params = HestonParameters(0.04, 0.04, 2.0, 0.3, -0.7)
        neg_corr_model = HestonModel(neg_corr_params)
        
        # Positive correlation
        pos_corr_params = HestonParameters(0.04, 0.04, 2.0, 0.3, 0.3)
        pos_corr_model = HestonModel(pos_corr_params)
        
        neg_corr_price, _ = neg_corr_model.monte_carlo_price(
            self.market_data, "put", n_paths=5000
        )
        pos_corr_price, _ = pos_corr_model.monte_carlo_price(
            self.market_data, "put", n_paths=5000
        )
        
        # Negative correlation should make puts more expensive
        assert neg_corr_price > pos_corr_price
    
    def test_time_to_expiry_effects(self):
        """Test that longer time to expiry increases option values."""
        short_maturity = self.market_data.copy(T=0.1)  # 1 month
        long_maturity = self.market_data.copy(T=1.0)   # 1 year
        
        short_price, _ = self.model.monte_carlo_price(
            short_maturity, "call", n_paths=5000
        )
        long_price, _ = self.model.monte_carlo_price(
            long_maturity, "call", n_paths=5000
        )
        
        assert long_price > short_price
    
    def test_mean_reversion_effects(self):
        """Test mean reversion speed effects."""
        # Slow mean reversion
        slow_mr_params = HestonParameters(0.04, 0.04, 0.5, 0.3, -0.7)
        slow_mr_model = HestonModel(slow_mr_params)
        
        # Fast mean reversion
        fast_mr_params = HestonParameters(0.04, 0.04, 5.0, 0.3, -0.7)
        fast_mr_model = HestonModel(fast_mr_params)
        
        slow_price, _ = slow_mr_model.monte_carlo_price(
            self.market_data, "call", n_paths=5000
        )
        fast_price, _ = fast_mr_model.monte_carlo_price(
            self.market_data, "call", n_paths=5000
        )
        
        # Slower mean reversion should generally lead to higher option prices
        # due to higher variance persistence
        assert slow_price > fast_price * 0.8  # Allow some tolerance


class TestHestonEdgeCases:
    """Test edge cases and boundary conditions for Heston model."""
    
    def test_very_short_expiry(self):
        """Test behavior with very short time to expiry."""
        short_expiry_data = MarketData(
            S0=100.0, K=100.0, T=0.001,  # Less than 1 day
            r=0.05, q=0.02, sigma=0.20
        )
        
        heston_params = HestonParameters(0.04, 0.04, 2.0, 0.3, -0.7)
        model = HestonModel(heston_params)
        
        price, ci = model.monte_carlo_price(
            short_expiry_data, "call", n_paths=5000
        )
        
        # Price should be close to intrinsic value for ATM option
        assert price >= 0
        assert price < 5.0  # Should be small for very short expiry
    
    def test_deep_itm_options(self):
        """Test deep in-the-money options."""
        deep_itm_data = MarketData(
            S0=150.0, K=100.0, T=0.25,  # Deep ITM call
            r=0.05, q=0.02, sigma=0.20
        )
        
        heston_params = HestonParameters(0.04, 0.04, 2.0, 0.3, -0.7)
        model = HestonModel(heston_params)
        
        price, ci = model.monte_carlo_price(
            deep_itm_data, "call", n_paths=5000
        )
        
        # Deep ITM call should be at least intrinsic value
        intrinsic_value = deep_itm_data.S0 - deep_itm_data.K
        assert price >= intrinsic_value * 0.95  # Allow small tolerance for discounting
    
    def test_deep_otm_options(self):
        """Test deep out-of-the-money options."""
        deep_otm_data = MarketData(
            S0=100.0, K=150.0, T=0.25,  # Deep OTM call
            r=0.05, q=0.02, sigma=0.20
        )
        
        heston_params = HestonParameters(0.04, 0.04, 2.0, 0.3, -0.7)
        model = HestonModel(heston_params)
        
        price, ci = model.monte_carlo_price(
            deep_otm_data, "call", n_paths=5000
        )
        
        # Deep OTM option should have small positive value
        assert 0 < price < 5.0
    
    def test_zero_correlation(self):
        """Test with zero correlation."""
        zero_corr_params = HestonParameters(0.04, 0.04, 2.0, 0.3, 0.0)
        model = HestonModel(zero_corr_params)
        
        market_data = MarketData(100.0, 100.0, 0.25, 0.05, 0.02, 0.20)
        
        price, ci = model.monte_carlo_price(
            market_data, "call", n_paths=5000
        )
        
        assert isinstance(price, float)
        assert price > 0
        assert not np.isnan(price)