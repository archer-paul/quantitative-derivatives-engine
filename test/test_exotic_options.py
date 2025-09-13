"""
Test module for exotic options pricing.

This module contains comprehensive tests for exotic option types including
barrier options, Asian options, lookback options, and other exotic structures.
"""

import pytest
import numpy as np
from derivatives_engine import (
    ExoticOptions,
    MarketData
)


class TestBarrierOptions:
    """Test cases for barrier options."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exotic_engine = ExoticOptions()
        self.market_data = MarketData(
            S0=100.0,
            K=100.0,
            T=0.25,
            r=0.05,
            q=0.02,
            sigma=0.20
        )
    
    def test_up_and_out_barrier_call(self):
        """Test up-and-out barrier call option."""
        barrier_level = 120.0  # Above current price
        
        price, ci = self.exotic_engine.barrier_option_mc(
            self.market_data, "call", "up_out", barrier_level, 
            n_paths=10000, n_steps=100
        )
        
        assert isinstance(price, float)
        assert isinstance(ci, float)
        assert price >= 0
        assert ci >= 0
        
        # Should be cheaper than vanilla call since it can be knocked out
        from derivatives_engine import BlackScholesModel
        vanilla_price = BlackScholesModel.price(self.market_data, "call")
        assert price <= vanilla_price
    
    def test_up_and_in_barrier_call(self):
        """Test up-and-in barrier call option."""
        barrier_level = 120.0
        
        up_in_price, _ = self.exotic_engine.barrier_option_mc(
            self.market_data, "call", "up_in", barrier_level, 
            n_paths=10000, n_steps=100
        )
        
        up_out_price, _ = self.exotic_engine.barrier_option_mc(
            self.market_data, "call", "up_out", barrier_level,
            n_paths=10000, n_steps=100
        )
        
        # up_in + up_out should approximately equal vanilla option
        from derivatives_engine import BlackScholesModel
        vanilla_price = BlackScholesModel.price(self.market_data, "call")
        
        # Allow for Monte Carlo error
        total_barrier_price = up_in_price + up_out_price
        relative_error = abs(total_barrier_price - vanilla_price) / vanilla_price
        assert relative_error < 0.05  # Within 5%
    
    def test_down_and_out_barrier_put(self):
        """Test down-and-out barrier put option."""
        barrier_level = 80.0  # Below current price
        
        price, ci = self.exotic_engine.barrier_option_mc(
            self.market_data, "put", "down_out", barrier_level,
            n_paths=10000, n_steps=100
        )
        
        assert price >= 0
        
        # Should be cheaper than vanilla put
        from derivatives_engine import BlackScholesModel
        vanilla_price = BlackScholesModel.price(self.market_data, "put")
        assert price <= vanilla_price
    
    def test_barrier_at_strike(self):
        """Test barrier option when barrier equals strike."""
        barrier_level = self.market_data.K  # Barrier at strike
        
        price, _ = self.exotic_engine.barrier_option_mc(
            self.market_data, "call", "up_out", barrier_level,
            n_paths=5000, n_steps=50
        )
        
        # Should be very small since barrier is at strike
        assert 0 <= price < 2.0
    
    def test_invalid_barrier_type(self):
        """Test invalid barrier type handling."""
        with pytest.raises(ValueError):
            self.exotic_engine.barrier_option_mc(
                self.market_data, "call", "invalid_type", 120.0
            )


class TestAsianOptions:
    """Test cases for Asian (average) options."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exotic_engine = ExoticOptions()
        self.market_data = MarketData(
            S0=100.0,
            K=100.0,
            T=0.25,
            r=0.05,
            q=0.02,
            sigma=0.20
        )
    
    def test_arithmetic_asian_call(self):
        """Test arithmetic Asian call option."""
        price, ci = self.exotic_engine.asian_option_mc(
            self.market_data, "call", "arithmetic",
            n_paths=10000, n_steps=100
        )
        
        assert isinstance(price, float)
        assert isinstance(ci, float)
        assert price >= 0
        assert ci >= 0
        
        # Should be cheaper than vanilla due to averaging effect
        from derivatives_engine import BlackScholesModel
        vanilla_price = BlackScholesModel.price(self.market_data, "call")
        assert price <= vanilla_price
    
    def test_geometric_asian_call(self):
        """Test geometric Asian call option."""
        price, ci = self.exotic_engine.asian_option_mc(
            self.market_data, "call", "geometric",
            n_paths=10000, n_steps=100
        )
        
        assert price >= 0
        
        # Geometric mean is always <= arithmetic mean
        arithmetic_price, _ = self.exotic_engine.asian_option_mc(
            self.market_data, "call", "arithmetic",
            n_paths=10000, n_steps=100
        )
        
        # Allow for Monte Carlo error
        assert price <= arithmetic_price * 1.1
    
    def test_arithmetic_asian_put(self):
        """Test arithmetic Asian put option."""
        price, ci = self.exotic_engine.asian_option_mc(
            self.market_data, "put", "arithmetic",
            n_paths=10000, n_steps=100
        )
        
        assert price >= 0
        
        # Should be cheaper than vanilla put
        from derivatives_engine import BlackScholesModel
        vanilla_price = BlackScholesModel.price(self.market_data, "put")
        assert price <= vanilla_price
    
    def test_invalid_average_type(self):
        """Test invalid average type handling."""
        with pytest.raises(ValueError):
            self.exotic_engine.asian_option_mc(
                self.market_data, "call", "invalid_average"
            )
    
    def test_asian_option_convergence(self):
        """Test convergence with increasing averaging points."""
        step_counts = [50, 100, 200]
        prices = []
        
        for n_steps in step_counts:
            price, _ = self.exotic_engine.asian_option_mc(
                self.market_data, "call", "arithmetic",
                n_paths=5000, n_steps=n_steps
            )
            prices.append(price)
        
        # Prices should converge as we increase steps
        diff1 = abs(prices[1] - prices[0])
        diff2 = abs(prices[2] - prices[1])
        
        # Expect some convergence (allow randomness)
        assert diff2 < diff1 * 2.0


class TestLookbackOptions:
    """Test cases for lookback options."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exotic_engine = ExoticOptions()
        self.market_data = MarketData(
            S0=100.0,
            K=100.0,
            T=0.25,
            r=0.05,
            q=0.02,
            sigma=0.20
        )
    
    def test_lookback_call(self):
        """Test lookback call option (max payoff)."""
        price, ci = self.exotic_engine.lookback_option_mc(
            self.market_data, "lookback_call",
            n_paths=10000, n_steps=100
        )
        
        assert isinstance(price, float)
        assert isinstance(ci, float)
        assert price >= 0
        
        # Lookback call should be more expensive than vanilla
        from derivatives_engine import BlackScholesModel
        vanilla_price = BlackScholesModel.price(self.market_data, "call")
        assert price >= vanilla_price
    
    def test_lookback_put(self):
        """Test lookback put option (min payoff)."""
        price, ci = self.exotic_engine.lookback_option_mc(
            self.market_data, "lookback_put",
            n_paths=10000, n_steps=100
        )
        
        assert price >= 0
        
        # Lookback put should be more expensive than vanilla
        from derivatives_engine import BlackScholesModel
        vanilla_price = BlackScholesModel.price(self.market_data, "put")
        assert price >= vanilla_price
    
    def test_invalid_lookback_type(self):
        """Test invalid lookback type handling."""
        with pytest.raises(ValueError):
            self.exotic_engine.lookback_option_mc(
                self.market_data, "invalid_lookback"
            )


class TestRainbowOptions:
    """Test cases for rainbow (multi-asset) options."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exotic_engine = ExoticOptions()
        
        # Create multi-asset market data
        self.market_data_list = [
            MarketData(100.0, 100.0, 0.25, 0.05, 0.02, 0.20),  # Asset 1
            MarketData(110.0, 100.0, 0.25, 0.05, 0.02, 0.25),  # Asset 2
            MarketData(90.0, 100.0, 0.25, 0.05, 0.02, 0.30)    # Asset 3
        ]
        
        # Correlation matrix (3x3)
        self.correlation_matrix = np.array([
            [1.0, 0.3, 0.1],
            [0.3, 1.0, 0.2],
            [0.1, 0.2, 1.0]
        ])
    
    def test_max_call_rainbow(self):
        """Test rainbow max call option."""
        price, ci = self.exotic_engine.rainbow_option_mc(
            self.market_data_list, "max_call", self.correlation_matrix,
            n_paths=5000, n_steps=100
        )
        
        assert isinstance(price, float)
        assert isinstance(ci, float)
        assert price >= 0
        
        # Max call should be more expensive than individual calls
        from derivatives_engine import BlackScholesModel
        individual_call = BlackScholesModel.price(self.market_data_list[0], "call")
        assert price >= individual_call
    
    def test_min_put_rainbow(self):
        """Test rainbow min put option."""
        price, ci = self.exotic_engine.rainbow_option_mc(
            self.market_data_list, "min_put", self.correlation_matrix,
            n_paths=5000, n_steps=100
        )
        
        assert price >= 0
    
    def test_correlation_matrix_validation(self):
        """Test correlation matrix dimension validation."""
        # Wrong dimension correlation matrix
        wrong_corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        
        with pytest.raises(ValueError):
            self.exotic_engine.rainbow_option_mc(
                self.market_data_list, "max_call", wrong_corr
            )
    
    def test_invalid_rainbow_type(self):
        """Test invalid rainbow option type."""
        with pytest.raises(ValueError):
            self.exotic_engine.rainbow_option_mc(
                self.market_data_list, "invalid_type", self.correlation_matrix
            )


class TestAmericanOptions:
    """Test cases for American options using binomial method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exotic_engine = ExoticOptions()
        self.market_data = MarketData(
            S0=100.0,
            K=100.0,
            T=0.25,
            r=0.05,
            q=0.02,
            sigma=0.20
        )
    
    def test_american_call(self):
        """Test American call option pricing."""
        price = self.exotic_engine.american_option_binomial(
            self.market_data, "call", n_steps=500
        )
        
        assert isinstance(price, float)
        assert price >= 0
        
        # Should be at least as valuable as European call
        from derivatives_engine import BlackScholesModel
        european_price = BlackScholesModel.price(self.market_data, "call")
        assert price >= european_price * 0.99  # Allow small numerical error
    
    def test_american_put(self):
        """Test American put option pricing."""
        price = self.exotic_engine.american_option_binomial(
            self.market_data, "put", n_steps=500
        )
        
        assert price >= 0
        
        # Should be at least as valuable as European put
        from derivatives_engine import BlackScholesModel
        european_price = BlackScholesModel.price(self.market_data, "put")
        assert price >= european_price
    
    def test_american_put_early_exercise_premium(self):
        """Test early exercise premium for American puts."""
        # Deep ITM put should have significant early exercise premium
        deep_itm_data = MarketData(
            S0=80.0, K=100.0, T=1.0,  # Deep ITM, long expiry
            r=0.10, q=0.0, sigma=0.20
        )
        
        american_price = self.exotic_engine.american_option_binomial(
            deep_itm_data, "put", n_steps=500
        )
        
        from derivatives_engine import BlackScholesModel
        european_price = BlackScholesModel.price(deep_itm_data, "put")
        
        early_exercise_premium = american_price - european_price
        assert early_exercise_premium > 0.5  # Should have meaningful premium


class TestQuantoOptions:
    """Test cases for quanto options."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exotic_engine = ExoticOptions()
        self.market_data = MarketData(
            S0=100.0,
            K=100.0,
            T=0.25,
            r=0.05,  # Domestic rate
            q=0.02,
            sigma=0.20
        )
    
    def test_quanto_call(self):
        """Test quanto call option."""
        fx_volatility = 0.15
        fx_correlation = -0.3  # Negative correlation typical
        foreign_rate = 0.03
        
        price, ci = self.exotic_engine.quanto_option_mc(
            self.market_data, "call", fx_volatility, fx_correlation, foreign_rate,
            n_paths=10000, n_steps=100
        )
        
        assert isinstance(price, float)
        assert isinstance(ci, float)
        assert price >= 0
    
    def test_quanto_correlation_effects(self):
        """Test quanto option sensitivity to FX correlation."""
        fx_volatility = 0.15
        foreign_rate = 0.03
        
        # Negative correlation
        neg_corr_price, _ = self.exotic_engine.quanto_option_mc(
            self.market_data, "call", fx_volatility, -0.5, foreign_rate,
            n_paths=5000
        )
        
        # Positive correlation
        pos_corr_price, _ = self.exotic_engine.quanto_option_mc(
            self.market_data, "call", fx_volatility, 0.5, foreign_rate,
            n_paths=5000
        )
        
        # Prices should differ based on correlation
        assert abs(neg_corr_price - pos_corr_price) > 0.1


class TestExoticOptionEdgeCases:
    """Test edge cases for exotic options."""
    
    def test_very_short_expiry(self):
        """Test exotic options with very short expiry."""
        short_expiry_data = MarketData(
            S0=100.0, K=100.0, T=0.001,  # Less than 1 day
            r=0.05, q=0.02, sigma=0.20
        )
        
        exotic_engine = ExoticOptions()
        
        # Test barrier option
        price, _ = exotic_engine.barrier_option_mc(
            short_expiry_data, "call", "up_out", 110.0, n_paths=1000
        )
        assert price >= 0
        assert price < 5.0
        
        # Test Asian option
        price, _ = exotic_engine.asian_option_mc(
            short_expiry_data, "call", "arithmetic", n_paths=1000
        )
        assert price >= 0
    
    def test_high_volatility(self):
        """Test exotic options with high volatility."""
        high_vol_data = MarketData(
            S0=100.0, K=100.0, T=0.25,
            r=0.05, q=0.02, sigma=1.0  # 100% volatility
        )
        
        exotic_engine = ExoticOptions()
        
        # Should still produce reasonable prices
        price, _ = exotic_engine.barrier_option_mc(
            high_vol_data, "call", "up_out", 150.0, n_paths=5000
        )
        assert price >= 0
        assert price < high_vol_data.S0  # Sanity check
    
    def test_zero_time_value(self):
        """Test options at expiration."""
        expired_data = MarketData(
            S0=110.0, K=100.0, T=0.0,  # At expiration
            r=0.05, q=0.02, sigma=0.20
        )
        
        exotic_engine = ExoticOptions()
        
        # Lookback call at expiration should equal max price - strike
        # Since there's only one observation (current price), max = current
        price, _ = exotic_engine.lookback_option_mc(
            expired_data, "lookback_call", n_paths=1000, n_steps=1
        )
        
        expected_payoff = max(expired_data.S0 - expired_data.K, 0)
        assert abs(price - expected_payoff) < 0.1  # Allow small numerical error