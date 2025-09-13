"""
Test module for Black-Scholes model implementation.

This module contains comprehensive tests for the Black-Scholes pricing model,
including price accuracy, Greeks calculation, and edge cases.
"""

import pytest
import numpy as np
from derivatives_engine import (
    BlackScholesModel,
    MarketData,
    OptionType
)


class TestBlackScholesModel:
    """Test cases for Black-Scholes model implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = BlackScholesModel()
        self.market_data = MarketData(
            S0=100.0,
            K=100.0,
            T=0.25,
            r=0.05,
            q=0.02,
            sigma=0.20
        )
    
    def test_call_option_pricing(self):
        """Test call option pricing accuracy."""
        call_price = self.model.price(self.market_data, OptionType.CALL)
        
        # Expected price calculated independently
        expected_price = 4.3804  # Approximate expected value
        
        assert isinstance(call_price, float)
        assert call_price > 0
        assert abs(call_price - expected_price) < 0.1  # Allow small tolerance
    
    def test_put_option_pricing(self):
        """Test put option pricing accuracy."""
        put_price = self.model.price(self.market_data, OptionType.PUT)
        
        assert isinstance(put_price, float)
        assert put_price > 0
        
        # For ATM options with dividend yield, put should be cheaper than call
        call_price = self.model.price(self.market_data, OptionType.CALL)
        assert put_price < call_price
    
    def test_put_call_parity(self):
        """Test put-call parity relationship."""
        call_price = self.model.price(self.market_data, OptionType.CALL)
        put_price = self.model.price(self.market_data, OptionType.PUT)
        
        # Put-call parity: C - P = S*e^(-qT) - K*e^(-rT)
        left_side = call_price - put_price
        right_side = (self.market_data.S0 * np.exp(-self.market_data.q * self.market_data.T) - 
                     self.market_data.K * np.exp(-self.market_data.r * self.market_data.T))
        
        assert abs(left_side - right_side) < 1e-10
    
    def test_greeks_calculation(self):
        """Test Greeks calculation."""
        greeks = self.model.greeks(self.market_data, OptionType.CALL)
        
        # Check that all Greeks are present
        expected_greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
        for greek in expected_greeks:
            assert greek in greeks
            assert isinstance(greeks[greek], float)
        
        # Test Greek ranges for ATM call
        assert 0.4 < greeks['delta'] < 0.6  # Delta around 0.5 for ATM
        assert greeks['gamma'] > 0  # Gamma always positive
        assert greeks['theta'] < 0  # Theta negative (time decay)
        assert greeks['vega'] > 0   # Vega positive for long options
        assert greeks['rho'] > 0    # Rho positive for calls
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        
        # Test zero time to expiration
        expired_market = self.market_data.copy(T=0.0)
        
        # ITM call at expiration
        itm_call_market = expired_market.copy(S0=110, K=100)
        itm_call_price = self.model.price(itm_call_market, OptionType.CALL)
        assert abs(itm_call_price - 10.0) < 1e-10
        
        # OTM call at expiration
        otm_call_market = expired_market.copy(S0=90, K=100)
        otm_call_price = self.model.price(otm_call_market, OptionType.CALL)
        assert abs(otm_call_price - 0.0) < 1e-10
    
    def test_moneyness_effects(self):
        """Test pricing behavior across different moneyness levels."""
        
        strikes = [80, 90, 100, 110, 120]
        call_prices = []
        
        for K in strikes:
            market = self.market_data.copy(K=K)
            price = self.model.price(market, OptionType.CALL)
            call_prices.append(price)
        
        # Call prices should decrease as strike increases
        for i in range(1, len(call_prices)):
            assert call_prices[i] < call_prices[i-1]
    
    def test_volatility_effects(self):
        """Test pricing sensitivity to volatility changes."""
        
        vols = [0.1, 0.15, 0.2, 0.25, 0.3]
        call_prices = []
        
        for vol in vols:
            market = self.market_data.copy(sigma=vol)
            price = self.model.price(market, OptionType.CALL)
            call_prices.append(price)
        
        # Call prices should increase with volatility
        for i in range(1, len(call_prices)):
            assert call_prices[i] > call_prices[i-1]
    
    def test_time_decay(self):
        """Test time decay effects on option pricing."""
        
        times = [0.5, 0.25, 0.1, 0.05, 0.01]
        call_prices = []
        
        for T in times:
            market = self.market_data.copy(T=T)
            price = self.model.price(market, OptionType.CALL)
            call_prices.append(price)
        
        # Call prices should generally decrease as time decreases (for ATM options)
        for i in range(1, len(call_prices)):
            assert call_prices[i] < call_prices[i-1]
    
    def test_interest_rate_effects(self):
        """Test interest rate sensitivity."""
        
        rates = [0.01, 0.03, 0.05, 0.07, 0.10]
        call_prices = []
        put_prices = []
        
        for r in rates:
            market = self.market_data.copy(r=r)
            call_price = self.model.price(market, OptionType.CALL)
            put_price = self.model.price(market, OptionType.PUT)
            call_prices.append(call_price)
            put_prices.append(put_price)
        
        # Call prices should increase with interest rates
        for i in range(1, len(call_prices)):
            assert call_prices[i] > call_prices[i-1]
        
        # Put prices should decrease with interest rates
        for i in range(1, len(put_prices)):
            assert put_prices[i] < put_prices[i-1]
    
    def test_dividend_effects(self):
        """Test dividend yield effects on pricing."""
        
        dividends = [0.0, 0.01, 0.02, 0.03, 0.05]
        call_prices = []
        put_prices = []
        
        for q in dividends:
            market = self.market_data.copy(q=q)
            call_price = self.model.price(market, OptionType.CALL)
            put_price = self.model.price(market, OptionType.PUT)
            call_prices.append(call_price)
            put_prices.append(put_price)
        
        # Call prices should decrease with dividend yield
        for i in range(1, len(call_prices)):
            assert call_prices[i] < call_prices[i-1]
        
        # Put prices should increase with dividend yield
        for i in range(1, len(put_prices)):
            assert put_prices[i] > put_prices[i-1]
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        from derivatives_engine.utils.exceptions import ValidationError

        with pytest.raises(ValidationError):
            invalid_market = MarketData(S0=-100, K=100, T=0.25, r=0.05, sigma=0.2)

        with pytest.raises(ValidationError):
            invalid_market = MarketData(S0=100, K=-100, T=0.25, r=0.05, sigma=0.2)

        with pytest.raises(ValidationError):
            invalid_market = MarketData(S0=100, K=100, T=-0.25, r=0.05, sigma=0.2)
        
        with pytest.raises(ValidationError):
            invalid_market = MarketData(S0=100, K=100, T=0.25, r=0.05, sigma=-0.2)
        
        # Test invalid option type
        with pytest.raises(ValueError):
            self.model.price(self.market_data, "invalid_type")


@pytest.fixture
def sample_market_data():
    """Fixture providing sample market data for tests."""
    return MarketData(
        S0=100.0,
        K=105.0,
        T=0.25,
        r=0.05,
        q=0.02,
        sigma=0.20
    )


def test_model_consistency(sample_market_data):
    """Test model consistency across multiple calls."""
    model = BlackScholesModel()
    
    # Multiple calls should return the same result
    price1 = model.price(sample_market_data, OptionType.CALL)
    price2 = model.price(sample_market_data, OptionType.CALL)
    
    assert abs(price1 - price2) < 1e-15
    
    # Greeks should also be consistent
    greeks1 = model.greeks(sample_market_data, OptionType.CALL)
    greeks2 = model.greeks(sample_market_data, OptionType.CALL)
    
    for greek_name in greeks1:
        assert abs(greeks1[greek_name] - greeks2[greek_name]) < 1e-15


def test_performance_benchmark():
    """Test that pricing is fast enough for production use."""
    import time
    
    model = BlackScholesModel()
    market_data = MarketData(S0=100, K=100, T=0.25, r=0.05, sigma=0.2)
    
    # Benchmark pricing speed
    n_iterations = 1000
    start_time = time.perf_counter()
    
    for _ in range(n_iterations):
        model.price(market_data, OptionType.CALL)
    
    end_time = time.perf_counter()
    avg_time_ms = (end_time - start_time) / n_iterations * 1000
    
    # Should be able to price at least 1000 options per second (< 1ms each)
    assert avg_time_ms < 1.0, f"Pricing too slow: {avg_time_ms:.3f} ms per option"