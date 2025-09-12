#!/usr/bin/env python3
"""
Basic Options Pricing Example

This example demonstrates the fundamental usage of the quantitative derivatives engine
for pricing European vanilla options using the Black-Scholes model and calculating Greeks.

Run this example:
    python examples/basic_pricing.py
"""

import sys
import os

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from derivatives_engine import (
    DerivativesPricingEngine,
    MarketData,
    OptionType
)


def main():
    """Demonstrate basic option pricing functionality."""
    
    print("=" * 60)
    print("BASIC OPTIONS PRICING EXAMPLE")
    print("=" * 60)
    
    # Initialize the pricing engine
    engine = DerivativesPricingEngine()
    
    # Define market conditions for a typical at-the-money option
    market_data = MarketData(
        S0=100.0,    # Current stock price: $100
        K=100.0,     # Strike price: $100 (at-the-money)
        T=0.25,      # Time to expiration: 3 months
        r=0.05,      # Risk-free rate: 5%
        q=0.02,      # Dividend yield: 2%
        sigma=0.20   # Volatility: 20%
    )
    
    print(f"Market Conditions:")
    print(f"  Stock Price (S0):     ${market_data.S0:,.2f}")
    print(f"  Strike Price (K):     ${market_data.K:,.2f}")
    print(f"  Time to Expiry (T):   {market_data.T:.3f} years")
    print(f"  Risk-free Rate (r):   {market_data.r:.2%}")
    print(f"  Dividend Yield (q):   {market_data.q:.2%}")
    print(f"  Volatility (sigma):   {market_data.sigma:.2%}")
    print(f"  Moneyness (S0/K):     {market_data.moneyness:.3f}")
    print()
    
    # Price call and put options
    print("BLACK-SCHOLES PRICING:")
    print("-" * 30)
    
    call_price = engine.bs_model.price(market_data, "call")
    put_price = engine.bs_model.price(market_data, "put")
    
    print(f"Call Option Price:     ${call_price:.4f}")
    print(f"Put Option Price:      ${put_price:.4f}")
    
    # Verify put-call parity
    import numpy as np
    pcp_left = call_price - put_price
    pcp_right = (market_data.S0 * np.exp(-market_data.q * market_data.T) - 
                market_data.K * np.exp(-market_data.r * market_data.T))
    
    print(f"\nPut-Call Parity Check:")
    print(f"  C - P = ${pcp_left:.6f}")
    print(f"  F - K*e^(-rT) = ${pcp_right:.6f}")
    print(f"  Difference: ${abs(pcp_left - pcp_right):.8f}")
    print()
    
    # Calculate Greeks for the call option
    print("GREEKS ANALYSIS (Call Option):")
    print("-" * 35)
    
    call_greeks = engine.bs_model.greeks(market_data, "call")
    
    print(f"Delta:                 {call_greeks['delta']:>8.4f}")
    print(f"Gamma:                 {call_greeks['gamma']:>8.4f}")
    print(f"Theta:                 {call_greeks['theta']:>8.4f}")
    print(f"Vega:                  {call_greeks['vega']:>8.4f}")
    print(f"Rho:                   {call_greeks['rho']:>8.4f}")
    print()
    
    # Interpret the Greeks
    print("GREEKS INTERPRETATION:")
    print("-" * 25)
    print(f"- For a $1 increase in stock price, call value changes by ${call_greeks['delta']:.4f}")
    print(f"- Delta changes by {call_greeks['gamma']:.4f} for each $1 stock move")
    print(f"- Option loses ${-call_greeks['theta']:.4f} in value per day (time decay)")
    print(f"- For 1% increase in volatility, call value increases by ${call_greeks['vega']:.4f}")
    print(f"- For 1% increase in interest rate, call value increases by ${call_greeks['rho']:.4f}")
    print()
    
    # Sensitivity analysis
    print("SENSITIVITY ANALYSIS:")
    print("-" * 22)
    
    scenarios = [
        ("Stock +10%", market_data.copy(S0=110)),
        ("Stock -10%", market_data.copy(S0=90)),
        ("Volatility +25%", market_data.copy(sigma=0.25)),
        ("Volatility -25%", market_data.copy(sigma=0.15)),
        ("1 Month Later", market_data.copy(T=0.167)),
    ]
    
    base_call_price = call_price
    
    for scenario_name, scenario_market in scenarios:
        scenario_price = engine.bs_model.price(scenario_market, "call")
        price_change = scenario_price - base_call_price
        percent_change = (price_change / base_call_price) * 100
        
        print(f"{scenario_name:>15}: ${scenario_price:7.4f} "
              f"(${price_change:+7.4f}, {percent_change:+6.2f}%)")
    
    print()
    print("=> Basic pricing example completed successfully!")
    

if __name__ == "__main__":
    main()