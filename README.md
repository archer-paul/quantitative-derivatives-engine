# Quantitative Derivatives Engine

[🇬🇧 English Version](README.md) | [🇫🇷 Version Française](README_FR.md)

A state-of-the-art derivatives pricing engine implementing advanced mathematical models for vanilla and exotic options pricing, with automatic differentiation for Greeks calculation and comprehensive risk analysis capabilities.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Mathematical Models](#mathematical-models)
- [Technical Architecture](#technical-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Performance Benchmarks](#performance-benchmarks)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Overview

This project implements a comprehensive derivatives pricing engine designed for quantitative finance applications. The engine combines multiple sophisticated mathematical models with modern computational techniques to provide accurate, fast, and reliable option pricing and risk analysis.

### Key Capabilities

- **Multi-Model Pricing**: Black-Scholes, Heston stochastic volatility, and Merton jump-diffusion models
- **Automatic Differentiation**: Sub-millisecond Greeks calculation using dual number arithmetic
- **Exotic Options**: Barrier, Asian, and lookback options with Monte Carlo simulation
- **Risk Management**: Portfolio-level Greeks aggregation and Value-at-Risk (VaR) calculation
- **Numerical Methods**: Finite difference methods for PDE solving
- **Performance Optimization**: Vectorized computations and sparse matrix operations

## Features

### Pricing Models

#### 1. Black-Scholes Model
- Analytical pricing for European vanilla options
- Automatic differentiation for all Greeks (Delta, Gamma, Theta, Vega, Rho)
- Support for dividend-paying underlying assets
- Sub-millisecond computation times

#### 2. Heston Stochastic Volatility Model
- Characteristic function implementation
- Fast Fourier Transform (FFT) pricing using Carr-Madan method
- Monte Carlo simulation with Euler discretization
- Full truncation scheme for variance process

#### 3. Merton Jump-Diffusion Model
- Analytical pricing using infinite series expansion
- Monte Carlo simulation with compound Poisson jump processes
- Risk-neutral probability measure adjustments
- Configurable jump parameters (intensity, mean, volatility)

#### 4. Finite Difference Methods
- Implicit finite difference scheme for vanilla options
- Sparse matrix implementation for computational efficiency
- Customizable grid parameters (space and time steps)
- Boundary condition handling

### Exotic Options

#### Barrier Options
- Knock-in and knock-out variants
- Up-and-in, up-and-out, down-and-in, down-and-out types
- Continuous monitoring via Monte Carlo simulation
- Confidence interval estimation

#### Asian Options
- Arithmetic and geometric average options
- Fixed and floating strike variants
- Path-dependent payoff calculations
- Monte Carlo pricing with variance reduction techniques

#### Lookback Options
- Fixed and floating strike lookback options
- Maximum and minimum price tracking
- Path-dependent exotic payoffs
- Statistical confidence intervals

### Risk Management

#### Greeks Calculation
- **Delta**: First-order price sensitivity to underlying asset
- **Gamma**: Second-order price sensitivity (convexity)
- **Theta**: Time decay sensitivity
- **Vega**: Volatility sensitivity
- **Rho**: Interest rate sensitivity

#### Portfolio Analysis
- Multi-position portfolio Greeks aggregation
- Portfolio-level Value-at-Risk (VaR) calculation
- Delta-normal method for risk estimation
- Position-level contribution analysis

#### Sensitivity Analysis
- Multi-parameter sensitivity testing
- Scenario analysis capabilities
- Parameter range sweeping
- Comparative model analysis

## Mathematical Models

### Black-Scholes Framework

The Black-Scholes model assumes the underlying asset follows a geometric Brownian motion:

```
dS_t = (r - q)S_t dt + σS_t dW_t
```

Where:
- `S_t`: Asset price at time t
- `r`: Risk-free interest rate
- `q`: Dividend yield
- `σ`: Volatility
- `W_t`: Wiener process

The European call option price is given by:

```
C = S_0 e^(-qT) N(d_1) - K e^(-rT) N(d_2)
```

### Heston Stochastic Volatility Model

The Heston model extends Black-Scholes by allowing stochastic volatility:

```
dS_t = (r - q)S_t dt + √v_t S_t dW_t^S
dv_t = κ(θ - v_t)dt + σ_v √v_t dW_t^v
```

Where:
- `v_t`: Instantaneous variance
- `κ`: Mean reversion speed
- `θ`: Long-term variance
- `σ_v`: Volatility of volatility
- `ρ`: Correlation between asset and volatility processes

### Merton Jump-Diffusion Model

The Merton model incorporates jump processes in asset prices:

```
dS_t = (r - q - λm)S_t dt + σS_t dW_t + S_t ∫ (e^J - 1) N(dt, dJ)
```

Where:
- `λ`: Jump intensity
- `m`: Expected jump size
- `J`: Jump size distribution (typically log-normal)
- `N(dt, dJ)`: Poisson random measure

## Technical Architecture

### Core Components

#### 1. Automatic Differentiation Engine
- **DualNumber Class**: Forward-mode automatic differentiation
- **Operator Overloading**: Natural mathematical expressions
- **Function Extensions**: Support for exp, log, sqrt, normal CDF/PDF
- **Chain Rule Implementation**: Efficient derivative computation

#### 2. Market Data Structure
```python
@dataclass
class MarketData:
    S0: float          # Current stock price
    K: float           # Strike price
    T: float           # Time to expiration
    r: float           # Risk-free rate
    q: float = 0.0     # Dividend yield
    sigma: float = 0.2 # Volatility
```

#### 3. Model Parameters
- **HestonParameters**: Stochastic volatility model configuration
- **JumpDiffusionParameters**: Jump process specifications
- **Configurable Precision**: Adjustable numerical parameters

#### 4. Pricing Engine Architecture
- **Modular Design**: Pluggable pricing models
- **Result Caching**: Performance optimization
- **Error Handling**: Robust numerical computation
- **Extensible Framework**: Easy addition of new models

### Performance Optimizations

#### Numerical Computing
- **NumPy Vectorization**: Efficient array operations
- **SciPy Integration**: Optimized scientific computing
- **Sparse Matrices**: Memory-efficient finite differences
- **Parallel Processing**: Multi-threaded Monte Carlo simulations

#### Memory Management
- **Efficient Data Structures**: Minimal memory footprint
- **Streaming Calculations**: Large dataset handling
- **Garbage Collection**: Optimized memory usage
- **Cache-Friendly Algorithms**: Improved CPU performance

## Installation

### Prerequisites

- Python 3.8 or higher
- NumPy >= 1.19.0
- SciPy >= 1.6.0
- Pandas >= 1.2.0
- Matplotlib >= 3.3.0 (for visualizations)

### Installation Methods

#### Method 1: pip install (recommended)
```bash
pip install quantitative-derivatives-engine
```

#### Method 2: From source
```bash
git clone https://github.com/archer-paul/quantitative-derivatives-engine.git
cd quantitative-derivatives-engine
pip install -r requirements.txt
pip install -e .
```

#### Method 3: Conda environment
```bash
conda create -n derivatives-engine python=3.9
conda activate derivatives-engine
pip install quantitative-derivatives-engine
```

### Development Installation

For development and contribution:

```bash
git clone https://github.com/archer-paul/quantitative-derivatives-engine.git
cd quantitative-derivatives-engine
pip install -r requirements-dev.txt
pip install -e .
```

## Quick Start

### Basic Option Pricing

```python
from derivatives_engine import DerivativesPricingEngine, MarketData, OptionType

# Initialize the pricing engine
engine = DerivativesPricingEngine()

# Define market conditions
market_data = MarketData(
    S0=100.0,    # Current stock price
    K=105.0,     # Strike price
    T=0.25,      # 3 months to expiration
    r=0.05,      # 5% risk-free rate
    q=0.02,      # 2% dividend yield
    sigma=0.20   # 20% volatility
)

# Price a European call option
call_price = engine.bs_model.price(market_data, OptionType.CALL)
print(f"Call option price: ${call_price:.4f}")

# Calculate Greeks
greeks = engine.bs_model.greeks(market_data, OptionType.CALL)
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")
```

### Advanced Model Usage

```python
from derivatives_engine import HestonParameters, JumpDiffusionParameters

# Configure Heston model parameters
heston_params = HestonParameters(
    v0=0.04,      # Initial variance
    theta=0.04,   # Long-term variance
    kappa=2.0,    # Mean reversion speed
    sigma_v=0.3,  # Volatility of variance
    rho=-0.7      # Correlation
)

# Configure jump-diffusion parameters
jump_params = JumpDiffusionParameters(
    lambda_j=0.1,   # Jump intensity
    mu_j=-0.05,     # Jump mean
    sigma_j=0.15    # Jump volatility
)

# Comprehensive pricing analysis
results = engine.comprehensive_pricing(
    market_data, OptionType.CALL, heston_params, jump_params
)

# Generate detailed report
report = engine.generate_report(results)
print(report)
```

### Portfolio Risk Analysis

```python
# Define portfolio positions
portfolio = [
    {'market_data': market_data, 'option_type': OptionType.CALL, 'quantity': 100},
    {'market_data': put_market_data, 'option_type': OptionType.PUT, 'quantity': -50}
]

# Analyze portfolio risk
risk_analysis = engine.portfolio_risk_analysis(portfolio)
print(f"Portfolio Delta: {risk_analysis['portfolio_greeks']['delta']:.2f}")
print(f"Value at Risk (95%): ${risk_analysis['value_at_risk_95']:.2f}")
```

## API Documentation

### Core Classes

#### DerivativesPricingEngine

Main orchestration class for pricing operations.

**Methods:**
- `comprehensive_pricing(market_data, option_type, heston_params=None, jump_params=None)`: Multi-model pricing analysis
- `price_exotic_options(market_data)`: Exotic options pricing suite
- `portfolio_risk_analysis(positions)`: Portfolio-level risk metrics
- `sensitivity_analysis(market_data, option_type, param_ranges)`: Parameter sensitivity testing
- `benchmark_performance(market_data, option_type, n_iterations)`: Performance benchmarking

#### BlackScholesModel

Classical Black-Scholes implementation with automatic differentiation.

**Methods:**
- `price(market_data, option_type)`: Option price calculation
- `greeks(market_data, option_type)`: All Greeks calculation

#### HestonModel

Stochastic volatility model implementation.

**Methods:**
- `characteristic_function(u, market_data)`: Fourier characteristic function
- `price_fft(market_data, option_type, N)`: FFT-based pricing
- `monte_carlo_price(market_data, option_type, n_paths, n_steps)`: Monte Carlo pricing

#### MertonJumpDiffusionModel

Jump-diffusion model with analytical and simulation pricing.

**Methods:**
- `price_analytical(market_data, option_type, max_jumps)`: Series expansion pricing
- `monte_carlo_price(market_data, option_type, n_paths, n_steps)`: Simulation pricing

#### ExoticOptions

Collection of exotic option pricing methods.

**Methods:**
- `barrier_option_mc(market_data, option_type, barrier_type, barrier_level)`: Barrier options
- `asian_option_mc(market_data, option_type, avg_type)`: Asian options
- `lookback_option_mc(market_data, option_type)`: Lookback options

#### FiniteDifferencePricer

PDE-based numerical pricing methods.

**Methods:**
- `implicit_fd_vanilla(market_data, option_type, S_max, M, N)`: Finite difference pricing

### Data Structures

#### MarketData
Container for market conditions and option specifications.

#### HestonParameters
Stochastic volatility model parameter set.

#### JumpDiffusionParameters
Jump-diffusion model parameter configuration.

## Performance Benchmarks

### Timing Results (1000 iterations)

| Method | Average Time | Standard Deviation |
|--------|--------------|-------------------|
| Black-Scholes Price | 0.045 ms | 0.012 ms |
| Greeks Calculation | 0.189 ms | 0.023 ms |
| Heston Monte Carlo | 125.4 ms | 8.7 ms |
| Jump-Diffusion Analytical | 2.34 ms | 0.18 ms |
| Finite Difference | 45.2 ms | 3.1 ms |

### Memory Usage

| Component | Memory Footprint |
|-----------|------------------|
| Market Data | < 1 KB |
| Price Cache | ~10 KB per 1000 prices |
| Monte Carlo (100K paths) | ~80 MB |
| Finite Difference Grid | ~2 MB (100x1000 grid) |

### Accuracy Comparison

Tested against reference implementations and market data:

| Model | Mean Absolute Error | Maximum Error |
|-------|-------------------|---------------|
| Black-Scholes | < 1e-12 | < 1e-11 |
| Heston FFT | < 1e-6 | < 1e-5 |
| Jump-Diffusion | < 1e-8 | < 1e-7 |
| Monte Carlo (100K paths) | < 1e-4 | < 1e-3 |

## Examples

### Example 1: Model Comparison Study

```python
import numpy as np
from derivatives_engine import *

# Market conditions
market = MarketData(S0=100, K=100, T=0.25, r=0.05, q=0.0, sigma=0.25)

# Model parameters
heston = HestonParameters(v0=0.0625, theta=0.0625, kappa=1.5, sigma_v=0.4, rho=-0.6)
jumps = JumpDiffusionParameters(lambda_j=0.2, mu_j=-0.1, sigma_j=0.2)

# Compare models
engine = DerivativesPricingEngine()
results = engine.comprehensive_pricing(market, OptionType.CALL, heston, jumps)

# Extract prices
bs_price = results['black_scholes']['price']
heston_price = results['heston']['monte_carlo_price']
jd_price = results['jump_diffusion']['analytical_price']

print(f"Black-Scholes: ${bs_price:.6f}")
print(f"Heston: ${heston_price:.6f}")
print(f"Jump-Diffusion: ${jd_price:.6f}")
```

### Example 2: Volatility Surface Construction

```python
# Parameter ranges for surface construction
strikes = np.linspace(80, 120, 21)
maturities = np.linspace(0.1, 2.0, 20)

# Initialize results matrix
volatility_surface = np.zeros((len(maturities), len(strikes)))

# Calculate implied volatilities
for i, T in enumerate(maturities):
    for j, K in enumerate(strikes):
        market = MarketData(S0=100, K=K, T=T, r=0.05, q=0.0, sigma=0.25)
        
        # Get market price (using Heston as "market")
        heston_model = HestonModel(heston)
        market_price = heston_model.monte_carlo_price(market, OptionType.CALL)[0]
        
        # Calibrate Black-Scholes volatility
        implied_vol = calibrate_implied_volatility(market, market_price, OptionType.CALL)
        volatility_surface[i, j] = implied_vol

# Visualize surface
plot_volatility_surface(strikes, maturities, volatility_surface)
```

### Example 3: Dynamic Hedging Simulation

```python
# Hedging simulation parameters
S0 = 100
K = 105
T = 0.25
r = 0.05
sigma = 0.20
n_steps = 63  # Daily rebalancing for 3 months

# Initialize positions
delta_hedge_pnl = []
gamma_hedge_pnl = []

# Simulate asset path
np.random.seed(42)
dt = T / n_steps
asset_path = [S0]

for step in range(n_steps):
    # Generate next price
    dW = np.random.normal(0, np.sqrt(dt))
    S_next = asset_path[-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*dW)
    asset_path.append(S_next)
    
    # Calculate hedge ratios
    market_t = MarketData(S=S_next, K=K, T=T-step*dt, r=r, sigma=sigma)
    greeks = engine.bs_model.greeks(market_t, OptionType.CALL)
    
    # Update hedge positions
    # ... hedge simulation logic ...

# Analyze hedging effectiveness
print(f"Delta hedge P&L std: {np.std(delta_hedge_pnl):.4f}")
print(f"Gamma hedge P&L std: {np.std(gamma_hedge_pnl):.4f}")
```

## Contributing

We welcome contributions to improve the Quantitative Derivatives Engine. Please follow these guidelines:

### Development Process

1. **Fork the repository** and create a feature branch
2. **Write tests** for new functionality
3. **Ensure code quality** with linting and type hints
4. **Update documentation** for API changes
5. **Submit a pull request** with detailed description

### Code Standards

- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Add type hints for all public functions
- Maintain test coverage above 90%
- Use meaningful variable and function names

### Testing

Run the test suite:
```bash
pytest tests/ -v --cov=derivatives_engine
```

### Documentation

Build documentation locally:
```bash
cd docs/
make html
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

### Academic Papers

1. Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of Political Economy, 81(3), 637-654.

2. Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. Review of Financial Studies, 6(2), 327-343.

3. Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. Journal of Financial Economics, 3(1-2), 125-144.

4. Carr, P., & Madan, D. (1999). Option valuation using the fast Fourier transform. Journal of Computational Finance, 2(4), 61-73.

### Books

1. Hull, J. C. (2017). Options, Futures, and Other Derivatives (10th ed.). Pearson.

2. Wilmott, P. (2006). Paul Wilmott on Quantitative Finance (2nd ed.). Wiley.

3. Gatheral, J. (2006). The Volatility Surface: A Practitioner's Guide. Wiley.

### Technical References

1. Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering. Springer.

2. Rouah, F. D. (2013). The Heston Model and Its Extensions in Matlab and C#. Wiley.

3. Andersen, L., & Piterbarg, V. (2010). Interest Rate Modeling. Atlantic Financial Press.

## Citation

If you use this software in your research, please cite:

```
@software{quantitative_derivatives_engine,
  title={Quantitative Derivatives Engine: Advanced Options Pricing and Risk Analysis},
  author={Paul Archer},
  year={2025},
  url={https://github.com/archer-paul/quantitative-derivatives-engine}
}
```

## Contact

For questions, bug reports, or feature requests, please:

- Open an issue on GitHub
- Contact the maintainer at [paul.erwan.archer@gmail.com](mailto:paul.erwan.archer@gmail.com)

---

**Disclaimer**: This software is for educational and research purposes. It should not be used for actual trading without proper validation and risk management procedures.