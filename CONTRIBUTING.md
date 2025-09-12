# Contributing to Quantitative Derivatives Engine

We welcome contributions to the Quantitative Derivatives Engine! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please be respectful and professional in all interactions.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of quantitative finance concepts
- Familiarity with NumPy, SciPy, and pandas

### Setting Up Development Environment

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/quantitative-derivatives-engine.git
   cd quantitative-derivatives-engine
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install development dependencies**:
   ```bash
   pip install -e .[dev]
   ```

5. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

6. **Verify installation**:
   ```bash
   python -c "import derivatives_engine; print('Installation successful!')"
   pytest tests/ -v
   ```

## Development Process

### Branch Strategy

- `main`: Stable release branch
- `develop`: Development branch for new features
- `feature/feature-name`: Feature development branches
- `bugfix/bug-description`: Bug fix branches
- `hotfix/critical-fix`: Critical production fixes

### Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code standards below

3. **Write tests** for new functionality

4. **Run the test suite**:
   ```bash
   pytest tests/ -v --cov=derivatives_engine
   ```

5. **Check code quality**:
   ```bash
   black derivatives_engine tests
   flake8 derivatives_engine tests
   mypy derivatives_engine
   ```

6. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request** on GitHub

## Code Standards

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting (line length: 88)
- Use [isort](https://isort.readthedocs.io/) for import sorting
- Use type hints for all public functions and methods

### Naming Conventions

- **Classes**: PascalCase (e.g., `BlackScholesModel`)
- **Functions/Methods**: snake_case (e.g., `calculate_price`)
- **Variables**: snake_case (e.g., `option_price`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `OPTION_TYPE`)
- **Private methods**: prefix with underscore (e.g., `_internal_method`)

### Documentation

- All public classes and functions must have docstrings
- Use [Google style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Include examples in docstrings where appropriate
- Document all parameters, return values, and exceptions

### Example Function Documentation

```python
def calculate_black_scholes_price(
    S0: float,
    K: float, 
    T: float,
    r: float,
    sigma: float,
    option_type: str
) -> float:
    """
    Calculate Black-Scholes option price.
    
    This function implements the classical Black-Scholes formula for
    European option pricing under constant volatility assumptions.
    
    Args:
        S0: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
        option_type: Either 'call' or 'put'
        
    Returns:
        Option price as a float
        
    Raises:
        ValueError: If option_type is not 'call' or 'put'
        ValueError: If any parameter is negative where it shouldn't be
        
    Example:
        >>> price = calculate_black_scholes_price(100, 105, 0.25, 0.05, 0.2, 'call')
        >>> print(f"Option price: ${price:.2f}")
        Option price: $4.32
    """
    # Implementation here
    pass
```

## Testing

### Test Structure

- Unit tests in `tests/` directory
- One test file per module (e.g., `test_black_scholes.py`)
- Integration tests for complete workflows
- Performance benchmarks for critical functions

### Test Requirements

- **Coverage**: Maintain >90% test coverage
- **Performance**: Include performance benchmarks for pricing functions
- **Edge Cases**: Test boundary conditions and error cases
- **Numerical Accuracy**: Verify results against known analytical solutions

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=derivatives_engine --cov-report=html

# Run specific test file
pytest tests/test_black_scholes.py -v

# Run performance tests only
pytest tests/ -v -m "performance"

# Skip slow tests
pytest tests/ -v -m "not slow"
```

### Writing Tests

```python
import pytest
import numpy as np
from derivatives_engine import BlackScholesModel, MarketData, OptionType

class TestBlackScholesModel:
    def setup_method(self):
        """Set up test fixtures."""
        self.model = BlackScholesModel()
        self.market_data = MarketData(
            S0=100.0, K=100.0, T=0.25, r=0.05, sigma=0.20
        )
    
    def test_call_option_pricing(self):
        """Test call option pricing accuracy."""
        price = self.model.price(self.market_data, OptionType.CALL)
        
        assert isinstance(price, float)
        assert price > 0
        assert 3.0 < price < 6.0  # Reasonable range for ATM option
    
    @pytest.mark.performance
    def test_pricing_performance(self):
        """Test that pricing is fast enough."""
        import time
        
        start = time.perf_counter()
        for _ in range(1000):
            self.model.price(self.market_data, OptionType.CALL)
        end = time.perf_counter()
        
        avg_time_ms = (end - start) / 1000 * 1000
        assert avg_time_ms < 1.0  # Less than 1ms per calculation
```

## Documentation

### Building Documentation

```bash
cd docs/
make html
```

### Documentation Standards

- Update docstrings for any modified functions
- Add examples to the examples/ directory for new features
- Update README.md if adding major features
- Include mathematical formulations for new models

## Pull Request Process

### Before Submitting

1. **Ensure all tests pass**:
   ```bash
   pytest tests/ -v --cov=derivatives_engine
   ```

2. **Check code quality**:
   ```bash
   black --check derivatives_engine tests
   flake8 derivatives_engine tests
   mypy derivatives_engine
   ```

3. **Update documentation** as needed

4. **Add tests** for new functionality

5. **Update CHANGELOG.md** with your changes

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)  
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have checked that code coverage remains above 90%

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
```

### Review Process

1. **Automated checks** must pass (CI/CD pipeline)
2. **Code review** by at least one maintainer
3. **Performance review** for performance-critical changes
4. **Documentation review** for user-facing changes

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: Backwards-compatible functionality additions
- **PATCH**: Backwards-compatible bug fixes

### Release Checklist

1. Update version numbers in `setup.py` and `__init__.py`
2. Update `CHANGELOG.md` with release notes
3. Create release branch: `git checkout -b release/vX.Y.Z`
4. Run full test suite including performance benchmarks
5. Build and test package: `python setup.py sdist bdist_wheel`
6. Create GitHub release with release notes
7. Upload to PyPI (maintainers only)

## Financial Domain Considerations

### Model Validation

- New pricing models must be validated against analytical solutions where available
- Include references to academic papers for model implementations
- Document model assumptions and limitations clearly
- Provide convergence tests for numerical methods

### Performance Requirements

- Pricing functions should complete in <1ms for production use
- Memory usage should be minimal for large portfolios
- Numerical stability is critical - avoid operations prone to overflow/underflow

### Risk Management

- Include appropriate warnings for model limitations
- Validate input parameters and provide meaningful error messages
- Document when models may not be appropriate (e.g., American options with Black-Scholes)

## Getting Help

- **Questions**: Open a GitHub issue with the "question" label
- **Bugs**: Open a GitHub issue with the "bug" label  
- **Feature Requests**: Open a GitHub issue with the "enhancement" label
- **Discussions**: Use GitHub Discussions for broader topics

## Recognition

Contributors will be recognized in:
- `CONTRIBUTORS.md` file
- GitHub contributors page
- Release notes for significant contributions

Thank you for contributing to the Quantitative Derivatives Engine!