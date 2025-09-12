"""
Automatic differentiation implementation for Greeks calculation.

This module provides forward-mode automatic differentiation using dual numbers
to efficiently compute option Greeks without numerical approximation errors.
"""

import numpy as np
from typing import Union
from scipy import stats


class DualNumber:
    """
    Dual number implementation for automatic differentiation.
    
    A dual number has the form a + b*ε where ε² = 0.
    This allows exact computation of derivatives using the chain rule.
    """
    
    def __init__(self, real: float, dual: float = 0.0):
        """
        Initialize dual number.
        
        Args:
            real: Real part (function value)
            dual: Dual part (derivative value)
        """
        self.real = float(real)
        self.dual = float(dual)
    
    def __add__(self, other: Union['DualNumber', float]) -> 'DualNumber':
        """Addition: (a + b*ε) + (c + d*ε) = (a+c) + (b+d)*ε"""
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real, self.dual + other.dual)
        return DualNumber(self.real + other, self.dual)
    
    def __radd__(self, other: Union['DualNumber', float]) -> 'DualNumber':
        """Right addition"""
        return self.__add__(other)
    
    def __sub__(self, other: Union['DualNumber', float]) -> 'DualNumber':
        """Subtraction: (a + b*ε) - (c + d*ε) = (a-c) + (b-d)*ε"""
        if isinstance(other, DualNumber):
            return DualNumber(self.real - other.real, self.dual - other.dual)
        return DualNumber(self.real - other, self.dual)
    
    def __rsub__(self, other: Union['DualNumber', float]) -> 'DualNumber':
        """Right subtraction"""
        return DualNumber(other - self.real, -self.dual)
    
    def __mul__(self, other: Union['DualNumber', float]) -> 'DualNumber':
        """Multiplication: (a + b*ε) * (c + d*ε) = ac + (ad + bc)*ε"""
        if isinstance(other, DualNumber):
            return DualNumber(
                self.real * other.real,
                self.real * other.dual + self.dual * other.real
            )
        return DualNumber(self.real * other, self.dual * other)
    
    def __rmul__(self, other: Union['DualNumber', float]) -> 'DualNumber':
        """Right multiplication"""
        return self.__mul__(other)
    
    def __truediv__(self, other: Union['DualNumber', float]) -> 'DualNumber':
        """Division: (a + b*ε) / (c + d*ε) = a/c + (bc - ad)/(c²)*ε"""
        if isinstance(other, DualNumber):
            return DualNumber(
                self.real / other.real,
                (self.dual * other.real - self.real * other.dual) / (other.real ** 2)
            )
        return DualNumber(self.real / other, self.dual / other)
    
    def __rtruediv__(self, other: Union['DualNumber', float]) -> 'DualNumber':
        """Right division"""
        return DualNumber(other / self.real, -other * self.dual / (self.real ** 2))
    
    def __pow__(self, other: Union['DualNumber', float]) -> 'DualNumber':
        """Power: (a + b*ε)^n = a^n + n*a^(n-1)*b*ε"""
        if isinstance(other, DualNumber):
            # Complex case: f^g where both are dual numbers
            # d/dx[f^g] = f^g * (g' * ln(f) + g * f'/f)
            log_f = log(self)
            return exp(other * log_f)
        else:
            # Simple case: f^c where c is constant
            return DualNumber(
                self.real ** other,
                other * (self.real ** (other - 1)) * self.dual
            )
    
    def __neg__(self) -> 'DualNumber':
        """Negation: -(a + b*ε) = -a + (-b)*ε"""
        return DualNumber(-self.real, -self.dual)
    
    def __abs__(self) -> 'DualNumber':
        """Absolute value (non-differentiable at 0)"""
        if self.real >= 0:
            return DualNumber(self.real, self.dual)
        else:
            return DualNumber(-self.real, -self.dual)
    
    def __repr__(self) -> str:
        """String representation"""
        return f"DualNumber({self.real}, {self.dual})"
    
    def __str__(self) -> str:
        """User-friendly string representation"""
        return f"{self.real} + {self.dual}ε"


# Mathematical functions for dual numbers

def exp(x: Union[DualNumber, float]) -> Union[DualNumber, float]:
    """Exponential function: d/dx[e^x] = e^x"""
    if isinstance(x, DualNumber):
        exp_real = np.exp(x.real)
        return DualNumber(exp_real, exp_real * x.dual)
    return np.exp(x)


def log(x: Union[DualNumber, float]) -> Union[DualNumber, float]:
    """Natural logarithm: d/dx[ln(x)] = 1/x"""
    if isinstance(x, DualNumber):
        if x.real <= 0:
            raise ValueError("Logarithm of non-positive number")
        return DualNumber(np.log(x.real), x.dual / x.real)
    return np.log(x)


def sqrt(x: Union[DualNumber, float]) -> Union[DualNumber, float]:
    """Square root: d/dx[√x] = 1/(2√x)"""
    if isinstance(x, DualNumber):
        if x.real < 0:
            raise ValueError("Square root of negative number")
        sqrt_real = np.sqrt(x.real)
        return DualNumber(sqrt_real, x.dual / (2 * sqrt_real))
    return np.sqrt(x)


def sin(x: Union[DualNumber, float]) -> Union[DualNumber, float]:
    """Sine function: d/dx[sin(x)] = cos(x)"""
    if isinstance(x, DualNumber):
        return DualNumber(np.sin(x.real), np.cos(x.real) * x.dual)
    return np.sin(x)


def cos(x: Union[DualNumber, float]) -> Union[DualNumber, float]:
    """Cosine function: d/dx[cos(x)] = -sin(x)"""
    if isinstance(x, DualNumber):
        return DualNumber(np.cos(x.real), -np.sin(x.real) * x.dual)
    return np.cos(x)


def norm_cdf(x: Union[DualNumber, float]) -> Union[DualNumber, float]:
    """
    Standard normal cumulative distribution function.
    d/dx[Φ(x)] = φ(x) = (1/√(2π)) * e^(-x²/2)
    """
    if isinstance(x, DualNumber):
        cdf_real = stats.norm.cdf(x.real)
        pdf_real = stats.norm.pdf(x.real)
        return DualNumber(cdf_real, pdf_real * x.dual)
    return stats.norm.cdf(x)


def norm_pdf(x: Union[DualNumber, float]) -> Union[DualNumber, float]:
    """
    Standard normal probability density function.
    d/dx[φ(x)] = -x * φ(x)
    """
    if isinstance(x, DualNumber):
        pdf_real = stats.norm.pdf(x.real)
        return DualNumber(pdf_real, -x.real * pdf_real * x.dual)
    return stats.norm.pdf(x)


def max_dual(x: Union[DualNumber, float], y: Union[DualNumber, float]) -> Union[DualNumber, float]:
    """
    Maximum function for dual numbers.
    Note: Not differentiable when x = y, but we handle this case.
    """
    if isinstance(x, DualNumber) or isinstance(y, DualNumber):
        # Convert to dual numbers if needed
        if not isinstance(x, DualNumber):
            x = DualNumber(x, 0.0)
        if not isinstance(y, DualNumber):
            y = DualNumber(y, 0.0)
        
        if x.real > y.real:
            return x
        elif x.real < y.real:
            return y
        else:
            # When equal, use average of derivatives (subdifferential)
            return DualNumber(x.real, (x.dual + y.dual) / 2)
    return max(x, y)


def min_dual(x: Union[DualNumber, float], y: Union[DualNumber, float]) -> Union[DualNumber, float]:
    """
    Minimum function for dual numbers.
    Note: Not differentiable when x = y, but we handle this case.
    """
    if isinstance(x, DualNumber) or isinstance(y, DualNumber):
        # Convert to dual numbers if needed
        if not isinstance(x, DualNumber):
            x = DualNumber(x, 0.0)
        if not isinstance(y, DualNumber):
            y = DualNumber(y, 0.0)
        
        if x.real < y.real:
            return x
        elif x.real > y.real:
            return y
        else:
            # When equal, use average of derivatives (subdifferential)
            return DualNumber(x.real, (x.dual + y.dual) / 2)
    return min(x, y)


# Utility functions

def make_dual(value: float, derivative: float = 1.0) -> DualNumber:
    """
    Create a dual number with specified value and derivative.
    
    Args:
        value: The function value
        derivative: The derivative value (default: 1.0 for primary variable)
        
    Returns:
        DualNumber with specified real and dual parts
    """
    return DualNumber(value, derivative)


def extract_derivative(result: Union[DualNumber, float]) -> float:
    """
    Extract the derivative from a dual number result.
    
    Args:
        result: Either a DualNumber or regular float
        
    Returns:
        The derivative value, or 0.0 if input is not a DualNumber
    """
    if isinstance(result, DualNumber):
        return result.dual
    return 0.0


def extract_value(result: Union[DualNumber, float]) -> float:
    """
    Extract the function value from a dual number result.
    
    Args:
        result: Either a DualNumber or regular float
        
    Returns:
        The function value
    """
    if isinstance(result, DualNumber):
        return result.real
    return float(result)


def is_dual(x: Union[DualNumber, float]) -> bool:
    """Check if input is a dual number."""
    return isinstance(x, DualNumber)


# Higher-order derivatives (using dual numbers recursively)

def second_derivative(func, x: float, h: float = 1e-8) -> float:
    """
    Compute second derivative using dual numbers.
    
    This uses the fact that f''(x) can be computed by applying
    automatic differentiation twice.
    
    Args:
        func: Function to differentiate
        x: Point at which to evaluate
        h: Step size for finite difference approximation
        
    Returns:
        Second derivative value
    """
    # Use finite difference on the first derivative
    x_dual_plus = DualNumber(x + h, 1.0)
    x_dual_minus = DualNumber(x - h, 1.0)
    
    f_prime_plus = func(x_dual_plus).dual
    f_prime_minus = func(x_dual_minus).dual
    
    return (f_prime_plus - f_prime_minus) / (2 * h)