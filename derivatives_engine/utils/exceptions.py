"""
Custom exception classes for the derivatives pricing engine.

This module defines specific exception classes for different types
of errors that can occur during pricing, calibration, and data handling.
"""


class PricingEngineError(Exception):
    """Base exception class for all pricing engine errors."""
    pass


class ValidationError(PricingEngineError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, parameter: str = None, value=None):
        super().__init__(message)
        self.parameter = parameter
        self.value = value
    
    def __str__(self):
        if self.parameter and self.value is not None:
            return f"Validation error for parameter '{self.parameter}' with value {self.value}: {super().__str__()}"
        return super().__str__()


class ModelError(PricingEngineError):
    """Raised when there's an error in model computation."""
    
    def __init__(self, message: str, model_name: str = None):
        super().__init__(message)
        self.model_name = model_name
    
    def __str__(self):
        if self.model_name:
            return f"Model error in {self.model_name}: {super().__str__()}"
        return super().__str__()


class CalibrationError(PricingEngineError):
    """Raised when model calibration fails."""
    
    def __init__(self, message: str, model_name: str = None, convergence_info: dict = None):
        super().__init__(message)
        self.model_name = model_name
        self.convergence_info = convergence_info or {}


class DataError(PricingEngineError):
    """Raised when there's an error with market data."""
    
    def __init__(self, message: str, data_source: str = None, symbol: str = None):
        super().__init__(message)
        self.data_source = data_source
        self.symbol = symbol
    
    def __str__(self):
        parts = []
        if self.data_source:
            parts.append(f"source: {self.data_source}")
        if self.symbol:
            parts.append(f"symbol: {self.symbol}")
        
        if parts:
            return f"Data error ({', '.join(parts)}): {super().__str__()}"
        return super().__str__()


class ConvergenceError(PricingEngineError):
    """Raised when numerical methods fail to converge."""
    
    def __init__(self, message: str, method: str = None, iterations: int = None, tolerance: float = None):
        super().__init__(message)
        self.method = method
        self.iterations = iterations
        self.tolerance = tolerance
    
    def __str__(self):
        parts = []
        if self.method:
            parts.append(f"method: {self.method}")
        if self.iterations is not None:
            parts.append(f"iterations: {self.iterations}")
        if self.tolerance is not None:
            parts.append(f"tolerance: {self.tolerance}")
        
        if parts:
            return f"Convergence error ({', '.join(parts)}): {super().__str__()}"
        return super().__str__()


class ParameterError(PricingEngineError):
    """Raised when model parameters are invalid or incompatible."""
    
    def __init__(self, message: str, parameter_name: str = None, parameter_value=None):
        super().__init__(message)
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value


class NumericalError(PricingEngineError):
    """Raised when numerical computation produces invalid results."""
    
    def __init__(self, message: str, computation: str = None, result=None):
        super().__init__(message)
        self.computation = computation
        self.result = result


class ConfigurationError(PricingEngineError):
    """Raised when there's an error in configuration or setup."""
    
    def __init__(self, message: str, config_section: str = None):
        super().__init__(message)
        self.config_section = config_section


class DependencyError(PricingEngineError):
    """Raised when a required dependency is not available."""
    
    def __init__(self, message: str, package_name: str = None, feature: str = None):
        super().__init__(message)
        self.package_name = package_name
        self.feature = feature
    
    def __str__(self):
        if self.package_name and self.feature:
            return f"Dependency error: {self.package_name} required for {self.feature}: {super().__str__()}"
        elif self.package_name:
            return f"Dependency error: {self.package_name} not available: {super().__str__()}"
        return super().__str__()


# Utility functions for error handling
def validate_positive(value, name: str, allow_zero: bool = False):
    """Validate that a value is positive (or non-negative if allow_zero=True)."""
    if allow_zero and value < 0:
        raise ValidationError(f"{name} must be non-negative", name, value)
    elif not allow_zero and value <= 0:
        raise ValidationError(f"{name} must be positive", name, value)


def validate_probability(value, name: str):
    """Validate that a value is a valid probability (between 0 and 1)."""
    if not (0 <= value <= 1):
        raise ValidationError(f"{name} must be between 0 and 1", name, value)


def validate_correlation(value, name: str):
    """Validate that a value is a valid correlation coefficient (between -1 and 1)."""
    if not (-1 <= value <= 1):
        raise ValidationError(f"{name} must be between -1 and 1", name, value)


def validate_option_type(option_type: str, valid_types: list = None):
    """Validate option type parameter."""
    if valid_types is None:
        valid_types = ["call", "put"]
    
    if option_type.lower() not in [t.lower() for t in valid_types]:
        raise ValidationError(f"Option type must be one of {valid_types}", "option_type", option_type)


def validate_finite(value, name: str):
    """Validate that a value is finite (not NaN or infinite)."""
    import numpy as np
    
    if np.isnan(value) or np.isinf(value):
        raise ValidationError(f"{name} must be finite", name, value)


def handle_numerical_result(result, computation_name: str, allow_negative: bool = False):
    """Validate and handle numerical computation results."""
    import numpy as np
    
    if np.isnan(result):
        raise NumericalError(f"{computation_name} produced NaN result", computation_name, result)
    
    if np.isinf(result):
        raise NumericalError(f"{computation_name} produced infinite result", computation_name, result)
    
    if not allow_negative and result < 0:
        raise NumericalError(f"{computation_name} produced negative result", computation_name, result)
    
    return result