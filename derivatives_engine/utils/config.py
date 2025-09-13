"""
Configuration management for the derivatives pricing engine.

This module provides centralized configuration management with support for
environment variables, configuration files, and default settings.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, asdict
from .exceptions import ConfigurationError
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class NumericalConfig:
    """Configuration for numerical computations."""
    default_monte_carlo_paths: int = 100000
    default_monte_carlo_steps: int = 252
    default_fd_grid_points: int = 100
    default_fd_time_steps: int = 1000
    default_binomial_steps: int = 1000
    convergence_tolerance: float = 1e-8
    max_iterations: int = 10000
    random_seed: Optional[int] = None


@dataclass
class CacheConfig:
    """Configuration for caching behavior."""
    enable_pricing_cache: bool = True
    enable_data_cache: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    max_cache_size: int = 1000
    cache_directory: str = "cache"


@dataclass
class LoggingConfig:
    """Configuration for logging behavior."""
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    log_directory: str = "logs"
    max_log_file_size_mb: int = 10
    max_log_files: int = 5


@dataclass
class DataConfig:
    """Configuration for data handling."""
    default_data_source: str = "yahoo"
    data_cache_days: int = 1
    retry_attempts: int = 3
    request_timeout: int = 30
    rate_limit_delay: float = 0.1


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    enable_jit: bool = True
    enable_vectorization: bool = True
    enable_parallel_processing: bool = True
    max_threads: Optional[int] = None
    memory_limit_mb: Optional[int] = None


@dataclass
class PricingEngineConfig:
    """Main configuration class for the pricing engine."""
    numerical: NumericalConfig
    cache: CacheConfig
    logging: LoggingConfig
    data: DataConfig
    performance: PerformanceConfig
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.numerical.default_monte_carlo_paths <= 0:
            raise ConfigurationError("Monte Carlo paths must be positive")
        
        if self.numerical.convergence_tolerance <= 0:
            raise ConfigurationError("Convergence tolerance must be positive")
        
        if self.cache.cache_ttl_seconds < 0:
            raise ConfigurationError("Cache TTL must be non-negative")
        
        if self.data.retry_attempts < 0:
            raise ConfigurationError("Retry attempts must be non-negative")
        
        logger.debug("Configuration validation passed")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PricingEngineConfig':
        """Create configuration from dictionary."""
        return cls(
            numerical=NumericalConfig(**config_dict.get('numerical', {})),
            cache=CacheConfig(**config_dict.get('cache', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            data=DataConfig(**config_dict.get('data', {})),
            performance=PerformanceConfig(**config_dict.get('performance', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'numerical': asdict(self.numerical),
            'cache': asdict(self.cache),
            'logging': asdict(self.logging),
            'data': asdict(self.data),
            'performance': asdict(self.performance)
        }
    
    def save_to_file(self, file_path: Union[str, Path]):
        """Save configuration to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {file_path}")
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'PricingEngineConfig':
        """Load configuration from JSON file."""
        try:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
            logger.info(f"Configuration loaded from {file_path}")
            return cls.from_dict(config_dict)
        except FileNotFoundError:
            logger.warning(f"Configuration file {file_path} not found, using defaults")
            return cls.get_default()
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}")


class ConfigManager:
    """Singleton configuration manager."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._load_config()
    
    def _load_config(self):
        """Load configuration from various sources."""
        # Try to load from file
        config_file = os.environ.get('PRICING_ENGINE_CONFIG', 'config/pricing_engine.json')
        
        if os.path.exists(config_file):
            try:
                self._config = PricingEngineConfig.load_from_file(config_file)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_file}: {e}")
                self._config = self._get_default_config()
        else:
            self._config = self._get_default_config()
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        logger.info("Configuration loaded successfully")
    
    def _get_default_config(self) -> PricingEngineConfig:
        """Get default configuration."""
        return PricingEngineConfig(
            numerical=NumericalConfig(),
            cache=CacheConfig(),
            logging=LoggingConfig(),
            data=DataConfig(),
            performance=PerformanceConfig()
        )
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        env_mappings = {
            'PRICING_ENGINE_LOG_LEVEL': ('logging', 'log_level'),
            'PRICING_ENGINE_MC_PATHS': ('numerical', 'default_monte_carlo_paths'),
            'PRICING_ENGINE_CACHE_TTL': ('cache', 'cache_ttl_seconds'),
            'PRICING_ENGINE_DATA_SOURCE': ('data', 'default_data_source'),
            'PRICING_ENGINE_ENABLE_JIT': ('performance', 'enable_jit'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                # Convert string values to appropriate types
                section_obj = getattr(self._config, section)
                current_value = getattr(section_obj, key)
                
                if isinstance(current_value, bool):
                    value = value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(current_value, int):
                    value = int(value)
                elif isinstance(current_value, float):
                    value = float(value)
                
                setattr(section_obj, key, value)
                logger.debug(f"Applied environment override: {env_var} = {value}")
    
    @property
    def config(self) -> PricingEngineConfig:
        """Get the current configuration."""
        return self._config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        for section_name, section_updates in updates.items():
            if hasattr(self._config, section_name):
                section = getattr(self._config, section_name)
                for key, value in section_updates.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
                    else:
                        logger.warning(f"Unknown configuration key: {section_name}.{key}")
            else:
                logger.warning(f"Unknown configuration section: {section_name}")
        
        # Re-validate after updates
        self._config._validate_config()
        logger.info("Configuration updated")
    
    def reset_to_defaults(self):
        """Reset configuration to default values."""
        self._config = self._get_default_config()
        logger.info("Configuration reset to defaults")


# Global configuration instance
config_manager = ConfigManager()

# Convenience function to get configuration
def get_config() -> PricingEngineConfig:
    """Get the global configuration instance."""
    return config_manager.config


# Configuration decorators
def with_config(config_section: str):
    """Decorator to inject configuration section into function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            config = get_config()
            section = getattr(config, config_section)
            kwargs[config_section + '_config'] = section
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_config(config_path: str):
    """Decorator to ensure a specific configuration value is set."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            config = get_config()
            sections = config_path.split('.')
            
            current = config
            for section in sections:
                if hasattr(current, section):
                    current = getattr(current, section)
                else:
                    raise ConfigurationError(f"Required configuration {config_path} not found")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator