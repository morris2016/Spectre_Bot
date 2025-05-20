"""
QuantumSpectre Core Infrastructure Module.

This module provides the foundation for the entire QuantumSpectre trading system,
including configuration management, service registry, and core utilities.
"""

__version__ = "1.0.0"

from .application import Application
from .config import Config, load_config
from .constants import EXCHANGE_TYPES, ORDER_TYPES, TIMEFRAMES
from .exceptions import (
    QuantumSpectreError, ConfigurationError, ExchangeError,
    ServiceError, ValidationError
)
from .logging_config import configure_logging, get_logger
from .service_registry import ServiceRegistry, service
from .utils import (
    async_retry, timeit, safe_execute, 
    create_id, current_timestamp_ms, 
    format_timestamp, serialize_to_json
)

# Initialize core components
configure_logging()
logger = get_logger(__name__)
config = load_config()
service_registry = ServiceRegistry()

__all__ = [
    "Application",
    "Config", "load_config", "config",
    "EXCHANGE_TYPES", "ORDER_TYPES", "TIMEFRAMES",
    "QuantumSpectreError", "ConfigurationError", "ExchangeError",
    "ServiceError", "ValidationError",
    "configure_logging", "get_logger", "logger",
    "ServiceRegistry", "service", "service_registry",
    "async_retry", "timeit", "safe_execute",
    "create_id", "current_timestamp_ms",
    "format_timestamp", "serialize_to_json",
]

logger.info(f"QuantumSpectre Core v{__version__} initialized")
