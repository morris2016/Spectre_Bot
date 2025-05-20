

#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Risk Manager Module Initialization

This module initializes the Risk Manager component of the QuantumSpectre Elite Trading System,
which is responsible for comprehensive risk management, position sizing, and capital preservation.
The risk management system is critical for achieving 80%+ win rates by ensuring losses are minimized
and risk is always carefully controlled.
"""

import logging
import importlib
import pkgutil
from typing import Dict, List, Any, Optional, Set, Type

from common.logger import get_logger
from common.utils import ClassRegistry

# Initialize module logger
logger = get_logger(__name__)

# Create registries for different types of risk components
position_sizers = ClassRegistry()
stop_loss_strategies = ClassRegistry()
take_profit_strategies = ClassRegistry()
exposure_managers = ClassRegistry()
circuit_breakers = ClassRegistry()
drawdown_protectors = ClassRegistry()

# Version info
__version__ = '1.0.0'

def register_all_components() -> None:
    """
    Discover and register all risk management components dynamically.
    This allows for easy extension with new risk management strategies.
    """
    logger.info("Registering all risk management components...")
    
    # Define the base package paths to scan for components
    base_paths = [
        'risk_manager.position_sizing',
        'risk_manager.stop_loss',
        'risk_manager.take_profit',
        'risk_manager.exposure',
        'risk_manager.circuit_breaker',
        'risk_manager.drawdown_protection',
    ]
    
    # Component base classes to compare against for registration
    from risk_manager.position_sizing import BasePositionSizer
    from risk_manager.stop_loss import BaseStopLossStrategy
    from risk_manager.take_profit import BaseTakeProfitStrategy
    from risk_manager.exposure import BaseExposureManager
    from risk_manager.circuit_breaker import BaseCircuitBreaker
    from risk_manager.drawdown_protection import BaseDrawdownProtector
    
    # Registration mapping
    registry_map = {
        BasePositionSizer: position_sizers,
        BaseStopLossStrategy: stop_loss_strategies,
        BaseTakeProfitStrategy: take_profit_strategies,
        BaseExposureManager: exposure_managers,
        BaseCircuitBreaker: circuit_breakers,
        BaseDrawdownProtector: drawdown_protectors
    }
    
    # Scan and register all components
    for base_path in base_paths:
        try:
            package = importlib.import_module(base_path)
            for _, name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
                if not is_pkg:
                    try:
                        module = importlib.import_module(name)
                        # Register all classes in the module that are risk components
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if isinstance(attr, type):
                                for base_class, registry in registry_map.items():
                                    if issubclass(attr, base_class) and attr != base_class:
                                        registry.register(attr)
                                        logger.debug(f"Registered {attr.__name__} to {registry}")
                    except (ImportError, AttributeError) as e:
                        logger.warning(f"Could not import module {name}: {e}")
        except ImportError as e:
            logger.warning(f"Could not import base package {base_path}: {e}")
    
    logger.info("All risk management components registered successfully")

def initialize() -> None:
    """
    Initialize the Risk Manager module, ensuring all components are registered
    and dependencies are set up.
    """
    logger.info(f"Initializing Risk Manager module v{__version__}...")
    register_all_components()
    logger.info("Risk Manager module initialized successfully")
    
    # Log the registered components for debugging
    logger.debug(f"Registered position sizers: {[cls.__name__ for cls in position_sizers.get_all()]}")
    logger.debug(f"Registered stop loss strategies: {[cls.__name__ for cls in stop_loss_strategies.get_all()]}")
    logger.debug(f"Registered take profit strategies: {[cls.__name__ for cls in take_profit_strategies.get_all()]}")
    logger.debug(f"Registered exposure managers: {[cls.__name__ for cls in exposure_managers.get_all()]}")
    logger.debug(f"Registered circuit breakers: {[cls.__name__ for cls in circuit_breakers.get_all()]}")
    logger.debug(f"Registered drawdown protectors: {[cls.__name__ for cls in drawdown_protectors.get_all()]}")

# Auto-initialize the module when imported
initialize()