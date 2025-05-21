"""
Configuration management for the QuantumSpectre trading system.

This module handles loading configuration from various sources (environment variables, 
config files) and provides a unified interface for accessing configuration values.
"""

import json
import os
import pathlib
import re
from typing import Any, Dict, List, Optional, Union

import yaml
from dotenv import load_dotenv

from .exceptions import ConfigurationError
from .logging_config import get_logger

logger = get_logger(__name__)

class Config:
    """
    Configuration manager for the QuantumSpectre trading system.
    
    This class provides a centralized repository for configuration values
    with support for hierarchical configurations, environment overrides,
    and configuration validation.
    """
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize a new Config instance.
        
        Args:
            config_dict: Optional initial configuration dictionary
        """
        self._config = config_dict or {}
        self._dynamic_config = {}
        self._listeners = []
        logger.debug("Configuration initialized")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation for nested configs)
            default: Default value to return if key is not found
            
        Returns:
            The configuration value
        """
        # Check dynamic config first (runtime updates)
        value = self._get_from_dict(self._dynamic_config, key)
        if value is not None:
            return value
        
        # Then check static config
        value = self._get_from_dict(self._config, key)
        if value is not None:
            return value
        
        return default
    
    def _get_from_dict(self, config_dict: Dict[str, Any], key: str) -> Any:
        """
        Get a value from a dictionary using dot notation.
        
        Args:
            config_dict: Dictionary to get value from
            key: Key using dot notation (e.g. 'database.host')
            
        Returns:
            The value or None if not found
        """
        keys = key.split('.')
        result = config_dict
        
        for k in keys:
            if isinstance(result, dict) and k in result:
                result = result[k]
            else:
                return None
        
        return result
    
    def set(self, key: str, value: Any, persist: bool = False) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
            persist: Whether to write to persistent storage
        """
        # Always update in-memory dynamic config
        self._set_in_dict(self._dynamic_config, key, value)
        
        # Optionally persist to storage
        if persist:
            # If we implement disk persistence, it would go here
            logger.warning("Persistence of configuration not implemented")
        
        # Notify listeners
        for listener in self._listeners:
            try:
                listener(key, value)
            except Exception as e:
                logger.error(f"Error in config listener: {str(e)}", exc_info=True)
    
    def _set_in_dict(self, config_dict: Dict[str, Any], key: str, value: Any) -> None:
        """
        Set a value in a dictionary using dot notation.
        
        Args:
            config_dict: Dictionary to set value in
            key: Key using dot notation (e.g. 'database.host')
            value: Value to set
        """
        keys = key.split('.')
        current = config_dict
        
        # Navigate to the right level
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the value
        current[keys[-1]] = value
    
    def add_listener(self, listener) -> None:
        """
        Add a listener function that's called when config changes.
        
        Args:
            listener: Function that takes (key, value) parameters
        """
        if callable(listener) and listener not in self._listeners:
            self._listeners.append(listener)
    
    def remove_listener(self, listener) -> None:
        """
        Remove a configuration change listener.
        
        Args:
            listener: The listener function to remove
        """
        if listener in self._listeners:
            self._listeners.remove(listener)
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Get the complete configuration as a dictionary.
        
        Returns:
            A merged dictionary of all configuration values
        """
        # Create a deep copy of the static config
        result = json.loads(json.dumps(self._config))
        
        # Merge with dynamic config
        self._deep_merge(result, self._dynamic_config)
        
        return result
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Deep merge source dictionary into target dictionary.
        
        Args:
            target: Target dictionary to merge into
            source: Source dictionary with values to merge
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def validate(self, schema: Dict[str, Any]) -> List[str]:
        """
        Validate the configuration against a schema.
        
        Args:
            schema: Validation schema with required keys and types
            
        Returns:
            List of validation errors, empty if valid
        """
        errors = []
        config = self.as_dict()
        
        def validate_item(path, schema_item, config_dict):
            if isinstance(schema_item, dict) and schema_item.get('_type') == 'object':
                # Validate object
                properties = schema_item.get('properties', {})
                required = schema_item.get('required', [])
                
                # Check required fields
                for field in required:
                    field_path = f"{path}.{field}" if path else field
                    if field not in config_dict:
                        errors.append(f"Missing required field: {field_path}")
                
                # Validate properties
                for prop_name, prop_schema in properties.items():
                    prop_path = f"{path}.{prop_name}" if path else prop_name
                    if prop_name in config_dict:
                        validate_item(prop_path, prop_schema, config_dict.get(prop_name, {}))
            
            elif isinstance(schema_item, dict) and '_type' in schema_item:
                # Validate value type
                expected_type = schema_item['_type']
                value = self.get(path)
                
                if value is not None:
                    valid = False
                    if expected_type == 'string' and isinstance(value, str):
                        valid = True
                    elif expected_type == 'number' and isinstance(value, (int, float)):
                        valid = True
                    elif expected_type == 'boolean' and isinstance(value, bool):
                        valid = True
                    elif expected_type == 'array' and isinstance(value, list):
                        valid = True
                    
                    if not valid:
                        errors.append(f"Type mismatch for {path}: expected {expected_type}")
                
                # Check constraints
                if 'min' in schema_item and value < schema_item['min']:
                    errors.append(f"Value for {path} is below minimum: {schema_item['min']}")
                if 'max' in schema_item and value > schema_item['max']:
                    errors.append(f"Value for {path} exceeds maximum: {schema_item['max']}")
                if 'pattern' in schema_item and not re.match(schema_item['pattern'], value):
                    errors.append(f"Value for {path} does not match pattern: {schema_item['pattern']}")
        
        # Validate the entire config
        for key, schema_item in schema.items():
            validate_item(key, schema_item, self._get_from_dict(config, key) or {})
        
        return errors


def load_config(config_path: Optional[str] = None, env_file: Optional[str] = None) -> Config:
    """
    Load configuration from files and environment variables.
    
    Args:
        config_path: Path to the configuration file (YAML or JSON)
        env_file: Path to the .env file for environment variables
        
    Returns:
        Config instance with loaded configuration
        
    Raises:
        ConfigurationError: If configuration loading fails
    """
    config_dict = {}
    
    # Try to determine config file path if not provided
    if not config_path:
        # Look in common locations
        potential_paths = [
            "config.yaml",
            "config.yml", 
            "config.json",
            "config/config.yaml",
            "config/config.yml",
            "config/config.json",
            os.path.expanduser("~/.quantum_spectre/config.yaml"),
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    # Load .env file if provided or exists
    env_paths = [env_file] if env_file else [".env", "config/.env"]
    for env_path in env_paths:
        if os.path.exists(env_path):
            logger.debug(f"Loading environment from {env_path}")
            load_dotenv(env_path)
            break
    
    # Load configuration file if exists
    if config_path and os.path.exists(config_path):
        logger.info(f"Loading configuration from {config_path}")
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith(('.yaml', '.yml')):
                    config_dict = yaml.safe_load(f)
                elif config_path.endswith('.json'):
                    config_dict = json.load(f)
                else:
                    raise ConfigurationError(f"Unsupported configuration file format: {config_path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration file: {str(e)}") from e
    
    # Overlay environment variables (QS_* or QUANTUM_SPECTRE_*)
    logger.debug("Applying environment variable configuration")
    env_prefix = ["QS_", "QUANTUM_SPECTRE_"]
    for key, value in os.environ.items():
        config_key = None
        
        # Check if key matches any of our prefixes
        for prefix in env_prefix:
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix):].lower()
                # Convert SNAKE_CASE to dot.notation
                config_key = config_key.replace('_', '.')
                break
        
        if config_key:
            # Try to parse value as JSON for complex types
            try:
                parsed_value = json.loads(value)
                logger.debug(f"Parsed environment variable {key} as JSON")
                set_nested_value(config_dict, config_key, parsed_value)
            except json.JSONDecodeError:
                # Not JSON, use as string
                set_nested_value(config_dict, config_key, value)
    
    # Create and return Config instance
    return Config(config_dict)


def set_nested_value(config_dict: Dict[str, Any], key: str, value: Any) -> None:
    """
    Set a nested value in a dictionary using dot notation.
    
    Args:
        config_dict: Dictionary to modify
        key: Key in dot notation (e.g., 'database.host')
        value: Value to set
    """
    keys = key.split('.')
    current = config_dict
    
    # Navigate to the right level
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        elif not isinstance(current[k], dict):
            # If the path exists but is not a dict, convert it
            current[k] = {}
        current = current[k]
    
    # Set the value
    current[keys[-1]] = value
