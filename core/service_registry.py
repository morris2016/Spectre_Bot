"""
Service registry for the QuantumSpectre trading system.

This module provides a central registry for services to enable dependency injection
and service discovery throughout the application.
"""

import inspect
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type

from .exceptions import ServiceError


class ServiceRegistry:
    """
    Registry for services that allows for dynamic service discovery and dependency injection.
    
    The ServiceRegistry maintains a mapping of service names to their implementations,
    allowing for looser coupling between components.
    """
    
    def __init__(self):
        """Initialize a new ServiceRegistry."""
        self._services = {}
        self._instances = {}
    
    def register(self, name: str, service_class: Type) -> None:
        """
        Register a service class with the registry.
        
        Args:
            name: Unique name for the service
            service_class: Class implementing the service
            
        Raises:
            ServiceError: If a service with the same name is already registered
        """
        if name in self._services:
            raise ServiceError(f"Service '{name}' is already registered")
        
        self._services[name] = service_class
    
    def get_service(self, name: str) -> Type:
        """
        Get the service class by name.
        
        Args:
            name: Name of the service
            
        Returns:
            The service class
            
        Raises:
            ServiceError: If the service is not registered
        """
        if name not in self._services:
            raise ServiceError(f"Service '{name}' is not registered")
        
        return self._services[name]
    
    def get_instance(self, name: str, config) -> Any:
        """
        Get or create an instance of a service.
        
        Args:
            name: Name of the service
            config: Configuration to pass to the service constructor
            
        Returns:
            An instance of the service
            
        Raises:
            ServiceError: If the service is not registered
        """
        if name not in self._instances:
            service_cls = self.get_service(name)
            self._instances[name] = service_cls(config)
        
        return self._instances[name]
    
    def list_services(self) -> List[str]:
        """
        Get a list of all registered service names.
        
        Returns:
            List of service names
        """
        return list(self._services.keys())
    
    def deregister(self, name: str) -> None:
        """
        Remove a service from the registry.
        
        Args:
            name: Name of the service to remove
            
        Raises:
            ServiceError: If the service is not registered
        """
        if name not in self._services:
            raise ServiceError(f"Service '{name}' is not registered")
        
        del self._services[name]
        if name in self._instances:
            del self._instances[name]
    
    def clear(self) -> None:
        """Remove all services from the registry."""
        self._services.clear()
        self._instances.clear()


def service(name: Optional[str] = None, dependencies: Optional[List[str]] = None):
    """
    Decorator to register a class as a service.
    
    Args:
        name: Optional name for the service (defaults to class name)
        dependencies: Optional list of service dependencies
        
    Returns:
        Decorated class
    
    Example:
        @service(name="data_service", dependencies=["database"])
        class DataService:
            def __init__(self, config):
                self.config = config
    """
    def decorator(cls):
        # Set service name
        service_name = name or cls.__name__.lower()
        
        # Add dependencies to class if provided
        if dependencies:
            cls.dependencies = dependencies
        elif not hasattr(cls, 'dependencies'):
            cls.dependencies = []
        
        # Register the function to auto-register with ServiceRegistry
        cls._service_name = service_name
        
        @wraps(cls)
        def wrapper(*args, **kwargs):
            return cls(*args, **kwargs)
        
        return cls
    
    # Handle case where decorator is used without arguments
    if callable(name):
        cls = name
        name = None
        return decorator(cls)
    
    return decorator


# Create global service finder
class ServiceFinder:
    """
    Utility to find services in modules to support auto-discovery.
    """
    
    @staticmethod
    def find_services_in_module(module) -> Dict[str, Type]:
        """
        Find all classes decorated with @service in a module.
        
        Args:
            module: Module to scan
            
        Returns:
            Dictionary of service names to service classes
        """
        services = {}
        
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and hasattr(obj, '_service_name'):
                services[obj._service_name] = obj
        
        return services
    
    @staticmethod
    def register_services_from_module(registry: ServiceRegistry, module) -> int:
        """
        Find and register all services in a module.
        
        Args:
            registry: ServiceRegistry to register services with
            module: Module to scan
            
        Returns:
            Number of services registered
        """
        services = ServiceFinder.find_services_in_module(module)
        
        for name, service_cls in services.items():
            registry.register(name, service_cls)
        
        return len(services)
