"""
Main application class for the QuantumSpectre trading system.

This class manages the lifecycle of the application and coordinates the
initialization and shutdown of all services.
"""

import asyncio
import signal
import sys
from typing import Dict, List, Optional, Set, Type, Union

from .config import Config
from .exceptions import ServiceError
from .logging_config import get_logger
from .service_registry import ServiceRegistry

logger = get_logger(__name__)

class Application:
    """
    Main application class for the QuantumSpectre trading system.
    
    This class handles:
    - Initialization of all services
    - Graceful shutdown on termination signals
    - Coordination between services
    - Application lifecycle management
    """
    
    def __init__(self, config: Config, service_registry: ServiceRegistry):
        """
        Initialize the application with configuration and service registry.
        
        Args:
            config: Application configuration
            service_registry: Registry of available services
        """
        self.config = config
        self.service_registry = service_registry
        self.services = {}
        self.running_tasks = set()
        self.shutdown_event = asyncio.Event()
        self.started = False
        self._configure_signal_handlers()
        
        logger.info("Application initialized")
    
    def _configure_signal_handlers(self) -> None:
        """Configure handlers for termination signals."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._handle_signal)
        
        logger.debug("Signal handlers configured")
    
    def _handle_signal(self, signum, frame) -> None:
        """
        Handle termination signals by initiating a graceful shutdown.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        sig_name = signal.Signals(signum).name
        logger.info(f"Received signal {sig_name}, initiating graceful shutdown")
        
        if not self.shutdown_event.is_set():
            # Use call_soon_threadsafe as signal handlers run in the main thread
            asyncio.get_event_loop().call_soon_threadsafe(
                self.shutdown_event.set
            )
        else:
            logger.warning("Forced shutdown initiated with SIGTERM")
            sys.exit(1)
    
    async def start(self, service_names: Optional[List[str]] = None) -> None:
        """
        Start the application and initialize all required services.
        
        Args:
            service_names: Optional list of service names to start.
                           If None, all registered services will be started.
        
        Raises:
            ServiceError: If a required service fails to start
        """
        if self.started:
            logger.warning("Application already started")
            return
        
        logger.info("Starting QuantumSpectre Trading System")
        
        try:
            # Start all services
            services_to_start = service_names or self.service_registry.list_services()
            
            # Resolve dependencies and determine start order
            start_order = self._resolve_dependencies(services_to_start)
            
            for service_name in start_order:
                service_cls = self.service_registry.get_service(service_name)
                logger.info(f"Initializing service: {service_name}")
                
                # Create and start the service
                service = service_cls(self.config)
                await service.initialize()
                self.services[service_name] = service
                
                # Start service's background tasks if any
                if hasattr(service, "run") and callable(service.run):
                    task = asyncio.create_task(
                        self._run_service_with_error_handling(service_name, service)
                    )
                    self.running_tasks.add(task)
                    task.add_done_callback(self.running_tasks.discard)
            
            self.started = True
            logger.info("All services started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start application: {str(e)}", exc_info=True)
            await self.shutdown()
            raise ServiceError(f"Application startup failed: {str(e)}") from e
    
    async def _run_service_with_error_handling(self, service_name: str, service) -> None:
        """
        Run a service with error handling to prevent crashes.
        
        Args:
            service_name: Name of the service
            service: Service instance
        """
        try:
            logger.debug(f"Starting service runtime: {service_name}")
            await service.run(self.shutdown_event)
            logger.debug(f"Service completed normally: {service_name}")
        except asyncio.CancelledError:
            logger.info(f"Service {service_name} task cancelled")
        except Exception as e:
            logger.error(f"Error in service {service_name}: {str(e)}", exc_info=True)
            # Don't bring down the whole system for a single service failure
            # But do signal that we need to shut down if configured to do so
            if self.config.get("core.fail_fast", False):
                logger.critical(f"Initiating application shutdown due to service failure: {service_name}")
                self.shutdown_event.set()
    
    def _resolve_dependencies(self, service_names: List[str]) -> List[str]:
        """
        Resolve service dependencies to determine correct startup order.
        
        Args:
            service_names: List of service names to resolve
            
        Returns:
            Ordered list of service names
            
        Raises:
            ServiceError: If circular dependencies are detected
        """
        # Build dependency graph
        graph = {}
        for name in service_names:
            service_cls = self.service_registry.get_service(name)
            dependencies = getattr(service_cls, "dependencies", [])
            graph[name] = dependencies
        
        # Topological sort
        result = []
        temp_mark = set()
        perm_mark = set()
        
        def visit(node):
            if node in temp_mark:
                cycles = " -> ".join(temp_mark)
                raise ServiceError(f"Circular dependency detected: {cycles} -> {node}")
            
            if node not in perm_mark:
                temp_mark.add(node)
                for dep in graph.get(node, []):
                    if dep not in graph:
                        raise ServiceError(f"Service {node} depends on {dep}, but {dep} is not registered")
                    visit(dep)
                
                temp_mark.remove(node)
                perm_mark.add(node)
                result.append(node)
        
        for node in graph:
            if node not in perm_mark:
                visit(node)
        
        return list(reversed(result))
    
    async def shutdown(self) -> None:
        """
        Gracefully shut down all services and the application.
        """
        if not self.started:
            logger.warning("Application shutdown called but it was not running")
            return
        
        logger.info("Shutting down application")
        
        # Cancel all running tasks
        if self.running_tasks:
            for task in self.running_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for all tasks to complete
            if self.running_tasks:
                await asyncio.gather(*self.running_tasks, return_exceptions=True)
        
        # Shutdown services in reverse order of startup
        for service_name in reversed(list(self.services.keys())):
            service = self.services[service_name]
            try:
                logger.info(f"Shutting down service: {service_name}")
                await service.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down service {service_name}: {str(e)}", exc_info=True)
        
        self.services.clear()
        self.started = False
        logger.info("Application shutdown complete")
    
    def get_service(self, service_name: str):
        """
        Get a running service instance by name.
        
        Args:
            service_name: Name of the service to retrieve
            
        Returns:
            The service instance
            
        Raises:
            ServiceError: If the service is not running
        """
        if service_name not in self.services:
            raise ServiceError(f"Service {service_name} is not running")
        
        return self.services[service_name]
    
    async def wait_for_shutdown(self) -> None:
        """
        Wait for the application to be shut down.
        """
        await self.shutdown_event.wait()
        logger.info("Shutdown event received")
        await self.shutdown()

    async def run_forever(self) -> None:
        """
        Run the application until a shutdown signal is received.
        """
        if not self.started:
            await self.start()
        
        try:
            await self.wait_for_shutdown()
        except asyncio.CancelledError:
            logger.info("Application run cancelled")
            await self.shutdown()
