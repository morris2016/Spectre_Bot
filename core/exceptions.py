"""
Exception classes for the QuantumSpectre trading system.

This module defines a hierarchy of exception classes that are used
throughout the system to provide consistent error handling.
"""

class QuantumSpectreError(Exception):
    """Base exception class for all QuantumSpectre errors."""
    
    def __init__(self, message: str = "An error occurred in the QuantumSpectre system"):
        self.message = message
        super().__init__(self.message)


class ConfigurationError(QuantumSpectreError):
    """Raised when there's an error in the system configuration."""
    
    def __init__(self, message: str = "Configuration error"):
        super().__init__(f"Configuration error: {message}")


class ServiceError(QuantumSpectreError):
    """Raised when there's an error with a system service."""
    
    def __init__(self, message: str = "Service error", service_name: str = None):
        msg = f"Service error"
        if service_name:
            msg = f"Service error in {service_name}"
        super().__init__(f"{msg}: {message}")


class ExchangeError(QuantumSpectreError):
    """Raised when there's an error communicating with an exchange."""
    
    def __init__(self, message: str = "Exchange error", exchange_name: str = None, 
                 status_code: int = None, response: str = None):
        msg = f"Exchange error"
        if exchange_name:
            msg = f"Exchange error ({exchange_name})"
        
        details = []
        if status_code:
            details.append(f"status: {status_code}")
        if response:
            # Truncate response if too long
            resp = response if len(response) < 100 else f"{response[:97]}..."
            details.append(f"response: {resp}")
        
        detail_str = ", ".join(details)
        if detail_str:
            msg = f"{msg} [{detail_str}]"
        
        super().__init__(f"{msg}: {message}")
        
        self.exchange_name = exchange_name
        self.status_code = status_code
        self.response = response


class ValidationError(QuantumSpectreError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str = "Validation error", field: str = None):
        msg = f"Validation error"
        if field:
            msg = f"Validation error for field '{field}'"
        super().__init__(f"{msg}: {message}")
        self.field = field


class AuthenticationError(QuantumSpectreError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(f"Authentication error: {message}")


class AuthorizationError(QuantumSpectreError):
    """Raised when a user doesn't have permission for an operation."""
    
    def __init__(self, message: str = "Not authorized", resource: str = None, action: str = None):
        msg = f"Authorization error"
        if resource and action:
            msg = f"Authorization error: Not permitted to {action} on {resource}"
        super().__init__(f"{msg}: {message}")


class DataError(QuantumSpectreError):
    """Raised when there's an issue with data processing or access."""
    
    def __init__(self, message: str = "Data error", data_type: str = None):
        msg = f"Data error"
        if data_type:
            msg = f"Data error for {data_type}"
        super().__init__(f"{msg}: {message}")


class NetworkError(QuantumSpectreError):
    """Raised when there's a network-related error."""
    
    def __init__(self, message: str = "Network error", host: str = None, 
                 timeout: bool = False, retry_count: int = None):
        msg = f"Network error"
        if host:
            msg = f"Network error with {host}"
        if timeout:
            msg = f"{msg} (timeout)"
        if retry_count is not None:
            msg = f"{msg} after {retry_count} retries"
        
        super().__init__(f"{msg}: {message}")
        
        self.host = host
        self.timeout = timeout
        self.retry_count = retry_count


class ExecutionError(QuantumSpectreError):
    """Raised when there's an error executing an order."""
    
    def __init__(self, message: str = "Order execution error", order_id: str = None, 
                 symbol: str = None, order_type: str = None):
        msg = f"Execution error"
        details = []
        if symbol:
            details.append(f"symbol: {symbol}")
        if order_type:
            details.append(f"type: {order_type}")
        if order_id:
            details.append(f"order_id: {order_id}")
        
        detail_str = ", ".join(details)
        if detail_str:
            msg = f"{msg} [{detail_str}]"
        
        super().__init__(f"{msg}: {message}")
        
        self.order_id = order_id
        self.symbol = symbol
        self.order_type = order_type


class StrategyError(QuantumSpectreError):
    """Raised when there's an error in a trading strategy."""
    
    def __init__(self, message: str = "Strategy error", strategy_name: str = None):
        msg = f"Strategy error"
        if strategy_name:
            msg = f"Strategy error in {strategy_name}"
        super().__init__(f"{msg}: {message}")
        self.strategy_name = strategy_name


class ModelError(QuantumSpectreError):
    """Raised when there's an error in a machine learning model."""
    
    def __init__(self, message: str = "Model error", model_name: str = None, 
                 stage: str = None):
        msg = f"Model error"
        details = []
        if model_name:
            details.append(f"model: {model_name}")
        if stage:
            details.append(f"stage: {stage}")
        
        detail_str = ", ".join(details)
        if detail_str:
            msg = f"{msg} [{detail_str}]"
        
        super().__init__(f"{msg}: {message}")
        
        self.model_name = model_name
        self.stage = stage


class ResourceError(QuantumSpectreError):
    """Raised when there's an error with system resources."""
    
    def __init__(self, message: str = "Resource error", resource_type: str = None,
                 current_usage: float = None, limit: float = None):
        msg = f"Resource error"
        if resource_type:
            msg = f"Resource error ({resource_type})"
        
        details = []
        if current_usage is not None and limit is not None:
            details.append(f"usage: {current_usage:.2f}/{limit:.2f}")
        
        detail_str = ", ".join(details)
        if detail_str:
            msg = f"{msg} [{detail_str}]"
        
        super().__init__(f"{msg}: {message}")
        
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit


class DatabaseError(QuantumSpectreError):
    """Raised when there's an error with database operations."""
    
    def __init__(self, message: str = "Database error", operation: str = None,
                 table: str = None, error_code: str = None):
        msg = f"Database error"
        details = []
        if operation:
            details.append(f"operation: {operation}")
        if table:
            details.append(f"table: {table}")
        if error_code:
            details.append(f"code: {error_code}")
        
        detail_str = ", ".join(details)
        if detail_str:
            msg = f"{msg} [{detail_str}]"
        
        super().__init__(f"{msg}: {message}")
        
        self.operation = operation
        self.table = table
        self.error_code = error_code


class TimeoutError(QuantumSpectreError):
    """Raised when an operation times out."""
    
    def __init__(self, message: str = "Operation timed out", operation: str = None,
                 timeout_sec: float = None):
        msg = f"Timeout error"
        details = []
        if operation:
            details.append(f"operation: {operation}")
        if timeout_sec is not None:
            details.append(f"after: {timeout_sec:.2f}s")
        
        detail_str = ", ".join(details)
        if detail_str:
            msg = f"{msg} [{detail_str}]"
        
        super().__init__(f"{msg}: {message}")
        
        self.operation = operation
        self.timeout_sec = timeout_sec


class NotFoundError(QuantumSpectreError):
    """Raised when a requested resource is not found."""
    
    def __init__(self, message: str = "Resource not found", resource_type: str = None,
                 resource_id: str = None):
        msg = f"Not found error"
        details = []
        if resource_type:
            details.append(f"type: {resource_type}")
        if resource_id:
            details.append(f"id: {resource_id}")
        
        detail_str = ", ".join(details)
        if detail_str:
            msg = f"{msg} [{detail_str}]"
        
        super().__init__(f"{msg}: {message}")
        
        self.resource_type = resource_type
        self.resource_id = resource_id


class DuplicateError(QuantumSpectreError):
    """Raised when a duplicate resource is detected."""
    
    def __init__(self, message: str = "Duplicate resource", resource_type: str = None,
                 resource_id: str = None):
        msg = f"Duplicate error"
        details = []
        if resource_type:
            details.append(f"type: {resource_type}")
        if resource_id:
            details.append(f"id: {resource_id}")
        
        detail_str = ", ".join(details)
        if detail_str:
            msg = f"{msg} [{detail_str}]"
        
        super().__init__(f"{msg}: {message}")
        
        self.resource_type = resource_type
        self.resource_id = resource_id
