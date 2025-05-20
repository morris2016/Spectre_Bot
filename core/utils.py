"""
Utility functions for the QuantumSpectre trading system.

This module provides common utility functions used throughout the system.
"""

import asyncio
import functools
import json
import time
import traceback
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from .exceptions import NetworkError, TimeoutError
from .logging_config import get_logger

logger = get_logger(__name__)

# Type variables for generics
T = TypeVar('T')
R = TypeVar('R')


def create_id() -> str:
    """
    Generate a unique ID.
    
    Returns:
        A unique ID string
    """
    return str(uuid.uuid4())


def current_timestamp_ms() -> int:
    """
    Get the current timestamp in milliseconds.
    
    Returns:
        Current timestamp in milliseconds
    """
    return int(time.time() * 1000)


def format_timestamp(timestamp_ms: int, format_str: str = '%Y-%m-%d %H:%M:%S.%f') -> str:
    """
    Format a timestamp (in milliseconds) to a string.
    
    Args:
        timestamp_ms: Timestamp in milliseconds
        format_str: Format string for the output
        
    Returns:
        Formatted timestamp string
    """
    dt = datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc)
    return dt.strftime(format_str)


def parse_timestamp(timestamp_str: str, format_str: str = '%Y-%m-%d %H:%M:%S.%f') -> int:
    """
    Parse a timestamp string to milliseconds.
    
    Args:
        timestamp_str: Timestamp string
        format_str: Format string of the input
        
    Returns:
        Timestamp in milliseconds
    """
    dt = datetime.strptime(timestamp_str, format_str)
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def timeit(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to measure execution time of a function.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function that logs execution time
        
    Example:
        @timeit
        def slow_function():
            time.sleep(1)
    """
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        logger.debug(f"{func.__name__} executed in {elapsed_time:.2f}ms")
        return result
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        logger.debug(f"{func.__name__} executed in {elapsed_time:.2f}ms")
        return result
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def safe_execute(
    func: Callable[..., T],
    default_value: Optional[T] = None,
    log_exceptions: bool = True,
    **kwargs
) -> T:
    """
    Safely execute a function catching any exceptions.
    
    Args:
        func: Function to execute
        default_value: Default value to return on exception
        log_exceptions: Whether to log exceptions
        **kwargs: Additional parameters to pass to the logger
        
    Returns:
        Function result or default value if an exception occurs
    """
    try:
        return func()
    except Exception as e:
        if log_exceptions:
            logger.exception(f"Error executing {func.__name__}: {str(e)}", **kwargs)
        return default_value


async def safe_execute_async(
    coro, 
    default_value: Optional[T] = None,
    log_exceptions: bool = True,
    **kwargs
) -> T:
    """
    Safely execute a coroutine catching any exceptions.
    
    Args:
        coro: Coroutine to execute
        default_value: Default value to return on exception
        log_exceptions: Whether to log exceptions
        **kwargs: Additional parameters to pass to the logger
        
    Returns:
        Coroutine result or default value if an exception occurs
    """
    try:
        return await coro
    except Exception as e:
        if log_exceptions:
            logger.exception(f"Error executing coroutine: {str(e)}", **kwargs)
        return default_value


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle Decimal, datetime, and other types."""
    
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, bytes):
            return obj.decode('utf-8')
        elif hasattr(obj, 'to_json'):
            return obj.to_json()
        return super().default(obj)


def serialize_to_json(obj: Any) -> str:
    """
    Serialize an object to JSON string.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON string
    """
    return json.dumps(obj, cls=CustomJSONEncoder)


def deserialize_from_json(json_str: str) -> Any:
    """
    Deserialize a JSON string to an object.
    
    Args:
        json_str: JSON string
        
    Returns:
        Deserialized object
    """
    return json.loads(json_str)


def round_decimal(value: Union[float, Decimal], precision: int = 8) -> Decimal:
    """
    Round a value to a specified precision.
    
    Args:
        value: Value to round
        precision: Number of decimal places
        
    Returns:
        Rounded Decimal value
    """
    if isinstance(value, float):
        value = Decimal(str(value))
    
    return value.quantize(Decimal('0.' + '0' * precision))


def format_number(value: Union[float, Decimal], precision: int = 8) -> str:
    """
    Format a number with specified precision.
    
    Args:
        value: Value to format
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    if isinstance(value, float):
        # Use string formatting for floats
        format_str = f"{{:.{precision}f}}"
        return format_str.format(value)
    elif isinstance(value, Decimal):
        # Use quantize for decimals
        return str(round_decimal(value, precision))
    
    # Return as is for other types
    return str(value)


async def async_retry(
    coro_func,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    errors: tuple = (Exception,),
    logger=None,
):
    """
    Retry an async function with exponential backoff.
    
    Args:
        coro_func: Async function to retry
        max_retries: Maximum number of retries
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Factor to increase delay by on each retry
        errors: Tuple of exceptions to catch and retry
        logger: Logger to use (defaults to module logger)
        
    Returns:
        Result of the coroutine
        
    Raises:
        NetworkError: If max_retries is exceeded
    """
    if logger is None:
        logger = get_logger(__name__)
    
    retries = 0
    last_exception = None
    
    while retries <= max_retries:
        try:
            return await coro_func()
        except errors as e:
            last_exception = e
            retries += 1
            
            if retries > max_retries:
                break
            
            delay = retry_delay * (backoff_factor ** (retries - 1))
            
            logger.warning(
                f"Retry {retries}/{max_retries} after error: {str(e)}. "
                f"Waiting {delay:.2f}s before next attempt."
            )
            
            await asyncio.sleep(delay)
    
    # If we get here, all retries failed
    raise NetworkError(
        f"Max retries ({max_retries}) exceeded", 
        retry_count=retries, 
        timeout=False
    ) from last_exception


def retry(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    errors: tuple = (Exception,),
):
    """
    Decorator to retry a function or coroutine with exponential backoff.
    
    Args:
        max_retries: Maximum number of retries
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Factor to increase delay by on each retry
        errors: Tuple of exceptions to catch and retry
        
    Returns:
        Decorated function or coroutine
        
    Example:
        @retry(max_retries=5, errors=(ConnectionError,))
        async def fetch_data():
            # ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await async_retry(
                lambda: func(*args, **kwargs),
                max_retries=max_retries,
                retry_delay=retry_delay,
                backoff_factor=backoff_factor,
                errors=errors,
            )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            retries = 0
            last_exception = None
            
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except errors as e:
                    last_exception = e
                    retries += 1
                    
                    if retries > max_retries:
                        break
                    
                    delay = retry_delay * (backoff_factor ** (retries - 1))
                    
                    logger.warning(
                        f"Retry {retries}/{max_retries} after error: {str(e)}. "
                        f"Waiting {delay:.2f}s before next attempt."
                    )
                    
                    time.sleep(delay)
            
            # If we get here, all retries failed
            raise NetworkError(
                f"Max retries ({max_retries}) exceeded", 
                retry_count=retries, 
                timeout=False
            ) from last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


async def timeout_after(seconds: float, coro):
    """
    Run a coroutine with a timeout.
    
    Args:
        seconds: Timeout in seconds
        coro: Coroutine to run
        
    Returns:
        Result of the coroutine
        
    Raises:
        TimeoutError: If the coroutine times out
    """
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation timed out after {seconds}s")


def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (overrides dict1)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge(result[key], value)
        else:
            # Override or add value
            result[key] = value
    
    return result


def chunks(items: List[T], chunk_size: int) -> List[List[T]]:
    """
    Split a list into chunks of a specified size.
    
    Args:
        items: List of items
        chunk_size: Size of each chunk
        
    Returns:
        List of chunked lists
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def truncate_string(s: str, max_length: int = 100, suffix: str = '...') -> str:
    """
    Truncate a string to a maximum length.
    
    Args:
        s: String to truncate
        max_length: Maximum length
        suffix: Suffix to add to truncated string
        
    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix


def get_exception_details(e: Exception) -> str:
    """
    Get detailed information about an exception.
    
    Args:
        e: Exception
        
    Returns:
        String with exception details
    """
    tb = traceback.format_exc()
    return f"{type(e).__name__}: {str(e)}\n{tb}"


def is_valid_uuid(uuid_str: str) -> bool:
    """
    Check if a string is a valid UUID.
    
    Args:
        uuid_str: String to check
        
    Returns:
        True if the string is a valid UUID, False otherwise
    """
    try:
        uuid.UUID(str(uuid_str))
        return True
    except (ValueError, AttributeError):
        return False


def camel_to_snake(name: str) -> str:
    """
    Convert a string from camelCase to snake_case.
    
    Args:
        name: String in camelCase
        
    Returns:
        String in snake_case
    """
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def snake_to_camel(name: str) -> str:
    """
    Convert a string from snake_case to camelCase.
    
    Args:
        name: String in snake_case
        
    Returns:
        String in camelCase
    """
    components = name.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


def limit_memory_usage(max_memory_mb: int) -> None:
    """
    Limit the memory usage of the process.
    
    Args:
        max_memory_mb: Maximum memory in megabytes
    
    Note:
        This only works on Unix-like systems
    """
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (max_memory_mb * 1024 * 1024, hard))
        logger.info(f"Memory usage limited to {max_memory_mb}MB")
    except (ImportError, ValueError, OSError) as e:
        logger.warning(f"Failed to limit memory usage: {e}")


def get_memory_usage() -> float:
    """
    Get the current memory usage of the process in megabytes.
    
    Returns:
        Memory usage in megabytes
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert to MB
    except ImportError:
        logger.warning("psutil not available, cannot get memory usage")
        return 0.0
