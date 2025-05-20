"""
Logging configuration for the QuantumSpectre trading system.

This module provides a centralized logging configuration with support for
different output formats, log levels, and destinations.
"""

import json
import logging
import logging.config
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

# Define custom log record format
class CustomFormatter(logging.Formatter):
    """Custom formatter that adds color to console output and formats JSON for files."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[41m',  # Red background
        'RESET': '\033[0m',      # Reset
    }
    
    def __init__(self, fmt: str = None, datefmt: str = None, style: str = '%',
                 use_colors: bool = True, json_format: bool = False):
        """
        Initialize the formatter.
        
        Args:
            fmt: Log format string
            datefmt: Date format string
            style: Format string style (% or {)
            use_colors: Whether to use colors in console output
            json_format: Whether to format logs as JSON
        """
        super().__init__(fmt, datefmt, style)
        self.use_colors = use_colors and sys.stdout.isatty()  # Only use colors if stdout is a terminal
        self.json_format = json_format
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log string
        """
        # Create a copy of the record to avoid modifying the original
        record_copy = logging.makeLogRecord(record.__dict__)
        
        # Add timestamp if not present
        if not hasattr(record_copy, 'created_formatted'):
            record_copy.created_formatted = datetime.fromtimestamp(
                record_copy.created
            ).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Format the message
        formatted_message = super().format(record_copy)
        
        # Convert to JSON if requested
        if self.json_format:
            log_data = {
                'timestamp': record_copy.created_formatted,
                'level': record_copy.levelname,
                'name': record_copy.name,
                'message': record_copy.getMessage(),
                'module': record_copy.module,
                'lineno': record_copy.lineno,
            }
            
            # Include exception info if available
            if record_copy.exc_info:
                log_data['exception'] = self.formatException(record_copy.exc_info)
            
            # Include any extra attributes
            for key, value in record_copy.__dict__.items():
                if key not in {
                    'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
                    'funcName', 'id', 'levelname', 'levelno', 'lineno', 'module',
                    'msecs', 'message', 'msg', 'name', 'pathname', 'process',
                    'processName', 'relativeCreated', 'stack_info', 'thread', 'threadName',
                    'created_formatted'
                } and not key.startswith('_'):
                    log_data[key] = value
            
            return json.dumps(log_data)
        
        # Add colors if enabled and not JSON format
        if self.use_colors:
            level_name = record_copy.levelname
            color = self.COLORS.get(level_name, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            formatted_message = f"{color}{formatted_message}{reset}"
        
        return formatted_message


def configure_logging(
    log_level: str = 'INFO',
    log_file: Optional[str] = None,
    json_logs: bool = False,
    log_to_console: bool = True,
    log_dir: Optional[str] = None,
    config_file: Optional[str] = None,
) -> None:
    """
    Configure the logging system.
    
    Args:
        log_level: Minimum log level to capture
        log_file: Path to the log file
        json_logs: Whether to format logs as JSON
        log_to_console: Whether to log to console
        log_dir: Directory to store log files
        config_file: Path to logging configuration file
    """
    # If config file is provided and exists, use it
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
        return
    
    # Get log level from string
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Basic configuration
    handlers = []
    log_format = '%(created_formatted)s [%(levelname)s] %(name)s: %(message)s'
    
    # Configure console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(CustomFormatter(log_format, use_colors=True, json_format=False))
        handlers.append(console_handler)
    
    # Configure file handler
    if log_file or log_dir:
        if log_dir:
            # Create log directory if it doesn't exist
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)
            
            # Generate filename based on current date/time
            if not log_file:
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = f"{log_dir_path}/quantum_spectre_{current_time}.log"
            else:
                log_file = f"{log_dir_path}/{log_file}"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(CustomFormatter(log_format, use_colors=False, json_format=json_logs))
        handlers.append(file_handler)
    
    # Apply configuration
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )
    
    # Add filter to provide formatted creation time
    class TimestampFilter(logging.Filter):
        def filter(self, record):
            record.created_formatted = datetime.fromtimestamp(
                record.created
            ).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            return True
    
    for handler in logging.root.handlers:
        handler.addFilter(TimestampFilter())
    
    # Set levels for some verbose libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    
    # Log the configuration
    logger = logging.getLogger("quantum_spectre.core.logging")
    logger.debug(f"Logging configured with level {log_level}")
    if log_file:
        logger.debug(f"Logging to file: {log_file}")


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (defaults to "quantum_spectre" if None)
        
    Returns:
        Logger instance
    """
    if not name:
        name = "quantum_spectre"
    elif not name.startswith("quantum_spectre."):
        name = f"quantum_spectre.{name}"
    
    return logging.getLogger(name)
