import logging
from typing import Optional
import os
from pathlib import Path

def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = 'INFO',
    format_str: Optional[str] = None
) -> logging.Logger:
    """Set up logger with specified configuration.
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_str: Optional custom log format string
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Set level
    level = getattr(logging, level.upper())
    logger.setLevel(level)
    
    # Default format if not specified
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_str)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log file specified
    if log_file:
        # Create log directory if needed
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 