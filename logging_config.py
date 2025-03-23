"""Logging configuration for the tutor system."""

import logging
import sys
import os
from datetime import datetime

def setup_logging(log_level=logging.INFO, log_to_file=True):
    """Set up logging for the application."""
    # Create logs directory if it doesn't exist
    if log_to_file and not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers if any
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_to_file:
        log_filename = f"logs/tutor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)
    
    # Configure specific module loggers
    logging.getLogger('core').setLevel(log_level)
    logging.getLogger('ui').setLevel(log_level)
    logging.getLogger('utils').setLevel(log_level)
    
    # Suppress some noisy libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    
    return root_logger