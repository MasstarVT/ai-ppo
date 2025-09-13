"""
Optimized debug logging configuration for AI PPO Trading System.
"""

import logging
import os
from datetime import datetime

def setup_debug_logging(enable_debug=False):
    """Set up optimized logging for the system."""
    
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Set logging level based on debug flag
    log_level = logging.DEBUG if enable_debug else logging.INFO
    
    # Configure root logger with minimal overhead
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            # Only console handler for performance
            logging.StreamHandler(),
        ]
    )
    
    # Only enable debug loggers when explicitly requested
    if enable_debug:
        debug_loggers = ['gui.app', 'training', 'data']
        for logger_name in debug_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)
    
    # Reduce noise from third-party libraries
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('PIL').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('requests').setLevel(logging.ERROR)
    logging.getLogger('streamlit').setLevel(logging.ERROR)
    
    if enable_debug:
        print("🐛 DEBUG LOGGING ENABLED")
    else:
        print("� OPTIMIZED LOGGING ENABLED")

if __name__ == "__main__":
    setup_debug_logging()

if __name__ == "__main__":
    setup_debug_logging()