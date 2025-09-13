"""
Debug logging configuration for AI PPO Trading System.
"""

import logging
import os
from datetime import datetime

def setup_debug_logging():
    """Set up comprehensive debug logging for the entire system."""
    
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # Console handler for real-time debugging
            logging.StreamHandler(),
            
            # File handler for persistent logging
            logging.FileHandler(f'logs/debug_{timestamp}.log'),
            
            # Separate handler for GUI events
            logging.FileHandler('logs/gui_debug.log'),
            
            # Separate handler for training events
            logging.FileHandler('logs/training_debug.log'),
            
            # Separate handler for data events
            logging.FileHandler('logs/data_debug.log')
        ]
    )
    
    # Set specific loggers to debug level
    debug_loggers = [
        'gui.app',
        'training',
        'data',
        'agents',
        'environments',
        'utils',
        '__main__'
    ]
    
    for logger_name in debug_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
    
    # Reduce noise from third-party libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    print("🐛 DEBUG LOGGING CONFIGURED")
    print(f"📝 Log files will be saved to: {os.path.abspath(logs_dir)}")
    print("🔍 Debug messages will be shown in console and saved to files")
    
    logger = logging.getLogger(__name__)
    logger.debug("Debug logging configuration completed")

if __name__ == "__main__":
    setup_debug_logging()