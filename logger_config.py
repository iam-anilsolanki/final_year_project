"""
Centralized logging configuration for the entire application.
Provides time-based log rotation (new file at midnight) and consistent formatting.
"""

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime


# Configuration
LOG_DIR = "logs"
LOG_LEVEL = logging.INFO
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging():
    """
    Sets up the logging directory and returns configured handlers.
    This is called internally by get_logger().
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        print(f"Created logs directory: {LOG_DIR}")
    
    return True


def get_logger(name):
    """
    Returns a configured logger instance for the given module name.
    
    Args:
        name (str): Name of the module/logger (typically __name__)
    
    Returns:
        logging.Logger: Configured logger instance
    
    Example:
        logger = get_logger(__name__)
        logger.info("Application started")
    """
    # Ensure logging is set up
    setup_logging()
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    
    # File Handler with Time-based Rotation (rotates at midnight)
    log_filename = os.path.join(LOG_DIR, "app.log")
    file_handler = TimedRotatingFileHandler(
        filename=log_filename,
        when='midnight',           # Rotate at midnight
        interval=1,                # Every 1 day
        backupCount=30,            # Keep 30 days of logs (configurable)
        encoding='utf-8'
    )
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(formatter)
    
    # Add suffix to rotated files (YYYY-MM-DD format)
    file_handler.suffix = "%Y-%m-%d"
    
    # Console Handler (for real-time monitoring)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# Module-level logger for this configuration file
_config_logger = None

def get_config_logger():
    """Returns a logger for the logger_config module itself."""
    global _config_logger
    if _config_logger is None:
        _config_logger = get_logger(__name__)
    return _config_logger


if __name__ == "__main__":
    # Test the logging configuration
    test_logger = get_logger("test_module")
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    test_logger.critical("This is a critical message")
    
    print(f"\nLog files are stored in: {os.path.abspath(LOG_DIR)}")
    print("Check the logs directory for the generated log file.")
