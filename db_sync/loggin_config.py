import os
import sys
import configparser
import logging.config

class LogManager:
    def load_logging_config():
        # Build absolute paths relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ini_path = os.path.join(base_dir, 'credentials', 'loggin_config.ini')
        log_file_path = os.path.join(base_dir, 'credentials', 'Sync_logfile.log')

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # Use forward slashes to avoid escape-sequence issues when fileConfig evals args
        log_file_path_safe = log_file_path.replace('\\', '/')

        # Configure logging using the INI and injected defaults
        logging.config.fileConfig(
            ini_path,
            defaults={'LOG_FILE': log_file_path_safe},
            disable_existing_loggers=False
        )
        return logging.getLogger(__name__)
     # Load logging configuration
    
    logger = load_logging_config()
    logger.info('loggin_config.py logger initialized')