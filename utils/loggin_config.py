import os
import logging.config

class LogManager:
    def load_logging_config():
        # Build absolute path to INI and configure logging
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ini_path = os.path.join(base_dir, 'config', 'loggin_config.ini')
        logging.config.fileConfig(ini_path, disable_existing_loggers=False)
        return logging.getLogger(__name__)
     # Load logging configuration
    
    logger = load_logging_config()
    logger.info('loggin_config.py logger initialized')