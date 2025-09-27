import configparser
import logging.config

class LogManager:
    def load_logging_config():
        # Load the configuration
        config = configparser.ConfigParser()
        config.read('credentials/loggin_config.ini')
        # Set up logging
        logging.config.fileConfig(config)
        return logging.getLogger(__name__)
     # Load logging configuration
    
    logger = load_logging_config()
    logger.info('loggin_config.py logger initialized')