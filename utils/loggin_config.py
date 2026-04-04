import os
import logging
import logging.config

class LogManager:
    def load_logging_config():
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Try new name first, then legacy name, then fall back to basicConfig
        for ini_name in ('logging.ini', 'loggin_config.ini'):
            ini_path = os.path.join(base_dir, 'config', ini_name)
            if os.path.exists(ini_path):
                try:
                    logging.config.fileConfig(ini_path, disable_existing_loggers=False)
                    return logging.getLogger(__name__)
                except Exception:
                    pass
        # Fallback: basic console logging so the app still starts
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(name)s: %(message)s'
        )
        return logging.getLogger(__name__)

    logger = load_logging_config()
    logger.info('loggin_config.py logger initialized')