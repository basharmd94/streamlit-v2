import sys
import logging, logging.config
from pathlib import Path

def load_logging_config():
    # Point to the INI in db_sync/credentials
    ini_path = (Path(__file__).parent / "credentials" / "loggin_config.ini").resolve()

    # Build the absolute logfile path next to this module
    log_file = (Path(__file__).parent / "credentials" / "Sync_logfile.log").resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Pass LOG_FILE into fileConfig so the INI can reference it
    logging.config.fileConfig(
        fname=str(ini_path),
        defaults={"LOG_FILE": log_file.as_posix()},  # use forward slashes
        disable_existing_loggers=False,
        encoding="utf-8-sig",
    )
    return logging.getLogger("sampleLogger")

class LogManager:
    logger = load_logging_config()