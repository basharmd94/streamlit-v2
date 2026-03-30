from configparser import ConfigParser
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent
LOG_INI    = CONFIG_DIR / "logging.ini"
DB_INI     = CONFIG_DIR / "database.ini"

def get_db_params(section: str = "postgresql") -> dict:
    parser = ConfigParser()
    parser.read(DB_INI)
    if not parser.has_section(section):
        raise ValueError(f"Section [{section}] not found in {DB_INI}")
    return dict(parser.items(section))
