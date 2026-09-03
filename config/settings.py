from configparser import ConfigParser
from pathlib import Path

CONFIG_DIR   = Path(__file__).resolve().parent
LOG_INI      = CONFIG_DIR / "logging.ini"
DB_INI       = CONFIG_DIR / "global_db.ini"
WHATSFLY_INI = CONFIG_DIR / "whatsfly.ini"

def get_db_params(section: str = "database") -> dict:
    parser = ConfigParser()
    parser.read(DB_INI)
    if not parser.has_section(section):
        raise ValueError(f"Section [{section}] not found in {DB_INI}")
    return dict(parser.items(section))


def get_whatsfly_params(section: str = "whatsfly") -> dict | None:
    """Returns {'api_token': ..., 'phone_number_id': ...} or None if
    config/whatsfly.ini (gitignored, like every other *.ini here) doesn't
    exist yet or is missing the section — never raises, so callers can show
    a friendly setup message instead of crashing the page."""
    if not WHATSFLY_INI.exists():
        return None
    parser = ConfigParser()
    parser.read(WHATSFLY_INI)
    if not parser.has_section(section):
        return None
    params = dict(parser.items(section))
    if not params.get("api_token") or not params.get("phone_number_id"):
        return None
    return params
