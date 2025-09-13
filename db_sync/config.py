import configparser
from pathlib import Path

def load_db_credentials(ini_file: str):
    # Make path relative to this file, not the CWD
    ini_path = (Path(__file__).parent / ini_file).resolve()

    cfg = configparser.ConfigParser()

    # Handle potential BOM and ensure the file is actually read
    loaded = cfg.read(ini_path, encoding="utf-8-sig")
    if not loaded:
        raise FileNotFoundError(f"INI file not found or unreadable: {ini_path}")

    if 'database' not in cfg:
        raise KeyError(f"'database' section missing in {ini_path}. Sections found: {cfg.sections()}")

    db = cfg['database']
    return {
        'dbname':   db.get('dbname'),
        'user':     db.get('user'),
        'host':     db.get('host', 'localhost'),
        'password': db.get('password'),
        'port':     db.getint('port', fallback=5432),
    }

local_db_credentials  = load_db_credentials('credentials/local_db.ini')
global_db_credentials = load_db_credentials('credentials/global_db.ini')