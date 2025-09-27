import configparser
import logging.config

def load_db_credentials(ini_file):
    config = configparser.ConfigParser()
    config.read(ini_file)
    return {
        'dbname': config['database']['dbname'],
        'user': config['database']['user'],
        'host': config['database']['host'],
        'password': config['database']['password']
    }

local_db_credentials = load_db_credentials('credentials/local_db.ini')
global_db_credentials = load_db_credentials('credentials/global_db.ini')

