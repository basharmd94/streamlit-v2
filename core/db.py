import psycopg2
from psycopg2 import pool
import pandas as pd
from config.settings import get_db_params
from utils.loggin_config import LogManager

_pool: pool.ThreadedConnectionPool | None = None


def _get_pool() -> pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        params = get_db_params()
        _pool = pool.ThreadedConnectionPool(minconn=1, maxconn=10, **params)
    return _pool


def get_data(query: str, *args):
    """Execute a query and return (records, column_names)."""
    conn = None
    try:
        conn = _get_pool().getconn()
        with conn.cursor() as cur:
            cur.execute(query, args if args else None)
            records = cur.fetchall()
            colnames = [d[0] for d in cur.description]
        return records, colnames
    except Exception as e:
        LogManager.logger.error(f"get_data error: {e}")
        return None, None
    finally:
        if conn:
            _get_pool().putconn(conn)


def get_dataframe(query: str, params: tuple) -> pd.DataFrame:
    """Execute a query and return a pandas DataFrame. Uses the connection pool."""
    conn = None
    try:
        conn = _get_pool().getconn()
        df = pd.read_sql(query, conn, params=list(params))
        return df
    except Exception as e:
        LogManager.logger.error(f"get_dataframe error: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            _get_pool().putconn(conn)
