import psycopg2
from config import local_db_credentials, global_db_credentials
import json
from psycopg2 import sql, extras
from loggin_config import LogManager

# ---------------- Connections ----------------

def connect_to_local_db():
    return psycopg2.connect(**local_db_credentials)

def connect_to_global_db():
    return psycopg2.connect(**global_db_credentials)

# ---------------- Schema helpers ----------------

def _load_schema_json(path: str = "schema_info.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)

def _tables_from_json(schema_json: dict) -> list:
    return schema_json.get("database", {}).get("tables", [])

def _table_exists(conn, schema_name: str, table_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = %s AND table_name = %s
            )
        """, (schema_name, table_name))
        return cur.fetchone()[0]

def _create_table_sql(schema_name: str, table_def: dict):
    cols = []
    for col in table_def.get("columns", []):
        col_name = col["name"]
        data_type = col["type"]
        constraints = " ".join(col.get("constraints", []))
        if constraints:
            cols.append(
                sql.SQL("{} {} {}").format(
                    sql.Identifier(col_name), sql.SQL(data_type), sql.SQL(constraints)
                )
            )
        else:
            cols.append(
                sql.SQL("{} {}").format(sql.Identifier(col_name), sql.SQL(data_type))
            )
    return sql.SQL("CREATE TABLE {}.{} ({})").format(
        sql.Identifier(schema_name),
        sql.Identifier(table_def["name"]),
        sql.SQL(", ").join(cols),
    )

# ---------------- Create schema & tables ----------------

def create_schema(conn, schema_name: str, schema_info_path: str = "schema_info.json", drop_and_recreate: bool = False) -> None:
    """
    Ensure the schema exists and tables from schema_info.json are present.
    - If drop_and_recreate=True: DROP each listed table (if exists) then CREATE fresh.
    - Else: CREATE missing tables; keep existing ones intact.
    """
    schema_json = _load_schema_json(schema_info_path)
    tables = _tables_from_json(schema_json)

    with conn:
        with conn.cursor() as cur:
            # Create schema if missing
            cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema_name)))

            for t in tables:
                table_name = t["name"]
                exists = _table_exists(conn, schema_name, table_name)

                if exists and drop_and_recreate:
                    LogManager.logger.info(f"Dropping table {table_name} in schema {schema_name}.")
                    cur.execute(
                        sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE")
                        .format(sql.Identifier(schema_name), sql.Identifier(table_name))
                    )
                    exists = False

                if not exists:
                    LogManager.logger.info(f"Creating table {table_name} in schema {schema_name}.")
                    cur.execute(_create_table_sql(schema_name, t))
                else:
                    LogManager.logger.info(f"Table {table_name} already exists in schema {schema_name}.")

    LogManager.logger.info(f"Schema {schema_name} is ready.")

# ---------------- Replace-or-create load helpers ----------------

def prepare_table_for_full_replace(conn, schema_name: str, table_name: str,
                                   schema_info_path: str = "schema_info.json",
                                   drop_and_recreate: bool = False) -> None:
    """
    Make sure the table is ready for a full reload.
    - If the table is missing: CREATE it from schema_info.json.
    - If present: TRUNCATE it (or DROP+CREATE if drop_and_recreate=True).
    """
    schema_json = _load_schema_json(schema_info_path)
    table_defs = {t["name"]: t for t in _tables_from_json(schema_json)}
    if table_name not in table_defs:
        raise ValueError(f"Table {table_name} not found in {schema_info_path}")

    with conn:
        with conn.cursor() as cur:
            exists = _table_exists(conn, schema_name, table_name)

            if exists and drop_and_recreate:
                LogManager.logger.info(f"Dropping table {table_name} in schema {schema_name}.")
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE")
                    .format(sql.Identifier(schema_name), sql.Identifier(table_name))
                )
                exists = False

            if not exists:
                LogManager.logger.info(f"Creating table {table_name} in schema {schema_name}.")
                cur.execute(_create_table_sql(schema_name, table_defs[table_name]))
            else:
                LogManager.logger.info(f"Truncating table {table_name} in schema {schema_name}.")
                cur.execute(
                    sql.SQL("TRUNCATE TABLE {}.{}")
                    .format(sql.Identifier(schema_name), sql.Identifier(table_name))
                )

def bulk_insert_rows(conn, schema_name: str, table_name: str, columns: list[str], rows: list[tuple], page_size: int = 10000) -> None:
    """
    Fast bulk insert into schema.table.
    Example:
        columns = ["uuid","zid","year","month",...]
        rows = [(...), (...), ...]
    """
    if not rows:
        LogManager.logger.info(f"No rows to insert into {table_name}. Skipping.")
        return

    with conn:
        with conn.cursor() as cur:
            cols_sql = sql.SQL(", ").join(sql.Identifier(c) for c in columns)
            insert_sql = sql.SQL("INSERT INTO {}.{} ({}) VALUES %s").format(
                sql.Identifier(schema_name), sql.Identifier(table_name), cols_sql
            )
            extras.execute_values(cur, insert_sql, rows, page_size=page_size)

    LogManager.logger.info(f"Inserted {len(rows)} rows into {table_name}.")
