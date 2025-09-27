import psycopg2
from config import local_db_credentials, global_db_credentials
import json
from psycopg2 import sql
from loggin_config import LogManager

def connect_to_local_db():
    return psycopg2.connect(**local_db_credentials)

def connect_to_global_db():
    return psycopg2.connect(**global_db_credentials)

def create_schema(conn, schema_name):
    # Read the schema_info.json file to get the schema information
    with open('schema_info.json', 'r') as f:
        schema_info = json.load(f)
        
    cur = conn.cursor()
    try:
        # Create schema
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema_name)))
        
        # Extract tables from the JSON
        tables = schema_info.get('database', {}).get('tables', [])
        
        # Iterate over the tables in the schema_info and create each table
        for table in tables:
            table_name = table.get('name')
            # Check if the table already exists
            cur.execute(sql.SQL("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = %s AND table_name = %s
                )
            """), (schema_name, table_name))
            table_exists = cur.fetchone()[0]
            
            if table_exists:
                LogManager.logger.info(f"Table {table_name} already exists in schema {schema_name}.")
                continue  # Skip to the next table
            
            # Construct the CREATE TABLE SQL and execute it if the table doesn't exist
            columns_sql = []
            for column in table.get('columns', []):
                column_name = column.get('name')
                data_type = column.get('type')
                constraints = ' '.join(column.get('constraints', []))
                columns_sql.append(sql.SQL("{} {} {}").format(
                    sql.Identifier(column_name), 
                    sql.SQL(data_type),
                    sql.SQL(constraints)
                ))
                
            create_table_sql = sql.SQL("""
                CREATE TABLE {}.{} ({})
            """).format(
                sql.Identifier(schema_name),
                sql.Identifier(table_name),
                sql.SQL(', ').join(columns_sql)
            )
            
            cur.execute(create_table_sql)
        
        # Commit changes
        conn.commit()
        
        LogManager.logger.info(f"Schema {schema_name} checked and tables created if not exist.")
        
    except Exception as e:
        LogManager.logger.error(f"Error creating schema {schema_name} or tables: {e}")
        # If any exception occurs, rollback the transaction
        conn.rollback()
    finally:
        if cur:
            cur.close()

def check_data_exists_in_all_tables(conn, schema_name):
    cur = conn.cursor()
    tables_dict = {}
    try:
        # Getting the list of all tables in the schema
        cur.execute(sql.SQL("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = %s
        """), (schema_name,))
        
        # Fetch the result of the query
        tables = cur.fetchall()

        for table in tables:
            table_name = table[0]
            table_info = {"itime": None, "utime": None}
            
            # Skip querying itime and utime for specific tables
            if table_name in ['stock', 'stock_value','page_permissions','users']:
                LogManager.logger.info(f"Skipping itime and utime check for table {table_name} in schema {schema_name}.")
                tables_dict[table_name] = table_info
                continue

            try:
                # Execute a SQL query to get the max itime and utime from the table
                cur.execute(sql.SQL("""
                    SELECT MAX(itime) AS max_itime, MAX(utime) AS max_utime FROM {}.{}
                """).format(sql.Identifier(schema_name), sql.Identifier(table_name)))
                
                # Fetch the result of the query
                max_times = cur.fetchone()
                table_info["itime"] = max_times[0]
                table_info["utime"] = max_times[1]
                
                if max_times[0] or max_times[1]:
                    LogManager.logger.info(f"Data exists in the table {table_name} in schema {schema_name}.")
                else:
                    LogManager.logger.info(f"No data exists in the table {table_name} in schema {schema_name}.")
                
            except Exception as e:
                LogManager.logger.info(f"Error getting max itime and utime from table {table_name} in schema {schema_name}: {e}")
                
            tables_dict[table_name] = table_info
                
    except Exception as e:
        LogManager.logger.info(f"Error checking data in schema {schema_name}: {e}")
    finally:
        if cur:
            cur.close()
    
    return tables_dict

# Example Usage:
# conn: Your database connection object
# schema_name: The name of the schema
# result = check_data_exists_in_all_tables(conn, "your_schema_name")
# print(result)


### if data exists need to take ztime or zutime instead of just data_exists. 

###scenarios
###database does not exist
###database exists but schema does not exist
###schema exists and tables exist but there is no data
###everything exists but tables are not up to date
###all tables are up to date.