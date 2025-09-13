from db_connector import connect_to_local_db, connect_to_global_db, create_schema, check_data_exists_in_all_tables
from loggin_config import LogManager
from sync_files import SyncManager

def main():
    ###local connection
    try:
        local_conn = connect_to_local_db()
    except Exception as e:
        LogManager.logger.info(f'Exception generated from local connection with error: {e}',exc_info = True)
    
    ###global connection
    try:
        global_conn = connect_to_global_db()
    except Exception as e:
         LogManager.logger.info(f'Exception generated from local connection with error: {e}',exc_info = True)

    tables_dict = None
    
    try:
        schema_name = 'public'
        create_schema(local_conn,schema_name) 
        tables_dict = check_data_exists_in_all_tables(local_conn, schema_name)
    except Exception as e:
        LogManager.logger.info(f'Exception generated from local connection with error: {e}',exc_info = True)

    if tables_dict:
        try:
            ###Sync database
            sync_manager = SyncManager(global_conn, local_conn)
            sync_manager.sync_all(tables_dict)
        except Exception as e:
            LogManager.logger.info(f'Exception generated from local connection with error: {e}',exc_info = True)
            
    local_conn.close()
    global_conn.close()
    LogManager.logger.info('Connection Closed')

if __name__ == "__main__":
    main()

