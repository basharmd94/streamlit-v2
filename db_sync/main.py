import sys
import subprocess
from pathlib import Path
from db_connector import connect_to_local_db, connect_to_global_db, create_schema
from loggin_config import LogManager
from sync_files import SyncManager

def run_setup_db():
    """Run setup_db.py before main sync process"""
    try:
        # Get the project root directory (parent of db_sync)
        project_root = Path(__file__).parent.parent
        setup_script = project_root / "auth" / "setup_db.py"
        
        if setup_script.exists():
            LogManager.logger.info(f"Running setup_db.py from {setup_script}")
            result = subprocess.run([sys.executable, str(setup_script)], 
                                  capture_output=True, text=True, check=True)
            LogManager.logger.info("setup_db.py completed successfully")
            if result.stdout:
                LogManager.logger.info(f"Setup output: {result.stdout}")
        else:
            LogManager.logger.warning(f"setup_db.py not found at {setup_script}")
    except subprocess.CalledProcessError as e:
        LogManager.logger.error(f"Error running setup_db.py: {e}")
        if e.stdout:
            LogManager.logger.error(f"Setup stdout: {e.stdout}")
        if e.stderr:
            LogManager.logger.error(f"Setup stderr: {e.stderr}")
    except Exception as e:
        LogManager.logger.error(f"Exception running setup_db.py: {e}", exc_info=True)

def main():
    local_conn = None
    global_conn = None

    # Config
    schema_name = "public"
    DROP_AND_RECREATE = False  # set True to DROP+CREATE tables from schema_info.json before syncing

    # ---- Connect ----
    try:
        local_conn = connect_to_local_db()
    except Exception as e:
        LogManager.logger.info(f"Exception from local connection: {e}", exc_info=True)
        return

    try:
        global_conn = connect_to_global_db()
    except Exception as e:
        LogManager.logger.info(f"Exception from global connection: {e}", exc_info=True)
        if local_conn:
            local_conn.close()
        return

    try:
        # ---- Ensure schema/tables exist (create missing; optional drop+recreate) ----
        create_schema(local_conn, schema_name, drop_and_recreate=DROP_AND_RECREATE)

        # ---- Full replace sync for all tables ----
        sync_manager = SyncManager(global_conn, local_conn, local_schema=schema_name)
        sync_manager.sync_all()

    except Exception as e:
        LogManager.logger.info(f"Exception during sync: {e}", exc_info=True)
    finally:
        # ---- Always close connections ----
        try:
            if local_conn:
                local_conn.close()
        finally:
            if global_conn:
                global_conn.close()
        LogManager.logger.info("Connection Closed")

    # Run setup_db.py after sync finishes
    run_setup_db()

if __name__ == "__main__":
    main()
