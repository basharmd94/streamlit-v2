from psycopg2 import sql, extras
import pandas as pd
import uuid
import numpy as np
from loggin_config import LogManager
import re

class SyncManager:
    def __init__(self, global_conn, local_conn):
        self.global_conn = global_conn
        self.local_conn = local_conn
        LogManager.logger.info('SyncManager Initialized')

    def execute_sync(self, select_sql, insert_sql, table_name, clear_table_first=False):
        """
        Syncs all data from the global database into the local table.
        If clear_table_first is True, truncates the local table before inserting.
        """
        LogManager.logger.info(f"Starting full sync for table: {table_name}")

        # Fetch all records from global database
        global_cur = self.global_conn.cursor()
        global_cur.execute(select_sql)
        records = global_cur.fetchall()
        column_names = [desc[0] for desc in global_cur.description]
        global_cur.close()

        if not records:
            LogManager.logger.info(f"No records found to sync for table: {table_name}")
            return

        # Load into DataFrame
        df = pd.DataFrame(records, columns=column_names)
        timestamp_columns = ['itime', 'utime']  
        for col in timestamp_columns:
            if col in df.columns:
                df[col].replace({pd.NaT: None}, inplace=True)

        # Generate a new UUID4 for each row
        df['uuid'] = [str(uuid.uuid4()) for _ in range(len(df))]

        # Reorder columns: uuid first
        df = df[['uuid'] + column_names]

        local_cur = self.local_conn.cursor()
        # Clear table if requested
        if clear_table_first:
            LogManager.logger.info(f"Truncating local table: {table_name}")
            local_cur.execute(f"TRUNCATE TABLE {table_name}")

        # Insert all records into local database
        values = list(df.itertuples(index=False, name=None))
        extras.execute_values(local_cur, insert_sql, values, page_size=100)
        self.local_conn.commit()
        local_cur.close()

        LogManager.logger.info(f"Inserted {len(df)} records into {table_name}")

    def sync_all(self, tables_dict):
        LogManager.logger.info("Starting full sync of all tables")

        # GL Master (glmst)
        select_sql_glmst = """
            SELECT
                glmst.ztime AS itime,
                glmst.zutime AS utime,
                glmst.zid,
                glmst.xacc AS ac_code,
                glmst.xdesc AS ac_name,
                glmst.xacctype AS ac_type,
                glmst.xhrc1 AS ac_lv1,
                glmst.xhrc2 AS ac_lv2,
                glmst.xhrc3 AS ac_lv3,
                glmst.xhrc4 AS ac_lv4,
                glmst.xaccusage AS usage
            FROM glmst
        """
        insert_sql_glmst = """
            INSERT INTO glmst (
                uuid, itime, utime, zid, ac_code, ac_name, ac_type,
                ac_lv1, ac_lv2, ac_lv3, ac_lv4, usage
            ) VALUES %s
        """
        self.execute_sync(select_sql_glmst, insert_sql_glmst, 'glmst', clear_table_first=True)

        # GL Details (gldetail)
        select_sql_gldetail = """
            SELECT
                gldetail.ztime AS itime,
                gldetail.zutime AS utime,
                gldetail.zid,
                gldetail.xacc AS ac_code,
                gldetail.xsub AS ac_sub,
                gldetail.xproj AS project,
                gldetail.xvoucher AS voucher,
                gldetail.xprime AS value
            FROM gldetail
        """
        insert_sql_gldetail = """
            INSERT INTO gldetail (
                uuid, itime, utime, zid, ac_code, ac_sub, project, voucher, value
            ) VALUES %s
        """
        self.execute_sync(select_sql_gldetail, insert_sql_gldetail, 'gldetail', clear_table_first=True)

        # GL Header (glheader)
        select_sql_glheader = """
            SELECT
                glheader.ztime AS itime,
                glheader.zutime AS utime,
                glheader.zid,
                glheader.xvoucher AS voucher,
                glheader.xdate AS date,
                glheader.xyear AS year,
                glheader.xper AS month
            FROM glheader
        """
        insert_sql_glheader = """
            INSERT INTO glheader (
                uuid, itime, utime, zid, voucher, date, year, month
            ) VALUES %s
        """
        self.execute_sync(select_sql_glheader, insert_sql_glheader, 'glheader', clear_table_first=True)

        # Purchase Orders (purchase)
        select_sql_purchase = """
            SELECT
                poodt.ztime AS itime,
                poodt.zutime AS utime,
                poord.zid AS zid,
                COALESCE(pogrn.xdate, poord.xdate) AS combinedate,
                poord.xpornum AS povoucher,
                pogrn.xgrnnum AS grnvoucher,
                poodt.xitem AS itemcode,
                poord.xcounterno AS shipmentname,
                poodt.xqtyord AS quantity,
                poodt.xrate AS cost,
                poord.xstatuspor AS status
            FROM poord
            JOIN poodt ON poord.xpornum = poodt.xpornum AND poord.zid = poodt.zid
            LEFT JOIN pogrn ON poord.xpornum = pogrn.xpornum AND poord.zid = pogrn.zid
            WHERE poord.xstatuspor IN ('5-Received','1-Open')
        """
        insert_sql_purchase = """
            INSERT INTO purchase (
                uuid, itime, utime, zid, combinedate, povoucher, grnvoucher,
                itemcode, shipmentname, quantity, cost, status
            ) VALUES %s
        """
        self.execute_sync(select_sql_purchase, insert_sql_purchase, 'purchase', clear_table_first=True)

        # Sales (sales)
        select_sql_sales = """
            SELECT
                opddt.ztime AS itime,
                opddt.zutime AS utime,
                opddt.zid AS zid,
                opdor.xdornum AS ordernumber,
                opdor.xdate AS date,
                opdor.xsp AS sp_id,
                opdor.xcus AS cusid,
                opddt.xitem AS itemcode,
                opddt.xqty AS quantity,
                opddt.xdtwotax AS altsales,
                opddt.xdtdisc AS proddiscount,
                opddt.xlineamt AS totalsales,
                imtrn.xval AS cost
            FROM opdor
            LEFT JOIN opddt ON opdor.xdornum = opddt.xdornum AND opdor.zid = opddt.zid
            LEFT JOIN imtrn ON opdor.xdornum = imtrn.xdocnum
                AND opddt.xdornum = imtrn.xdocnum
                AND opddt.xitem = imtrn.xitem
                AND opddt.zid = imtrn.zid
                AND opdor.zid = imtrn.zid
                AND opddt.xrow = imtrn.xdocrow
        """
        insert_sql_sales = """
            INSERT INTO sales (
                uuid, itime, utime, zid, ordernumber, date, sp_id, cusid,
                itemcode, quantity, altsales, proddiscount, totalsales, cost
            ) VALUES %s
        """
        self.execute_sync(select_sql_sales, insert_sql_sales, 'sales', clear_table_first=True)

        # Returns (returns)
        select_sql_return = """
            SELECT
                opcdt.ztime AS itime,
                opcdt.zutime AS utime,
                opcdt.zid AS zid,
                opcrn.xcrnnum AS revoucher,
                opcrn.xdate AS date,
                opcrn.xcus AS cusid,
                opcrn.xemp AS sp_id,
                opcdt.xitem AS itemcode,
                opcdt.xqty AS returnqty,
                opcdt.xlineamt AS treturnamt,
                imtrn.xval AS returncost
            FROM opcdt
            LEFT JOIN opcrn ON opcrn.xcrnnum = opcdt.xcrnnum AND opcrn.zid = opcdt.zid
            LEFT JOIN imtrn ON opcrn.xcrnnum = imtrn.xdocnum
                AND opcdt.xcrnnum = imtrn.xdocnum
                AND opcdt.xitem = imtrn.xitem
                AND opcdt.zid = imtrn.zid
                AND opcrn.zid = imtrn.zid
                AND opcdt.xrow = imtrn.xdocrow
        """
        insert_sql_return = """
            INSERT INTO return (
                uuid, itime, utime, zid, revoucher, date, cusid, sp_id,
                itemcode, returnqty, treturnamt, returncost
            ) VALUES %s
        """
        self.execute_sync(select_sql_return, insert_sql_return, 'return', clear_table_first=True)

        # Stock (stock)
        select_sql_stock = """
            SELECT
                imtrn.zid AS zid,
                imtrn.xitem AS itemcode,
                SUM(imtrn.xqty * imtrn.xsign) AS stockqty,
                SUM(imtrn.xval * imtrn.xsign) AS stockvalue
            FROM imtrn
            GROUP BY imtrn.zid, imtrn.xitem
        """
        insert_sql_stock = """
            INSERT INTO stock (
                uuid, zid, itemcode, stockqty, stockvalue
            ) VALUES %s
        """
        self.execute_sync(select_sql_stock, insert_sql_stock, 'stock', clear_table_first=True)

        # Stock Value (stock_value)
        select_sql_stock_value = """
            SELECT
                imtrn.zid AS zid,
                imtrn.xyear AS year,
                imtrn.xper AS month,
                imtrn.xwh AS warehouse,
                SUM(imtrn.xval * imtrn.xsign) AS stockvalue
            FROM imtrn
            GROUP BY imtrn.zid, imtrn.xyear, imtrn.xper, imtrn.xwh
        """
        insert_sql_stock_value = """
            INSERT INTO stock_value (
                uuid, zid, year, month, warehouse, stockvalue
            ) VALUES %s
        """
        self.execute_sync(select_sql_stock_value, insert_sql_stock_value, 'stock_value', clear_table_first=True)

        # Customers (cacus)
        select_sql_cacus = """
            SELECT
                cacus.ztime AS itime,
                cacus.zutime AS utime,
                cacus.zid AS zid,
                cacus.xcus AS cusid,
                cacus.xshort AS cusname,
                cacus.xadd2 AS cusadd,
                cacus.xcity AS cuscity,
                cacus.xstate AS cusstate,
                cacus.xmobile AS cusmobile,
                cacus.xtaxnum AS cusmobile2
            FROM cacus
        """
        insert_sql_cacus = """
            INSERT INTO cacus (
                uuid, itime, utime, zid, cusid, cusname, cusadd,
                cuscity, cusstate, cusmobile, cusmobile2
            ) VALUES %s
        """
        self.execute_sync(select_sql_cacus, insert_sql_cacus, 'cacus', clear_table_first=True)

        # Items (caitem)
        select_sql_caitem = """
            SELECT
                caitem.ztime AS itime,
                caitem.zutime AS utime,
                caitem.zid AS zid,
                caitem.xitem AS itemcode,
                caitem.xdesc AS itemname,
                caitem.xgitem AS itemgroup,
                caitem.xabc AS itemgroup2,
                caitem.xdrawing AS packcode
            FROM caitem
        """
        insert_sql_caitem = """
            INSERT INTO caitem (
                uuid, itime, utime, zid, itemcode, itemname,
                itemgroup, itemgroup2, packcode
            ) VALUES %s
        """
        self.execute_sync(select_sql_caitem, insert_sql_caitem, 'caitem', clear_table_first=True)

        # For prmst table
        select_sql_prmst = """
            SELECT 
                prmst.ztime as itime,
                prmst.zutime as utime,
                prmst.zid as zid,
                prmst.xemp as spid,
                prmst.xname as spname,
                prmst.xdept as department,
                prmst.xdesig as designation
            FROM prmst
        """

        insert_sql_prmst = """
            INSERT INTO employee (
                uuid, itime, utime, zid, spid, spname, department, designation
            ) VALUES %s
        """
        self.execute_sync(select_sql_prmst, insert_sql_prmst, 'employee', clear_table_first=True)

         # Items (caitem)
        select_sql_casup = """
            SELECT
                casup.ztime AS itime,
                casup.zutime AS utime,
                casup.zid AS zid,
                casup.xsup AS supid,
                casup.xorg AS supname,
                casup.xcity AS supadd
            FROM casup
        """
        insert_sql_casup = """
            INSERT INTO casup (
                uuid, itime, utime, zid, supid, supname, supadd
            ) VALUES %s
        """
        self.execute_sync(select_sql_casup, insert_sql_casup, 'casup', clear_table_first=True)