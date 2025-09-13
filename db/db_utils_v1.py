import streamlit as st
import psycopg2
from configparser import ConfigParser
from db import sql_scripts
from sqlalchemy import create_engine
import pandas as pd



# db/db_utils.py

def config(filename='config/database.ini', section='postgresql'):
    parser = ConfigParser()
    parser.read(filename)
    
    db_params = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db_params[param[0]] = param[1]
        
    return db_params

@st.cache_data(ttl=600) 
def get_data(query, *args):
    """Fetch data using a specific SQL query."""
    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cursor = conn.cursor()
        if args:
            cursor.execute(query, args)
        else:
            cursor.execute(query)
            
        records = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]
        cursor.close()
        return records, colnames
    except Exception as error:
        print(f"Error encountered: {error}")
        return None, None
    finally:
        if conn:
            conn.close()

@st.cache_data(ttl=600) 
def get_gl_details(zid, project=None, year=None, smonth=None, emonth=None, is_bs=False, is_project=False):
    DB_CONFIG = config()  # Fetch the database configuration using the config function

    # Create the engine using the configuration
    engine = create_engine('postgresql://{user}:{password}@{host}:{port}/{database}'.format(**DB_CONFIG))
    where_clauses = [
        "glmst.zid = %(zid)s",
        "gldetail.zid = %(zid)s",
        "glheader.zid = %(zid)s"
    ]
    
    select_columns = [
        "glmst.zid", "glmst.ac_code", "glheader.year", "glheader.month", "SUM(gldetail.value) as sum"
    ]
    
    group_by_columns = [
        "glmst.zid", "glmst.ac_code", "glheader.year", "glheader.month","glmst.usage"
    ]
    
    if is_project:
        select_columns.extend(["glmst.ac_type", "glmst.ac_lv1", "glmst.ac_lv2", "glmst.usage"])
        group_by_columns.extend(["glmst.ac_type", "glmst.ac_lv1", "glmst.ac_lv2"])
        where_clauses.append("gldetail.project = %(project)s")
        
    if is_bs:
        where_clauses.append("(glmst.ac_type = 'Asset' OR glmst.ac_type = 'Liability')")
    else:
        where_clauses.append("(glmst.ac_type = 'Income' OR glmst.ac_type = 'Expenditure')")
        
    if year:
        where_clauses.append("glheader.year = %(year)s")
        
    if is_bs:
        if emonth:
            where_clauses.append("glheader.month <= %(emonth)s")
    else:
        if smonth:
            where_clauses.append("glheader.month >= %(smonth)s")
        if emonth:
            where_clauses.append("glheader.month <= %(emonth)s")
    
    where_clause = " AND ".join(where_clauses)
    select_clause = ", ".join(select_columns)
    group_by_clause = ", ".join(group_by_columns)
    
    query = f"""
        SELECT {select_clause}
        FROM glmst
        JOIN gldetail ON glmst.ac_code = gldetail.ac_code
        JOIN glheader ON gldetail.voucher = glheader.voucher
        WHERE {where_clause}
        GROUP BY {group_by_clause}
        ORDER BY glheader.month ASC, glmst.ac_type
    """
    
    params = {
        'zid': zid,
        'project': project,
        'year': year,
        'smonth': smonth,
        'emonth': emonth
    } 
    print(query)
    try:
        df = pd.read_sql(query, con=engine, params=params)
    finally:
        engine.dispose()
    return df

@st.cache(allow_output_mutation=True) 
def get_gl_details_ap_project(zid, project, year, xacc, emonth, sup_list):
    DB_CONFIG = config()  # Fetch the database configuration using the config function

    # Create the engine using the configuration
    engine = create_engine('postgresql://{user}:{password}@{host}:{port}/{database}'.format(**DB_CONFIG))
    sub_conditions = [
        ("INTERNAL", "IN" if isinstance(sup_list, tuple) else "="),
        ("EXTERNAL", "NOT IN" if isinstance(sup_list, tuple) else "!=")
    ]
    
    dfs = []
    try:
        for label, condition in sub_conditions:
            query = f"""
                SELECT '{label}' as source, SUM(gldetail.xprime) as total
                FROM glmst
                JOIN gldetail ON glmst.xacc = gldetail.xacc
                JOIN glheader ON gldetail.xvoucher = glheader.xvoucher
                WHERE glmst.zid = %(zid)s
                AND gldetail.zid = %(zid)s
                AND glheader.zid = %(zid)s
                AND gldetail.xproj = %(project)s
                AND glmst.xacc = %(xacc)s
                AND glheader.xyear = %(year)s
                AND glheader.xper <= %(emonth)s
                AND gldetail.xsub {condition} %(sup_list)s
            """
            df = pd.read_sql(query, con=engine, params={'zid': zid, 'project': project, 'xacc': xacc, 'year': year, 'emonth': emonth, 'sup_list': sup_list})
            dfs.append(df)
    finally:
        engine.dispose()
    
    return pd.concat(dfs, axis=0)

@st.cache(allow_output_mutation=True) 
def get_gl_master(zid):
    DB_CONFIG = config()  # Fetch the database configuration using the config function

    # Create the engine using the configuration
    engine = create_engine('postgresql://{user}:{password}@{host}:{port}/{database}'.format(**DB_CONFIG))
    try:
        query = f"SELECT ac_code, ac_name, ac_type, ac_lv1, ac_lv2, ac_lv3, ac_lv4 FROM glmst WHERE glmst.zid = %(zid)s"
        df = pd.read_sql(query, con=engine, params={'zid': zid})
    finally:
        engine.dispose()  # Explicitly close the connection
    
    return df
