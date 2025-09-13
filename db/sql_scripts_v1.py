# db/sql_scripts.py
def get_sales_data():
    return """SELECT 
                sales.zid,
                sales.ordernumber as voucher,
                sales.date,
                EXTRACT(YEAR FROM sales.date) AS year,
                EXTRACT(MONTH FROM sales.date) AS month,
                sales.sp_id as spid,
                employee.spname,
                sales.cusid,
                cacus.cusname,
                cacus.cuscity as area,
                sales.itemcode,
                caitem.itemname,
                caitem.itemgroup2 as itemgroup,
                sales.quantity,
                sales.totalsales,
                sales.cost
            FROM 
                sales 
            JOIN 
                employee ON sales.sp_id = employee.spid AND sales.zid = employee.zid
            JOIN 
                cacus ON sales.cusid = cacus.cusid AND sales.zid = cacus.zid
            JOIN 
                caitem ON sales.itemcode = caitem.itemcode AND sales.zid = caitem.zid
            WHERE 
                sales.zid = %s"""

def get_return_data():
    return """SELECT
                return.zid,
                return.revoucher,
                return.date,
                EXTRACT(YEAR FROM return.date) AS year,
                EXTRACT(MONTH FROM return.date) AS month,
                return.sp_id as spid,
                employee.spname,
                return.cusid,
                cacus.cusname,
                cacus.cuscity as area,
                return.itemcode,
                caitem.itemname,
                caitem.itemgroup2 as itemgroup,
                return.returnqty,
                return.treturnamt,
                return.returncost
            FROM 
                return
            JOIN 
                employee ON return.sp_id = employee.spid AND return.zid = employee.zid
            JOIN 
                cacus ON return.cusid = cacus.cusid AND return.zid = cacus.zid
            JOIN 
                caitem ON return.itemcode = caitem.itemcode AND return.zid = caitem.zid
            WHERE 
                return.zid = %s"""

def get_purchase_data():
    return """SELECT 
                purchase.zid,
                purchase.combinedate,
                purchase.povoucher,
                purchase.grnvoucher,
                CASE
                    WHEN caitem.packcode IS NOT NULL AND caitem.packcode <> '' AND caitem.packcode != 'NO' THEN caitem.packcode
                    ELSE purchase.itemcode
                END AS itemcode,
                caitem.itemname,
                purchase.shipmentname,
                purchase.quantity,
                purchase.cost,
                purchase.status
            FROM 
                purchase 
            JOIN 
                caitem ON purchase.itemcode =  caitem.itemcode AND purchase.zid = caitem.zid
            WHERE 
                purchase.zid IN (%s,%s)
                AND purchase.povoucher LIKE 'IP--%%'
                AND purchase.status IN ('5-Received','1-Open')
            """


def get_product_inventory_data():
    return """SELECT 
                stock.zid,
                CASE
                    WHEN caitem.packcode IS NOT NULL 
                    AND caitem.packcode <> '' 
                    AND caitem.packcode != 'NO'
                    AND LEFT(caitem.packcode,2) != 'KH' THEN caitem.packcode
                    ELSE stock.itemcode
                END AS itemcode,
                caitem.itemname,
                caitem.itemgroup,
                stock.stockqty,
                stock.stockvalue
            FROM 
                stock
            JOIN
                caitem ON stock.itemcode = caitem.itemcode AND stock.zid = caitem.zid
            WHERE 
                stock.zid IN  (%s,%s)"""

def get_inventory_value_data():
    return """SELECT 
                stock_value.zid,
                stock_value.year,
                stock_value.month,
                stock_value.warehouse,
                stock_value.stockvalue
            FROM stock_value WHERE zid = (%s)"""

def get_collection_data():
    return """SELECT
                glmst.zid,
                gldetail.voucher as glvoucher,
                gldetail.ac_sub as cusid,
                glheader.date,
                glheader.year,
                glheader.month,
                ABS(SUM(gldetail.value)) as value
            FROM
                glmst
            JOIN
                gldetail ON glmst.ac_code = gldetail.ac_code AND glmst.zid = gldetail.zid
            JOIN
                glheader ON gldetail.voucher = glheader.voucher AND glmst.zid = glheader.zid
            WHERE
                glmst.zid = %s
            AND (
                gldetail.voucher LIKE 'JV--%%'
                OR gldetail.voucher LIKE 'RCT-%%'
                OR gldetail.voucher LIKE 'CRCT%%'
                OR gldetail.voucher LIKE 'STJV%%'
                OR gldetail.voucher LIKE 'BRCT%%'
            )
            AND glmst.usage IN ('AR')
            GROUP BY
                glmst.zid,gldetail.voucher,gldetail.ac_sub,glmst.usage,glheader.date,glheader.year,glheader.month"""



def get_payment_data():
    return """SELECT
                glmst.zid as zid,
                gldetail.voucher as glvoucher,
                gldetail.ac_sub as account,
                glmst.ac_name as description,
                glheader.date,
                glheader.year,
                glheader.month,
                SUM(gldetail.value) as value
            FROM
                glmst
            JOIN
                gldetail ON glmst.ac_code = gldetail.ac_code AND glmst.zid = gldetail.zid
            JOIN
                glheader ON gldetail.voucher = glheader.voucher AND glmst.zid = glheader.zid
            WHERE
                glmst.zid = 100001
                AND gldetail.zid = 100001
                AND glheader.zid = 100001
                AND (
                    gldetail.voucher LIKE 'CPAY%%'
                    OR gldetail.voucher LIKE 'BPAY%%'
                    OR gldetail.voucher LIKE 'STJV%%'
                )
            GROUP BY
                glmst.zid,
                gldetail.voucher,
                gldetail.ac_sub,
                glmst.ac_name,
                glheader.date,
                glheader.year,
                glheader.month
            HAVING
                SUM(gldetail.value) > 0
            ORDER BY
                SUM(gldetail.value) DESC"""

