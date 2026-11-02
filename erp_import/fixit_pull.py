#!/usr/bin/env python3
"""Fixit -> company 100007: stock, selling price and cost of the Fixit items the shop sells.

Fixit is a separate ERP on its own server. This script (run once a day on the HMBR server) reads Fixit's
`final_items_view` over a read-only login and keeps one row per shop Fixit item in `um_fixit_item`
(in the HMBR `da` database). erp_import.py then uses that table for the stock it sends to the shop, the
cost price, the selling price report and the purchase-order rate.

  followed items : 100007 caitem rows of Fixit (xitemnew = 'Fixit', or an empty xitemnew with a Fixit-shaped
                   xalias such as 18-018). Only these are read: never the whole Fixit catalogue.
  stock          : added over Fixit Gulshan (100001), Central (100002) and Ecommerce (100003); a store with
                   negative stock counts as 0 (the same rule as the master list the shop was filled from)
  selling price  : final_items_view.std_price of Gulshan (config [fixit] price_zid), else Ecommerce, else Central
  cost           : final_items_view.store_buy_price of Gulshan (= Central's sales rate), else Ecommerce
  (the master list rule 'stock above 5 and sales not 0' is only for NEW items and is not applied here)

Dry run unless --commit.   python fixit_pull.py [--commit] [--only-skus 18-018,26-4323]
"""
import argparse
import datetime as dt
import logging
import os
import sys
from decimal import Decimal

import psycopg2
import psycopg2.extras

import erp_import as E

LOG = logging.getLogger("fixit_pull")
LOCK_KEY = E.LOCK_KEY + 1
FIXIT_ZIDS = (100001, 100002, 100003)


def fixit_connect(cp):
    return psycopg2.connect(host=E.cfg_get(cp, "fixit", "host", "localhost"), port=E.cfg_get(cp, "fixit", "port", "5432"),
                            dbname=E.cfg_get(cp, "fixit", "dbname", "fixit"), user=E.cfg_get(cp, "fixit", "user", "postgres"),
                            password=E.cfg_get(cp, "fixit", "password", ""))


def followed_skus(cur, st):
    cur.execute("SELECT xitem, xalias FROM caitem WHERE zid = %s AND coalesce(xalias,'') <> '' AND"
                " (lower(xitemnew) = 'fixit' OR (coalesce(xitemnew,'') = '' AND xalias ~ '^[0-9]{2}-[0-9]{3,5}$'))"
                " ORDER BY xitem", (st.zid,))
    return {r["xalias"]: r["xitem"] for r in cur.fetchall()}


def read_fixit(fconn, skus):
    fc = fconn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    fc.execute("SELECT zid, trim(item_id) AS sku, item_name, item_group, std_price, stock, avg_monthly_sales,"
               " central_buy_price, store_buy_price FROM final_items_view"
               " WHERE zid IN %s AND trim(item_id) = ANY(%s)", (FIXIT_ZIDS, list(skus)))
    by = {}
    for r in fc.fetchall():
        by.setdefault(r["sku"], {})[r["zid"]] = r
    return by


def summarise(rows, price_zid):
    """One um_fixit_item row from the per-company rows of one SKU (or None when Fixit does not know it)."""
    if not rows:
        return None
    stock = sum(max(Decimal(str(r["stock"] or 0)), Decimal(0)) for r in rows.values())
    sales = sum(Decimal(str(r["avg_monthly_sales"] or 0)) for r in rows.values())
    order = [price_zid] + [z for z in (100003, 100002, 100001) if z != price_zid]
    price_row = next((rows[z] for z in order if z in rows and rows[z]["std_price"] and Decimal(str(rows[z]["std_price"])) > 0), None)
    cost_row = next((rows[z] for z in (price_zid, 100003) if z in rows and rows[z]["store_buy_price"]
                     and Decimal(str(rows[z]["store_buy_price"])) > 0), None)
    ref = price_row or rows.get(price_zid) or next(iter(rows.values()))
    return {"fixit_zid": (price_row or ref)["zid"], "stock": stock,
            "std_price": Decimal(str(price_row["std_price"])) if price_row else None,
            "cost": Decimal(str(cost_row["store_buy_price"])) if cost_row else None,
            "item_group": ref["item_group"], "sales": sales}


def save(cur, sku, s):
    cur.execute("SELECT 1 FROM um_fixit_item WHERE sku = %s", (sku,))
    exists = cur.fetchone() is not None
    if s is None:
        vals = (None, None, None, None, None, None, False)
    else:
        vals = (s["fixit_zid"], s["stock"], s["std_price"], s["cost"], s["item_group"], s["sales"], True)
    if exists:
        cur.execute("UPDATE um_fixit_item SET fixit_zid=%s, stock=%s, std_price=%s, cost=%s, item_group=%s, sales=%s,"
                    " found=%s, as_of=now() WHERE sku=%s", vals + (sku,))
    else:
        cur.execute("INSERT INTO um_fixit_item (fixit_zid, stock, std_price, cost, item_group, sales, found, sku)"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s,%s)", vals + (sku,))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=os.path.join(E.HERE, "erp_import.ini"))
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--only-skus", default="")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    cp = E.load_config(args.config)
    st = E.Settings(cp)
    E.setup_logging(args.verbose)
    LOG.info("=== fixit_pull start | mode=%s ===", "COMMIT" if args.commit else "DRY-RUN (nothing is kept)")
    conn = E.connect(cp)
    lock = E.connect(cp)
    lc = lock.cursor()
    lc.execute("SELECT pg_try_advisory_lock(%s)", (LOCK_KEY,))
    if not lc.fetchone()[0]:
        LOG.info("another fixit_pull is running; exiting")
        return 0
    try:
        cur = E.dict_cur(conn)
        E.ensure_um_tables(cur)
        skus = followed_skus(cur, st)
        if args.only_skus:
            want = set(args.only_skus.split(","))
            skus = {k: v for k, v in skus.items() if k in want}
        LOG.info("followed Fixit items in 100007: %d", len(skus))
        if not skus:
            return 0
        fconn = fixit_connect(cp)
        by = read_fixit(fconn, skus)
        fconn.close()
        n_found = n_missing = n_nocost = n_noprice = 0
        for sku in sorted(skus):
            s = summarise(by.get(sku), st.fixit_price_zid)
            if s is None:
                n_missing += 1
                E.attention(cur, "fixit_missing", sku, "Fixit item %s (%s) is not in the Fixit database view" % (sku, skus[sku]))
            else:
                n_found += 1
                n_nocost += s["cost"] is None
                n_noprice += s["std_price"] is None
            save(cur, sku, s)
        LOG.info("read from Fixit: %d found, %d missing; without cost: %d; without price: %d", n_found, n_missing, n_nocost, n_noprice)
        for sku in sorted(skus)[:5]:
            cur.execute("SELECT * FROM um_fixit_item WHERE sku = %s", (sku,))
            r = cur.fetchone()
            LOG.info("   e.g. %s -> stock=%s price=%s cost=%s group=%s sales=%s (zid %s)", sku, r["stock"], r["std_price"],
                     r["cost"], r["item_group"], r["sales"], r["fixit_zid"])
        if args.commit:
            conn.commit()
            LOG.info("saved")
        else:
            conn.rollback()
            LOG.info("DRY-RUN: rolled back")
        return 0
    finally:
        try:
            lc.execute("SELECT pg_advisory_unlock(%s)", (LOCK_KEY,))
        except Exception:
            pass
        conn.close()
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
