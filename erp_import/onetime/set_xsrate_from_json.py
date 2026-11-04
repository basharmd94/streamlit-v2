#!/usr/bin/env python
"""ONE-TIME LOAD: write the online shop's current sales prices into caitem.xsrate of 100001 (HMBR) and 100000 (GI).

RETIRED 2026-11-03. Do not run the load again: the ERP uses xsrate of 100001/100000 for its own calculations, and this load
broke them (it was undone with --undo; see fix_xsrate_zepto_gi.py). The sell-price rule no longer reads that column for
HMBR/GI (it uses their xstdprice x markup). --undo still works; --commit now needs the long flag below, on purpose.

Run it once, on the HMBR server, so that xsrate in both companies matches what the shop charges today. After that the importer's
sell-price rule (erp_import.py --job price) reads xsrate from the source company, so 100007 and the shop stay consistent.

The input is a JSON file made from the shop's live catalogue (dev_tools/make_xsrate_json.py):
    {"taken_at": "...", "items": [{"zid": 100001, "xitem": "0718", "price": 80, "name": "..."}, ...]}
`xitem` is the shop SKU = the source company's caitem.xitem. Only these rows are touched; every other xsrate stays as it is.

    python set_xsrate_from_json.py                 DRY RUN: reads only, prints the plan, writes logs/xsrate_upload_<time>.csv
    python set_xsrate_from_json.py --commit        writes xsrate (one transaction), old values saved in table um_xsrate_backup
    python set_xsrate_from_json.py --undo          shows what would be put back (add --commit to do it)

Safe to run twice (a row whose xsrate already equals the price is left alone). Stops without writing if more than 5% of the JSON
items do not exist in the database (wrong database?), unless --force. Needs only psycopg2; Python 3.6+; PostgreSQL 9.4.
"""
import argparse
import configparser
import csv
import datetime as dt
import glob
import json
import os
import sys
from decimal import Decimal, InvalidOperation

import psycopg2
import psycopg2.extras

HERE = os.path.dirname(os.path.abspath(__file__))
ZIDS = (100001, 100000)
BACKUP = "um_xsrate_backup"


def safe(s):
    """Item names may hold characters a Windows console cannot print."""
    return str(s).encode("ascii", "replace").decode("ascii")


def connect(config_path):
    cp = configparser.ConfigParser(interpolation=None)
    if not cp.read(config_path, encoding="utf-8"):
        sys.exit("config not found: %s (use --config)" % config_path)
    g = lambda k, d="": cp.get("database", k, fallback=d).split("#")[0].strip()
    conn = psycopg2.connect(host=g("host", "localhost"), port=g("port", "5432"), dbname=g("dbname", "da"),
                            user=g("user", "postgres"), password=g("password"))
    conn.set_client_encoding("UTF8")
    return conn, "%s:%s/%s" % (g("host", "localhost"), g("port", "5432"), g("dbname", "da"))


def load_items(path):
    doc = json.load(open(path, encoding="utf-8"))
    items, seen = [], set()
    for n, it in enumerate(doc.get("items") or [], 1):
        try:
            zid, xitem, price = int(it["zid"]), str(it["xitem"]).strip(), Decimal(str(it["price"]))
        except (KeyError, ValueError, InvalidOperation):
            sys.exit("item %d in the JSON is malformed: %r" % (n, it))
        if zid not in ZIDS or not xitem or not price.is_finite() or price <= 0:
            sys.exit("item %d in the JSON is not allowed (zid must be 100001 or 100000, xitem set, price above 0): %r" % (n, it))
        if (zid, xitem) in seen:
            sys.exit("the JSON lists %s/%s twice" % (zid, xitem))
        seen.add((zid, xitem))
        items.append({"zid": zid, "xitem": xitem, "price": price, "name": it.get("name") or "", "basis": it.get("basis") or ""})
    if not items:
        sys.exit("no items in %s" % path)
    return doc, items


def build_plan(cur, items):
    rows = {}
    for zid in ZIDS:
        xs = tuple(i["xitem"] for i in items if i["zid"] == zid)
        if xs:
            cur.execute("SELECT zid, xitem, xdesc, xstdprice, xsrate FROM caitem WHERE zid = %s AND xitem IN %s", (zid, xs))
            for r in cur.fetchall():
                rows[(r["zid"], r["xitem"])] = r
    skus = tuple(i["xitem"] for i in items)
    cur.execute("SELECT trim(xalias) AS sku, xitemnew FROM caitem WHERE zid = 100007 AND trim(xalias) IN %s", (skus,))
    pair = {r["sku"]: str(r["xitemnew"]).strip() for r in cur.fetchall()}
    plan = []
    for it in items:
        r = rows.get((it["zid"], it["xitem"]))
        row = {"zid": it["zid"], "xitem": it["xitem"], "name": it["name"], "new": it["price"], "old": None, "std": None,
               "status": "", "notes": []}
        if r is None:
            row["status"] = "missing"
            row["notes"].append("no such item in caitem of this company")
        else:
            row["old"] = r["xsrate"]
            row["std"] = r["xstdprice"]
            row["name"] = row["name"] or r["xdesc"] or ""
            if r["xsrate"] is not None and abs(Decimal(r["xsrate"]) - it["price"]) < Decimal("0.005"):
                row["status"] = "same"
            elif r["xsrate"] is None or r["xsrate"] == 0:
                row["status"] = "set_empty"
            else:
                row["status"] = "overwrite"
            if r["xstdprice"] and it["price"] < r["xstdprice"]:
                row["notes"].append("shop price is BELOW the std price %s" % r["xstdprice"])
            elif r["xstdprice"] and it["price"] > 2 * r["xstdprice"]:
                row["notes"].append("shop price is more than DOUBLE the std price %s" % r["xstdprice"])
        if it.get("basis"):
            row["notes"].append(it["basis"])
        if pair.get(it["xitem"]) != str(it["zid"]):
            row["notes"].append("100007 does not pair this SKU with company %s (pairing: %s)" % (it["zid"], pair.get(it["xitem"], "none")))
        plan.append(row)
    return plan


def write_csv(plan):
    os.makedirs(os.path.join(HERE, "..", "logs"), exist_ok=True)
    path = os.path.abspath(os.path.join(HERE, "..", "logs", "xsrate_upload_%s.csv" % dt.datetime.now().strftime("%Y%m%d_%H%M%S")))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["zid", "xitem", "name", "status", "old_xsrate", "new_xsrate", "std_price", "notes"])
        for r in plan:
            w.writerow([r["zid"], r["xitem"], r["name"], r["status"], r["old"], r["new"], r["std"], "; ".join(r["notes"])])
    return path


def do_load(conn, cur, plan, commit):
    todo = [r for r in plan if r["status"] in ("set_empty", "overwrite")]
    if not commit or not todo:
        return 0
    cur.execute("SELECT to_regclass(%s)", ("public." + BACKUP,))
    if list(cur.fetchone().values())[0] is None:
        cur.execute("CREATE TABLE " + BACKUP + " (id serial PRIMARY KEY, zid integer, xitem text, old_xsrate numeric,"
                    " new_xsrate numeric, saved_at timestamp DEFAULT now())")
    for r in todo:
        cur.execute("INSERT INTO " + BACKUP + " (zid, xitem, old_xsrate, new_xsrate) VALUES (%s, %s, %s, %s)",
                    (r["zid"], r["xitem"], r["old"], r["new"]))
        cur.execute("UPDATE caitem SET xsrate = %s, zutime = now() WHERE zid = %s AND xitem = %s", (r["new"], r["zid"], r["xitem"]))
        if cur.rowcount != 1:
            conn.rollback()
            sys.exit("STOPPED: updating %s/%s touched %d rows. Nothing was written." % (r["zid"], r["xitem"], cur.rowcount))
    conn.commit()
    return len(todo)


def do_undo(conn, cur, commit):
    cur.execute("SELECT to_regclass(%s)", ("public." + BACKUP,))
    if list(cur.fetchone().values())[0] is None:
        print("no backup table (%s): nothing was ever loaded here." % BACKUP)
        return
    cur.execute("SELECT b.id, b.zid, b.xitem, b.old_xsrate, b.new_xsrate, c.xsrate AS now FROM " + BACKUP + " b"
                " JOIN caitem c ON c.zid = b.zid AND c.xitem = b.xitem ORDER BY b.id DESC")
    rows = cur.fetchall()
    ok = [r for r in rows if r["now"] is not None and r["now"] == r["new_xsrate"]]
    print("backup rows: %d; can be put back (xsrate still holds the loaded value): %d; changed since by someone: %d"
          % (len(rows), len(ok), len(rows) - len(ok)))
    if not commit:
        print("DRY RUN: nothing was put back. Add --commit to do it.")
        return
    n = 0
    for r in ok:                                    # newest load first, so repeated loads unwind in order
        cur.execute("UPDATE caitem SET xsrate = %s, zutime = now() WHERE zid = %s AND xitem = %s AND xsrate = %s",
                    (r["old_xsrate"], r["zid"], r["xitem"], r["new_xsrate"]))
        if cur.rowcount == 1:
            cur.execute("DELETE FROM " + BACKUP + " WHERE id = %s", (r["id"],))
            n += 1
    conn.commit()
    print("put back %d values; the backup rows for them were removed." % n)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    newest = sorted(glob.glob(os.path.join(HERE, "xsrate_shop_prices_*.json")))
    ap.add_argument("--json", default=newest[-1] if newest else None, help="default: the newest xsrate_shop_prices_*.json next to this script")
    ap.add_argument("--config", default=os.path.join(HERE, "..", "erp_import.ini"))
    ap.add_argument("--commit", action="store_true", help="write (default: dry run)")
    ap.add_argument("--undo", action="store_true", help="put the old xsrate values back from the backup table")
    ap.add_argument("--force", action="store_true", help="go on even if many JSON items are missing in the database")
    ap.add_argument("--i-know-xsrate-is-used-by-the-erp", dest="retired_ok", action="store_true",
                    help="needed for --commit: this load is retired")
    args = ap.parse_args()
    if args.commit and not args.undo and not args.retired_ok:
        sys.exit("RETIRED: this load changed xsrate values the ERP uses for its own calculations and was undone on 3 Nov 2026.\n"
                 "Not writing. (--undo still works.)")

    conn, where = connect(args.config)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    print("database : %s" % where)
    print("mode     : %s" % ("UNDO" if args.undo else "LOAD") + ("  --commit (WRITES)" if args.commit else "  (dry run, nothing is written)"))
    if args.undo:
        do_undo(conn, cur, args.commit)
        return 0

    if not args.json:
        sys.exit("no JSON file found next to this script; pass --json")
    doc, items = load_items(args.json)
    print("json     : %s (shop prices taken %s, %d items)" % (os.path.basename(args.json), doc.get("taken_at", "?"), len(items)))
    plan = build_plan(cur, items)

    count = {}
    for r in plan:
        count[(r["zid"], r["status"])] = count.get((r["zid"], r["status"]), 0) + 1
    print("\n%-8s %-10s %7s" % ("company", "status", "items"))
    for k in sorted(count):
        print("%-8s %-10s %7d" % (k[0], k[1], count[k]))
    for status, label in (("missing", "NOT FOUND in this database"), ("overwrite", "xsrate already had a different value (will be replaced; old value is in the backup)")):
        rows = [r for r in plan if r["status"] == status]
        if rows:
            print("\n%s: %d" % (label, len(rows)))
            for r in rows[:15]:
                print("    %s/%s  %s  old=%s new=%s" % (r["zid"], r["xitem"], safe(r["name"])[:40], r["old"], r["new"]))
            if len(rows) > 15:
                print("    ... and %d more (all in the CSV)" % (len(rows) - 15))
    notes = [r for r in plan if r["notes"] and r["status"] != "missing"]
    if notes:
        print("\nto look at (still written): %d" % len(notes))
        for r in notes[:20]:
            print("    %s/%s  %s  %s" % (r["zid"], r["xitem"], safe(r["name"])[:34], "; ".join(r["notes"])))
    csv_path = write_csv(plan)
    print("\nfull plan: %s" % csv_path)

    missing = sum(1 for r in plan if r["status"] == "missing")
    if missing > 0.05 * len(plan) and not args.force:
        sys.exit("\nSTOPPED: %d of %d JSON items are not in this database. Is this the right database? (--force to go on)" % (missing, len(plan)))
    n = do_load(conn, cur, plan, args.commit)
    if args.commit and n == 0:
        print("\nNothing to write: xsrate already matches the JSON for every item.")
    elif args.commit:
        print("\nWROTE xsrate on %d items. Old values: table %s (undo: python set_xsrate_from_json.py --undo --commit)." % (n, BACKUP))
    else:
        todo = sum(1 for r in plan if r["status"] in ("set_empty", "overwrite"))
        print("\nDRY RUN: %d items would be written. Add --commit to do it." % todo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
