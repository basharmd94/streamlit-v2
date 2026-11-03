#!/usr/bin/env python
"""EMERGENCY FIX (3 Nov 2026): put Zepto's xsrate (the MRP) back, and clear the shop prices still left in GI's xsrate.

What happened: the one-time shop-price load (set_xsrate_from_json.py) was undone, but
  * all 792 Zepto (100005) xsrate values were later set to 0 by something outside these scripts, and
  * 58 GI (100000) items still hold values: the undo leaves an item alone when someone changed it after the load.

    python fix_xsrate_zepto_gi.py              DRY RUN: reads only, prints the plan, writes logs/xsrate_fix_<time>.csv
    python fix_xsrate_zepto_gi.py --commit     writes (one transaction); old values saved in table um_xsrate_fix_backup
    python fix_xsrate_zepto_gi.py --undo       shows what would be put back (add --commit to do it)

Rules (nothing else is touched: not HMBR 100001, not any other column except zutime of the rows it changes):
  Zepto : an item whose xsrate is 0 or empty gets the MRP from zepto_xsrate_mrp_*.json (taken from the local copy of the ERP, owner
          confirmed the MRPs have not changed). An item that already holds ANY other value is left alone and listed.
  GI    : every item whose xsrate is above 0 is set to 0.
Stops without writing if more than 5% of the JSON items are not in the database (wrong database?), unless --force.
Needs only psycopg2; Python 3.6+; PostgreSQL 9.4.
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
ZEPTO, GI = 100005, 100000
BACKUP = "um_xsrate_fix_backup"
TOL = Decimal("0.005")


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


def load_mrp(path):
    doc = json.load(open(path, encoding="utf-8"))
    mrp = {}
    for n, it in enumerate(doc.get("items") or [], 1):
        try:
            xitem, val = str(it["xitem"]).strip(), Decimal(str(it["xsrate"]))
        except (KeyError, ValueError, InvalidOperation):
            sys.exit("item %d in the JSON is malformed: %r" % (n, it))
        if not xitem or not val.is_finite() or val <= 0 or xitem in mrp:
            sys.exit("item %d in the JSON is not allowed (xitem set once, xsrate above 0): %r" % (n, it))
        mrp[xitem] = val
    if not mrp:
        sys.exit("no items in %s" % path)
    return doc, mrp


def build_plan(cur, mrp):
    """Every row gets an action: restore / zero / same / differs (left alone) / not_in_json (left alone) / missing."""
    plan = []
    cur.execute("SELECT xitem, xdesc, xsrate FROM caitem WHERE zid = %s ORDER BY xitem", (ZEPTO,))
    live = {}
    for r in cur.fetchall():
        live[r["xitem"]] = r
        old, want = r["xsrate"], mrp.get(r["xitem"])
        if want is None:
            act = "not_in_json"
        elif old is None or old == 0:
            act = "restore"
        elif abs(Decimal(old) - want) < TOL:
            act = "same"
        else:
            act = "differs"
        plan.append({"zid": ZEPTO, "xitem": r["xitem"], "name": r["xdesc"] or "", "old": old,
                     "new": want if act == "restore" else None, "action": act})
    for xitem, want in sorted(mrp.items()):
        if xitem not in live:
            plan.append({"zid": ZEPTO, "xitem": xitem, "name": "", "old": None, "new": None, "action": "missing"})
    cur.execute("SELECT xitem, xdesc, xsrate FROM caitem WHERE zid = %s AND xsrate > 0 ORDER BY xitem", (GI,))
    for r in cur.fetchall():
        plan.append({"zid": GI, "xitem": r["xitem"], "name": r["xdesc"] or "", "old": r["xsrate"], "new": Decimal(0), "action": "zero"})
    return plan


def write_csv(plan):
    os.makedirs(os.path.join(HERE, "..", "logs"), exist_ok=True)
    path = os.path.abspath(os.path.join(HERE, "..", "logs", "xsrate_fix_%s.csv" % dt.datetime.now().strftime("%Y%m%d_%H%M%S")))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["zid", "xitem", "name", "action", "old_xsrate", "new_xsrate"])
        for r in plan:
            w.writerow([r["zid"], r["xitem"], r["name"], r["action"], r["old"], r["new"]])
    return path


def counts(cur):
    cur.execute("SELECT zid, count(*) AS items, sum(CASE WHEN xsrate IS NULL THEN 1 ELSE 0 END) AS xs_null,"
                " sum(CASE WHEN xsrate = 0 THEN 1 ELSE 0 END) AS xs_zero, sum(CASE WHEN xsrate > 0 THEN 1 ELSE 0 END) AS xs_filled"
                " FROM caitem WHERE zid IN (100000, 100001, 100005) GROUP BY zid ORDER BY zid")
    print("\n%-8s %7s %8s %8s %10s" % ("company", "items", "empty", "zero", "filled"))
    for r in cur.fetchall():
        print("%-8s %7d %8d %8d %10d" % (r["zid"], r["items"], r["xs_null"], r["xs_zero"], r["xs_filled"]))


def do_fix(conn, cur, todo):
    cur.execute("SELECT to_regclass(%s)", ("public." + BACKUP,))
    if list(cur.fetchone().values())[0] is None:
        cur.execute("CREATE TABLE " + BACKUP + " (id serial PRIMARY KEY, zid integer, xitem text, old_xsrate numeric,"
                    " new_xsrate numeric, saved_at timestamp DEFAULT now())")
    for r in todo:
        cur.execute("INSERT INTO " + BACKUP + " (zid, xitem, old_xsrate, new_xsrate) VALUES (%s, %s, %s, %s)",
                    (r["zid"], r["xitem"], r["old"], r["new"]))
        cur.execute("UPDATE caitem SET xsrate = %s, zutime = now() WHERE zid = %s AND xitem = %s"
                    " AND xsrate IS NOT DISTINCT FROM %s", (r["new"], r["zid"], r["xitem"], r["old"]))
        if cur.rowcount != 1:
            conn.rollback()
            sys.exit("STOPPED: updating %s/%s touched %d rows (someone changed it while this ran?). Nothing was written."
                     % (r["zid"], r["xitem"], cur.rowcount))
    conn.commit()
    return len(todo)


def do_undo(conn, cur, commit):
    cur.execute("SELECT to_regclass(%s)", ("public." + BACKUP,))
    if list(cur.fetchone().values())[0] is None:
        print("no backup table (%s): this fix was never run here." % BACKUP)
        return
    cur.execute("SELECT b.id, b.zid, b.xitem, b.old_xsrate, b.new_xsrate, c.xsrate AS now FROM " + BACKUP + " b"
                " JOIN caitem c ON c.zid = b.zid AND c.xitem = b.xitem ORDER BY b.id DESC")
    rows = cur.fetchall()
    ok = [r for r in rows if r["now"] is not None and r["new_xsrate"] is not None and r["now"] == r["new_xsrate"]]
    print("backup rows: %d; can be put back (xsrate still holds what this fix wrote): %d; changed since by someone: %d"
          % (len(rows), len(ok), len(rows) - len(ok)))
    if not commit:
        print("DRY RUN: nothing was put back. Add --commit to do it.")
        return
    n = 0
    for r in ok:
        cur.execute("UPDATE caitem SET xsrate = %s, zutime = now() WHERE zid = %s AND xitem = %s AND xsrate = %s",
                    (r["old_xsrate"], r["zid"], r["xitem"], r["new_xsrate"]))
        if cur.rowcount == 1:
            cur.execute("DELETE FROM " + BACKUP + " WHERE id = %s", (r["id"],))
            n += 1
    conn.commit()
    print("put back %d values; the backup rows for them were removed." % n)
    counts(cur)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    newest = sorted(glob.glob(os.path.join(HERE, "zepto_xsrate_mrp_*.json")))
    ap.add_argument("--json", default=newest[-1] if newest else None, help="default: the newest zepto_xsrate_mrp_*.json next to this script")
    ap.add_argument("--config", default=os.path.join(HERE, "..", "erp_import.ini"))
    ap.add_argument("--commit", action="store_true", help="write (default: dry run)")
    ap.add_argument("--undo", action="store_true", help="put the values this fix replaced back, from the backup table")
    ap.add_argument("--force", action="store_true", help="go on even if many JSON items are missing in the database")
    args = ap.parse_args()

    conn, where = connect(args.config)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    print("database : %s" % where)
    print("mode     : %s" % ("UNDO" if args.undo else "FIX") + ("  --commit (WRITES)" if args.commit else "  (dry run, nothing is written)"))
    if args.undo:
        do_undo(conn, cur, args.commit)
        return 0

    if not args.json:
        sys.exit("no zepto_xsrate_mrp_*.json found next to this script; pass --json")
    doc, mrp = load_mrp(args.json)
    print("json     : %s (%d Zepto items; %s)" % (os.path.basename(args.json), len(mrp), doc.get("source", "")))
    plan = build_plan(cur, mrp)

    n = {}
    for r in plan:
        n[(r["zid"], r["action"])] = n.get((r["zid"], r["action"]), 0) + 1
    print("\n%-8s %-12s %6s" % ("company", "action", "items"))
    for k in sorted(n):
        print("%-8s %-12s %6d" % (k[0], k[1], n[k]))
    print("\nNOW (before any change):")
    counts(cur)

    for act, label in (("differs", "Zepto items that already hold another value (LEFT ALONE)"),
                       ("not_in_json", "Zepto items not in the JSON (LEFT ALONE)"),
                       ("missing", "JSON items NOT FOUND in this database")):
        rows = [r for r in plan if r["action"] == act]
        if rows:
            print("\n%s: %d" % (label, len(rows)))
            for r in rows[:15]:
                print("    %s  %s  now=%s json=%s" % (r["xitem"], safe(r["name"])[:36], r["old"], mrp.get(r["xitem"])))
            if len(rows) > 15:
                print("    ... and %d more (all in the CSV)" % (len(rows) - 15))
    gi = [r for r in plan if r["action"] == "zero"]
    if gi:
        print("\nGI items that will be set to 0 (old value is kept in %s): %d" % (BACKUP, len(gi)))
        for r in gi:
            print("    %s  %s  %s" % (r["xitem"], safe(r["name"])[:36], r["old"]))
    print("\nfull plan: %s" % write_csv(plan))

    missing = sum(1 for r in plan if r["action"] == "missing")
    if missing > 0.05 * len(mrp) and not args.force:
        sys.exit("\nSTOPPED: %d of %d JSON items are not in this database. Is this the right database? (--force to go on)" % (missing, len(mrp)))
    todo = [r for r in plan if r["action"] in ("restore", "zero")]
    if not args.commit:
        print("\nDRY RUN: %d items would be written. Add --commit to do it." % len(todo))
        return 0
    if not todo:
        print("\nNothing to write.")
        return 0
    do_fix(conn, cur, todo)
    print("\nWROTE xsrate on %d items. Old values: table %s (undo: python fix_xsrate_zepto_gi.py --undo --commit)." % (len(todo), BACKUP))
    print("\nNOW (after the change):")
    counts(cur)
    return 0


if __name__ == "__main__":
    sys.exit(main())
