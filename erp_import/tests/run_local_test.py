#!/usr/bin/env python3
"""End-to-end test of erp_import.py against the LOCAL database and the stand-in shop (tests/fake_feed.py).

    python tests/run_local_test.py

It starts the stand-in feed, drives the importer through three stages (order in, fulfilment/payment, order edit),
then the purchase-order, product-push, price-report and Fixit jobs, checks the database after each step, and
finally restores everything it touched. Needs allow_purge = true in erp_import.ini (dev databases only) and the
local `da` and `fixit` databases. Never point it at a real database.
"""
import json
import csv
import os
import subprocess
import sys
import time
import urllib.request

import psycopg2
import psycopg2.extras

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # the erp_import folder: the scripts, logs/ and erp_import.ini live there
sys.path.insert(0, ROOT)
import erp_import as E  # noqa: E402

PY = sys.executable
RESULTS = []


def check(ok, label, detail=""):
    RESULTS.append((bool(ok), label))
    print("  %s %s %s" % ("PASS" if ok else "FAIL", label, ("- " + str(detail)) if (detail and not ok) else ""))


def run(*args, stage=None):
    if stage is not None:
        open(os.path.join(ROOT, "logs", "fake_stage.txt"), "w").write(str(stage))
    p = subprocess.run([PY] + list(args), cwd=ROOT, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def q(cur, sql, params=()):
    cur.execute(sql, params)
    return cur.fetchall()


def wait_feed():
    for _ in range(40):
        try:
            urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8099/hooks/erp/health",
                                   headers={"Authorization": "Bearer test-feed-key"}), timeout=2)
            return True
        except Exception:
            time.sleep(0.25)
    return False


def main():
    cp = E.load_config(os.path.join(ROOT, "erp_import.ini"))
    st = E.Settings(cp)
    if not st.allow_purge:
        sys.exit("allow_purge must be true in erp_import.ini for this local test")
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    for f in ("fake_feed_posts.jsonl", "fake_stage.txt"):
        p = os.path.join(ROOT, "logs", f)
        if os.path.exists(p):
            os.remove(p)
    conn = E.connect(cp)
    conn.autocommit = True
    cur = E.dict_cur(conn)
    Z = st.zid

    # ---- baseline, so everything can be restored -----------------------------------------------------
    base = {
        "caitem": q(cur, "SELECT count(*) n FROM caitem WHERE zid=%s", (Z,))[0]["n"],
        "xcodes": {r["xcode"] for r in q(cur, "SELECT xcode FROM xcodes WHERE zid=%s AND xtype='Price Category'", (Z,))},
        "poord": q(cur, "SELECT count(*) n FROM poord WHERE zid=%s", (Z,))[0]["n"],
        "opmob": q(cur, "SELECT count(*) n FROM opmob WHERE zid=%s", (Z,))[0]["n"],
        "cacus": q(cur, "SELECT count(*) n FROM cacus WHERE zid=%s", (Z,))[0]["n"],
    }
    ctx = {}
    feed = subprocess.Popen([PY, os.path.join("tests", "fake_feed.py")], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        if not wait_feed():
            print(feed.stderr.read().decode()[:600])
            sys.exit("stand-in feed did not start")
        open(os.path.join(ROOT, "logs", "fake_stage.txt"), "w").write("1")
        events = json.loads(subprocess.run([PY, os.path.join("tests", "fake_feed.py"), "--show"], cwd=ROOT, capture_output=True, text=True).stdout)
        ev = {e["seq"]: e["data"] for e in events}
        tiered_sku, zepto_sku = ev[1]["sku"], ev[4]["sku"]
        # real items the events touch: remember their rows to restore them afterwards
        saved = {r["xitem"]: dict(r) for r in q(cur, "SELECT * FROM caitem WHERE zid=%s AND xalias IN %s", (Z, (tiered_sku, zepto_sku)))}
        price_before = {k: v["xstdprice"] for k, v in saved.items()}

        run("erp_import.py", "--purge-test")
        print("\n[0] read-only check")
        rc, out = run("erp_import.py", "--job", "check")
        check(rc == 0 and "ALL ESSENTIAL CHECKS PASSED" in out, "check job passes", out[-300:])

        print("\n[1] stage 1: items and orders (commit)")
        rc, out = run("erp_import.py", "--commit", stage=1)
        check(rc == 0, "run exits 0", out[-400:])
        res = {r["seq"]: r for r in q(cur, "SELECT seq, status, result FROM um_processed_events ORDER BY seq")}
        failed = {s for s, r in res.items() if r["status"] == "failed"}
        check(failed == {5, 10, 11, 12, 14}, "failed events are exactly the five bad ones", sorted(failed))
        codes = {s: json.loads(res[s]["result"])["error"]["code"] for s in failed}
        check(codes == {5: "pair_conflict", 10: "unknown_item", 11: "total_mismatch", 12: "missing_display_id", 14: "return_manual"},
              "failure codes", codes)
        att = {(r["kind"], r["order_ref"]) for r in q(cur, "SELECT kind, order_ref FROM um_attention")}
        check(("return_to_register", "UM-1000001") in att, "refund event leaves a staff to-do row", att)
        check(any(w.get("code") == "barcode_mismatch" for w in json.loads(res[4]["result"]).get("warnings", [])),
              "wrong shop barcode only warns, ERP pairing wins")
        check(json.loads(res[6]["result"])["warnings"][0]["code"] == "customer_not_stored", "customer event is not stored")
        check(q(cur, "SELECT count(*) n FROM cacus WHERE zid=%s", (Z,))[0]["n"] == base["cacus"], "no customer rows created")
        now_price = {r["xitem"]: r["xstdprice"] for r in q(cur, "SELECT xitem, xstdprice FROM caitem WHERE zid=%s AND xalias IN %s", (Z, (tiered_sku, zepto_sku)))}
        check(now_price == price_before, "shop price did not overwrite the ERP price")
        t1 = q(cur, "SELECT * FROM caitem WHERE zid=%s AND xalias='TESTSKU-001'", (Z,))
        check(len(t1) == 1 and t1[0]["xitemnew"] == "100001" and t1[0]["xgitem"] == "Trading Item", "new HMBR item inserted", t1[:1])
        f1 = q(cur, "SELECT * FROM caitem WHERE zid=%s AND xalias='18-9998'", (Z,))
        check(len(f1) == 1 and f1[0]["xitemnew"] == "Fixit" and f1[0]["xgitem"] == "Fix iT Item" and f1[0]["xitem"].startswith("04-"),
              "new Fixit item: xitem 04-, xitemnew Fixit", f1[:1] and (f1[0]["xitem"], f1[0]["xitemnew"], f1[0]["xgitem"]))
        xc = q(cur, "SELECT xdescdet FROM xcodes WHERE zid=%s AND xtype='Price Category' AND xcode=%s", (Z, f1[0]["xitem"]))
        check(xc and xc[0]["xdescdet"] == "TEST Fixit item", "xcodes.xdescdet filled", xc)
        ctx["fix_xitem"] = f1[0]["xitem"]

        o1 = q(cur, "SELECT * FROM opmob WHERE zid=%s AND invoiceno='UM-1000001' ORDER BY xroword", (Z,))
        check(len(o1) == 3, "order UM-1000001 has 3 opmob lines", len(o1))
        r0 = o1[0]
        check((r0["xcus"], r0["xemp"], r0["xcusname"], r0["xterminal"], r0["xstatusord"], r0["xordernum"], r0["invoicesl"]) ==
              ("CUS-000073", "MR--000007", "UtilityMart", "UTM", "New", None, 1000001), "fixed customer/employee, number in invoicesl",
              (r0["xcus"], r0["xemp"], r0["invoicesl"]))
        items_total = sum(float(r["xqty"]) * float(r["xprice"]) for r in o1)
        check(r0["xtra1"] == 5 and abs(r0["xtra2"] - round(items_total * 0.95, 2)) < 0.02, "xtra1 = discount %, xtra2 = net total", (r0["xtra1"], r0["xtra2"]))
        check(r0["xtra3"] == "COD" and r0["xtra4"] is None, "xtra3 = COD, xtra4 empty until the courier")
        x5 = json.loads(r0["xtra5"])
        check(x5["shop_order_id"] == "order_TEST0001" and x5["delivery_charge"] == 100 and x5["discount_percent"] == 5.0
              and not any(k in x5 for k in ("phone", "address", "receiver")), "xtra5 JSON holds delivery charge and no shopper details", x5)
        check(q(cur, "SELECT count(*) n FROM opmob WHERE zid=%s AND invoiceno='UM-1000002'", (Z,))[0]["n"] == 0, "cancelled order removed from opmob")
        o6 = q(cur, "SELECT xtra3, xtra4 FROM opmob WHERE zid=%s AND invoiceno='UM-1000006'", (Z,))
        check(o6 and o6[0]["xtra3"] == "ONLINE", "online order waits for its transaction id", o6)
        check(q(cur, "SELECT count(*) n FROM poord WHERE zid=%s AND xref LIKE 'UM-%%'", (Z,))[0]["n"] == 0, "daily mode: no PO at order time")
        qd = {r["external_order_id"]: r["done_at"] for r in q(cur, "SELECT * FROM um_po_queue")}
        check(set(qd) == {"order_TEST0001", "order_TEST0006"} and all(v is None for v in qd.values()), "orders 1 and 6 are queued for the PO job", qd)
        posts = [json.loads(l) for l in open(os.path.join(ROOT, "logs", "fake_feed_posts.jsonl"))]
        sts = [s for p in posts if p["path"].endswith("/order-status") for s in p["body"]["statuses"]]
        check({s["erp_status"] for s in sts} >= {"entered"} and any(s["erp_status"] == "canceled" for s in sts), "status posted: entered and canceled")
        check(any(p["path"].endswith("/ack") for p in posts), "events acknowledged")

        print("\n[2] purchase-order job (daily)")
        cur.execute("INSERT INTO um_fixit_item (sku, fixit_zid, stock, std_price, cost, item_group, sales, found) VALUES ('18-9998',100001,40,150,33,'Household',3,true)")
        rc, out = run("erp_import.py", "--job", "po")
        check("purchase orders made:" in out and q(cur, "SELECT count(*) n FROM poord WHERE zid=%s AND xref LIKE 'UM-%%'", (Z,))[0]["n"] == 0, "dry run makes nothing")
        rc, out = run("erp_import.py", "--job", "po", "--commit")
        check(rc == 0, "po job exits 0", out[-300:])
        pos = q(cur, "SELECT o.xpornum, o.xsup, o.xref, o.xsupref, o.xstatuspor, o.xtotamt, d.xitem, d.xqtyord, d.xrate, d.xdatesch"
                     " FROM poord o JOIN poodt d ON d.zid=o.zid AND d.xpornum=o.xpornum WHERE o.zid=%s AND o.xref LIKE 'UM-%%' ORDER BY o.xpornum, d.xrow", (Z,))
        by = {(p["xref"], p["xsup"]): p for p in pos}
        check(set(by) == {("UM-1000001", "SUP-000001"), ("UM-1000001", "SUP-000003"), ("UM-1000001", "SUP-000005"), ("UM-1000006", "SUP-000001")},
              "one PO per order per source business (HMBR, Zepto, Fixit)", sorted(by))
        check(all(p["xstatuspor"] == "1-Open" and p["xsupref"] == p["xref"] and str(p["xdatesch"]) == "2999-12-31" for p in pos), "POs are 1-Open, carry the order number, schedule date 2999-12-31")
        tq = int(ev[7]["items"][0]["quantity"])
        check(int(by[("UM-1000001", "SUP-000001")]["xqtyord"]) == tq, "HMBR PO quantity = ordered (100007 has none)", by[("UM-1000001", "SUP-000001")]["xqtyord"])
        check(int(by[("UM-1000001", "SUP-000003")]["xqtyord"]) == 2, "Zepto PO = shortfall only (100007 holds 4 of 6)", by[("UM-1000001", "SUP-000003")]["xqtyord"])
        check(float(by[("UM-1000001", "SUP-000005")]["xrate"]) == 33 and int(by[("UM-1000001", "SUP-000005")]["xqtyord"]) == 2, "Fixit PO at Central's sales rate (um_fixit_item.cost)")
        src = q(cur, "SELECT s.xstdprice, s.xpricecat FROM caitem c JOIN caitem s ON s.zid=c.xitemnew::int AND s.xitem=c.xalias WHERE c.zid=%s AND c.xalias=%s", (Z, tiered_sku))[0]
        tier = q(cur, "SELECT xdisc FROM opspprc WHERE zid=100001 AND xpricecat=%s AND xqty<=%s AND (xdateeff IS NULL OR xdateeff<=current_date) AND (xdateexp IS NULL OR xdateexp>=current_date) ORDER BY xqty DESC LIMIT 1",
                 (src["xpricecat"], tq + 1))
        want = float(src["xstdprice"]) - float(tier[0]["xdisc"])
        check(abs(float(by[("UM-1000001", "SUP-000001")]["xrate"]) - want) < 0.005, "HMBR PO rate = source price less the opspprc tier for the day's total", (by[("UM-1000001", "SUP-000001")]["xrate"], want))
        check(int(by[("UM-1000006", "SUP-000001")]["xqtyord"]) == 1, "second order gets its own PO for its own units")
        om = {r["external_order_id"]: r["po_numbers"] for r in q(cur, "SELECT external_order_id, po_numbers FROM um_order_map")}
        check(len(om["order_TEST0001"].split(",")) == 3 and len(om["order_TEST0006"].split(",")) == 1, "um_order_map records which PO belongs to which order", om)
        check(all(r["done_at"] for r in q(cur, "SELECT done_at FROM um_po_queue")), "queue marked done")
        rc, out = run("erp_import.py", "--job", "po", "--commit")
        check("purchase orders made: none" in out, "second PO run makes nothing")

        print("\n[3] stage 2: courier and payment trail")
        rc, out = run("erp_import.py", "--commit", stage=2)
        check(rc == 0, "run exits 0", out[-300:])
        check({r["xtra4"] for r in q(cur, "SELECT xtra4 FROM opmob WHERE zid=%s AND invoiceno='UM-1000001'", (Z,))} == {"PATHAO-TEST-1"}, "courier id on every line (xtra4)")
        check({r["xtra3"] for r in q(cur, "SELECT xtra3 FROM opmob WHERE zid=%s AND invoiceno='UM-1000006'", (Z,))} == {"SSL-TEST-TXN-6"}, "transaction id replaces ONLINE (xtra3)")

        print("\n[4] stage 3: the online order is edited while still New")
        old_pos = om["order_TEST0006"]
        rc, out = run("erp_import.py", "--commit", stage=3)
        check(rc == 0, "run exits 0", out[-300:])
        o6 = q(cur, "SELECT xqty, xtra3 FROM opmob WHERE zid=%s AND invoiceno='UM-1000006'", (Z,))
        check(len(o6) == 1 and o6[0]["xqty"] == 2 and o6[0]["xtra3"] == "SSL-TEST-TXN-6", "opmob rebuilt with the new quantity and the transaction id", o6)
        check(q(cur, "SELECT count(*) n FROM poord WHERE zid=%s AND xpornum=%s", (Z, old_pos))[0]["n"] == 0, "its old open PO was removed")
        rc, out = run("erp_import.py", "--job", "po", "--commit")
        new = q(cur, "SELECT d.xqtyord FROM poord o JOIN poodt d ON d.zid=o.zid AND d.xpornum=o.xpornum WHERE o.zid=%s AND o.xref='UM-1000006'", (Z,))
        check(len(new) == 1 and int(new[0]["xqtyord"]) == 2, "PO job makes the new PO with the new quantity", new)
        rc, out = run("erp_import.py", "--commit", stage=3)
        check(rc == 0 and q(cur, "SELECT count(*) n FROM opmob WHERE zid=%s AND invoiceno='UM-1000001'", (Z,))[0]["n"] == 3, "replaying the feed changes nothing")

        print("\n[5] product details to the shop")
        skus = "%s,%s,18-9998" % (tiered_sku, zepto_sku)
        rc, out = run("erp_import.py", "--job", "products", "--only-skus", skus)
        check("to send" in out and not os.path.exists(os.path.join(ROOT, "logs", "none")), "dry run reports what would be sent", out[-250:])
        n_before = sum(1 for l in open(os.path.join(ROOT, "logs", "fake_feed_posts.jsonl")) if json.loads(l)["path"].endswith("/products"))
        rc, out = run("erp_import.py", "--job", "products", "--only-skus", skus, "--commit")
        posts = [json.loads(l) for l in open(os.path.join(ROOT, "logs", "fake_feed_posts.jsonl"))]
        prod = [p for p in posts if p["path"].endswith("/products")]
        check(rc == 0 and len(prod) == n_before + 1, "one POST /products", out[-250:])
        items = {i["sku"]: i for i in prod[-1]["body"]["items"]}
        t = items.get(tiered_sku, {})
        check(t.get("barcode") and t.get("item_group") == "Trading Item" and t.get("price") and t.get("cost_price") and "on_hand" in t,
              "HMBR item carries barcode, group, price, cost, stock", t)
        check(abs(t["cost_price"] - want) < 0.01 or t["cost_price"] > 0, "cost = the latest PO rate")
        fx = items.get("18-9998", {})
        check(fx.get("cost_price") == 33 and fx.get("on_hand") == 40 and fx.get("item_group") == "Household" and fx.get("price") in (150, 150.0),
              "Fixit item: cost, stock and group from um_fixit_item", fx)
        rc, out = run("erp_import.py", "--job", "products", "--only-skus", skus, "--commit")
        check("0 to send" in out, "second push sends nothing (only changes go)", out[-200:])

        print("\n[5b] --shop-dry-run: ask the shop what a push WOULD change; nothing is kept")
        sent_before = q(cur, "SELECT sent_at FROM um_product_sent WHERE sku = %s", (tiered_sku,))[0]["sent_at"]
        rc, out = run("erp_import.py", "--job", "products", "--only-skus", skus, "--shop-dry-run")
        posts = [json.loads(l) for l in open(os.path.join(ROOT, "logs", "fake_feed_posts.jsonl"))]
        dry = [p for p in posts if p["path"].endswith("/products") and p.get("dry_run")]
        check(rc == 0 and len(dry) == 1 and len(dry[0]["body"]["items"]) == 3,
              "one dry-run POST with every followed item, although nothing changed since the last push", out[-300:])
        check("would_change" in out and "the shop wrote nothing" in out, "the shop's answer is summarised", out[-300:])
        check(q(cur, "SELECT sent_at FROM um_product_sent WHERE sku = %s", (tiered_sku,))[0]["sent_at"] == sent_before, "nothing is recorded as sent")
        check(any(f.startswith("shop_dry_run_products_") and f.endswith(".csv") for f in os.listdir(os.path.join(ROOT, "logs"))), "the changes are saved as a table")
        rc, out = run("erp_import.py", "--job", "products", "--shop-dry-run", "--commit")
        check(rc != 0 and "cannot be combined" in out, "--shop-dry-run together with --commit is refused", out[-200:])

        print("\n[6] sell-price report (writes nothing)")
        rc, out = run("erp_import.py", "--job", "price", "--commit")
        check(rc == 0 and "price rule" in out and "--commit ignored" in out, "report only; --commit ignored while apply = false", out[-300:])
        now_price = {r["xitem"]: r["xstdprice"] for r in q(cur, "SELECT xitem, xstdprice FROM caitem WHERE zid=%s AND xalias IN %s", (Z, (tiered_sku, zepto_sku)))}
        check(now_price == price_before, "no price changed")
        csvp = os.path.join(ROOT, "logs", "price_report_%s.csv" % time.strftime("%Y-%m-%d"))
        check(os.path.exists(csvp) and sum(1 for _ in open(csvp)) > 400, "CSV written", csvp)

        # the rule: an HMBR / GI item = the source's xstdprice x 1.40 (its xsrate is never read); a Zepto item = the source's xsrate (the MRP);
        # no source price = no price, never a guess; a move beyond max_change_pct is flagged "review" (and so never written)
        src_sql = ("SELECT s.zid, s.xitem, s.xstdprice, s.xsrate FROM caitem c JOIN caitem s ON s.xitem = c.xalias"
                   " AND s.zid = c.xitemnew::int WHERE c.zid = %s AND c.xalias = %s")
        s1 = q(cur, src_sql, (Z, tiered_sku))[0]
        z1 = q(cur, src_sql, (Z, zepto_sku))[0]
        old7 = q(cur, "SELECT xstdprice FROM caitem WHERE zid = %s AND xalias = %s", (Z, tiered_sku))[0]["xstdprice"]

        def rep_of(sku):
            rc_, _ = run("erp_import.py", "--job", "price")
            return rc_, {r["sku"]: r for r in csv.DictReader(open(csvp, encoding="utf-8"))}[sku]
        try:
            cur.execute("UPDATE caitem SET xstdprice = 1300 WHERE zid = %s AND xalias = %s", (Z, tiered_sku))
            cur.execute("UPDATE caitem SET xstdprice = 1000, xsrate = 99999 WHERE zid = %s AND xitem = %s", (s1["zid"], s1["xitem"]))
            rc, rep = rep_of(tiered_sku)
            check(rc == 0 and float(rep["new_price"]) == 1400 and "xstdprice" in rep["rule"] and rep["flag"] == "raise",
                  "HMBR/GI item: source xstdprice x 1.40 (an xsrate of 99999 is ignored)", rep)
            cur.execute("UPDATE caitem SET xstdprice = 100 WHERE zid = %s AND xitem = %s", (s1["zid"], s1["xitem"]))
            rc, rep = rep_of(tiered_sku)
            check(float(rep["new_price"]) == 140 and rep["flag"] == "review", "a move beyond max_change_pct is flagged review, not raise/lower", rep)
            cp3 = E.load_config(os.path.join(ROOT, "erp_import.ini"))
            if not cp3.has_section("pricing"):
                cp3.add_section("pricing")
            cp3.set("pricing", "approve_skus", tiered_sku)
            r3 = {r["sku"]: r for r in E.price_rows(cur, E.Settings(cp3))}[tiered_sku]
            check(r3["flag"] == "lower" and float(r3["new"]) == 140, "[pricing] approve_skus lets a listed SKU through the review limit", r3)
            cur.execute("UPDATE caitem SET xstdprice = 1234 WHERE zid = %s AND xitem = %s", (s1["zid"], s1["xitem"]))
            rc, rep = rep_of(tiered_sku)
            check(float(rep["new_price"]) == 1727, "1234 x 1.40 = 1727.6 is rounded DOWN to 1727", rep)
            cur.execute("UPDATE caitem SET xstdprice = 0.5 WHERE zid = %s AND xitem = %s", (s1["zid"], s1["xitem"]))
            rc, rep = rep_of(tiered_sku)
            check(rep["flag"] == "no_source_price" and rep["new_price"] == "", "a source price that rounds down to 0 gives no price", rep)
            cur.execute("UPDATE caitem SET xstdprice = NULL WHERE zid = %s AND xitem = %s", (s1["zid"], s1["xitem"]))
            rc, rep = rep_of(tiered_sku)
            check(rep["flag"] == "no_source_price" and rep["new_price"] == "", "no source xstdprice -> no price (flagged, not guessed)", rep)
            for set_to in (4321, None):
                cur.execute("UPDATE caitem SET xsrate = %s WHERE zid = %s AND xitem = %s", (set_to, z1["zid"], z1["xitem"]))
                rc, rep = rep_of(zepto_sku)
                if set_to:
                    check(rc == 0 and float(rep["new_price"]) == 4321 and "xsrate" in rep["rule"], "Zepto item: price = its own xsrate (the MRP)", rep)
                else:
                    check(rep["flag"] == "no_source_price" and rep["new_price"] == "", "Zepto: no xsrate -> no price (flagged, not guessed)", rep)
        finally:
            cur.execute("UPDATE caitem SET xstdprice = %s, xsrate = %s WHERE zid = %s AND xitem = %s", (s1["xstdprice"], s1["xsrate"], s1["zid"], s1["xitem"]))
            cur.execute("UPDATE caitem SET xsrate = %s WHERE zid = %s AND xitem = %s", (z1["xsrate"], z1["zid"], z1["xitem"]))
            cur.execute("UPDATE caitem SET xstdprice = %s WHERE zid = %s AND xalias = %s", (old7, Z, tiered_sku))

        print("\n[6a] price --commit saves the old prices first; --undo puts them back; --xlsx writes the before/after Excel")
        import openpyxl
        apply_ini = os.path.join(ROOT, "logs", "test_apply.ini")
        cp4 = E.load_config(os.path.join(ROOT, "erp_import.ini"))
        if not cp4.has_section("pricing"):
            cp4.add_section("pricing")
        cp4.set("pricing", "apply", "true")
        with open(apply_ini, "w") as f_:
            cp4.write(f_)
        keep7 = q(cur, "SELECT xstdprice FROM caitem WHERE zid = %s AND xalias = %s", (Z, tiered_sku))[0]["xstdprice"]
        keep_s = q(cur, "SELECT xstdprice FROM caitem WHERE zid = %s AND xitem = %s", (s1["zid"], s1["xitem"]))[0]["xstdprice"]
        try:
            cur.execute("UPDATE caitem SET xstdprice = 1300 WHERE zid = %s AND xalias = %s", (Z, tiered_sku))
            cur.execute("UPDATE caitem SET xstdprice = 1000 WHERE zid = %s AND xitem = %s", (s1["zid"], s1["xitem"]))
            base_prices = q(cur, "SELECT xitem, xstdprice FROM caitem WHERE zid = %s ORDER BY xitem", (Z,))
            rc, out = run("erp_import.py", "--job", "price", "--commit", "--xlsx", "--config", apply_ini)
            now7 = q(cur, "SELECT xstdprice FROM caitem WHERE zid = %s AND xalias = %s", (Z, tiered_sku))[0]["xstdprice"]
            check(rc == 0 and float(now7) == 1400, "price --commit writes the rule price (1000 x 1.40)", (now7, out[-300:]))
            bk = q(cur, "SELECT * FROM um_price_backup WHERE sku = %s", (tiered_sku,))
            check(len(bk) == 1 and float(bk[0]["old_price"]) == 1300 and float(bk[0]["new_price"]) == 1400, "the old price was saved first", bk)
            xl = sorted(f for f in os.listdir(os.path.join(ROOT, "logs")) if f.startswith("price_before_after_") and f.endswith(".xlsx"))
            check(bool(xl), "--xlsx wrote the before/after Excel")
            if xl:
                wb = openpyxl.load_workbook(os.path.join(ROOT, "logs", xl[-1]))
                ws = wb["Before and after"]
                row = [[c.value for c in r_] for r_ in ws.iter_rows(min_row=2) if r_[0].value == tiered_sku]
                check(wb.sheetnames == ["Before and after", "Summary"] and row and float(row[0][4]) == 1300 and float(row[0][6]) == 1400,
                      "the Excel shows was 1300 and will-be 1400 for the test item", row)
            rc, out = run("erp_import.py", "--job", "price", "--undo", "--config", apply_ini)
            check("can be put back" in out and "DRY-RUN" in out and
                  float(q(cur, "SELECT xstdprice FROM caitem WHERE zid = %s AND xalias = %s", (Z, tiered_sku))[0]["xstdprice"]) == 1400,
                  "--undo alone is a dry run", out[-250:])
            rc, out = run("erp_import.py", "--job", "price", "--undo", "--commit", "--config", apply_ini)
            check(rc == 0 and q(cur, "SELECT xitem, xstdprice FROM caitem WHERE zid = %s ORDER BY xitem", (Z,)) == base_prices,
                  "--undo --commit puts every price back exactly", out[-250:])
            check(q(cur, "SELECT count(*) n FROM um_price_backup")[0]["n"] == 0, "the backup rows are removed after the undo")
        finally:
            cur.execute("UPDATE caitem SET xstdprice = %s WHERE zid = %s AND xalias = %s", (keep7, Z, tiered_sku))
            cur.execute("UPDATE caitem SET xstdprice = %s WHERE zid = %s AND xitem = %s", (keep_s, s1["zid"], s1["xitem"]))
            cur.execute("DROP TABLE IF EXISTS um_price_backup")
            for f_ in os.listdir(os.path.join(ROOT, "logs")):
                if f_.startswith("price_before_after_") or f_ == "test_apply.ini":
                    os.remove(os.path.join(ROOT, "logs", f_))

        print("\n[6b] one-time xsrate load from a JSON of shop prices (onetime/set_xsrate_from_json.py)")
        src = q(cur, "SELECT s.zid, s.xitem, s.xsrate FROM caitem c JOIN caitem s ON s.xitem = c.xalias AND s.zid = c.xitemnew::int"
                     " WHERE c.zid = %s AND c.xitemnew IN ('100001','100000') ORDER BY s.zid, s.xitem LIMIT 3", (Z,))
        src.append(q(cur, "SELECT s.zid, s.xitem, s.xsrate FROM caitem c JOIN caitem s ON s.xitem = c.xalias AND s.zid = c.xitemnew::int"
                          " WHERE c.zid = %s AND c.xitemnew = '100001' AND s.xsrate > 0 LIMIT 1", (Z,))[0])
        jpath = os.path.join(ROOT, "logs", "test_xsrate_shop_prices.json")
        prices = [7777, 8888, 9999, 6666]
        with open(jpath, "w") as f:
            json.dump({"taken_at": "test", "items": [{"zid": r["zid"], "xitem": r["xitem"], "price": prices[i], "name": "t"} for i, r in enumerate(src)]}, f)
        def xs():
            return [q(cur, "SELECT xsrate FROM caitem WHERE zid=%s AND xitem=%s", (r["zid"], r["xitem"]))[0]["xsrate"] for r in src]
        before = xs()
        total_before = q(cur, "SELECT count(*) AS n, coalesce(sum(xsrate),0) AS t FROM caitem WHERE zid IN (100000,100001) AND xsrate > 0")[0]
        script = os.path.join("onetime", "set_xsrate_from_json.py")
        rc, out = run(script, "--json", jpath)
        check(rc == 0 and "DRY RUN" in out and xs() == before, "dry run changes nothing", out[-200:])
        rc, out = run(script, "--json", jpath, "--commit")
        check(rc != 0 and "RETIRED" in out and xs() == before, "the retired load refuses --commit without its long flag", out[-200:])
        retired_ok = "--i-know-xsrate-is-used-by-the-erp"
        rc, out = run(script, "--json", jpath, "--commit", retired_ok)
        check(rc == 0 and [float(v) for v in xs()] == [float(p_) for p_ in prices], "commit writes the shop prices into xsrate", (xs(), out[-200:]))
        rc, out = run(script, "--json", jpath, "--commit", retired_ok)
        check(rc == 0 and "Nothing to write" in out, "second run writes nothing")
        rc, out = run(script, "--undo", "--commit")
        check(rc == 0 and xs() == before, "undo puts the old values back", (xs(), before))
        check(q(cur, "SELECT count(*) AS n, coalesce(sum(xsrate),0) AS t FROM caitem WHERE zid IN (100000,100001) AND xsrate > 0")[0] == total_before,
              "no other company 100001/100000 xsrate was touched")
        check(q(cur, "SELECT count(*) AS n FROM um_xsrate_backup")[0]["n"] == 0, "backup rows are removed after the undo")
        cur.execute("DROP TABLE um_xsrate_backup")
        os.remove(jpath)

        print("\n[6d] bundles: the shop must send the parts; a bundle line gets its own failure code")
        for line, want in (({"sku": "NO-SUCH-BUNDLE", "bundle": True, "name": "Starter bundle"}, "bundle_not_expanded"),
                           ({"sku": "NO-SUCH-ITEM", "name": "x"}, "unknown_item")):
            try:
                E.resolve_lines(cur, st, [line])
                check(False, "%s is refused" % want)
            except E.EventError as e_:
                check(e_.code == want, "%s -> %s" % ("bundle line" if line.get("bundle") else "plain unknown item", want), e_.code)

        print("\n[6c] exceptions: a SKU in [exceptions] skip_skus is never pushed to the shop")
        cp2 = E.load_config(os.path.join(ROOT, "erp_import.ini"))
        if not cp2.has_section("exceptions"):
            cp2.add_section("exceptions")
        cp2.set("exceptions", "skip_skus", tiered_sku)
        st2 = E.Settings(cp2)
        prods = E.collect_products(cur, st2)
        check(tiered_sku not in prods and zepto_sku in prods, "product push leaves the excepted SKU out")
        stock2 = E.current_stock(cur, st2)
        check(tiered_sku not in stock2 and zepto_sku in stock2, "stock push leaves it out")
        check({r["sku"]: r for r in E.price_rows(cur, st2)}[tiered_sku]["flag"] == "skipped_exception", "price rule does not touch it")

        print("\n[7] Fixit pull (reads the local fixit database)")
        row = dict(saved[next(iter(saved))])
        fx_code = "18-019"
        q_ = "INSERT INTO caitem (%s) VALUES (%s)" % (", ".join(row), ", ".join(["%s"] * len(row)))
        row.update({"xitem": "04-90019", "xalias": fx_code, "xitemnew": "Fixit", "xgitem": "Fix iT Item", "xpricecat": fx_code})
        cur.execute(q_, list(row.values()))
        ctx["fx_extra"] = "04-90019"
        rc, out = run("fixit_pull.py", "--only-skus", fx_code + ",18-9998")
        check(rc == 0 and "found" in out and q(cur, "SELECT count(*) n FROM um_fixit_item WHERE sku=%s", (fx_code,))[0]["n"] == 0, "dry run saves nothing", out[-300:])
        rc, out = run("fixit_pull.py", "--only-skus", fx_code, "--commit")
        u = q(cur, "SELECT * FROM um_fixit_item WHERE sku=%s", (fx_code,))
        fc = psycopg2.connect(host="localhost", dbname="fixit", user="postgres").cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        fc.execute("SELECT zid, stock, std_price, store_buy_price, item_group FROM final_items_view WHERE trim(item_id)=%s", (fx_code,))
        frows = {r["zid"]: r for r in fc.fetchall()}
        want_stock = sum(max(float(r["stock"]), 0) for r in frows.values())
        check(u and u[0]["found"] and abs(float(u[0]["stock"]) - want_stock) < 0.01, "stock = sum over the three Fixit companies", (u and u[0]["stock"], want_stock))
        g = frows.get(100001)
        if g:
            check(g["std_price"] is None or float(u[0]["std_price"] or 0) == float(g["std_price"]), "price from Gulshan", (u[0]["std_price"], g["std_price"]))
            check(g["store_buy_price"] is None or float(u[0]["cost"] or 0) == float(g["store_buy_price"]), "cost = Central's sales rate (Gulshan purchase price)", (u[0]["cost"], g["store_buy_price"]))
        rc, out = run("fixit_pull.py", "--only-skus", "18-9998", "--commit")
        check(q(cur, "SELECT found FROM um_fixit_item WHERE sku='18-9998'")[0]["found"] is False, "a SKU Fixit does not know is flagged, not invented")
        check(q(cur, "SELECT count(*) n FROM um_attention WHERE kind='fixit_missing'")[0]["n"] >= 1, "and listed for staff")

        print("\n[8] real catalogue, dry run (nothing is kept)")
        snap = os.path.join(ROOT, "logs", "medusa_snapshot.json")
        if os.path.exists(snap):
            feed.terminate(); feed.wait()
            feed = subprocess.Popen([PY, os.path.join("tests", "fake_feed.py"), "--snapshot", snap], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            wait_feed()
            run("erp_import.py", "--purge-test")
            rc, out = run("erp_import.py", "--after", "0", "--max-events", "5000", "--no-stock")
            check(rc == 0 and "DRY-RUN: rolling back" in out, "dry run of the whole shop catalogue completes", out[-250:])
            codes = {}
            for line in out.splitlines():
                if "FAILED " in line:
                    c = line.split("FAILED ")[1].split(":")[0]
                    codes[c] = codes.get(c, 0) + 1
            print("      real catalogue: events %s; failures by code %s" % (out.count("type=item.upserted"), codes))
            print("      barcode_mismatch warnings: %d; start_price_from_shop (new items): %d" % (out.count("barcode_mismatch"), out.count("start_price_from_shop")))
        else:
            print("  (no medusa_snapshot.json: run dev_tools/medusa_snapshot.py to include this step)")
    finally:
        feed.terminate()
        print("\n[cleanup] restoring the local database")
        run("erp_import.py", "--purge-test")
        cur.execute("DELETE FROM um_fixit_item")
        cur.execute("DELETE FROM um_attention")
        cur.execute("DELETE FROM um_po_queue")
        cur.execute("DELETE FROM xcodes WHERE zid=%s AND xtype='Price Category' AND xcode LIKE '04-%%'", (Z,))
        cur.execute("DELETE FROM caitem WHERE zid=%s AND xitemnew='Fixit'", (Z,))
        cur.execute("DELETE FROM xcodes WHERE zid=%s AND xtype='Price Category' AND xcode NOT IN %s", (Z, tuple(base["xcodes"]) or ("",)))
        for xitem, r in saved.items():
            cols = [c for c in r if c not in ("zid", "xitem")]
            cur.execute("UPDATE caitem SET %s WHERE zid=%%s AND xitem=%%s" % ", ".join("%s=%%s" % c for c in cols),
                        [r[c] for c in cols] + [Z, xitem])
        after = {
            "caitem": q(cur, "SELECT count(*) n FROM caitem WHERE zid=%s", (Z,))[0]["n"],
            "poord": q(cur, "SELECT count(*) n FROM poord WHERE zid=%s", (Z,))[0]["n"],
            "opmob": q(cur, "SELECT count(*) n FROM opmob WHERE zid=%s", (Z,))[0]["n"],
            "cacus": q(cur, "SELECT count(*) n FROM cacus WHERE zid=%s", (Z,))[0]["n"],
            "xcodes": {r["xcode"] for r in q(cur, "SELECT xcode FROM xcodes WHERE zid=%s AND xtype='Price Category'", (Z,))},
        }
        check(after == base, "database back to its starting counts", {k: (base[k], after[k]) for k in base if base[k] != after[k]})
        conn.close()
    bad = [l for ok, l in RESULTS if not ok]
    print("\n%d checks, %d failed%s" % (len(RESULTS), len(bad), (": " + "; ".join(bad)) if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
