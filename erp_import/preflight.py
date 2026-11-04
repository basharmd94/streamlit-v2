"""Pre-flight check before going live: `erp_import.py --job preflight`.

Reads only. It looks at four places and lists every problem in one Excel (logs/preflight_<time>.xlsx) and a short console summary:
  HMBR server   the source companies (100001 HMBR, 100000 GI, 100005 Zepto) behind every followed 100007 item
  Fixit server  the final_items_view for the Fixit items 100007 follows
  100007        the item table itself (price, cost, stock, names, duplicate SKUs)
  Live shop     what the shop would change if the products were pushed now (asked with ?dry_run=1: nothing changes there)
FAIL = fix before going live, WARN = look at it, INFO = for your information. Needs openpyxl for the Excel (else a CSV is written).
"""
import csv
import datetime as dt
import os
import re
import sys
from collections import Counter, OrderedDict
from decimal import Decimal

FAIL, WARN, INFO = "FAIL", "WARN", "INFO"
A_HMBR, A_FIXIT, A_ITEMS, A_SHOP = "HMBR server", "Fixit server", "100007 products", "Live shop"
AREAS = [A_HMBR, A_FIXIT, A_ITEMS, A_SHOP]
HEAD = ["Area", "Level", "Problem", "SKU", "100007 item", "Name", "Detail"]
WIDTH = [16, 8, 26, 12, 12, 42, 80]
FILL = {FAIL: "F8CBAD", WARN: "FFF2CC"}
STALE_HOURS = 36


def _f(x):
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def run(E, cur, st, cp, shop, args):
    """E is the erp_import module (passed in, so it is not imported twice). Returns 1 when there is a FAIL, else 0."""
    rows = []

    def add(area, level, code, sku="", item="", name="", detail=""):
        rows.append({"area": area, "level": level, "code": code, "sku": sku or "", "item": item or "", "name": name or "", "detail": detail})

    # ---- 100007 and the HMBR server's source companies -------------------------------------------------------
    cur.execute("SELECT xitem, xalias, xdesc, xitemnew, xstdprice FROM caitem WHERE zid = %s AND coalesce(xalias,'') <> '' ORDER BY xitem", (st.zid,))
    items = cur.fetchall()
    cur.execute("SELECT s.zid, s.xitem, s.xstdprice, s.xsrate FROM caitem s JOIN caitem c ON c.xalias = s.xitem"
                " AND s.zid = (CASE WHEN c.xitemnew ~ '^[0-9]+$' THEN c.xitemnew::int END) WHERE c.zid = %s", (st.zid,))
    source = {(str(r["zid"]), r["xitem"]): r for r in cur.fetchall()}
    per_source = Counter(E.source_of(it) or "none" for it in items)
    add(A_ITEMS, INFO, "followed_items", detail="%d followed items: %s" % (len(items), ", ".join("%s %d" % kv for kv in sorted(per_source.items()))))

    seen = {}
    for it in items:
        seen.setdefault(E.clean(it["xalias"]), []).append(it["xitem"])
    for sku, xs in sorted(seen.items()):
        if len(xs) > 1:
            add(A_ITEMS, FAIL, "duplicate_sku", sku, ", ".join(xs), detail="more than one 100007 item has this SKU (xalias)")

    for it in items:
        sku, xitem, name, src = E.clean(it["xalias"]), it["xitem"], E.clean(it["xdesc"]), E.source_of(it)
        if not name:
            add(A_ITEMS, WARN, "no_name", sku, xitem, detail="the item has no description")
        if not src:
            add(A_ITEMS, FAIL, "no_source", sku, xitem, name, "xitemnew = %r is not a source company or Fixit" % E.clean(it["xitemnew"]))
            continue
        want = E.PREFIX_XITEMNEW.get(xitem[:2])
        if want and E.clean(it["xitemnew"]) not in (want, "") :
            add(A_ITEMS, WARN, "prefix_mismatch", sku, xitem, name, "code prefix %s belongs to %s, xitemnew says %s" % (xitem[:2], want, E.clean(it["xitemnew"])))
        if src in E.NUMERIC_SOURCES:
            r = source.get((src, sku))
            if not r:
                add(A_HMBR, FAIL, "source_missing", sku, xitem, name, "no item %s in company %s" % (sku, src))
            elif src == "100005":
                if not r["xsrate"] or r["xsrate"] <= 1:
                    add(A_HMBR, WARN, "placeholder_price", sku, xitem, name, "Zepto MRP (xsrate) is %s" % r["xsrate"])
            elif not r["xstdprice"] or r["xstdprice"] <= 1:
                add(A_HMBR, WARN, "placeholder_price", sku, xitem, name, "source standard price in company %s is %s" % (src, r["xstdprice"]))

    # the price rule against what 100007 holds
    for r in E.price_rows(cur, st):
        f, old, new = r["flag"], r["old"], r["new"]
        if f == "review":
            add(A_ITEMS, WARN, "price_held", r["sku"], r["xitem"], r["name"], "the rule would move it %.0f%% (%s to %s); it keeps %s" % (r["change_pct"], old, new, old))
        elif f == "no_source_price":
            add(A_ITEMS, FAIL if not old else WARN, "no_rule_price", r["sku"], r["xitem"], r["name"],
                "the source has no price for the rule; 100007 holds %s" % (old if old else "nothing"))
        elif f in ("raise", "lower", "new_item"):
            add(A_ITEMS, WARN, "price_not_applied", r["sku"], r["xitem"], r["name"], "100007 holds %s, the rule says %s (--job price --commit not run yet?)" % (old, new))
        elif f == "skipped_exception":
            add(A_ITEMS, INFO, "price_exception", r["sku"], r["xitem"], r["name"], "in [exceptions] skip_skus: the shop keeps its own figures")

    products = E.collect_products(cur, st)
    names = {E.clean(it["xalias"]): (it["xitem"], E.clean(it["xdesc"])) for it in items}
    for sku, p in sorted(products.items()):
        xitem, name = names.get(sku, ("", ""))
        if p["price"] is None:
            add(A_ITEMS, FAIL, "no_price", sku, xitem, name, "100007 has no price to send")
        if p["cost_price"] is None:
            add(A_ITEMS, WARN, "no_cost", sku, xitem, name, "no cost price (no PO, no receipt, no source price)")
        if p["on_hand"] is None:
            add(A_ITEMS, WARN, "no_stock", sku, xitem, name, "no stock figure (not in final_items_view)")

    # ---- Fixit server -----------------------------------------------------------------------------------------
    try:
        sys.modules.setdefault("erp_import", E)
        import fixit_pull
        fconn = fixit_pull.fixit_connect(cp)
        fc = fconn.cursor()
        fc.execute("SELECT zid, count(*) FROM final_items_view GROUP BY zid ORDER BY zid")
        add(A_FIXIT, INFO, "fixit_view", detail="final_items_view rows per company: " + ", ".join("%s %d" % (z, n) for z, n in fc.fetchall()))
        fskus = fixit_pull.followed_skus(cur, st)
        if not fskus:
            add(A_FIXIT, INFO, "no_fixit_items", detail="no Fixit items in 100007 yet: they arrive with the first feed load (about 1,466)")
        else:
            by = fixit_pull.read_fixit(fconn, fskus)
            for sku, xitem in sorted(fskus.items()):
                s = fixit_pull.summarise(by.get(sku), st.fixit_price_zid)
                if s is None:
                    add(A_FIXIT, WARN, "fixit_unknown", sku, xitem, detail="not in the Fixit server's final_items_view")
                    continue
                if not s["std_price"]:
                    add(A_FIXIT, WARN, "fixit_no_price", sku, xitem, detail="Fixit has no selling price")
                if not s["cost"]:
                    add(A_FIXIT, WARN, "fixit_no_cost", sku, xitem, detail="Fixit has no buying price")
        cur.execute("SELECT count(*) AS n, max(as_of) AS last FROM um_fixit_item")
        r = cur.fetchone()
        if r["n"] and r["last"] and (dt.datetime.now() - r["last"]) > dt.timedelta(hours=STALE_HOURS):
            add(A_FIXIT, WARN, "fixit_stale", detail="fixit_pull last ran %s: run it again" % r["last"].strftime("%Y-%m-%d %H:%M"))
        fconn.close()
    except Exception as e:                                    # a down server must be reported, not crash the report
        add(A_FIXIT, FAIL, "fixit_unreachable", detail="%s: %s" % (type(e).__name__, str(e).strip()[:200]))

    # ---- live shop: what a push would change ------------------------------------------------------------------
    changes = []
    try:
        h = shop.health()
        add(A_SHOP, INFO, "shop_health", detail="environment=%s, sync_enabled=%s" % (h.get("environment"), h.get("sync_enabled")))
        todo = [{k: v for k, v in p.items() if v is not None} for _, p in sorted(products.items())]      # SKU order, as the push sends them
        answers = E.shop_dry_run_answers(shop, todo, True)
        n_calls = -(-len(todo) // 500)
        main, retry = answers[:n_calls], answers[n_calls:]
        unknown = sorted(set().union(*[E._skus_in(a.get("unknown_skus")) for a in main]) if main else [])
        for sku in unknown:
            xitem, name = names.get(sku, ("", ""))
            add(A_SHOP, WARN, "not_in_shop", sku, xitem, name, "an ERP item the shop does not have (typing error in the shop's SKU?)")
        left = (retry[-1].get("conflicts") or []) if retry else [c for a in main for c in (a.get("conflicts") or [])]
        for c in left:
            sku = c.get("sku") if isinstance(c, dict) else c
            xitem, name = names.get(sku, ("", ""))
            add(A_SHOP, WARN, "barcode_conflict", sku, xitem, name, "%s%s" % (c.get("reason", "") if isinstance(c, dict) else "", " (even when sent together)" if retry else ""))
        changes = E.would_change_rows(answers)
        counts = Counter()
        for w in changes:
            sku, field, old, new = w.get("sku"), w.get("field"), w.get("from"), w.get("to")
            xitem, name = names.get(sku, ("", ""))
            counts[field] += 1
            if field == "price" and _f(old) and _f(new) is not None:
                pct = (_f(new) - _f(old)) / _f(old) * 100
                if st.price_review_pct and abs(pct) > st.price_review_pct:
                    add(A_SHOP, WARN, "shop_price_moves", sku, xitem, name, "the shop price would move %.0f%%: %s to %s" % (pct, old, new))
            elif field == "barcode" and old and old != new:
                add(A_SHOP, WARN, "shop_barcode_replaced", sku, xitem, name, "the shop's barcode %s would become %s" % (old, new))
            elif field in ("on_hand", "stock") and _f(new) == 0 and _f(old):
                add(A_SHOP, INFO, "shop_sold_out", sku, xitem, name, "stock %s in the shop would become 0" % old)
        add(A_SHOP, INFO, "shop_changes", detail="the shop would change %d values: %s" % (len(changes), ", ".join("%s %d" % kv for kv in sorted(counts.items()))))
    except (E.ShopUnavailable, E.ShopRejected) as e:
        add(A_SHOP, FAIL, "shop_unreachable", detail=str(e).strip()[:200])

    return report(rows, changes, st)


def report(rows, changes, st):
    """Console summary + the Excel; returns 1 when any FAIL."""
    level_n = {a: Counter(r["level"] for r in rows if r["area"] == a) for a in AREAS}
    print("\nPRE-FLIGHT   FAIL = fix before going live | WARN = look at it | INFO = for your information")
    print("%-16s %5s %5s %5s" % ("area", FAIL, WARN, INFO))
    for a in AREAS:
        print("%-16s %5d %5d %5d" % (a, level_n[a][FAIL], level_n[a][WARN], level_n[a][INFO]))
    for level in (FAIL, WARN):
        by_code = OrderedDict()
        for r in rows:
            if r["level"] == level:
                by_code.setdefault((r["area"], r["code"]), []).append(r)
        for (area, code), rs in by_code.items():
            print("  %s  %-16s %-18s %3d   e.g. %s" % (level, area, code, len(rs), ", ".join(x["sku"] or "-" for x in rs[:5])))
            if level == FAIL:
                print("        %s" % rs[0]["detail"][:110])
    for r in rows:
        if r["level"] == INFO and not r["sku"]:
            print("  INFO  %-16s %s" % (r["area"], r["detail"]))
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "preflight_%s" % dt.datetime.now().strftime("%Y%m%d_%H%M"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        out = write_xlsx(path + ".xlsx", rows, changes, level_n)
    except ImportError:
        out = path + ".csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(HEAD)
            w.writerows([[r["area"], r["level"], r["code"], r["sku"], r["item"], r["name"], r["detail"]] for r in rows])
        print("  (openpyxl is not installed, so this is a CSV: venv\\Scripts\\pip install openpyxl for the Excel)")
    print("\nfull list: %s" % out)
    return 1 if any(r["level"] == FAIL for r in rows) else 0


def write_xlsx(path, rows, changes, level_n):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    def sheet(ws, head, data, widths, fills=None):
        ws.append(head)
        for c in ws[1]:
            c.font = Font(name="Arial", bold=True)
            c.fill = PatternFill("solid", fgColor="D9D9D9")
            c.alignment = Alignment(horizontal="center", wrap_text=True)
        for i, line in enumerate(data):
            ws.append(line)
            for c in ws[ws.max_row]:
                c.font = Font(name="Arial")
                if fills and fills[i]:
                    c.fill = PatternFill("solid", fgColor=fills[i])
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

    wb = Workbook()
    sm = wb.active
    sm.title = "Summary"
    data = [[a] + [level_n[a][k] for k in (FAIL, WARN, INFO)] for a in AREAS]
    sheet(sm, ["Area", FAIL, WARN, INFO], data, [18, 8, 8, 8])
    sm.append([])
    sm.append(["Problem type", "Level", "Count", "Area"])
    for c in sm[sm.max_row]:
        c.font = Font(name="Arial", bold=True)
    kinds = Counter((r["area"], r["level"], r["code"]) for r in rows if r["sku"] or r["level"] != INFO)
    for (area, level, code), n in sorted(kinds.items(), key=lambda kv: (AREAS.index(kv[0][0]), [FAIL, WARN, INFO].index(kv[0][1]), kv[0][2])):
        sm.append([code, level, n, area])
        for c in sm[sm.max_row]:
            c.font = Font(name="Arial")
    sm.column_dimensions["A"].width = 26
    order = {FAIL: 0, WARN: 1, INFO: 2}
    rs = sorted(rows, key=lambda r: (order[r["level"]], AREAS.index(r["area"]), r["code"], r["sku"]))
    sheet(wb.create_sheet("Problems"), HEAD, [[r["area"], r["level"], r["code"], r["sku"], r["item"], r["name"], r["detail"]] for r in rs], WIDTH,
          [FILL.get(r["level"]) for r in rs])
    ch = []
    for w in changes:
        old, new = _f(w.get("from")), _f(w.get("to"))
        ch.append([w.get("sku"), w.get("field"), w.get("from"), w.get("to"),
                   (new - old) / old if w.get("field") in ("price", "cost_price", "on_hand") and old and new is not None else None])
    sheet(wb.create_sheet("Shop would change"), ["SKU", "Field", "Shop now", "ERP would send", "Change"], ch, [12, 14, 20, 20, 10])
    for row in wb["Shop would change"].iter_rows(min_row=2, min_col=5, max_col=5):
        row[0].number_format = "0.0%"
    wb.save(path)
    return path
