#!/usr/bin/env python3
"""A stand-in for the shop's /hooks/erp endpoints, for testing erp_import.py without the real shop.

    python tests/fake_feed.py                      # serves http://127.0.0.1:8099/hooks/erp with the TEST events
    python tests/fake_feed.py --snapshot logs/medusa_snapshot.json   # item events built from the real catalogue
    python tests/fake_feed.py --show               # print the events it would serve and exit

It is also the executable contract for the shop session (see docs/SHOP_CONTRACT.md): it checks the Bearer key and the
HMAC signature on POSTs exactly as the real shop must, validates the bodies of /ack, /order-status, /stock and
/products, and answers 422 with the reason when a body is wrong. Every accepted POST is appended to
logs/fake_feed_posts.jsonl.

TEST events are built from REAL items in the local `da` database (so the importer's lookups are real). Every
shop id contains TEST so `erp_import.py --purge-test` can remove what a test run created. Events carry a
"stage"; only events up to the stage written in logs/fake_stage.txt (default 99) are served, so a test can
release the later events (fulfilment, payment, edits) after the first run.
"""
import argparse
import configparser
import hashlib
import hmac
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

import psycopg2
import psycopg2.extras

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # the erp_import folder (logs/ and erp_import.ini live there)
STAGE_FILE = os.path.join(ROOT, "logs", "fake_stage.txt")
CFG = {"key": "test-feed-key", "secret": "test-shared-secret", "env": "live"}
EVENTS = []
ACKED = []


def connect(cp):
    return psycopg2.connect(host=cp.get("database", "host", fallback="localhost"), port=cp.get("database", "port", fallback="5432"),
                            dbname=cp.get("database", "dbname", fallback="da"), user=cp.get("database", "user", fallback="postgres"),
                            password=cp.get("database", "password", fallback=""))


def pick_items(cp):
    """Real 100007 items: a source item with a wholesale tier and no 100007 stock, a Zepto item 100007 half-covers, a spare."""
    conn = connect(cp)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        WITH s7 AS (SELECT xitem, coalesce(sum(xqty*xsign),0) st FROM imtrn WHERE zid=100007 GROUP BY xitem)
        SELECT c.xitem, c.xalias, c.xitemnew, c.xdesc, c.xlong, c.xlinks, c.xstdprice, coalesce(s7.st,0) AS st7,
               f.stock::numeric AS src_stock, t.xqty AS tier_qty, t.xdisc AS tier_disc, src.xstdprice AS src_price
        FROM caitem c
        JOIN caitem src ON src.zid = c.xitemnew::int AND src.xitem = c.xalias
        JOIN opspprc t ON t.zid = src.zid AND t.xpricecat = src.xpricecat AND t.xdisc > 0 AND t.xqty BETWEEN 3 AND 20
             AND (t.xdateeff IS NULL OR t.xdateeff <= current_date) AND (t.xdateexp IS NULL OR t.xdateexp >= current_date)
        JOIN final_items_view f ON f.zid = c.xitemnew::int AND f.item_id = c.xalias
        LEFT JOIN s7 ON s7.xitem = c.xitem
        WHERE c.zid = 100007 AND c.xitemnew = '100001' AND c.xstdprice > 0 AND f.stock::numeric > 100 AND coalesce(s7.st,0) = 0
        ORDER BY c.xitem LIMIT 1""")
    tiered = cur.fetchone()
    cur.execute("""
        WITH s7 AS (SELECT xitem, coalesce(sum(xqty*xsign),0) st FROM imtrn WHERE zid=100007 GROUP BY xitem)
        SELECT c.xitem, c.xalias, c.xitemnew, c.xdesc, c.xlong, c.xlinks, c.xstdprice, s7.st AS st7, f.stock::numeric AS src_stock
        FROM caitem c JOIN s7 ON s7.xitem = c.xitem
        JOIN final_items_view f ON f.zid = c.xitemnew::int AND f.item_id = c.xalias
        WHERE c.zid = 100007 AND c.xitemnew = '100005' AND s7.st BETWEEN 1 AND 6 AND c.xstdprice > 0 AND f.stock::numeric > 20
        ORDER BY c.xitem LIMIT 1""")
    zepto = cur.fetchone()
    cur.execute("""
        SELECT c.xitem, c.xalias, c.xitemnew, c.xdesc, c.xlong, c.xlinks, c.xstdprice FROM caitem c
        WHERE c.zid = 100007 AND c.xitemnew = '100000' AND c.xstdprice > 0 ORDER BY c.xitem LIMIT 1""")
    gi = cur.fetchone()
    conn.close()
    return tiered, zepto, gi


def ev(seq, typ, data, stage=1, version=1):
    return {"seq": seq, "id": "evt_TEST%04d" % seq, "type": typ, "at": "2026-11-02T08:%02d:00Z" % (seq % 60),
            "version": version, "data": data, "stage": stage}


def line(n, it, qty, price=None):
    price = float(price if price is not None else it["xstdprice"])
    lt = round(qty * price, 2)
    return {"line": n, "sku": it["xalias"], "barcode": it["xitem"], "erp_source": "HMBR", "name": it["xdesc"], "quantity": qty,
            "unit_price": price, "discount": 0.0, "line_total": lt}


def order(ext, disp, lines, ship=100.0, discount_pct=0.0, payment=None, total_override=None, notes="Call at the gate", drop_display=False):
    gross = round(sum(l["line_total"] for l in lines), 2)
    disc = round(gross * discount_pct / 100.0, 2)
    total = total_override if total_override is not None else round(gross - disc + ship, 2)
    o = {"external_order_id": ext, "display_id": disp, "placed_at": "2026-11-02T08:30:00Z", "status": "open",
         "customer": {"shop_id": "cus_TEST001", "kind": "guest", "name": "Test Customer", "phone": "01799000001"},
         "delivery_address": {"name": "Test Customer", "phone": "01799000001", "line1": "House 5, Road 2 (বাড়ি ৫, রোড ২)",
                              "area": "Mirpur 10", "city": "Dhaka"},
         "payment": payment or {"method": "cod", "status": "unpaid", "gateway": None, "transaction_id": None},
         "items": lines, "shipping": {"method": "Pathao home delivery", "charge": ship, "free_delivery_applied": False},
         "discounts": ([{"code": "TEST%d" % discount_pct, "amount": disc}] if disc else []),
         "totals": {"items": gross, "discount": disc, "shipping": ship, "total": total, "currency": "BDT", "prices_include_vat": True},
         "notes": notes, "channel": "web"}
    if drop_display:
        o.pop("display_id")
    return o


def build_events(tiered, zepto, gi):
    q_tier = int(tiered["tier_qty"])
    q_zep = int(zepto["st7"]) + 2
    fix_sku = "18-9998"
    fix = {"xitem": "04-?????", "xalias": fix_sku, "xdesc": "TEST Fixit item", "xstdprice": 150.0}
    o1 = order("order_TEST0001", 1000001, [line(1, tiered, q_tier), line(2, zepto, q_zep), line(3, fix, 2)], discount_pct=5.0)
    o2 = order("order_TEST0002", 1000002, [line(1, gi, 1)])
    o6_pay = {"method": "sslcommerz", "status": "unpaid", "gateway": "sslcommerz", "transaction_id": None}
    o6 = order("order_TEST0006", 1000006, [line(1, tiered, 1)], payment=o6_pay, ship=60.0)
    o6b_pay = {"method": "sslcommerz", "status": "captured", "gateway": "sslcommerz", "transaction_id": "SSL-TEST-TXN-6"}
    o6b = order("order_TEST0006", 1000006, [line(1, tiered, 2)], payment=o6b_pay, ship=60.0)
    bad_sku = {"xalias": "NOPE-999", "xitem": "", "xdesc": "x", "xstdprice": 100.0}
    return [
        ev(1, "item.upserted", {"sku": tiered["xalias"], "barcode": tiered["xitem"], "name": tiered["xdesc"], "variant_title": "",
                                "product_title": tiered["xlong"] or tiered["xdesc"], "product_handle": tiered["xlinks"] or "test-handle",
                                "options": {}, "category": {"shop_id": "ha-9-4", "name": "x", "path": "x"},
                                "price": float(tiered["xstdprice"]), "active": True, "erp_source": "HMBR"}),
        ev(2, "item.upserted", {"sku": "TESTSKU-001", "barcode": "", "name": "Test Item Online 50mm", "variant_title": "50mm",
                                "product_title": "Test Item Online", "product_handle": "test-item-online", "options": {"Size": "50mm"},
                                "category": {"shop_id": "ha-9-4", "name": "x", "path": "x"}, "price": 250.0, "active": True, "erp_source": "HMBR"}),
        ev(3, "item.upserted", {"sku": fix_sku, "barcode": "", "name": "TEST Fixit item", "variant_title": "", "product_title": "TEST Fixit item",
                                "product_handle": "test-fixit-item", "options": {}, "category": {"shop_id": "x", "name": "x", "path": "x"},
                                "price": 150.0, "active": True, "erp_source": "FIXIT"}),
        ev(4, "item.upserted", {"sku": zepto["xalias"], "barcode": "03-99999", "name": zepto["xdesc"], "variant_title": "",
                                "product_title": zepto["xlong"] or zepto["xdesc"], "product_handle": zepto["xlinks"] or "test-handle2",
                                "options": {}, "category": {"shop_id": "x", "name": "x", "path": "x"}, "price": 1.0, "active": True, "erp_source": "ZEPTO"}),
        ev(5, "item.upserted", {"sku": "TESTSKU-002", "barcode": tiered["xitem"], "name": "Conflicting pair", "variant_title": "",
                                "product_title": "Conflicting pair", "product_handle": "conflict", "options": {},
                                "category": {"shop_id": "x", "name": "x", "path": "x"}, "price": 10.0, "active": True, "erp_source": "HMBR"}),
        ev(6, "customer.upserted", {"customer": {"shop_id": "cus_TEST001", "kind": "guest", "name": "Test Customer", "phone": "01799000001"}}),
        ev(7, "order.created", o1),
        ev(8, "order.created", o2),
        ev(9, "order.canceled", {"external_order_id": "order_TEST0002", "reason": "customer asked", "at": "2026-11-02T09:00:00Z"}),
        ev(10, "order.created", order("order_TEST0003", 1000003, [line(1, bad_sku, 1)])),
        ev(11, "order.created", order("order_TEST0004", 1000004, [line(1, tiered, 1)], total_override=999)),
        ev(12, "order.created", order("order_TEST0005", 0, [line(1, tiered, 1)], drop_display=True)),
        ev(13, "order.created", o6),
        ev(14, "order.refunded", {"external_order_id": "order_TEST0001", "amount": 100, "method": "cash", "reason": "test", "at": "2026-11-02T10:00:00Z"}),
        ev(15, "order.fulfilled", {"external_order_id": "order_TEST0001", "consignments": [
            {"id": "PATHAO-TEST-1", "courier": "pathao", "fee_estimate": 90}], "items": []}, stage=2),
        ev(16, "payment.captured", {"external_order_id": "order_TEST0006", "amount": 100, "method": "sslcommerz",
                                    "transaction_id": "SSL-TEST-TXN-6", "captured_at": "2026-11-02T11:00:00Z"}, stage=2),
        ev(17, "order.updated", o6b, stage=3, version=2),
    ]


MEDUSA_SOURCE = {"fixit": "FIXIT", "trading": "HMBR", "gi": "GI", "zepto": "ZEPTO", "zepto cleaning combo": "ZEPTO"}


def snapshot_events(path):
    prods = json.load(open(path))
    out, seq = [], 0
    for p in prods:
        coll = ((p.get("collection") or {}).get("title") or "").strip().lower()
        src = MEDUSA_SOURCE.get(coll)
        for v in p.get("variants") or []:
            sku = (v.get("sku") or "").strip()
            if not sku or not src:
                continue
            price = next((x["amount"] for x in (v.get("prices") or []) if x.get("currency_code") == "bdt"), None)
            seq += 1
            out.append(ev(seq, "item.upserted", {
                "sku": sku, "barcode": (v.get("barcode") or "").strip(), "name": v.get("title") or p.get("title"),
                "variant_title": v.get("title") or "", "product_title": p.get("title"), "product_handle": p.get("handle"),
                "options": {}, "category": {"shop_id": "", "name": "", "path": ""}, "price": price,
                "active": p.get("status") == "published", "erp_source": src}))
    return out


def current_stage():
    try:
        return int(open(STAGE_FILE).read().strip())
    except Exception:
        return 99


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def validate(path, body):
    """Return an error text, or None. These are the rules the real shop must enforce too."""
    if path.endswith("/ack"):
        for r in body.get("results") or []:
            if "seq" not in r or "id" not in r or r.get("status") not in ("ok", "failed"):
                return "each result needs seq, id and status ok|failed"
    elif path.endswith("/order-status"):
        for s in body.get("statuses") or []:
            if not s.get("external_order_id") or s.get("erp_status") not in (
                    "entered", "order_created", "confirmed", "credited", "canceled", "removed_in_erp"):
                return "each status needs external_order_id and a known erp_status"
    elif path.endswith("/stock"):
        items = body.get("items")
        if not isinstance(items, list) or len(items) > 500:
            return "items must be a list of at most 500"
        for i in items:
            if not i.get("sku") or not isinstance(i.get("on_hand"), int) or i["on_hand"] < 0:
                return "each item needs sku and a whole on_hand >= 0"
    elif path.endswith("/products"):
        items = body.get("items")
        if not isinstance(items, list) or len(items) > 500:
            return "items must be a list of at most 500"
        for i in items:
            if not i.get("sku"):
                return "each item needs sku"
            for k in ("price", "cost_price"):
                if k in i and (not isinstance(i[k], (int, float)) or i[k] < 0):
                    return "%s must be a number >= 0" % k
            if "on_hand" in i and (not isinstance(i["on_hand"], int) or i["on_hand"] < 0):
                return "on_hand must be a whole number >= 0"
    else:
        return "unknown path"
    return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth(self):
        return self.headers.get("Authorization") == "Bearer " + CFG["key"]

    def do_GET(self):
        u = urlparse(self.path)
        if not self._auth():
            return self._send(401, {"error": "bad key"})
        if u.path.endswith("/health"):
            return self._send(200, {"ok": True, "server_time": now_iso(), "environment": CFG["env"]})
        if u.path.endswith("/feed"):
            q = parse_qs(u.query)
            after, limit = int(q.get("after", ["0"])[0]), int(q.get("limit", ["100"])[0])
            stage = current_stage()
            ready = [e for e in EVENTS if e["seq"] > after and e.get("stage", 1) <= stage]
            page = [{k: v for k, v in e.items() if k != "stage"} for e in ready[:limit]]
            return self._send(200, {"events": page, "last_seq": page[-1]["seq"] if page else after,
                                    "has_more": len(ready) > limit, "environment": CFG["env"], "server_time": now_iso()})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n)
        if not self._auth():
            return self._send(401, {"error": "bad key"})
        ts, sig = self.headers.get("X-UM-Timestamp", ""), self.headers.get("X-UM-Signature", "")
        want = "sha256=" + hmac.new(CFG["secret"].encode(), ts.encode() + b"." + raw, hashlib.sha256).hexdigest()
        if not ts or abs(time.time() - int(ts or 0)) > 300 or not hmac.compare_digest(want, sig):
            return self._send(403, {"error": "bad signature or clock"})
        try:
            body = json.loads(raw.decode("utf-8"))
        except ValueError:
            return self._send(422, {"error": "body is not JSON"})
        path = urlparse(self.path).path
        err = validate(path, body)
        if err:
            return self._send(422, {"error": err})
        os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
        with open(os.path.join(ROOT, "logs", "fake_feed_posts.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"path": path, "body": body}, ensure_ascii=False) + "\n")
        self._send(200, {"ok": True})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "erp_import.ini"))
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--snapshot", default="", help="build item events from a medusa_snapshot.json instead of the TEST events")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    cp = configparser.ConfigParser(interpolation=None)
    cp.read(args.config, encoding="utf-8")
    CFG["key"] = cp.get("shop", "feed_key", fallback=CFG["key"])
    CFG["secret"] = cp.get("shop", "shared_secret", fallback=CFG["secret"])
    if args.snapshot:
        EVENTS.extend(snapshot_events(args.snapshot))
    else:
        tiered, zepto, gi = pick_items(cp)
        if not (tiered and zepto and gi):
            raise SystemExit("no suitable 100007 test items found in the local database (%s, %s, %s)" % (bool(tiered), bool(zepto), bool(gi)))
        EVENTS.extend(build_events(tiered, zepto, gi))
    if args.show:
        print(json.dumps(EVENTS, indent=1, ensure_ascii=False))
        return
    print("stand-in feed on http://127.0.0.1:%d/hooks/erp  (%d events)" % (args.port, len(EVENTS)))
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
