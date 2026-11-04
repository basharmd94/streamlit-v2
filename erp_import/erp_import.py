#!/usr/bin/env python3
"""ERP importer for the online shop (Medusa) -> company 100007 in the `da` database.

One script, several jobs (pick with --job; every job is a DRY RUN unless --commit is given):

  run       (every 5 min, default)  shop feed -> ERP: items -> caitem/xcodes, orders -> opmob;
                                    order status + changed stock go back to the shop
  po        (once a day)            open purchase orders (poord/poodt, status 1-Open), one per order per
                                    source business, rate = source price less the source's opspprc tier
  products  (once a day)            barcode, price, cost, item group, stock -> shop (POST /products)
  price     (on demand)             REPORT of the scheduled sell-price rule; writes only if enabled in the config
  check                             read-only health check of the database, fixed codes and the shop

All online sales go to ONE ERP customer (config `customer`, CUS-000073 UtilityMart) and one employee
(`salesman`, MR--000007). The shop order number is the opmob invoiceno ("UM-1000001"); the ERP copies it
into opord.xref / opdor.xref. Shopper details are not stored in the ERP.

Written for PostgreSQL 9.4 (no ON CONFLICT, no IF NOT EXISTS) and Python 3.6+ with only psycopg2 outside
the standard library (openpyxl only for `--job price --xlsx`). See README.md (run guide) and docs/CYCLE_MAP.md (the agreed column map).
"""
import argparse
import configparser
import csv
import datetime as dt
import hashlib
import hmac
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
import urllib.parse
import uuid
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP

import psycopg2
import psycopg2.extras

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = logging.getLogger("erp_import")

ERP_SOURCE_PREFIX = {"HMBR": "01", "GI": "02", "ZEPTO": "03", "FIXIT": "04", "OUTSIDE": "05"}
PREFIX_XITEMNEW = {"01": "100001", "02": "100000", "03": "100005", "04": "Fixit", "05": "100007"}
PREFIX_XGITEM = {
    "01": "Trading Item", "02": "GI (Home Product) Item", "03": "Zepto Item",
    "04": "Fix iT Item", "05": "Outside Item",
}
NUMERIC_SOURCES = ("100001", "100000", "100005")    # companies of this ERP that hold the stock
FIXIT = "FIXIT"                                      # source key for Fixit items (their own server)
FIXIT_SKU_RE = re.compile(r"^\d{2}-\d{3,5}$")        # Fixit master codes: 18-018, 26-4323
DEFAULT_SUPPLIERS = {"100001": "SUP-000001", "100000": "SUP-000002", "100005": "SUP-000003", FIXIT: "SUP-000005"}
PO_OPEN_STATUS = "1-Open"
PO_NO_DATE = dt.date(2999, 12, 31)                   # the ERP's "no schedule date" on PO lines
LOCK_KEY = 70070001


# ───────────────────────────── errors ─────────────────────────────

class EventError(Exception):
    """A problem with ONE event: recorded as failed, acknowledged, skipped."""
    def __init__(self, code, message, retryable=False, after=None):
        Exception.__init__(self, message)
        self.code, self.message, self.retryable = code, message, retryable
        self.after = after        # runs AFTER the event's savepoint is rolled back (e.g. a staff to-do row)


class ShopUnavailable(Exception):
    """Network / 5xx / 429 / missing endpoint: stop quietly, the next run resumes from the same place."""


class ShopRejected(Exception):
    """401/403/422: configuration or body problem; stop the run and say why."""


# ───────────────────────────── config ─────────────────────────────

def load_config(path):
    cp = configparser.ConfigParser(interpolation=None)
    if not os.path.exists(path):
        sys.exit("Config file not found: %s (copy erp_import.ini.example)" % path)
    cp.read(path, encoding="utf-8")
    return cp


def cfg_get(cp, section, key, default=None):
    try:
        return cp.get(section, key)
    except (configparser.NoSectionError, configparser.NoOptionError):
        return default


def cfg_bool(cp, section, key, default=False):
    v = cfg_get(cp, section, key)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


class Settings(object):
    def __init__(self, cp):
        self.cp = cp
        self.base_url = cfg_get(cp, "shop", "base_url", "").rstrip("/")
        self.feed_key = cfg_get(cp, "shop", "feed_key", "")
        self.secret = cfg_get(cp, "shop", "shared_secret", "")
        self.allow_staging = cfg_bool(cp, "shop", "allow_staging_feed", False)
        self.timeout = int(cfg_get(cp, "shop", "timeout", "30"))
        self.zid = int(cfg_get(cp, "erp", "zid", "100007"))
        self.customer = cfg_get(cp, "erp", "customer", "CUS-000073")
        self.customer_name = cfg_get(cp, "erp", "customer_name", "UtilityMart")
        self.customer_address = cfg_get(cp, "erp", "customer_address", "UtilityMart online shop")
        self.salesman = cfg_get(cp, "erp", "salesman", "MR--000007")
        self.username = cfg_get(cp, "erp", "username", "utilitymart")
        self.terminal = cfg_get(cp, "erp", "terminal", "UTM")
        self.order_prefix = cfg_get(cp, "erp", "order_prefix", "UM-")
        self.warehouse = cfg_get(cp, "erp", "warehouse", "Sales Warehouse Online Shop")
        self.project = cfg_get(cp, "erp", "project", "HMBR Online Shop")
        self.user_email = cfg_get(cp, "erp", "user_email", "")
        self.po_mode = cfg_get(cp, "behaviour", "po_mode", "daily").strip().lower()     # daily | immediate | off
        self.allow_purge = cfg_bool(cp, "behaviour", "allow_purge", False)
        self.push_stock = cfg_bool(cp, "stock", "push", True)
        self.full_after_hour = int(cfg_get(cp, "stock", "full_after_hour", "2"))
        self.price_apply = cfg_bool(cp, "pricing", "apply", False)
        self.price_markup = Decimal(cfg_get(cp, "pricing", "markup", "1.40"))      # HMBR / GI sell price = their xstdprice x this
        self.price_review_pct = float(cfg_get(cp, "pricing", "max_change_pct", "25"))   # a bigger move is flagged "review", not written (0 = off)
        # SKUs whose rule price is accepted even when the move is bigger than max_change_pct (checked by the owner)
        self.price_approve = {x.strip() for x in cfg_get(cp, "pricing", "approve_skus", "").split(",") if x.strip()}
        # SKUs the ERP must not push to the shop (stock, price, barcode, cost): the shop keeps its own figures for them
        self.skip_skus = {x.strip() for x in cfg_get(cp, "exceptions", "skip_skus", "").split(",") if x.strip()}
        self.fixit_price_zid = int(cfg_get(cp, "fixit", "price_zid", "100001"))
        self.suppliers = dict(DEFAULT_SUPPLIERS)
        if cp.has_section("suppliers"):
            for k, v in cp.items("suppliers"):
                self.suppliers[k.strip().upper() if k.strip().upper() == FIXIT else k.strip()] = v.strip()


# ───────────────────────────── shop client ─────────────────────────────

class Shop(object):
    def __init__(self, st):
        self.st = st

    def _call(self, method, path, body=None, query=None):
        url = self.st.base_url + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        headers = {"Authorization": "Bearer " + self.st.feed_key, "Accept": "application/json"}
        data = None
        if body is not None:
            raw = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ts = str(int(time.time()))
            sig = hmac.new(self.st.secret.encode("utf-8"), ts.encode("ascii") + b"." + raw,
                           hashlib.sha256).hexdigest()
            headers.update({"Content-Type": "application/json", "X-UM-Timestamp": ts,
                            "X-UM-Signature": "sha256=" + sig})
            data = raw
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            resp = urllib.request.urlopen(req, timeout=self.st.timeout)
            text = resp.read().decode("utf-8")
            return json.loads(text) if text.strip() else {}
        except urllib.error.HTTPError as e:
            txt = e.read().decode("utf-8", "replace")[:500]
            if e.code in (401, 403, 422):
                raise ShopRejected("%s %s -> HTTP %s: %s" % (method, path, e.code, txt))
            raise ShopUnavailable("%s %s -> HTTP %s: %s" % (method, path, e.code, txt))
        except (urllib.error.URLError, OSError) as e:
            raise ShopUnavailable("%s %s -> %s" % (method, path, e))

    def health(self):
        return self._call("GET", "/health")

    def feed(self, after, limit=100, retry=False):
        q = {"after": after, "limit": limit}
        if retry:
            q["retry"] = 1
        return self._call("GET", "/feed", query=q)

    def ack(self, results):
        return self._call("POST", "/ack", {"results": results})

    def order_status(self, statuses):
        return self._call("POST", "/order-status", {"statuses": statuses})

    def stock(self, items, full):
        return self._call("POST", "/stock", {"as_of": utc_now_iso(), "items": items, "full": full})

    def products(self, items, full, dry_run=False):
        # ?dry_run=1: the shop answers what it WOULD change and writes nothing (even while its ERP link is switched off)
        return self._call("POST", "/products", {"as_of": utc_now_iso(), "items": items, "full": full},
                          query={"dry_run": 1} if dry_run else None)


def utc_now_iso():
    return dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# ───────────────────────────── database helpers ─────────────────────────────

UM_DDL = [
    ("um_sync_state",
     "CREATE TABLE um_sync_state (id integer PRIMARY KEY, last_seq bigint NOT NULL DEFAULT 0,"
     " last_full_stock_date date, last_full_status_date date, updated_at timestamp DEFAULT now())"),
    ("um_processed_events",
     "CREATE TABLE um_processed_events (event_id text PRIMARY KEY, seq bigint, type text, key text,"
     " status text, result text, processed_at timestamp DEFAULT now())"),
    ("um_order_map",
     "CREATE TABLE um_order_map (external_order_id text PRIMARY KEY, display_id integer, invoiceno text,"
     " invoicesl bigint, customer_code text, version integer, state text, po_numbers text,"
     " last_status text, last_status_at timestamp, info text,"
     " created_at timestamp DEFAULT now(), updated_at timestamp DEFAULT now())"),
    ("um_po_queue",
     "CREATE TABLE um_po_queue (external_order_id text PRIMARY KEY, queued_at timestamp DEFAULT now(),"
     " done_at timestamp)"),
    ("um_stock_sent",
     "CREATE TABLE um_stock_sent (sku text PRIMARY KEY, on_hand integer, sent_at timestamp DEFAULT now())"),
    ("um_product_sent",
     "CREATE TABLE um_product_sent (sku text PRIMARY KEY, barcode text, std_price numeric, cost_price numeric,"
     " item_group text, on_hand integer, sent_at timestamp DEFAULT now())"),
    ("um_fixit_item",
     "CREATE TABLE um_fixit_item (sku text PRIMARY KEY, fixit_zid integer, stock numeric, std_price numeric,"
     " cost numeric, item_group text, sales numeric, found boolean DEFAULT true, as_of timestamp DEFAULT now())"),
    ("um_attention",
     "CREATE TABLE um_attention (id serial PRIMARY KEY, created_at timestamp DEFAULT now(), kind text,"
     " order_ref text, details text, resolved boolean DEFAULT false)"),
]


def row_vals(r):
    return list(r.values()) if isinstance(r, dict) else list(r)


def scalar(cur):
    """First column of the next row, whether the cursor returns tuples or dicts."""
    r = cur.fetchone()
    return None if r is None else row_vals(r)[0]


def ensure_um_tables(cur):
    made = []
    for name, ddl in UM_DDL:
        cur.execute("SELECT to_regclass(%s)", ("public." + name,))
        if scalar(cur) is None:
            LOG.info("creating table %s", name)
            cur.execute(ddl)
            made.append(name)
    cur.execute("SELECT count(*) FROM um_sync_state WHERE id = 1")
    if scalar(cur) == 0:
        cur.execute("INSERT INTO um_sync_state (id, last_seq) VALUES (1, 0)")
    return made


_COLMETA = {}


def colmeta(cur, table):
    if table not in _COLMETA:
        cur.execute("SELECT column_name, data_type, character_maximum_length FROM information_schema.columns"
                    " WHERE table_name = %s", (table,))
        _COLMETA[table] = {v[0]: (v[1], v[2]) for v in (row_vals(r) for r in cur.fetchall())}
    return _COLMETA[table]


def fit_row(cur, table, row, warnings):
    """Truncate over-long text values to the column's real limit and warn (never fail an order)."""
    meta = colmeta(cur, table)
    for col, val in list(row.items()):
        if isinstance(val, str) and col in meta and meta[col][1] and len(val) > meta[col][1]:
            row[col] = val[:meta[col][1]]
            warnings.append({"code": "field_truncated", "field": "%s.%s" % (table, col), "max": meta[col][1]})
    return row


def insert_row(cur, table, row):
    meta = colmeta(cur, table)
    cols = [c for c in row if c in meta]
    sql = "INSERT INTO %s (%s) VALUES (%s)" % (table, ", ".join(cols), ", ".join(["%s"] * len(cols)))
    cur.execute(sql, [row[c] for c in cols])


def dict_cur(conn):
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


def advisory_xact_lock(cur, name):
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (name,))


def next_code(cur, table, col, zid, prefix, width, extra_where=""):
    """max numeric suffix + 1 (the ERP numbers documents this way; xtrn counters are not maintained)."""
    advisory_xact_lock(cur, "um:%s:%s:%s" % (table, col, prefix))
    cur.execute("SELECT max(%s) FROM %s WHERE zid = %%s AND %s ~ %%s %s" % (col, table, col, extra_where),
                (zid, "^" + re.escape(prefix) + "[0-9]+$"))
    top = scalar(cur)
    n = int(top[len(prefix):]) + 1 if top else 1
    return "%s%0*d" % (prefix, width, n)


def d(x, default="0"):
    return Decimal(str(x if x is not None else default))


def money_int(x):
    return int(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def clean(s):
    return (s or "").strip()


def dhaka_date(iso):
    try:
        t = dt.datetime.strptime(iso[:19], "%Y-%m-%dT%H:%M:%S") + dt.timedelta(hours=6)
        return t.date()
    except Exception:
        return dt.date.today()


def source_of(it):
    """Which business holds the stock of a 100007 item: '100001' / '100000' / '100005' / 'FIXIT' / ''."""
    s = clean(it.get("xitemnew"))
    if s in NUMERIC_SOURCES:
        return s
    if s.lower() == "fixit" or (not s and FIXIT_SKU_RE.match(clean(it.get("xalias")))):
        return FIXIT
    return ""


# ───────────────────────────── items (shop -> caitem) ─────────────────────────────

def find_item(cur, zid, barcode, sku, warnings=None):
    """SKU first: the ERP's xalias/xitem pairing is the truth, the shop's barcode may be wrong."""
    if sku:
        cur.execute("SELECT * FROM caitem WHERE zid = %s AND xalias = %s ORDER BY xitem LIMIT 1", (zid, sku))
        row = cur.fetchone()
        if row:
            if barcode and barcode != row["xitem"] and warnings is not None:
                warnings.append({"code": "barcode_mismatch", "sku": sku, "shop_barcode": barcode,
                                 "erp_barcode": row["xitem"]})
            return row
    if barcode:
        cur.execute("SELECT * FROM caitem WHERE zid = %s AND xitem = %s", (zid, barcode))
        row = cur.fetchone()
        if row:
            if sku and clean(row.get("xalias")) and clean(row["xalias"]) != sku:
                raise EventError("pair_conflict", "barcode %s is SKU %s in the ERP but the shop says %s (the ERP pairing "
                                 "wins; fix the shop)" % (barcode, row["xalias"], sku))
            return row
    return None


def options_note(options):
    pairs = []
    for k, v in (options or {}).items():
        if clean(str(k)) and clean(str(v)):
            pairs.append("%s:%s" % (clean(str(k)), clean(str(v))))
    return ",".join(pairs)


def resolve_scode(cur, zid, handle, variant_title):
    """VG code: reuse the one already on this product handle, else allocate the next VG####."""
    if not handle:
        return None
    cur.execute("SELECT xscode FROM caitem WHERE zid = %s AND xlinks = %s AND coalesce(xscode,'') <> '' LIMIT 1",
                (zid, handle))
    r = cur.fetchone()
    if r:
        return r["xscode"]
    if not clean(variant_title) or clean(variant_title).lower() in ("default", "default title"):
        return None   # lone item: the ERP gives those no VG code
    advisory_xact_lock(cur, "um:vg")
    cur.execute("SELECT max(xscode) FROM caitem WHERE zid = %s AND xscode ~ '^VG[0-9]+$'", (zid,))
    top = cur.fetchone()["max"]
    n = int(top[2:]) + 1 if top else 1
    return "VG%04d" % n


def ensure_xcodes(cur, st, xitem, xalias, desc, active):
    cur.execute("SELECT 1 FROM xcodes WHERE zid = %s AND xtype = 'Price Category' AND xcode = %s", (st.zid, xitem))
    flag = "1" if active else "0"
    now = dt.datetime.now()
    if cur.fetchone():
        cur.execute("UPDATE xcodes SET zactive = %s, zutime = %s WHERE zid = %s AND xtype = 'Price Category'"
                    " AND xcode = %s AND coalesce(zactive,'') <> %s", (flag, now, st.zid, xitem, flag))
    else:
        row = {"ztime": now, "zid": st.zid, "xtype": "Price Category", "xcode": xitem, "xcodealt": xalias,
               "xdescdet": (desc or "")[:250], "zactive": flag}
        insert_row(cur, "xcodes", row)


def handle_item(cur, st, evt):
    """Shop is master for title, options, handle, VG code; the ERP is master for price (never written here,
    except as the starting price of a brand-new item)."""
    data = evt["data"]
    warnings = []
    sku, barcode = clean(data.get("sku")), clean(data.get("barcode"))
    if not sku and not barcode:
        raise EventError("missing_sku", "item has neither SKU nor barcode")
    name = clean(data.get("name")) or clean(data.get("product_title"))
    base = clean(data.get("product_title"))
    note = options_note(data.get("options"))
    handle = clean(data.get("product_handle"))
    price = data.get("price")
    active = data.get("active", True) is not False
    now = dt.datetime.now()
    row = find_item(cur, st.zid, barcode, sku, warnings)

    if row:
        upd = {"xdesc": name, "xlong": base, "xnote": note, "xlinks": handle, "zutime": now}
        if sku and not clean(row.get("xalias")):
            upd["xalias"] = sku
        if not clean(row.get("xscode")):
            code = resolve_scode(cur, st.zid, handle, data.get("variant_title"))
            if code:
                upd["xscode"] = code
        fit_row(cur, "caitem", upd, warnings)
        sets = ", ".join("%s = %%s" % c for c in upd)
        cur.execute("UPDATE caitem SET %s WHERE zid = %%s AND xitem = %%s" % sets,
                    list(upd.values()) + [st.zid, row["xitem"]])
        ensure_xcodes(cur, st, row["xitem"], row["xalias"] or sku, name, active)
        return {"status": "ok", "erp_ref": {"xitem": row["xitem"]}, "warnings": warnings}

    # not found -> INSERT with the next code for the business the shop says it belongs to
    source = clean(data.get("erp_source")).upper()
    if source not in ERP_SOURCE_PREFIX:
        raise EventError("unknown_erp_source", "erp_source %r is not one of %s" % (source, sorted(ERP_SOURCE_PREFIX)))
    prefix = ERP_SOURCE_PREFIX[source]
    advisory_xact_lock(cur, "um:caitem:%s" % prefix)
    cur.execute("SELECT max(xitem) FROM caitem WHERE zid = %s AND xitem ~ %s", (st.zid, "^" + prefix + "-[0-9]{5}$"))
    top = cur.fetchone()["max"]
    xitem = "%s-%05d" % (prefix, int(top.split("-")[1]) + 1 if top else 1)

    cur.execute("SELECT * FROM caitem WHERE zid = %s AND xcatful = 'Auto' AND coalesce(xunitstk,'') <> ''"
                " AND xitem LIKE %s ORDER BY ztime DESC LIMIT 1", (st.zid, prefix + "-%"))
    tpl = cur.fetchone()
    if not tpl:
        cur.execute("SELECT * FROM caitem WHERE zid = %s AND xcatful = 'Auto' AND coalesce(xunitstk,'') <> ''"
                    " ORDER BY ztime DESC LIMIT 1", (st.zid,))
        tpl = cur.fetchone()
    new = dict(tpl) if tpl else {}
    new.update({
        "zid": st.zid, "xitem": xitem, "xalias": sku or None, "xitemnew": PREFIX_XITEMNEW[prefix],
        "xdesc": name, "xlong": base, "xnote": note or None, "xlinks": handle or None,
        "xscode": resolve_scode(cur, st.zid, handle, data.get("variant_title")),
        "xgitem": PREFIX_XGITEM[prefix], "xpricecat": sku or None,
        "xstdprice": d(price) if price is not None else None, "xstdcost": None,
        "xcatful": "Auto", "xwh": st.warehouse, "xbarcode": None, "xdrawing": None,
        "ztime": now, "zutime": None,
    })
    fit_row(cur, "caitem", new, warnings)
    insert_row(cur, "caitem", new)
    ensure_xcodes(cur, st, xitem, sku, name, active)
    if price is not None:
        warnings.append({"code": "start_price_from_shop", "sku": sku, "price": float(d(price))})
    return {"status": "ok", "erp_ref": {"xitem": xitem}, "warnings": warnings}


# ───────────────────────────── customers (not stored) ─────────────────────────────

def handle_customer_ignored(cur, st, evt):
    """All online sales go to the one UtilityMart customer; shopper details stay in the shop."""
    return {"status": "ok", "warnings": [{"code": "customer_not_stored"}]}


# ───────────────────────────── orders (shop -> opmob) ─────────────────────────────

def order_map_get(cur, ext_id):
    cur.execute("SELECT * FROM um_order_map WHERE external_order_id = %s", (ext_id,))
    return cur.fetchone()


def order_map_save(cur, ext_id, **fields):
    row = order_map_get(cur, ext_id)
    fields["updated_at"] = dt.datetime.now()
    if row:
        cur.execute("UPDATE um_order_map SET %s WHERE external_order_id = %%s" %
                    ", ".join("%s = %%s" % c for c in fields), list(fields.values()) + [ext_id])
    else:
        fields["external_order_id"] = ext_id
        cols = list(fields)
        cur.execute("INSERT INTO um_order_map (%s) VALUES (%s)" % (", ".join(cols), ", ".join(["%s"] * len(cols))),
                    [fields[c] for c in cols])


def order_info(cur, ext_id):
    row = order_map_get(cur, ext_id)
    try:
        return json.loads(row["info"]) if row and row["info"] else {}
    except ValueError:
        return {}


def attention(cur, kind, order_ref, details):
    cur.execute("INSERT INTO um_attention (kind, order_ref, details) VALUES (%s, %s, %s)", (kind, order_ref, details))


def order_numbers(st, data):
    """(invoiceno 'UM-1000001', invoicesl 1000001) from the shop's order number."""
    disp = data.get("display_id")
    try:
        n = int(disp)
    except (TypeError, ValueError):
        raise EventError("missing_display_id", "order has no usable display_id (%r)" % (disp,))
    return "%s%d" % (st.order_prefix, n), n


def resolve_lines(cur, st, items):
    out, missing, bundles = [], [], []
    for ln in items:
        it = find_item(cur, st.zid, clean(ln.get("barcode")), clean(ln.get("sku")))
        if not it:
            (bundles if ln.get("bundle") else missing).append(clean(ln.get("sku")) or clean(ln.get("barcode")) or clean(ln.get("name")) or "?")
            continue
        out.append((ln, it))
    if bundles:
        raise EventError("bundle_not_expanded", "bundle lines must arrive as their parts, the ERP has no bundle items: " + ", ".join(bundles))
    if missing:
        raise EventError("unknown_item", "no ERP item for: " + ", ".join(missing))
    return out


def check_totals(data):
    t = data.get("totals") or {}
    lines_sum = sum(d(x.get("line_total")) for x in data.get("items", []))
    ship = d(t.get("shipping"))
    disc = d(t.get("discount"))
    items_total = d(t.get("items"))
    total = d(t.get("total"))
    problems = []
    if abs(lines_sum - items_total) > 1:
        problems.append("sum of lines %s != totals.items %s" % (lines_sum, items_total))
    if abs(items_total + ship - disc - total) > 1:
        problems.append("items %s + shipping %s - discount %s != total %s" % (items_total, ship, disc, total))
    if problems:
        raise EventError("total_mismatch", "; ".join(problems))
    return items_total - disc          # net of discount, before delivery


def payment_text(pay):
    """xtra3: cash on delivery, or the online payment's transaction id."""
    method = clean(pay.get("method")).lower() or "cod"
    if method == "cod":
        return "COD"
    return clean(pay.get("transaction_id")) or "ONLINE"


def consignment_text(consignments):
    return ",".join(clean(c.get("id")) for c in (consignments or []) if clean(c.get("id")))


def write_opmob(cur, st, data, resolved, invoiceno, invoicesl, warnings):
    ext_id = data["external_order_id"]
    net = check_totals(data)
    t = data.get("totals") or {}
    items_total, disc = d(t.get("items")), d(t.get("discount"))
    pct = (disc / items_total * 100).quantize(Decimal("0.01")) if items_total > 0 else Decimal(0)
    pay = data.get("payment") or {}
    ship = data.get("shipping") or {}
    extra = {
        "shop_order_id": ext_id, "delivery_charge": float(d(ship.get("charge"))),
        "free_delivery": bool(ship.get("free_delivery_applied")), "shipping_method": clean(ship.get("method")),
        "note": clean(data.get("notes")), "items_total": float(items_total), "discount_amount": float(disc),
        "discount_percent": float(pct), "payment_method": clean(pay.get("method")).lower() or "cod",
        "gateway": clean(pay.get("gateway")), "channel": clean(data.get("channel")),
    }
    now = dt.datetime.now()
    xdate = dhaka_date(data.get("placed_at") or "")
    trail_pay = payment_text(pay)
    trail_courier = consignment_text(data.get("consignments")) or None
    for i, (ln, it) in enumerate(resolved, 1):
        qty = int(Decimal(str(ln.get("quantity") or 0)))
        price = d(ln.get("unit_price"))
        row = {
            "zid": st.zid, "ztime": now, "zutime": now, "invoiceno": invoiceno, "invoicesl": invoicesl,
            "username": st.username, "xemp": st.salesman, "xcus": st.customer, "xcusname": st.customer_name,
            "xcusadd": st.customer_address, "xitem": it["xitem"],
            "xdesc": clean(it.get("xdesc")) or clean(ln.get("name")),
            "xqty": qty, "xprice": float(price), "xstatusord": "New", "xordernum": None, "xroword": i,
            "xterminal": st.terminal, "xdate": xdate, "xsl": str(uuid.uuid4()), "xlat": None, "xlong": None,
            "xlinetotal": money_int(price * qty), "xtra1": money_int(pct), "xtra2": float(net),
            "xtra3": trail_pay, "xtra4": trail_courier, "xtra5": json.dumps(extra, ensure_ascii=False),
        }
        fit_row(cur, "opmob", row, warnings)
        insert_row(cur, "opmob", row)
    return extra


def queue_po(cur, st, ext_id):
    """Daily mode: remember the order; the po job makes the purchase orders at the fixed time."""
    cur.execute("SELECT 1 FROM um_po_queue WHERE external_order_id = %s", (ext_id,))
    if cur.fetchone():
        cur.execute("UPDATE um_po_queue SET queued_at = now(), done_at = NULL WHERE external_order_id = %s", (ext_id,))
    else:
        cur.execute("INSERT INTO um_po_queue (external_order_id) VALUES (%s)", (ext_id,))


def handle_order_write(cur, st, evt, replace):
    data = evt["data"]
    warnings = []
    ext_id = data["external_order_id"]
    invoiceno, invoicesl = order_numbers(st, data)
    resolved = resolve_lines(cur, st, data.get("items") or [])
    if not resolved:
        raise EventError("no_items", "order has no lines")
    cur.execute("SELECT xstatusord FROM opmob WHERE zid = %s AND invoiceno = %s FOR UPDATE", (st.zid, invoiceno))
    rows = cur.fetchall()
    mapped = order_map_get(cur, ext_id)
    if rows:
        if any(r["xstatusord"] != "New" for r in rows):
            raise EventError("locked", "order already moved on in the ERP (%s); staff decide" % rows[0]["xstatusord"])
        if not replace:
            return {"status": "ok", "erp_ref": {"order_no": invoiceno, "document": "opmob"},
                    "warnings": [{"code": "already_entered"}]}
        cur.execute("DELETE FROM opmob WHERE zid = %s AND invoiceno = %s", (st.zid, invoiceno))
        if mapped:
            delete_open_pos(cur, st, mapped["po_numbers"], invoiceno, warnings)
    elif replace and mapped and mapped["state"] == "canceled":
        raise EventError("locked", "order was canceled")
    write_opmob(cur, st, data, resolved, invoiceno, invoicesl, warnings)
    order_map_save(cur, ext_id, display_id=int(invoicesl), invoiceno=invoiceno, invoicesl=invoicesl,
                   customer_code=st.customer, version=evt.get("version"), state="active", po_numbers="",
                   info=json.dumps({"payment": data.get("payment") or {}}))
    pos = []
    if st.po_mode == "immediate":
        pos = make_pos_for_orders(cur, st, [ext_id], warnings)
    elif st.po_mode == "daily":
        queue_po(cur, st, ext_id)
    ref = {"order_no": invoiceno, "document": "opmob", "id": str(invoicesl), "customer_code": st.customer}
    if pos:
        ref["purchase_orders"] = pos
    return {"status": "ok", "erp_ref": ref, "warnings": warnings}


def handle_order_created(cur, st, evt):
    return handle_order_write(cur, st, evt, replace=False)


def handle_order_updated(cur, st, evt):
    return handle_order_write(cur, st, evt, replace=True)


def handle_order_canceled(cur, st, evt):
    data = evt["data"]
    ext_id = data.get("external_order_id")
    warnings = []
    mapped = order_map_get(cur, ext_id)
    if not mapped:
        return {"status": "ok", "warnings": [{"code": "order_unknown_in_erp"}]}
    invoiceno = mapped["invoiceno"]
    cur.execute("SELECT xstatusord FROM opmob WHERE zid = %s AND invoiceno = %s FOR UPDATE", (st.zid, invoiceno))
    rows = cur.fetchall()
    if rows and all(r["xstatusord"] == "New" for r in rows):
        cur.execute("DELETE FROM opmob WHERE zid = %s AND invoiceno = %s", (st.zid, invoiceno))
        delete_open_pos(cur, st, mapped["po_numbers"], invoiceno, warnings)
        cur.execute("DELETE FROM um_po_queue WHERE external_order_id = %s", (ext_id,))
        order_map_save(cur, ext_id, state="canceled", po_numbers="")
        return {"status": "ok", "warnings": warnings}
    if not rows:
        order_map_save(cur, ext_id, state="canceled")
        return {"status": "ok", "warnings": [{"code": "removed_in_erp_already"}]}
    def after(c):
        attention(c, "credit_note_needed", invoiceno, "Order already moved on in the ERP: register a return / credit "
                  "note and a purchase return to the source business (manual)")
        order_map_save(c, ext_id, state="locked")
    raise EventError("locked", "already converted/created in the ERP; staff must register the return", after=after)


def set_trail(cur, st, invoiceno, xtra3=None, xtra4=None, extra=None):
    """Keep the payment (xtra3) and courier (xtra4) trail on every line of the order, whatever its status."""
    cur.execute("SELECT xsl, xtra5 FROM opmob WHERE zid = %s AND invoiceno = %s", (st.zid, invoiceno))
    for r in cur.fetchall():
        sets, vals = [], []
        if xtra3 is not None:
            sets.append("xtra3 = %s"); vals.append(xtra3)
        if xtra4 is not None:
            sets.append("xtra4 = %s"); vals.append(xtra4)
        if extra:
            try:
                x = json.loads(r["xtra5"]) if r["xtra5"] else {}
            except ValueError:
                x = {}
            x.update(extra)
            sets.append("xtra5 = %s"); vals.append(json.dumps(x, ensure_ascii=False))
        if sets:
            cur.execute("UPDATE opmob SET %s WHERE zid = %%s AND xsl = %%s" % ", ".join(sets), vals + [st.zid, r["xsl"]])


def handle_order_info(cur, st, evt):
    """payment.captured / order.fulfilled / order.delivered: kept with the order and on the opmob trail."""
    data = evt["data"]
    ext_id = data.get("external_order_id")
    mapped = order_map_get(cur, ext_id)
    if not mapped:
        raise EventError("unknown_order", "no ERP order for %s" % ext_id)
    info = order_info(cur, ext_id)
    t = evt["type"]
    invoiceno = mapped["invoiceno"]
    if t == "payment.captured":
        info["payment_captured"] = {k: data.get(k) for k in ("amount", "method", "transaction_id", "captured_at")}
        txn = clean(data.get("transaction_id"))
        if txn and clean(data.get("method")).lower() != "cod":
            set_trail(cur, st, invoiceno, xtra3=txn)
    elif t == "order.delivered":
        info["delivered_at"] = data.get("at")
    elif t == "order.fulfilled":
        info["consignments"] = data.get("consignments") or []
        ids = consignment_text(info["consignments"])
        if ids:
            set_trail(cur, st, invoiceno, xtra4=ids, extra={"consignments": info["consignments"]})
    order_map_save(cur, ext_id, info=json.dumps(info, ensure_ascii=False))
    return {"status": "ok", "warnings": []}


def handle_manual_return(cur, st, evt):
    data = evt["data"]
    ext_id = data.get("external_order_id")
    mapped = order_map_get(cur, ext_id) if ext_id else None
    ref = mapped["invoiceno"] if mapped else (ext_id or "?")
    def after(c):
        attention(c, "return_to_register", ref, "%s: returns, refunds and purchase returns are manual. Payload: %s"
                  % (evt["type"], json.dumps(data, ensure_ascii=False)[:600]))
    raise EventError("return_manual", "returns are not automated: staff register the return in the ERP", after=after)


# ───────────────────────────── purchase orders ─────────────────────────────

def po_shortfall(cur, st, xitem, qty, invoiceno):
    """How many units 100007 must still buy: this order + other open web demand - own stock - open PO quantity."""
    cur.execute("SELECT coalesce(sum(xqty * xsign), 0) AS s FROM imtrn WHERE zid = %s AND xitem = %s", (st.zid, xitem))
    stock = cur.fetchone()["s"]
    cur.execute("SELECT coalesce(sum(m.xqty), 0) AS s FROM opmob m WHERE m.zid = %s AND m.xitem = %s"
                " AND m.invoiceno IS DISTINCT FROM %s AND (m.xstatusord = 'New' OR (m.xstatusord = 'Order Created'"
                " AND NOT EXISTS (SELECT 1 FROM opdor d WHERE d.zid = m.zid AND d.xordernum = m.xordernum)))",
                (st.zid, xitem, invoiceno))
    demand = cur.fetchone()["s"]
    cur.execute("SELECT coalesce(sum(GREATEST(coalesce(d.xqtyord,0) - coalesce(d.xqtygrn,0), 0)), 0) AS s"
                " FROM poodt d JOIN poord o ON o.zid = d.zid AND o.xpornum = d.xpornum"
                " WHERE d.zid = %s AND d.xitem = %s AND coalesce(o.xstatuspor,'') NOT IN ('5-Received')",
                (st.zid, xitem))
    supply = cur.fetchone()["s"]
    need = Decimal(qty) + demand - stock - supply
    return max(Decimal(0), min(Decimal(qty), need))


def po_rate(cur, st, it, total_qty, warnings):
    """Source company's xstdprice less its opspprc quantity tier (highest tier the day's total reaches).
    Fixit: Central's sales rate (um_fixit_item.cost)."""
    src, sku = source_of(it), clean(it.get("xalias"))
    if src == FIXIT:
        cur.execute("SELECT cost FROM um_fixit_item WHERE sku = %s AND found", (sku,))
        r = cur.fetchone()
        cost = d(r["cost"]) if r and r["cost"] is not None else Decimal(0)
        if cost == 0:
            warnings.append({"code": "po_price_missing", "sku": sku, "source": "fixit"})
        return cost
    cur.execute("SELECT xstdprice, xpricecat FROM caitem WHERE zid = %s AND xitem = %s", (int(src), sku))
    r = cur.fetchone()
    base = d(r["xstdprice"]) if r and r["xstdprice"] is not None else Decimal(0)
    if base == 0:
        warnings.append({"code": "po_price_missing", "sku": sku, "source": src})
        return base
    rate = base
    if r["xpricecat"]:
        cur.execute("SELECT xqty, xdisc FROM opspprc WHERE zid = %s AND xpricecat = %s AND xqty <= %s"
                    " AND (xdateeff IS NULL OR xdateeff <= current_date) AND (xdateexp IS NULL OR xdateexp >= current_date)"
                    " ORDER BY xqty DESC LIMIT 1", (int(src), r["xpricecat"], total_qty))
        t = cur.fetchone()
        if t and t["xdisc"]:
            rate = max(base - d(t["xdisc"]), Decimal(0))
            warnings.append({"code": "tier_rate", "sku": sku, "tier_qty": float(t["xqty"]), "base": float(base),
                             "rate": float(rate)})
    return rate


def make_pos_for_orders(cur, st, ext_ids, warnings):
    """One OPEN PO per order per source business, for the units 100007 does not already have."""
    orders = []
    for e in ext_ids:
        m = order_map_get(cur, e)
        if m and m["state"] == "active":
            cur.execute("SELECT xitem, sum(xqty) AS q FROM opmob WHERE zid = %s AND invoiceno = %s GROUP BY xitem",
                        (st.zid, m["invoiceno"]))
            orders.append((m, cur.fetchall()))
    day_qty = {}
    for _, lines in orders:
        for ln in lines:
            day_qty[ln["xitem"]] = day_qty.get(ln["xitem"], Decimal(0)) + d(ln["q"])
    all_made = []
    for m, lines in orders:
        made = make_order_pos(cur, st, m, lines, day_qty, warnings)
        order_map_save(cur, m["external_order_id"], po_numbers=",".join(made))
        cur.execute("UPDATE um_po_queue SET done_at = now() WHERE external_order_id = %s", (m["external_order_id"],))
        all_made += made
    return all_made


def make_order_pos(cur, st, order, lines, day_qty, warnings):
    invoiceno, ext_id = order["invoiceno"], order["external_order_id"]
    by_src, manual = {}, []
    for ln in lines:
        cur.execute("SELECT * FROM caitem WHERE zid = %s AND xitem = %s", (st.zid, ln["xitem"]))
        it = cur.fetchone()
        src = source_of(it) if it else ""
        if src:
            qty = po_shortfall(cur, st, it["xitem"], int(d(ln["q"])), invoiceno)
            if qty > 0:
                by_src.setdefault(src, []).append((it, qty))
        else:
            manual.append(ln["xitem"])
    if manual:
        attention(cur, "manual_sourcing", invoiceno, "Items with no source business need sourcing by hand: " + ", ".join(manual))
        warnings.append({"code": "manual_sourcing", "items": manual})
    cur.execute("SELECT * FROM poord WHERE zid = %s ORDER BY ztime DESC LIMIT 1", (st.zid,))
    tpl_h = cur.fetchone()
    cur.execute("SELECT * FROM poodt WHERE zid = %s ORDER BY ztime DESC LIMIT 1", (st.zid,))
    tpl_l = cur.fetchone()
    today, now, made = dt.date.today(), dt.datetime.now(), []
    for src, plines in sorted(by_src.items()):
        supplier = st.suppliers.get(src)
        if not supplier:
            attention(cur, "po_not_created", invoiceno, "no supplier configured for source %s" % src)
            warnings.append({"code": "no_supplier", "source": src})
            continue
        pnum = next_code(cur, "poord", "xpornum", st.zid, "PO--", 6)
        rows, total = [], Decimal(0)
        for n, (it, qty) in enumerate(plines, 1):
            rate = po_rate(cur, st, it, day_qty.get(it["xitem"], qty), warnings)
            amt = rate * qty
            total += amt
            ln = dict(tpl_l) if tpl_l else {}
            ln.update({"zid": st.zid, "ztime": now, "zutime": None, "xpornum": pnum, "xrow": n, "xcode": it["xitem"],
                       "xcodebasis": "Our Code", "xitem": it["xitem"], "xlong": "", "xstatuspdt": "Priced",
                       "xstype": "Stock-N-Sell", "xwh": st.warehouse, "xqtyord": qty, "xqtygrn": 0, "xqtystk": 0,
                       "xcfpur": 1, "xrate": rate, "xprice": rate, "xdisc": 0, "xdtdisc": 0, "xdttax": 0,
                       "xdtwotax": amt, "xlineamt": amt, "xdatesch": PO_NO_DATE, "xline": 0, "xpricebasis": "Entered"})
            rows.append(ln)
        hdr = dict(tpl_h) if tpl_h else {}
        hdr.update({"zid": st.zid, "ztime": now, "zutime": None, "xpornum": pnum, "xdate": today, "xsup": supplier,
                    "xdatesupref": today, "xstatuspor": PO_OPEN_STATUS, "xappamt": 0, "xpartial": "0",
                    "xcur": "BDT", "xexch": 1, "xwh": st.warehouse, "xproj": st.project, "xdisc": 0,
                    "xdtwotax": total, "xdtdisc": 0, "xdttax": 0, "xval": total, "xdiscamt": 0, "xtotamt": total,
                    "xtrnpor": "PO--", "xtypepor": "Local", "xwhoption": "Header", "xgrnnum": None,
                    "xref": invoiceno, "xsupref": invoiceno,
                    "xrem": "Auto-created from online order %s (%s)" % (invoiceno, ext_id)})
        if st.user_email:
            hdr["zemail"] = st.user_email
            hdr["xmember"] = st.user_email
        fit_row(cur, "poord", hdr, warnings)
        insert_row(cur, "poord", hdr)
        for ln in rows:
            fit_row(cur, "poodt", ln, warnings)
            insert_row(cur, "poodt", ln)
        made.append(pnum)
    return made


def delete_open_pos(cur, st, po_numbers, ref, warnings):
    """Remove importer-made POs that staff have not confirmed; flag the rest for a purchase return."""
    for pnum in [p for p in (po_numbers or "").split(",") if p]:
        cur.execute("SELECT xstatuspor FROM poord WHERE zid = %s AND xpornum = %s", (st.zid, pnum))
        r = cur.fetchone()
        if not r:
            continue
        if r["xstatuspor"] == PO_OPEN_STATUS:
            cur.execute("DELETE FROM poodt WHERE zid = %s AND xpornum = %s", (st.zid, pnum))
            cur.execute("DELETE FROM poord WHERE zid = %s AND xpornum = %s", (st.zid, pnum))
        else:
            attention(cur, "purchase_return_needed", ref,
                      "PO %s is %s: return the goods to the source business (manual purchase return)" % (pnum, r["xstatuspor"]))
            warnings.append({"code": "purchase_return_needed", "po": pnum, "status": r["xstatuspor"]})


# ───────────────────────────── event dispatch ─────────────────────────────

HANDLERS = {
    "item.upserted": handle_item,
    "customer.upserted": handle_customer_ignored,
    "customer.erased": handle_customer_ignored,
    "order.created": handle_order_created,
    "order.updated": handle_order_updated,
    "order.canceled": handle_order_canceled,
    "payment.captured": handle_order_info,
    "order.fulfilled": handle_order_info,
    "order.delivered": handle_order_info,
    "order.refunded": handle_manual_return,
    "return.created": handle_manual_return,
    "return.received": handle_manual_return,
    "return.refunded": handle_manual_return,
}


def event_key(evt):
    d_ = evt.get("data") or {}
    return (d_.get("external_order_id") or d_.get("external_return_id") or d_.get("sku")
            or (d_.get("customer") or {}).get("shop_id") or d_.get("shop_id") or "")


def process_event(conn, st, evt, commit):
    cur = dict_cur(conn)
    cur.execute("SELECT status, result FROM um_processed_events WHERE event_id = %s", (evt["id"],))
    prev = cur.fetchone()
    if prev:
        res = json.loads(prev["result"])
        res.update({"seq": evt["seq"], "id": evt["id"]})
        LOG.info("seq=%s type=%s key=%s -> already processed (%s), re-acknowledging", evt["seq"], evt["type"],
                 event_key(evt), prev["status"])
        return res
    cur.execute("SAVEPOINT ev")
    handler = HANDLERS.get(evt["type"])
    try:
        if handler is None:
            result = {"status": "ok", "warnings": [{"code": "ignored_event_type", "type": evt["type"]}]}
        else:
            result = handler(cur, st, evt)
        cur.execute("RELEASE SAVEPOINT ev")
    except EventError as e:
        cur.execute("ROLLBACK TO SAVEPOINT ev")
        if e.after:
            e.after(cur)
        result = {"status": "failed", "error": {"code": e.code, "message": e.message, "retryable": e.retryable}}
    except (psycopg2.OperationalError, psycopg2.InterfaceError):
        raise
    except psycopg2.Error as e:
        cur.execute("ROLLBACK TO SAVEPOINT ev")
        code = "encoding" if isinstance(e, (psycopg2.DataError,)) and "encoding" in str(e).lower() else "database_error"
        result = {"status": "failed", "error": {"code": code, "message": str(e).strip()[:300], "retryable": False}}
    except Exception as e:                                   # never let one event stop the run
        cur.execute("ROLLBACK TO SAVEPOINT ev")
        LOG.exception("unexpected error on event %s", evt.get("id"))
        result = {"status": "failed", "error": {"code": "importer_error", "message": repr(e)[:300],
                                                "retryable": False}}
    result["seq"], result["id"] = evt["seq"], evt["id"]
    stored = {k: v for k, v in result.items() if k not in ("seq", "id")}
    cur.execute("INSERT INTO um_processed_events (event_id, seq, type, key, status, result) VALUES (%s,%s,%s,%s,%s,%s)",
                (evt["id"], evt["seq"], evt["type"], event_key(evt), result["status"],
                 json.dumps(stored, ensure_ascii=False, default=str)))
    cur.execute("UPDATE um_sync_state SET last_seq = %s, updated_at = now() WHERE id = 1 AND last_seq < %s",
                (evt["seq"], evt["seq"]))
    LOG.info("seq=%s type=%s key=%s -> %s %s%s", evt["seq"], evt["type"], event_key(evt), result["status"],
             json.dumps(result.get("erp_ref") or {}, ensure_ascii=False),
             (" warnings=" + json.dumps(result["warnings"], ensure_ascii=False)) if result.get("warnings") else "")
    if result["status"] == "failed":
        LOG.warning("   FAILED %s: %s", result["error"]["code"], result["error"]["message"])
    if commit:
        conn.commit()
    return result


# ───────────────────────────── order status (ERP -> shop) ─────────────────────────────

def compute_status(cur, st, row):
    refs = {"opmob": row["invoiceno"], "order": None, "sale": None, "credit_note": None}
    cur.execute("SELECT xstatusord, xordernum FROM opmob WHERE zid = %s AND invoiceno = %s LIMIT 1",
                (st.zid, row["invoiceno"]))
    m = cur.fetchone()
    if not m:
        return ("canceled" if row["state"] == "canceled" else "removed_in_erp"), refs
    s = m["xstatusord"]
    if s == "New":
        return "entered", refs
    if s == "Order Created" and m["xordernum"]:
        refs["order"] = m["xordernum"]
        cur.execute("SELECT xdornum FROM opdor WHERE zid = %s AND xordernum = %s LIMIT 1", (st.zid, m["xordernum"]))
        so = cur.fetchone()
        if not so:
            return "order_created", refs
        refs["sale"] = so["xdornum"]
        cur.execute("SELECT xcrnnum FROM opcrn WHERE zid = %s AND xdornum = %s LIMIT 1", (st.zid, so["xdornum"]))
        cr = cur.fetchone()
        if cr:
            refs["credit_note"] = cr["xcrnnum"]
            return "credited", refs
        return "confirmed", refs
    refs["note"] = s
    return "entered", refs


def collect_statuses(conn, st, resend_all):
    cur = dict_cur(conn)
    cur.execute("SELECT * FROM um_order_map ORDER BY created_at")
    out = []
    for row in cur.fetchall():
        status, refs = compute_status(cur, st, row)
        final = status in ("canceled", "credited")
        if status != row["last_status"] or (resend_all and not (final and row["last_status"] == status)):
            out.append({"external_order_id": row["external_order_id"], "display_id": row["display_id"],
                        "erp_status": status, "at": utc_now_iso(), "refs": refs})
    return out


def mark_statuses(conn, statuses):
    cur = conn.cursor()
    for s in statuses:
        cur.execute("UPDATE um_order_map SET last_status = %s, last_status_at = now() WHERE external_order_id = %s",
                    (s["erp_status"], s["external_order_id"]))


# ───────────────────────────── stock and product details (ERP -> shop) ─────────────────────────────

STOCK_SQL = """
    SELECT c.xalias AS sku, floor(max(f.stock::numeric)) AS on_hand
    FROM caitem c
    JOIN final_items_view f ON f.item_id = c.xalias
         AND f.zid = (CASE WHEN c.xitemnew ~ '^[0-9]+$' THEN c.xitemnew::int END)
    WHERE c.zid = %s AND c.xitemnew IN %s AND coalesce(c.xalias, '') <> ''
    GROUP BY c.xalias
"""
FIXIT_STOCK_SQL = """
    SELECT c.xalias AS sku, floor(greatest(u.stock, 0)) AS on_hand
    FROM caitem c JOIN um_fixit_item u ON u.sku = c.xalias AND u.found
    WHERE c.zid = %s AND lower(c.xitemnew) = 'fixit' AND coalesce(c.xalias, '') <> ''
"""


def current_stock(cur, st):
    cur.execute(STOCK_SQL, (st.zid, NUMERIC_SOURCES))
    out = {r["sku"]: max(int(r["on_hand"] or 0), 0) for r in cur.fetchall()}
    cur.execute(FIXIT_STOCK_SQL, (st.zid,))
    out.update({r["sku"]: max(int(r["on_hand"] or 0), 0) for r in cur.fetchall()})
    return {k: v for k, v in out.items() if k not in st.skip_skus}


def collect_stock(conn, st, full):
    cur = dict_cur(conn)
    current = current_stock(cur, st)
    if full:
        return current, current
    cur.execute("SELECT sku, on_hand FROM um_stock_sent")
    sent = {r["sku"]: r["on_hand"] for r in cur.fetchall()}
    return {k: v for k, v in current.items() if sent.get(k) != v}, current


def push_stock(conn, shop, items, full):
    cur = conn.cursor()
    rows = [{"sku": k, "on_hand": v} for k, v in sorted(items.items())]
    if not rows:
        return 0
    chunks = [rows[i:i + 500] for i in range(0, len(rows), 500)]
    one_call_full = full and len(chunks) == 1      # >500 SKUs: several calls, none claims to be the whole list
    for chunk in chunks:
        shop.stock(chunk, one_call_full)
        for r in chunk:
            cur.execute("SELECT 1 FROM um_stock_sent WHERE sku = %s", (r["sku"],))
            if cur.fetchone():
                cur.execute("UPDATE um_stock_sent SET on_hand = %s, sent_at = now() WHERE sku = %s",
                            (r["on_hand"], r["sku"]))
            else:
                cur.execute("INSERT INTO um_stock_sent (sku, on_hand) VALUES (%s, %s)", (r["sku"], r["on_hand"]))
    return len(rows)


def r2(x):
    return None if x is None else float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def collect_products(cur, st, only=None):
    """barcode, price, cost, item group and stock for every followed item, keyed by SKU."""
    cur.execute("SELECT xitem, xalias, xitemnew, xstdprice, xgitem FROM caitem WHERE zid = %s"
                " AND coalesce(xalias,'') <> '' ORDER BY xitem", (st.zid,))
    items = cur.fetchall()
    cur.execute("SELECT DISTINCT ON (d.xitem) d.xitem, d.xrate FROM poodt d JOIN poord o ON o.zid = d.zid"
                " AND o.xpornum = d.xpornum WHERE d.zid = %s AND coalesce(d.xrate, 0) > 0"
                " ORDER BY d.xitem, o.xdate DESC, o.xpornum DESC", (st.zid,))
    po_rate_of = {r["xitem"]: r["xrate"] for r in cur.fetchall()}
    cur.execute("SELECT DISTINCT ON (xitem) xitem, xval / xqty AS unit FROM imtrn WHERE zid = %s"
                " AND substr(ximtrnnum, 1, 4) = 'RE--' AND xqty > 0 AND xval > 0"
                " ORDER BY xitem, xdate DESC, ximtrnnum DESC", (st.zid,))
    receipt_of = {r["xitem"]: r["unit"] for r in cur.fetchall()}
    cur.execute("SELECT c.xitem, s.xstdprice AS src FROM caitem c JOIN caitem s ON s.xitem = c.xalias"
                " AND s.zid = (CASE WHEN c.xitemnew ~ '^[0-9]+$' THEN c.xitemnew::int END) WHERE c.zid = %s", (st.zid,))
    source_price = {r["xitem"]: r["src"] for r in cur.fetchall()}
    cur.execute("SELECT sku, cost, item_group FROM um_fixit_item WHERE found")
    fixit = {r["sku"]: r for r in cur.fetchall()}
    stock = current_stock(cur, st)
    out = {}
    for it in items:
        src, sku = source_of(it), it["xalias"]
        if not src or sku in st.skip_skus or (only and sku not in only):
            continue
        group = it["xgitem"]
        if src == FIXIT:
            fx = fixit.get(sku, {})
            cost = fx.get("cost") or po_rate_of.get(it["xitem"]) or receipt_of.get(it["xitem"])
            group = fx.get("item_group") or group
        else:
            cost = po_rate_of.get(it["xitem"]) or receipt_of.get(it["xitem"]) or source_price.get(it["xitem"])
        out[sku] = {"sku": sku, "barcode": it["xitem"], "price": r2(it["xstdprice"]) if it["xstdprice"] else None,
                    "cost_price": r2(cost) if cost else None, "item_group": group or None, "on_hand": stock.get(sku)}
    return out


def _skus_in(x):
    """The SKUs named in a list of the shop's answer: plain strings, or objects with a `sku`."""
    return {(i.get("sku") if isinstance(i, dict) else i) for i in (x or []) if i}


def shop_dry_run_answers(shop, todo, full):
    """POST /products?dry_run=1 for every item (in calls of 500): the shop says what it WOULD change and writes nothing."""
    chunks = [todo[i:i + 500] for i in range(0, len(todo), 500)]
    answers = [shop.products(chunk, bool(full) and len(chunks) == 1, dry_run=True) for chunk in chunks]
    conflicts = [c for a in answers for c in (a.get("conflicts") or []) if isinstance(c, dict)]
    if conflicts:
        # a barcode wanted by one item is held by another that changes in a different call of 500; the real push sends the
        # conflicting items together at the end, so ask the same way: them plus the items named as holding the barcode
        names = _skus_in(conflicts) | {m for c in conflicts for m in re.findall(r"in use by (\S+)", str(c.get("reason", "")))}
        again = [p for p in todo if p["sku"] in names]
        LOG.info("barcode conflicts: %d; asking again with them and the items holding their barcodes together (%d items)", len(conflicts), len(again))
        answers.append(shop.products(again, False, dry_run=True))
    return answers


def shop_dry_run_products(shop, todo, full):
    """The dry run for `--job products --shop-dry-run`: asks, saves the shop's answer, prints a summary. Nothing is kept here."""
    answers = shop_dry_run_answers(shop, todo, full)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    path = os.path.join(HERE, "logs", "shop_dry_run_products_%s.json" % dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=1)
    LOG.info("shop dry run: %d call(s), the shop wrote nothing; its full answer: %s", len(answers), path)
    for n, a in enumerate(answers, 1):
        for k, v in sorted(a.items()):
            if isinstance(v, (list, dict)):
                first = (v[0] if isinstance(v, list) else next(iter(v.items()))) if v else None
                LOG.info("   call %d  %-14s %d%s", n, k, len(v), "" if first is None else "   e.g. " + json.dumps(first, ensure_ascii=False)[:150])
            else:
                LOG.info("   call %d  %-14s %s", n, k, v)
    rows = [x for a in answers for x in (a.get("would_change") or []) if isinstance(x, dict)]
    if rows:                                    # the shop's own list of changes, flattened for a spreadsheet
        cols = sorted({k for r in rows for k in r}, key=lambda c: (c != "sku", c))
        csv_path = path[:-len(".json")] + ".csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                w.writerow([r.get(c, "") for c in cols])
        LOG.info("   the changes as a table: %s (%d rows)", csv_path, len(rows))
    return len(todo)


def products_job(cur, conn, st, shop, args, commit):
    cur.execute("SELECT * FROM um_product_sent")
    sent = {r["sku"]: r for r in cur.fetchall()}
    cur_items = collect_products(cur, st, set(args.only_skus.split(",")) if args.only_skus else None)

    def key(x):
        return (x.get("barcode"), x.get("price"), x.get("cost_price"), x.get("item_group"), x.get("on_hand"))

    todo = []
    for sku, p in sorted(cur_items.items()):
        s = sent.get(sku)
        prev = None if not s else (s["barcode"], r2(s["std_price"]) if s["std_price"] is not None else None,
                                   r2(s["cost_price"]) if s["cost_price"] is not None else None, s["item_group"], s["on_hand"])
        if args.full or prev != key(p):
            todo.append({k: v for k, v in p.items() if v is not None})
    LOG.info("products: %d followed items, %d to send (%s)", len(cur_items), len(todo), "full" if args.full else "changed only")
    for p in todo[:5]:
        LOG.info("   e.g. %s", json.dumps(p, ensure_ascii=False))
    miss_cost = sum(1 for p in cur_items.values() if p["cost_price"] is None)
    miss_price = sum(1 for p in cur_items.values() if p["price"] is None)
    if miss_cost or miss_price:
        LOG.warning("   without cost: %d, without price: %d", miss_cost, miss_price)
    if args.shop_dry_run:
        return shop_dry_run_products(shop, todo, args.full) if todo else 0
    if not commit or not todo:
        return len(todo)
    def record(p):
        row = (p["sku"], p.get("barcode"), p.get("price"), p.get("cost_price"), p.get("item_group"), p.get("on_hand"))
        cur.execute("SELECT 1 FROM um_product_sent WHERE sku = %s", (p["sku"],))
        if cur.fetchone():
            cur.execute("UPDATE um_product_sent SET barcode=%s, std_price=%s, cost_price=%s, item_group=%s,"
                        " on_hand=%s, sent_at=now() WHERE sku=%s", row[1:] + (p["sku"],))
        else:
            cur.execute("INSERT INTO um_product_sent (sku, barcode, std_price, cost_price, item_group, on_hand)"
                        " VALUES (%s,%s,%s,%s,%s,%s)", row)

    chunks = [todo[i:i + 500] for i in range(0, len(todo), 500)]
    retry, not_applied = [], {}
    for chunk in chunks:
        ans = shop.products(chunk, bool(args.full) and len(chunks) == 1)
        refused = _skus_in(ans.get("unknown_skus")) | _skus_in(ans.get("skipped"))
        conflict = _skus_in(ans.get("conflicts"))
        retry += [p for p in chunk if p["sku"] in conflict]
        not_applied.update({s: "not in the shop / skipped" for s in refused})
        for p in chunk:
            if p["sku"] not in refused and p["sku"] not in conflict:      # only what the shop took is recorded as sent
                record(p)
    if retry:
        # a barcode taken from another item that changes in a LATER call of 500: send those items together, in one call
        ans = shop.products(retry, False)
        refused = _skus_in(ans.get("unknown_skus")) | _skus_in(ans.get("skipped")) | _skus_in(ans.get("conflicts"))
        for p in retry:
            if p["sku"] in refused:
                not_applied[p["sku"]] = "barcode conflict or not in the shop"
            else:
                record(p)
        LOG.info("barcode conflicts sent again together: %d items, %d resolved", len(retry), len(retry) - sum(1 for p in retry if p["sku"] in refused))
    conn.commit()
    if not_applied:
        LOG.warning("not applied by the shop, tried again on the next run (%d): %s", len(not_applied), ", ".join(sorted(not_applied)[:20]))
    return len(todo)


# ───────────────────────────── sell price rule (report first) ─────────────────────────────

def price_rows(cur, st):
    """What the scheduled in-ERP price rule would set on every followed 100007 item."""
    cur.execute("SELECT c.xitem, c.xalias, c.xdesc, c.xitemnew, c.xstdprice AS old, s.xsrate AS src_rate, s.xstdprice AS src_std"
                " FROM caitem c LEFT JOIN caitem s ON s.xitem = c.xalias"
                " AND s.zid = (CASE WHEN c.xitemnew ~ '^[0-9]+$' THEN c.xitemnew::int END)"
                " WHERE c.zid = %s AND coalesce(c.xalias,'') <> '' ORDER BY c.xitem", (st.zid,))
    items = cur.fetchall()
    cur.execute("SELECT sku, std_price FROM um_fixit_item WHERE found")
    fixit = {r["sku"]: r["std_price"] for r in cur.fetchall()}
    out = []
    for it in items:
        src = source_of(it)
        new, rule = None, ""
        if src == "100005":
            # Zepto: the sales rate (its MRP) kept in its own caitem.xsrate
            if it["src_rate"] and it["src_rate"] > 0:
                new, rule = d(it["src_rate"]), "Zepto caitem.xsrate (MRP)"
        elif src in ("100001", "100000"):
            # HMBR / GI: their standard price x the markup. Their xsrate is never read or written here: the ERP itself
            # uses it for its own calculations (the one-time shop-price load of 3 Nov 2026 broke that and was reverted)
            if it["src_std"] and it["src_std"] > 0:
                new = (d(it["src_std"]) * st.price_markup).quantize(Decimal("1"), rounding=ROUND_FLOOR)      # whole taka, rounded down
                rule = "source xstdprice x %s, rounded down" % st.price_markup
                if new <= 0:
                    new, rule = None, ""          # a source price under 1 taka would give a zero price: no price, never a guess
        elif src == FIXIT:
            if fixit.get(it["xalias"]):
                new, rule = d(fixit[it["xalias"]]), "Fixit std price (Gulshan)"
        if not src:
            continue
        old = d(it["old"]) if it["old"] is not None else None
        if it["xalias"] in st.skip_skus:
            out.append({"xitem": it["xitem"], "sku": it["xalias"], "name": it["xdesc"] or "", "source": src, "old": old, "new": None, "flag": "skipped_exception",
                        "rule": "[exceptions] skip_skus: the shop keeps its own price", "change_pct": None})
            continue
        flag = "no_source_price" if new is None else ("new_item" if not old else
                ("same" if abs(new - old) < Decimal("0.005") else ("raise" if new > old else "lower")))
        pct = None if not old or new is None else float((new - old) / old * 100)
        if flag in ("raise", "lower") and st.price_review_pct and abs(pct) > st.price_review_pct and it["xalias"] not in st.price_approve:
            flag = "review"             # a jump this big is usually a wrong source price (a placeholder 1, say): reported, never written unseen
        out.append({"xitem": it["xitem"], "sku": it["xalias"], "name": it["xdesc"] or "", "source": src, "old": old, "new": new, "flag": flag,
                    "rule": rule, "change_pct": pct})
    return out


def price_job(cur, conn, st, commit, xlsx=False):
    rows = price_rows(cur, st)
    by = {}
    for r in rows:
        by.setdefault((r["source"], r["flag"]), 0)
        by[(r["source"], r["flag"])] += 1
    for k in sorted(by):
        LOG.info("price rule  %-7s %-16s %5d", k[0], k[1], by[k])
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    path = os.path.join(HERE, "logs", "price_report_%s.csv" % dt.date.today().isoformat())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["xitem", "sku", "source", "old_price", "new_price", "change_pct", "flag", "rule"])
        for r in rows:
            w.writerow([r["xitem"], r["sku"], r["source"], r["old"], r["new"],
                        "" if r["change_pct"] is None else round(r["change_pct"], 1), r["flag"], r["rule"]])
    LOG.info("price report written: %s (%d items)", path, len(rows))
    if xlsx:
        try:
            import price_xlsx
            xl = price_xlsx.write(os.path.join(HERE, "logs", "price_before_after_%s.xlsx" % dt.datetime.now().strftime("%Y%m%d_%H%M")),
                                  rows, st.price_review_pct)
            LOG.info("before/after Excel written: %s", xl)
        except ImportError:
            LOG.warning("--xlsx needs openpyxl: run  venv\\Scripts\\pip install openpyxl  and try again")
    if not commit:
        return 0
    if not st.price_apply:
        LOG.warning("--commit ignored: [pricing] apply is false in the config. The report is all that was done.")
        return 0
    todo = [r for r in rows if r["new"] is not None and r["flag"] in ("raise", "lower", "new_item")]
    if todo:
        # the old prices are saved first, so "--job price --undo" can put them back
        cur.execute("SELECT to_regclass('public.um_price_backup')")
        if list(cur.fetchone().values())[0] is None:
            cur.execute("CREATE TABLE um_price_backup (id serial PRIMARY KEY, batch text, xitem text, sku text, old_price numeric,"
                        " new_price numeric, saved_at timestamp DEFAULT now())")
        batch = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        for r in todo:
            cur.execute("INSERT INTO um_price_backup (batch, xitem, sku, old_price, new_price) VALUES (%s, %s, %s, %s, %s)",
                        (batch, r["xitem"], r["sku"], r["old"], r["new"]))
            cur.execute("UPDATE caitem SET xstdprice = %s, zutime = now() WHERE zid = %s AND xitem = %s",
                        (r["new"], st.zid, r["xitem"]))
    conn.commit()
    LOG.info("prices written: %d%s", len(todo), (" (old prices saved in um_price_backup, batch %s; undo: --job price --undo --commit)" % batch) if todo else "")
    return len(todo)


def price_undo(cur, conn, st, commit):
    """Put back the prices the latest `--job price --commit` replaced (only where 100007 still holds what that run wrote)."""
    cur.execute("SELECT to_regclass('public.um_price_backup')")
    if list(cur.fetchone().values())[0] is None:
        LOG.info("no price backup (um_price_backup): the price job never wrote here")
        return 0
    cur.execute("SELECT max(batch) AS b FROM um_price_backup")
    batch = cur.fetchone()["b"]
    if not batch:
        LOG.info("the price backup is empty: nothing to put back")
        return 0
    cur.execute("SELECT b.id, b.xitem, b.sku, b.old_price, b.new_price, c.xstdprice AS now FROM um_price_backup b"
                " JOIN caitem c ON c.zid = %s AND c.xitem = b.xitem WHERE b.batch = %s ORDER BY b.id DESC", (st.zid, batch))
    rows = cur.fetchall()
    ok = [r for r in rows if r["now"] is not None and r["now"] == r["new_price"]]
    LOG.info("price backup batch %s: %d rows; can be put back (price still what the run wrote): %d; changed since by someone: %d",
             batch, len(rows), len(ok), len(rows) - len(ok))
    if not commit:
        LOG.info("DRY-RUN: nothing was put back. Add --commit to do it.")
        return 0
    n = 0
    for r in ok:
        cur.execute("UPDATE caitem SET xstdprice = %s, zutime = now() WHERE zid = %s AND xitem = %s AND xstdprice = %s",
                    (r["old_price"], st.zid, r["xitem"], r["new_price"]))
        if cur.rowcount == 1:
            cur.execute("DELETE FROM um_price_backup WHERE id = %s", (r["id"],))
            n += 1
    conn.commit()
    LOG.info("put back %d prices; their backup rows were removed", n)
    return n


# ───────────────────────────── jobs ─────────────────────────────

def setup_logging(verbose):
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    path = os.path.join(HERE, "logs", "erp_import_%s.log" % dt.date.today().isoformat())
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for h in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        root.addHandler(h)


def connect(cp):
    conn = psycopg2.connect(host=cfg_get(cp, "database", "host", "localhost"),
                            port=cfg_get(cp, "database", "port", "5432"),
                            dbname=cfg_get(cp, "database", "dbname", "da"),
                            user=cfg_get(cp, "database", "user", "postgres"),
                            password=cfg_get(cp, "database", "password", ""))
    conn.set_client_encoding("UTF8")
    return conn


def fixed_code_warnings(cur, st):
    cur.execute("SELECT count(*) AS n FROM prmst WHERE zid = %s AND xemp = %s", (st.zid, st.salesman))
    if cur.fetchone()["n"] == 0:
        LOG.warning("employee %s is not in prmst for zid %s", st.salesman, st.zid)
    cur.execute("SELECT count(*) AS n FROM cacus WHERE zid = %s AND xcus = %s", (st.zid, st.customer))
    if cur.fetchone()["n"] == 0:
        LOG.warning("customer %s is not in cacus for zid %s", st.customer, st.zid)


def locked_session(args, job):
    """Config, settings, logging, database connection and the one-run-at-a-time lock."""
    cp = load_config(args.config)
    st = Settings(cp)
    setup_logging(args.verbose)
    LOG.info("=== erp_import %s start | mode=%s | zid=%s ===", job, "COMMIT" if args.commit else "DRY-RUN (nothing is kept)", st.zid)
    conn = connect(cp)
    lockconn = connect(cp)
    lc = lockconn.cursor()
    lc.execute("SELECT pg_try_advisory_lock(%s)", (LOCK_KEY,))
    if not lc.fetchone()[0]:
        LOG.info("another run is active; exiting")
        conn.close(); lockconn.close()
        return None
    return cp, st, conn, lockconn, lc


def release(conn, lockconn, lc):
    try:
        lc.execute("SELECT pg_advisory_unlock(%s)", (LOCK_KEY,))
    except Exception:
        pass
    conn.close()
    lockconn.close()


def run(args):
    sess = locked_session(args, "run")
    if not sess:
        return 0
    cp, st, conn, lockconn, lc = sess
    commit = args.commit
    try:
        cur = conn.cursor()
        cur.execute("SHOW server_encoding")
        enc = cur.fetchone()[0]
        LOG.info("database server_encoding=%s%s", enc, "" if enc == "UTF8" else "  (!! Bangla text may not store)")
        ensure_um_tables(cur)
        if commit:
            conn.commit()
        fixed_code_warnings(dict_cur(conn), st)

        shop = Shop(st)
        h = shop.health()
        env = h.get("environment")
        LOG.info("shop health ok, environment=%s", env)
        if env == "staging" and not st.allow_staging:
            LOG.error("feed is STAGING but allow_staging_feed is false: refusing to run")
            return 2

        cur = dict_cur(conn)
        cur.execute("SELECT * FROM um_sync_state WHERE id = 1")
        state = cur.fetchone()
        cursor = args.after if args.after is not None else state["last_seq"]
        LOG.info("cursor=%s", cursor)

        results, total = [], 0
        while total < args.max_events:
            page = shop.feed(cursor, 100, args.retry)
            events = page.get("events") or []
            for evt in events:
                results.append(process_event(conn, st, evt, commit))
                cursor = max(cursor, evt["seq"])
                total += 1
            if not events or not page.get("has_more"):
                break
        LOG.info("events processed: %d (failed: %d)", total, sum(1 for r in results if r["status"] == "failed"))

        statuses, stock_items, stock_full = [], {}, False
        if not args.no_status:
            cur.execute("SELECT last_full_status_date FROM um_sync_state WHERE id = 1")
            lfs = cur.fetchone()["last_full_status_date"]
            statuses = collect_statuses(conn, st, resend_all=(lfs != dt.date.today()))
            LOG.info("order statuses to report: %d", len(statuses))
            for s in statuses:
                LOG.info("   %s (%s) -> %s %s", s["external_order_id"], s["display_id"], s["erp_status"], json.dumps(s["refs"]))
        if st.push_stock and not args.no_stock:
            cur.execute("SELECT last_full_stock_date FROM um_sync_state WHERE id = 1")
            lfd = cur.fetchone()["last_full_stock_date"]
            stock_full = (lfd != dt.date.today()) and dt.datetime.now().hour >= st.full_after_hour
            stock_items, _ = collect_stock(conn, st, stock_full)
            LOG.info("stock to push: %d SKUs (%s)", len(stock_items), "full resend" if stock_full else "changed only")

        if not commit:
            LOG.info("DRY-RUN: rolling back; nothing posted to the shop, cursor not advanced")
            conn.rollback()
            return 0

        if results:
            shop.ack([{k: v for k, v in r.items() if v is not None} for r in results])
            LOG.info("acknowledged %d events", len(results))
        if statuses:
            shop.order_status(statuses)
            mark_statuses(conn, statuses)
            conn.commit()
            LOG.info("order status posted for %d orders", len(statuses))
        if stock_items:
            n = push_stock(conn, shop, stock_items, stock_full)
            conn.commit()
            LOG.info("stock posted for %d SKUs", n)
        c2 = conn.cursor()
        if stock_full and stock_items:
            c2.execute("UPDATE um_sync_state SET last_full_stock_date = current_date WHERE id = 1")
        if not args.no_status:
            c2.execute("UPDATE um_sync_state SET last_full_status_date = current_date WHERE id = 1")
        conn.commit()
        LOG.info("=== erp_import run done ===")
        return 0
    except ShopUnavailable as e:
        conn.rollback()
        LOG.warning("shop not reachable / busy, will resume next run: %s", e)
        return 0
    except ShopRejected as e:
        conn.rollback()
        LOG.error("shop rejected the call (check key/secret/clock/body): %s", e)
        return 1
    finally:
        release(conn, lockconn, lc)


def job_po(args):
    sess = locked_session(args, "po")
    if not sess:
        return 0
    cp, st, conn, lockconn, lc = sess
    try:
        cur = dict_cur(conn)
        ensure_um_tables(cur)
        cur.execute("SELECT q.external_order_id FROM um_po_queue q JOIN um_order_map m ON m.external_order_id = q.external_order_id"
                    " WHERE q.done_at IS NULL AND m.state = 'active' ORDER BY m.created_at, q.external_order_id")
        ids = [r["external_order_id"] for r in cur.fetchall()]
        LOG.info("orders waiting for purchase orders: %d", len(ids))
        warnings = []
        made = make_pos_for_orders(cur, st, ids, warnings) if ids else []
        for w in warnings:
            LOG.info("   warning %s", json.dumps(w, ensure_ascii=False))
        LOG.info("purchase orders made: %s", ", ".join(made) or "none")
        if args.commit:
            conn.commit()
        else:
            LOG.info("DRY-RUN: rolled back")
            conn.rollback()
        return 0
    finally:
        release(conn, lockconn, lc)


def job_products(args):
    sess = locked_session(args, "products")
    if not sess:
        return 0
    cp, st, conn, lockconn, lc = sess
    try:
        cur = dict_cur(conn)
        ensure_um_tables(cur)
        shop = Shop(st)
        if args.shop_dry_run:
            args.full = True                    # ask about every followed item, whatever was sent before
            h = shop.health()
            LOG.info("shop health ok, environment=%s, sync_enabled=%s", h.get("environment"), h.get("sync_enabled"))
        elif args.commit:
            shop.health()
        products_job(cur, conn, st, shop, args, args.commit)
        if not args.commit:
            LOG.info("DRY-RUN: nothing %s", "changed in the shop or recorded here (--shop-dry-run)" if args.shop_dry_run else "posted to the shop")
            conn.rollback()
        return 0
    except ShopUnavailable as e:
        conn.rollback()
        LOG.warning("shop not reachable or has no /products yet: %s", e)
        return 0
    except ShopRejected as e:
        conn.rollback()
        LOG.error("shop rejected the call: %s", e)
        return 1
    finally:
        release(conn, lockconn, lc)


def job_preflight(args):
    """Read-only check of the HMBR source companies, the Fixit server, the 100007 items and the live shop (see preflight.py)."""
    sess = locked_session(args, "preflight")
    if not sess:
        return 0
    cp, st, conn, lockconn, lc = sess
    try:
        cur = dict_cur(conn)
        ensure_um_tables(cur)
        import preflight
        return preflight.run(sys.modules[__name__], cur, st, cp, Shop(st), args)
    finally:
        conn.rollback()
        release(conn, lockconn, lc)


def job_price(args):
    sess = locked_session(args, "price")
    if not sess:
        return 0
    cp, st, conn, lockconn, lc = sess
    try:
        cur = dict_cur(conn)
        ensure_um_tables(cur)
        if args.undo:
            price_undo(cur, conn, st, args.commit)
        else:
            price_job(cur, conn, st, args.commit, args.xlsx)
        conn.rollback()
        return 0
    finally:
        release(conn, lockconn, lc)


def job_check(args):
    """Read-only health check. Exit code 1 if anything essential is missing."""
    cp = load_config(args.config)
    st = Settings(cp)
    setup_logging(args.verbose)
    bad = []

    def line(ok, label, detail=""):
        LOG.info("%-5s %s %s", "OK" if ok is True else ("WARN" if ok is None else "FAIL"), label, detail)
        if ok is False:
            bad.append(label)

    if st.skip_skus:
        line(None, "exceptions: not pushed to the shop (stock, price, barcode, cost)", ", ".join(sorted(st.skip_skus)))
    conn = connect(cp)
    cur = dict_cur(conn)
    cur.execute("SHOW server_version"); ver = cur.fetchone()["server_version"]
    cur.execute("SHOW server_encoding"); enc = cur.fetchone()["server_encoding"]
    line(True, "database", "PostgreSQL %s, encoding %s" % (ver, enc))
    line(enc == "UTF8" or None, "encoding is UTF8 (Bangla text)")
    cur.execute("SELECT xorg FROM cacus WHERE zid = %s AND xcus = %s", (st.zid, st.customer))
    r = cur.fetchone()
    line(bool(r), "customer %s exists in cacus" % st.customer, r["xorg"] if r else "")
    cur.execute("SELECT xname FROM prmst WHERE zid = %s AND xemp = %s", (st.zid, st.salesman))
    r = cur.fetchone()
    line(bool(r), "employee %s exists in prmst" % st.salesman, r["xname"] if r else "")
    for src, sup in sorted(st.suppliers.items()):
        cur.execute("SELECT xorg FROM casup WHERE zid = %s AND xsup = %s", (st.zid, sup))
        r = cur.fetchone()
        line(bool(r), "supplier %s (%s) exists in casup" % (sup, src), r["xorg"] if r else "")
    cur.execute("SELECT count(*) AS n FROM caitem WHERE zid = %s AND xwh = %s", (st.zid, st.warehouse))
    line(cur.fetchone()["n"] > 0 or None, "warehouse '%s' used by caitem" % st.warehouse)
    cur.execute("SELECT xitemnew, count(*) AS n FROM caitem WHERE zid = %s GROUP BY 1 ORDER BY 1", (st.zid,))
    LOG.info("      caitem by source: %s", ", ".join("%s=%d" % (r["xitemnew"] or "(blank)", r["n"]) for r in cur.fetchall()))
    cur.execute("SELECT count(*) AS n FROM caitem WHERE zid = %s AND xitemnew !~ '^[0-9]*$' AND lower(xitemnew) <> 'fixit'", (st.zid,))
    line(cur.fetchone()["n"] == 0, "no unknown non-numeric xitemnew values")
    cur.execute("SELECT count(*) AS n FROM information_schema.columns WHERE table_name = 'opmob' AND column_name IN ('xtra1','xtra2','xtra3','xtra4','xtra5')")
    line(cur.fetchone()["n"] == 5, "opmob has xtra1..xtra5")
    cur.execute("SELECT data_type FROM information_schema.columns WHERE table_name = 'opmob' AND column_name = 'xtra1'")
    r = cur.fetchone()
    LOG.info("      opmob.xtra1 type: %s (discount %% goes here as a whole number)", r["data_type"] if r else "?")
    # the um tables: create inside a transaction that is rolled back
    made = ensure_um_tables(cur)
    LOG.info("      tables the first run would create: %s", ", ".join(made) or "none, all exist")
    conn.rollback()
    conn.close()
    if st.base_url:
        try:
            h = Shop(st).health()
            line(True, "shop /health", "environment=%s" % h.get("environment"))
        except (ShopUnavailable, ShopRejected) as e:
            line(None, "shop /health", str(e)[:120])
    else:
        line(None, "shop base_url not set")
    fx = cfg_get(cp, "fixit", "dbname")
    if fx:
        try:
            fconn = psycopg2.connect(host=cfg_get(cp, "fixit", "host", "localhost"), port=cfg_get(cp, "fixit", "port", "5432"),
                                     dbname=fx, user=cfg_get(cp, "fixit", "user", "postgres"),
                                     password=cfg_get(cp, "fixit", "password", ""))
            fc = fconn.cursor(); fc.execute("SELECT count(*) FROM final_items_view")
            line(True, "fixit database", "final_items_view rows: %d" % fc.fetchone()[0])
            fconn.close()
        except psycopg2.Error as e:
            line(False, "fixit database", str(e).strip()[:120])
    LOG.info("%s", "ALL ESSENTIAL CHECKS PASSED" if not bad else "FAILED: " + "; ".join(bad))
    return 1 if bad else 0


def purge_test(args):
    """DEV ONLY: remove everything created from the fake feed's TEST ids and reset the cursor."""
    cp = load_config(args.config)
    st = Settings(cp)
    setup_logging(args.verbose)
    if not st.allow_purge:
        sys.exit("allow_purge is false in the config: refusing (never enable this on the live database)")
    conn = connect(cp)
    cur = dict_cur(conn)
    ensure_um_tables(cur)
    cur.execute("SELECT external_order_id, invoiceno, po_numbers FROM um_order_map WHERE external_order_id LIKE 'order_TEST%%'")
    for o in cur.fetchall():
        cur.execute("DELETE FROM opmob WHERE zid = %s AND invoiceno = %s", (st.zid, o["invoiceno"]))
        cur.execute("DELETE FROM um_attention WHERE order_ref = %s", (o["invoiceno"],))
        for p in [x for x in (o["po_numbers"] or "").split(",") if x]:
            cur.execute("DELETE FROM poodt WHERE zid = %s AND xpornum = %s", (st.zid, p))
            cur.execute("DELETE FROM poord WHERE zid = %s AND xpornum = %s", (st.zid, p))
    cur.execute("DELETE FROM um_po_queue WHERE external_order_id LIKE 'order_TEST%%'")
    cur.execute("DELETE FROM um_order_map WHERE external_order_id LIKE 'order_TEST%%'")
    cur.execute("SELECT xitem FROM caitem WHERE zid = %s AND (xalias LIKE 'TESTSKU%%' OR xalias = '18-9998')", (st.zid,))   # 18-9998 = the stand-in's Fixit test item
    for r in cur.fetchall():
        cur.execute("DELETE FROM xcodes WHERE zid = %s AND xtype = 'Price Category' AND xcode = %s", (st.zid, r["xitem"]))
        cur.execute("DELETE FROM caitem WHERE zid = %s AND xitem = %s", (st.zid, r["xitem"]))
    cur.execute("DELETE FROM um_fixit_item WHERE sku = '18-9998'")
    cur.execute("DELETE FROM um_processed_events")
    cur.execute("DELETE FROM um_stock_sent")
    cur.execute("DELETE FROM um_product_sent")
    cur.execute("UPDATE um_sync_state SET last_seq = 0, last_full_stock_date = NULL, last_full_status_date = NULL")
    conn.commit()
    conn.close()
    print("test data purged and cursor reset")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--job", choices=["run", "po", "products", "price", "check", "preflight"], default="run")
    ap.add_argument("--config", default=os.path.join(HERE, "erp_import.ini"))
    ap.add_argument("--commit", action="store_true", help="write to the database and post to the shop (default: dry run)")
    ap.add_argument("--after", type=int, default=None, help="run: start from this feed seq instead of the saved cursor")
    ap.add_argument("--max-events", type=int, default=1000)
    ap.add_argument("--retry", action="store_true", help="run: also ask the feed for retryable failed events")
    ap.add_argument("--no-status", action="store_true")
    ap.add_argument("--no-stock", action="store_true")
    ap.add_argument("--only-skus", default="", help="products: comma-separated SKUs to limit the push to")
    ap.add_argument("--full", action="store_true", help="products: send every followed item, not just the changes")
    ap.add_argument("--xlsx", action="store_true", help="price: also write logs/price_before_after_<time>.xlsx (what is today / what the rule leaves); needs openpyxl")
    ap.add_argument("--undo", action="store_true", help="price: put back the prices the latest `--job price --commit` replaced (add --commit to do it)")
    ap.add_argument("--shop-dry-run", action="store_true",
                    help="products: send every followed item with ?dry_run=1; the shop answers what it WOULD change and writes nothing")
    ap.add_argument("--purge-test", action="store_true", help="DEV ONLY: delete fake-feed test data and reset cursor")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if args.shop_dry_run and (args.commit or args.job != "products"):
        ap.error("--shop-dry-run is only for --job products, and cannot be combined with --commit")
    if (args.xlsx or args.undo) and args.job != "price":
        ap.error("--xlsx and --undo are only for --job price")
    if args.purge_test:
        sys.exit(purge_test(args))
    jobs = {"run": run, "po": job_po, "products": job_products, "price": job_price, "check": job_check, "preflight": job_preflight}
    sys.exit(jobs[args.job](args))


if __name__ == "__main__":
    main()
