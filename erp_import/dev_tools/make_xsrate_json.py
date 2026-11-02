#!/usr/bin/env python3
"""DEVELOPER TOOL (runs on the Mac, never on the ERP server): turn the shop's CURRENT prices into the JSON that
onetime/set_xsrate_from_json.py loads into caitem.xsrate of companies 100001 (HMBR) and 100000 (GI).

Input : logs/medusa_snapshot.json (take a fresh one first: python dev_tools/medusa_snapshot.py; read-only GETs)
        + the ERP pairing in 100007 (caitem.xalias = shop SKU, caitem.xitemnew = the source company)
Output: onetime/xsrate_shop_prices_<date>.json   one entry per shop variant that is an HMBR or GI item:
        {"zid": 100001, "xitem": "0718", "price": 80, "name": "<shop title>"}
        xitem is the shop SKU, which is the source company's caitem.xitem.

Rules (owner decisions, 2026-11-02):
  * price = the shop's current BDT price;
  * where the shop price is BELOW the source company's std price (xstdprice), price = floor(xstdprice x 1.35) instead, so the ERP
    corrects the shop when the product push runs. The entry then also carries "shop_price" and "basis";
  * --exclude SKU,SKU leaves items out (listed under "excluded" in the file). CF000003 Korean Putty: the ERP unit is 1 kg, the shop sells
    100 g packs, so the two prices are not comparable until the shop side has a unit exception.

    python dev_tools/make_xsrate_json.py --exclude CF000003 [--snapshot logs/medusa_snapshot.json] [--out onetime/xsrate_shop_prices_YYYYMMDD.json]
"""
import argparse
import collections
import datetime as dt
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import erp_import as E  # noqa: E402  (the importer's own config reader and connection)

SOURCES = (100001, 100000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=os.path.join(ROOT, "logs", "medusa_snapshot.json"))
    ap.add_argument("--config", default=os.path.join(ROOT, "erp_import.ini"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--exclude", default="", help="comma-separated SKUs to leave out")
    ap.add_argument("--below-std-factor", type=float, default=1.35, help="price used when the shop price is below the std price: std x this, rounded down")
    args = ap.parse_args()

    snap = json.load(open(args.snapshot, encoding="utf-8"))
    taken = dt.datetime.fromtimestamp(os.path.getmtime(args.snapshot))
    out_path = args.out or os.path.join(ROOT, "onetime", "xsrate_shop_prices_%s.json" % taken.strftime("%Y%m%d"))

    cp = E.load_config(args.config)
    conn = E.connect(cp)
    cur = E.dict_cur(conn)
    cur.execute("SELECT trim(xalias) AS sku, xitem AS barcode, xitemnew FROM caitem WHERE zid = 100007 AND coalesce(xalias,'') <> ''")
    pair = {r["sku"]: r for r in cur.fetchall()}

    exclude = {x.strip() for x in args.exclude.split(",") if x.strip()}
    items, skipped, excluded = [], collections.defaultdict(list), []
    for p in snap:
        for v in p.get("variants") or []:
            sku = (v.get("sku") or "").strip()
            if not sku:
                continue
            pr = [x["amount"] for x in v.get("prices") or [] if x["currency_code"] == "bdt"]
            p100007 = pair.get(sku)
            if not p100007 or str(p100007["xitemnew"]).strip() not in [str(z) for z in SOURCES]:
                continue                                    # Zepto, Fixit, or not an ERP item: out of scope here
            zid = int(p100007["xitemnew"])
            if not pr or pr[0] is None or pr[0] <= 0:
                skipped["shop price missing or 0"].append("%s/%s %s" % (zid, sku, p["title"]))
                continue
            if sku in exclude:
                excluded.append({"zid": zid, "xitem": sku, "shop_price": pr[0], "name": p["title"]})
                continue
            cur.execute("SELECT xstdprice FROM caitem WHERE zid = %s AND xitem = %s", (zid, sku))
            src = cur.fetchone()
            if not src:
                skipped["SKU is not an item of the source company"].append("%s/%s" % (zid, sku))
                continue
            name = p["title"] if (v.get("title") or "").lower() in ("default", "default variant", "") else "%s - %s" % (p["title"], v["title"])
            shop = int(pr[0]) if float(pr[0]).is_integer() else float(pr[0])
            entry = {"zid": zid, "xitem": sku, "price": shop, "name": name}
            std = src["xstdprice"]
            if std and shop < std:
                entry["price"] = int(math.floor(float(std) * args.below_std_factor))
                entry["shop_price"] = shop
                entry["basis"] = "shop price %s is below the std price %s: %s x std price, rounded down" % (shop, std, args.below_std_factor)
            items.append(entry)

    items.sort(key=lambda r: (r["zid"], r["xitem"]))
    doc = {"taken_at": taken.strftime("%Y-%m-%d %H:%M"),
           "what": "current BDT sales price in the online shop (medusa.utilitymart.shop) of every variant that is a 100001 (HMBR) or 100000 (GI) item",
           "use": "onetime/set_xsrate_from_json.py writes price into caitem.xsrate where zid and xitem match; xitem is the shop SKU",
           "rules": ["price = the shop's current BDT price",
                     "if the shop price is below the std price (xstdprice) of the source item, price = floor(xstdprice x %s) and the entry says so (basis)" % args.below_std_factor],
           "excluded": excluded,
           "items": items}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    by = collections.Counter(i["zid"] for i in items)
    print("wrote %s" % out_path)
    print("items: %d  (100001: %d, 100000: %d)   shop snapshot taken %s" % (len(items), by[100001], by[100000], doc["taken_at"]))
    changed = [i for i in items if "basis" in i]
    print("price taken as %s x std price because the shop price was below it: %d" % (args.below_std_factor, len(changed)))
    for i in changed:
        print("    %s/%s  %s: shop %s -> %s" % (i["zid"], i["xitem"], i["name"][:40], i["shop_price"], i["price"]))
    for e in excluded:
        print("left out on request: %s/%s %s (shop price %s)" % (e["zid"], e["xitem"], e["name"], e["shop_price"]))
    for why, rows in skipped.items():
        print("left out, %s: %d" % (why, len(rows)))
        for r in rows:
            print("    " + r)
    conn.close()


if __name__ == "__main__":
    main()
