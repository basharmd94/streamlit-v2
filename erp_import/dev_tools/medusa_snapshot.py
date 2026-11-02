#!/usr/bin/env python3
"""DEVELOPER TOOL, never deployed to the ERP server: read-only snapshot of the live shop's catalogue.

Uses the Medusa ADMIN key (GET requests only) from a git-ignored file and writes
erp_import/logs/medusa_snapshot.json (logs/ is git-ignored). The stand-in feed (tests/fake_feed.py --snapshot)
turns that file into realistic item events so the importer can be dry-run against the real catalogue.

    python dev_tools/medusa_snapshot.py [--env ../UtilityMartERPIntegration/.env.live]

The key is read from the file and never printed. See UtilityMartERPIntegration/MEDUSA_ACCESS_FOR_ERP_SESSION.md.
"""
import argparse
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://medusa.utilitymart.shop"
FIELDS = ("id,title,handle,status,metadata,type.value,collection.title,categories.name,variants.id,variants.title,"
          "variants.sku,variants.barcode,variants.metadata,variants.prices.amount,variants.prices.currency_code")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=os.path.join(HERE, "..", "..", "UtilityMartERPIntegration", ".env.live"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "logs", "medusa_snapshot.json"))
    args = ap.parse_args()
    key = None
    for line in open(args.env):
        m = re.match(r"^LIVE_MEDUSA_ADMIN_API_KEY=(.*)$", line.rstrip("\n"))
        if m:
            key = m[1].strip().strip('"').strip("'")
    auth = "Basic " + base64.b64encode(("%s:" % key).encode()).decode()

    def get(path):
        req = urllib.request.Request(BASE + path, headers={"Authorization": auth})
        return json.load(urllib.request.urlopen(req, timeout=60))

    out, off = [], 0
    while True:
        r = get("/admin/products?limit=100&offset=%d&fields=%s" % (off, FIELDS))
        out += r["products"]
        off += 100
        if off >= r["count"]:
            break
        time.sleep(0.25)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print("saved %d products to %s" % (len(out), os.path.abspath(args.out)))


if __name__ == "__main__":
    main()
