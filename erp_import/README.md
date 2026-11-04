# ERP importer: online shop (Medusa) <-> company 100007

Written 2026-11-02 from the owner's decisions. Everything is in `docs/`: the agreed column map is `docs/CYCLE_MAP.md` (short) and `docs/CYCLE_COLUMNS.md` (every column),
`docs/CYCLE_VISUAL.html` is the picture, `docs/SHOP_CONTRACT.md` is what the shop session must provide, `docs/DEPLOY_HMBR_SERVER.md` is the server step-by-step.
Where this file and `docs/ERP_INTEGRATION_SPEC.md` (the shop's original v2.2 spec) differ, this file wins.

## Layout
```
erp_import/
  erp_import.py  fixit_pull.py  erp_service.py   the three scripts that run on the HMBR server
  erp_import.ini.example                         config template -> erp_import.ini (git-ignored: keys, passwords)
  onetime/    RETIRED load of the shop's prices into caitem.xsrate of 100001/100000 (it broke the ERP's own xsrate use; undone), and fix_xsrate_zepto_gi.py, the repair of 3 Nov 2026 (dry run by default, backup, --undo)
  sql/        prep SQL: delete two stray items; optional least-privilege login; the Fixit server's read-only login
  docs/       cycle map, column reference, picture, shop contract, server step-by-step, the shop's original spec
  tests/      fake_feed.py (stand-in for the shop's /hooks/erp, also the contract) and run_local_test.py (71 checks on the LOCAL database)
  dev_tools/  developer only, never deployed: medusa_snapshot.py (read-only copy of the live catalogue), make_xsrate_json.py (builds the retired load's JSON)
  logs/       run logs, price reports, the snapshot (git-ignored)
```

## Jobs (all are DRY RUNS unless `--commit`)
| Job | Schedule | What it does |
|---|---|---|
| `python erp_import.py` (`--job run`) | every 5 min | shop feed -> ERP: items -> `caitem`/`xcodes`, orders -> `opmob`; changed stock and order status back to the shop |
| `python erp_import.py --job po` | once a day, fixed time | open purchase orders `poord`/`poodt` (`1-Open`), one per order per source business, rate = source `xstdprice` less its `opspprc` tier for the day's total (Fixit: Central's sales rate) |
| `python erp_import.py --job products` | once a day | barcode, price, cost, item group, stock -> shop `POST /products` (`--only-skus a,b`, `--full`) |
| `python erp_import.py --job price` | once a day (05:45, after the Fixit pull) | the in-ERP sell-price rule: 100007 `xstdprice` = Zepto's own `caitem.xsrate` (the MRP), or HMBR/GI `caitem.xstdprice` x `[pricing] markup` (1.40), or the Fixit Gulshan price. Always writes a REPORT (`logs/price_report_<date>.csv`); writes prices only when `[pricing] apply = true` AND `--commit`; a move over `max_change_pct` (25) is flagged `review` and not written |
| `python erp_import.py --job check` | any time | read-only health check: database, fixed codes, suppliers, shop `/health`, Fixit database |
| `python fixit_pull.py` | once a day | Fixit stock, selling price (Gulshan) and cost (Central's sales rate) into `um_fixit_item` |

`--purge-test` (dev only, needs `allow_purge = true`) removes what the stand-in's TEST events created.

## Fixed values (config `[erp]`)
Customer `CUS-000073` UtilityMart, employee `MR--000007`, terminal `UTM`, warehouse "Sales Warehouse Online Shop", project "HMBR Online Shop",
suppliers `SUP-000001` (100001), `-02` (100000), `-03` (100005), `-05` (Fixit). Shopper details are never stored in the ERP.

## What lands where (short; full detail in docs/CYCLE_MAP.md)
* **Order** -> `opmob`, one row per line. `invoiceno` = `UM-1000001`, `invoicesl` = 1000001; `xtra1` discount % (whole number, exact in `xtra5`);
  `xtra2` net total after discount; `xtra3` payment trail (`COD`, `ONLINE`, then the SSLCommerz transaction id); `xtra4` Pathao consignment id;
  `xtra5` JSON (delivery charge, shop order id, shipping method, note, exact discount). The ERP copies `invoiceno` into `opord.xref` / `opdor.xref`.
* **Cancel** while still `New`: `opmob` rows and the still-open POs are deleted. Afterwards: a staff to-do in `um_attention`.
* **Items**: matched SKU first (`caitem.xalias`), the ERP's SKU/barcode pairing wins. The shop is master for title, options, handle, VG code. The ERP is master for price.
* **Fixit** items: `xitemnew = 'Fixit'`, code `04-nnnnn`, `xgitem = 'Fix iT Item'`, SKU = Fixit code (`18-018`).
* **Exceptions**: SKUs in `[exceptions] skip_skus` (now CF000003 Korean Putty: ERP unit kg, shop sells 100 g packs) are never pushed to the shop: no stock, price, barcode or cost.
* **Returns, refunds, purchase returns**: manual (acknowledged `failed: return_manual` + a `um_attention` row).

## Tables the importer creates
`um_sync_state`, `um_processed_events`, `um_order_map` (shop order -> invoice -> PO numbers), `um_po_queue`, `um_stock_sent`, `um_product_sent`, `um_fixit_item`, `um_attention` (staff to-do list).

## Run it locally
```
python tests/fake_feed.py &           # the stand-in shop
python erp_import.py --job check
python erp_import.py                  # dry run: everything, rolled back
python tests/run_local_test.py        # the whole cycle, with cleanup
```

## On the live server
See `docs/DEPLOY_HMBR_SERVER.md` (HMBR server with the NSSM service, Fixit server, shop). In short: run the prep SQL in `sql/`, Python 3.6+ venv with `psycopg2-binary`, `erp_import.ini` (real keys; `allow_purge = false`, `allow_staging_feed = false`; never the Medusa admin key), `--job check`, a dry run, then the by-hand first live day, then switch the NSSM service from `--dry-run` to live.
Database rights if you use a dedicated login: `sql/prep_02_importer_login.sql` (tested locally). `SHOW server_encoding;` must be UTF8 (Bangla text). Written for PostgreSQL 9.4: no `ON CONFLICT`, no `IF NOT EXISTS`.

## Known limits
* Tested on PostgreSQL 14 only (71 checks). The live server is 9.4: repeat `--job check` and a dry run there first.
* `xtra1` is an integer column here, so the discount percent is whole; check its type on live with `--job check`.
* `xitemnew = 'Fixit'` is text in a column that otherwise holds company numbers: any ERP report that casts `xitemnew` to a number would fail on those rows.
* The shop's `/hooks/erp` (feed, ack, order-status, stock) and the new `/products` do not exist yet (2026-11-02): the importer treats a missing endpoint as "shop not reachable" and retries next run.
* The master-list rule for NEW Fixit items (stock above 5, sales not 0, prefix the shop already has) is a reminder, not built.
