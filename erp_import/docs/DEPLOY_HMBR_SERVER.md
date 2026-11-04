# Putting the online-shop importer in place: HMBR server, Fixit server, the shop

Written 2026-11-02. Windows commands (cmd / PowerShell), `<repo>` = the folder where the Streamlit repo lives on the HMBR server.
Everything below was run locally against PostgreSQL 14 and the stand-in shop (`tests/run_local_test.py`: 71 checks pass). **Nothing has run on PostgreSQL 9.4 or against the real shop yet**, so Part A ends with checks before anything is switched on.

## How the three places talk

```
   Medusa shop (live)                    HMBR server                                  Fixit server
   medusa.utilitymart.shop               PostgreSQL 9.4, database da                  PostgreSQL, final_items_view
   /hooks/erp  (feed, ack,     <-- HTTPS, the importer calls OUT --  erp_service.py (NSSM)  -- reads (port 5432) -->
   order-status, stock,                 erp_import.py / fixit_pull.py
   products)  <- built by the shop session
```
**The HMBR server opens no port and needs no Caddy route**: the importer only calls out. The `/hooks/erp` endpoints are built on the shop side (`SHOP_CONTRACT.md`).
The NSSM service on the HMBR server is only a clock that starts the jobs (it replaces four Task Scheduler entries).

---

## Part A: HMBR server

### A1. Server facts
Confirmed: Python 3.9.6 (64-bit), PostgreSQL 9.4, backup available. Still to check, 1 minute:
```bat
psql -U postgres -d da -c "SHOW server_encoding;"
nssm version
```
`server_encoding` must be `UTF8` (Bangla item names).

### A2. Backup
Done. For the record, the steps below touch `caitem` and `xcodes` (prep), then `caitem`, `xcodes`, `opmob`, `poord`, `poodt` plus the new `um_*` tables (importer).

### A3. Code and Python environment (same pattern as the webhook service)
After `erp_import/` is committed and pushed (say when): `cd <repo>` and `git pull origin main`. Secrets are not in git (`erp_import.ini`, `logs\`, `venv\` are ignored). Without git, copy the whole `erp_import` folder.
```bat
cd <repo>\erp_import
py -3.9 -m venv venv
venv\Scripts\python -m pip install psycopg2-binary
mkdir logs
```
Nothing else is needed (the rest is the Python standard library).

### A4. Minimal config for the prep steps
```bat
copy erp_import.ini.example erp_import.ini
notepad erp_import.ini
```
For now only `[database]` matters (host, port, `dbname = da`, user, password). The rest is filled in A8.

### A5. Prep 1: delete the two stray items
`psql -U postgres -d da -f erp_import\sql\prep_01_delete_two_stray_items.sql` (or paste it into pgAdmin). Step 1 of the file shows 2 rows, the last check 0 rows. It stops with an error, deleting nothing, if any stock/order/PO row uses them. Do this before the first importer run: the importer numbers new Fixit items from `04-00001`.

### A6. RETIRED (2026-11-03): do NOT load shop prices into `caitem.xsrate`
**This step was done on 3 Nov and undone the same day: the ERP uses `xsrate` of 100001/100000 for its own calculations and the load broke them.** The sell-price rule now uses HMBR/GI `xstdprice` x 1.40 and never touches their `xsrate`. The script refuses `--commit` without a long flag. The text below stays only as a record.

`xsrate` in 100001 (HMBR) and 100000 (GI) was set to the price the shop charges, so ERP and shop start identical. The input is `onetime\xsrate_shop_prices_<date>.json`: one entry per shop item that is an HMBR or GI item, `{"zid": 100001, "xitem": "0718", "price": 80, "name": "..."}` (`xitem` is the shop SKU = the source company's `caitem.xitem`). Only those rows are touched. Two decisions are already in the file:
* **Shop price below the ERP standard price** (4 aluminium ladders, SF000004/5/6/24): the price is `floor(1.35 x xstdprice)` (4252, 5103, 5953, 7290) instead of the shop's, and the entry says so (`shop_price`, `basis`). The first product push then raises them in the shop.
* **CF000003 Korean Putty is left out** (the ERP unit is 1 kg, the shop sells 100 g packs, so the prices are not comparable). It is listed under `excluded` in the JSON and in `[exceptions] skip_skus` of the config, so the ERP never pushes its price, stock or cost to the shop. Remove it from the list when the shop has a unit exception.

1. **Is the JSON fresh?** Made on the Mac from a read-only snapshot of the live shop. If the shop's prices have been edited since (check the `taken_at` line), refresh first: `python dev_tools/medusa_snapshot.py` then `python dev_tools/make_xsrate_json.py --exclude CF000003`, and take the new file.
2. **Dry run** (reads only):
   ```bat
   venv\Scripts\python onetime\set_xsrate_from_json.py
   ```
   It prints the database it is connected to, counts per company (`set_empty` / `overwrite` / `same` / `missing`), the items whose `xsrate` already held a different value (these are replaced), and "to look at" lines. The full plan is in `logs\xsrate_upload_<time>.csv`. Local copy: 527 items (455 HMBR, 72 GI); 36 HMBR hand-set values would be replaced; 12 lines to look at (the 4 ladders, and 8 shop prices more than double the ERP standard price, among them four Spirit Level Ultra items whose ERP standard price is a placeholder 1). To leave an item out, delete its entry from the JSON.
3. **Write it:**
   ```bat
   venv\Scripts\python onetime\set_xsrate_from_json.py --commit
   ```
   One transaction; the old values go to table `um_xsrate_backup`. Running it again writes nothing. It stops without writing if more than 5% of the items are missing in the database (wrong database). **Undo:** `onetime\set_xsrate_from_json.py --undo --commit` (puts back only values nobody changed since).

### A7. Optional: a database login that can only do what the importer needs
Edit the password in `sql\prep_02_importer_login.sql`, run it as a superuser on `da`. It reads the ERP tables it needs and writes only `caitem`, `xcodes`, `opmob`, `poord`, `poodt` plus its own `um_*` tables. Tested locally: every job (dry run and `--commit`) runs under it. Skip it and use the Streamlit app's login if you prefer; switch later by changing `[database]`.

### A8. Full config
| Section | Set to |
|---|---|
| `[shop]` | `base_url = https://medusa.utilitymart.shop/hooks/erp`, `feed_key` and `shared_secret` (Part C), `allow_staging_feed = false` |
| `[database]` | as in A4, or the login from A7 |
| `[behaviour]` | `po_mode = daily`, **`allow_purge = false`** (never true on the server) |
| `[exceptions]` | `skip_skus = CF000003` (already in the example file) |
| `[pricing]` | `apply = false` for now |
| `[fixit]` | Fixit host/port/dbname and the read-only login from Part B |
| `[schedule]` | `run_every_minutes = 5`, `fixit_time = 05:30`, `price_time = 05:45`, `products_time = 06:00`, `po_time = 23:30` |

The file holds passwords and keys: keep it out of git, readable only by the service account (`icacls erp_import.ini /inheritance:r /grant:r Administrators:F SYSTEM:F`).

### A9. Health check (read-only)
```bat
venv\Scripts\python erp_import.py --job check
```
Expect OK lines for: database + UTF8, customer `CUS-000073`, employee `MR--000007`, the four suppliers, the warehouse, `opmob.xtra1..xtra5`. It also prints `opmob.xtra1 type` (whole-number discount % if `integer`) and "tables the first run would create" (the eight `um_*` tables). Until Parts B and C exist, only the **shop /health** and **Fixit database** lines may fail. Send me the output.

### A10. Dry run (writes nothing)
```bat
venv\Scripts\python erp_import.py
venv\Scripts\python erp_service.py --once --dry-run
```
Until the shop's hooks exist the feed call fails and the run says the shop is not reachable: that is expected and harmless. Read `logs\erp_import_<date>.log`.

### A11. The NSSM service (in dry-run mode first)
```bat
cd <repo>\erp_import
nssm install UtilityMartERP "<repo>\erp_import\venv\Scripts\python.exe" "erp_service.py --dry-run"
nssm set UtilityMartERP DisplayName "UtilityMart ERP importer"
nssm set UtilityMartERP AppDirectory "<repo>\erp_import"
nssm set UtilityMartERP AppStdout "<repo>\erp_import\logs\service_stdout.log"
nssm set UtilityMartERP AppStderr "<repo>\erp_import\logs\service_stderr.log"
nssm set UtilityMartERP AppRotateFiles 1
nssm set UtilityMartERP AppRotateOnline 1
nssm set UtilityMartERP AppRotateBytes 5000000
nssm set UtilityMartERP AppEnvironmentExtra PYTHONIOENCODING=utf-8
nssm set UtilityMartERP AppExit Default Restart
nssm set UtilityMartERP AppRestartDelay 30000
nssm set UtilityMartERP Start SERVICE_AUTO_START
nssm start UtilityMartERP
```
`logs\erp_service.log` shows every job start/finish/failure (a failing job prints its last 15 lines). Stop: `nssm stop UtilityMartERP`. **Going live** (A13): `nssm set UtilityMartERP AppParameters "erp_service.py"` then `nssm restart UtilityMartERP`.

### A12. Watching it
* Logs: `logs\erp_service.log` (the clock), `logs\erp_import_<date>.log` (each job), `logs\price_report_<date>.csv`.
* Staff to-do list the importer writes: `SELECT created_at, kind, order_ref, details FROM um_attention WHERE NOT resolved ORDER BY id;` (refunds/returns to do by hand, cancels after shipping, missing Fixit items, locked orders).
* Stop everything at once: `nssm stop UtilityMartERP`. Open POs the importer made are ordinary `1-Open` POs with `xref = UM-<order>`: staff can delete them like any other.

### A13. Going live (by hand, in this order)
**ERP** = you on the HMBR server (`D:\streamlit-v2\erp_import`, venv active). **SHOP** = the shop session / Coolify. Each step says what to check before the next one. Stop at the first surprise and send the output.
Do not switch the service to live before Phase 5: a live service would push whatever 100007 holds to the shop on its first products run.

**Before you start (SHOP).** The code is committed and deployed with `ERP_FEED_KEY` and `ERP_SHARED_SECRET` set and `ERP_SYNC_ENABLED` NOT set (the shop is unchanged). The shop session answered on 4 Nov (`ERP_SHOP_SIDE.md` section 9): the backfill sends every order the link has no record of except cancelled ones, and the live shop has only one cancelled test order, so no order goes; `RUN_ONCE_ARGS="commit items"` sends only the items. Variants without a SKU: 82, of which 29 belong to bundles and need none; the other **53 ordinary variants need a SKU typed in the Medusa admin** (32 are on sale; "Clean Ol" 5 Litre and "Zepto Liquid Dish Wash" 1 Liter / Green matter most, they are parts of two on-sale bundles). Until a SKU is typed such an item is not sent and an order containing it waits in the shop (listed on its ERP sync page). Bundles go out as parts (`SHOP_CONTRACT.md` point 8).

**Phase 1: connect, the shop link still off (nothing changes in the shop)**
1. `git pull origin main`, then `venv\Scripts\python erp_import.py --job check` must say `OK shop /health ... environment=live` (404 = the shop is not deployed; 401/403 = key, secret or server clock).
2. `venv\Scripts\pip install openpyxl` (once), then `venv\Scripts\python erp_import.py --job price --xlsx`. Open `logs\price_before_after_<time>.xlsx`: every item with what is in 100007 today ("Was"), the rule's price and what it will be ("Will be"), sorted with the HELD rows first. A move over 25% is HELD: it is NOT written and the item keeps today's price. Decide those rows: fix the source price in the ERP, accept the rule price by adding the SKU to `approve_skus` under `[pricing]`, or set the price you want by hand in 100007. Note that the first products push sends 100007's price to the shop, so a held item's old 100007 price goes to the shop as it is.
3. Apply the rule: set `apply = true` under `[pricing]` in `erp_import.ini`, then `venv\Scripts\python erp_import.py --job price --commit`. It saves every old price in table `um_price_backup` BEFORE it writes; keep the Excel from step 2 as your own record. **Undo:** `venv\Scripts\python erp_import.py --job price --undo` (dry run: shows what would be put back), then the same with `--commit`. It puts back the latest run's prices, only where 100007 still holds what that run wrote.
4. `venv\Scripts\python erp_import.py --job products --shop-dry-run`: the shop says what a push WOULD change (price, barcode, cost, group, stock) and writes nothing. Read `logs\shop_dry_run_products_<time>.csv` (and `.json`).

**Final check before the switch-on, and again before Phase 5:** `venv\Scripts\python erp_import.py --job preflight` (read only). One Excel, `logs\preflight_<time>.xlsx`, over the HMBR server's source companies (100001, 100000, 100005), the Fixit server, the 100007 items and the live shop's differences (what a push would change, asked with `?dry_run=1`), with a short summary on the console. FAIL = fix first (no price, no source item, duplicate SKU, a server that cannot be reached); WARN = look at it (price held, price not applied yet, placeholder source price, no cost or stock, an ERP item the shop does not have, a shop price or barcode that would change a lot). Run it again after typing the shop's SKUs.

**Phase 2: switch the shop link on (SHOP, after telling the ERP side)**
5. SHOP: the backfill dry run (`RUN_ONCE_TASK=erp-backfill`, no args), read the report, fix what it lists, delete the variable.
6. ERP: back up the item table: `"C:\Program Files\PostgreSQL\9.4\bin\pg_dump.exe" -U postgres -h 127.0.0.1 -Fc -t caitem -f D:\backups\caitem_before_um_load.backup da` (the folder must exist; the nightly full backup is the other safety net).
7. SHOP: `ERP_SYNC_ENABLED=true`, restart, then the backfill with `RUN_ONCE_ARGS="commit items"` (items only; orders reach the ERP as events from now on). The outbox now holds about 2,200 item events. The ERP service is still in `--dry-run`, so nothing is consumed yet. Real orders placed from now on wait in the outbox too.

**Phase 3: load the items into the ERP (ERP)**
8. `venv\Scripts\python erp_import.py` (dry run on the real feed; nothing is acknowledged, so repeat it as often as you like). Expect about 2,100 item events, 0 failures, about 17 `barcode_mismatch` warnings (the ERP pairing wins) and about 1,466 NEW Fixit items (`04-00001` upward). Send the output.
9. `venv\Scripts\python erp_import.py --commit --no-stock`; repeat until it reports 0 events (one run takes up to 1000). `--no-stock` keeps the first stock push for step 12. Real orders waiting in the outbox are entered by these runs.
10. `venv\Scripts\python fixit_pull.py --commit`: `followed Fixit items in 100007` should now be about 1466; anything Fixit does not know is listed.
11. `venv\Scripts\python erp_import.py --job price`, then `--job price --commit` (the new Fixit items get their Gulshan price). Check a few Fixit items in the ERP screens (`xitemnew = 'Fixit'` is text in a column that otherwise holds company numbers).
12. `venv\Scripts\python erp_import.py --job products --shop-dry-run`, read it, then `--job products --commit`: the first real push of barcodes (the ERP item codes), price, cost, group and stock into Medusa. Look at the shop's "ERP sync" page and a few products.

**Phase 4: test orders (the service is still in dry-run, so you decide when an order is entered)**
13. Place test orders in the shop: one cash order with one item; one mixing HMBR, GI, Zepto and Fixit items; a bundle; an online payment; a discount code; fulfil one (Pathao parcel) and cancel one. After each, run `venv\Scripts\python erp_import.py --commit` and look:
    ```sql
    SELECT invoiceno, xitem, xqty, xprice, xlinetotal, xstatusord, xtra1 AS discount_pct, xtra3 AS payment, xtra4 AS consignment FROM opmob WHERE zid = 100007 ORDER BY ztime DESC LIMIT 20;
    SELECT created_at, kind, order_ref, details FROM um_attention WHERE NOT resolved ORDER BY id;
    ```
    The order sits under customer `CUS-000073` as `UM-<number>`; staff process it in the ERP as usual (it shows in the DO Creation Audit while pending). Check: the totals and discount %, the payment trail, the courier id, the status and stock coming back to the shop, the purchase orders (`--job po` is a dry run; `--job po --commit` makes them).

**Phase 5: the service goes live**
14. `nssm set UtilityMartERP AppParameters "erp_service.py"` then `nssm restart UtilityMartERP` (`C:\caddy\nssm.exe`). Watch `logs\erp_service.log` and `logs\erp_import_<date>.log` for a day (05:30 Fixit, 05:45 prices, 06:00 products, 23:30 purchase orders; a run every 5 minutes).

**Stopping and undoing.** ERP side: `nssm stop UtilityMartERP`. Shop side: remove `ERP_SYNC_ENABLED` and restart (no more events; real `/stock` and `/products` refused; nothing already in the outbox is lost). Prices: `--job price --undo --commit` (step 3). Items: restore from `caitem_before_um_load.backup`. Purchase orders the importer made are ordinary `1-Open` POs with `xref = UM-<order>`: staff can delete them.

## Part B: Fixit server

The importer only READS the Fixit database (`fixit_pull.py`, run from the HMBR server): stock = sum over Fixit Gulshan, Central and Ecommerce; price = Gulshan; cost = Central's sales rate.
1. Check the version: `SELECT version();` (materialized views need 9.3+, `REFRESH ... CONCURRENTLY` 9.4+).
2. See whether the view exists: `\dm public.final_items_view`. If not, run `db\sql_scripts\fixit_create_final_items_view.sql` in the Fixit database (it creates the materialized view and its index).
3. **It is a snapshot.** Refresh it every night BEFORE 05:30 (the Fixit pull): a Task Scheduler / pgAgent entry on the Fixit server running
   `psql -U postgres -d <fixitdb> -c "REFRESH MATERIALIZED VIEW CONCURRENTLY public.final_items_view;"` at about 05:00. Without it the shop would get stale Fixit stock.
4. Read-only login: `sql\fixit_01_reader_login.sql` (change the password). Tested: it can read the view and nothing else.
5. Reachability from the HMBR server: `listen_addresses` includes the network address, `pg_hba.conf` gets `host <fixitdb> um_fixit_reader <HMBR server IP>/32 md5` (then `pg_ctl reload`), the Windows firewall allows 5432 from that IP only. Test from the HMBR server: `psql -h <fixit host> -U um_fixit_reader -d <fixitdb> -c "SELECT count(*) FROM final_items_view;"`.
6. Put host, db name, login in `[fixit]` of `erp_import.ini`; `venv\Scripts\python fixit_pull.py` (dry run) should list the Fixit items found.

Reminder (not built): the rule for NEW Fixit items (stock above 5, sales not 0, a prefix the shop already has) is still to be written.

---

## Part C: Medusa shop

1. Give the shop session `erp_import\docs\SHOP_CONTRACT.md` (what to build: `/hooks/erp` feed, ack, order-status, stock, and the new `POST /products`; whole-number `display_id`; payment `transaction_id`; courier consignment ids). `tests/fake_feed.py` is the same contract as running code.
2. Make two secrets (do not paste them in chat or commit them):
   ```bat
   venv\Scripts\python -c "import secrets; print(secrets.token_hex(32))"
   ```
   Run it twice: one is `ERP_FEED_KEY`, the other `ERP_SHARED_SECRET`. Put both in the shop's environment AND in `[shop] feed_key / shared_secret` of the HMBR server's `erp_import.ini`.
3. The shop answers `GET /health` with `"environment": "live"`; then A8 shows green and A12 can start.
