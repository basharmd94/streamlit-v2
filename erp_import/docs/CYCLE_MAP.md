# UtilityMart shop <-> ERP 100007: the cycle, the columns hit, the live-server plan

Draft 2026-11-02 (updated with the owner's answers). No code changed. Picture of all servers and flows: `CYCLE_VISUAL.html`. Every column of the written tables, with types and examples: `CYCLE_COLUMNS.md`.
Written by: **I** = the importer (my script), **ERP** = the ERP's own order creation, **Shop** = Medusa.

## 1. Settled

- Everything happens in 100007. Fixed: customer **CUS-000073** UtilityMart, employee **MR--000007**, terminal UTM, warehouse "Sales Warehouse Online Shop", project "HMBR Online Shop".
- Only the shop's own variants are followed, matched on **SKU** (= source-company `xitem` = `caitem.xalias`) and **barcode** (= 100007 `xitem`). Nothing else is pulled.
- Order number goes in `opmob.invoiceno` as `UM-1000001` and the number in `invoicesl`; the ERP copies `invoiceno` into `opord.xref` and `opdor.xref`. Discount % goes in `xtra1` (whole number), the payment trail in `xtra3` (SSLCommerz transaction id, or COD), the courier delivery id in `xtra4`, delivery charge and the rest in `xtra5`. I find an order's rows again by `invoiceno`.
- Shopper details stay in the shop. Returns, refunds and purchase returns are manual.
- **Master of each field:** ERP = SKU, barcode, std price, cost price, item group, stock. Shop = variant code (product handle / VG), variant title, options, category, images, description.
- **ERP price overwrites the shop price.** Inside the ERP a scheduled job sets 100007 `xstdprice` from the SOURCE company (rule decided 2026-11-04): **Zepto (100005) = its own `caitem.xsrate` (the MRP); HMBR (100001) and GI (100000) = their `caitem.xstdprice` x 1.40, rounded down to a whole taka** (`[pricing] markup`); Fixit = Fixit Gulshan `std_price`. An item with no source price gets no price, never a guess. The `xsrate` of HMBR/GI is never read or written: the ERP uses it for its own calculations (a one-time load of shop prices into it on 2 Nov broke that and was undone). A move over `max_change_pct` (25) is flagged `review` and not written. It starts as a report (`[pricing] apply = false`).
- **Fixit:** follow exactly the Fixit items the shop sells (the master list). Stock = sum over the three Fixit companies, selling price = Gulshan, cost and PO rate = Central's sales rate (`store_buy_price`). `xitemnew = 'Fixit'`. The rule for NEW Fixit items (prefix the shop has, stock above 5, sales not 0) is parked as a reminder.
- **ERP pairing of SKU and barcode in 100007 is the truth**; the shop is corrected from it. The 82 SKU-less variants and `1527`, `1528` are ignored for now.
- Cost price for never-bought items = the source company's `xstdprice`. Cost is pushed to the shop with the other product details.
- Everything is tested locally first (local `da`, local `fixit`, live Medusa read-only). Real test orders on the live shop are allowed once the hooks exist.
- Parked: nothing for the overselling gap. The spec (section 8) has the shop subtract its own reserved and shipped units from the stock I send, so I send physical stock only.

## 2. What connecting showed (2 Nov, read-only)

| Check | Result |
|---|---|
| Shop access (read-only admin key) | Works. 1,598 products, 2,191 variants, 2,164 inventory items, 0 orders, 2 customers. |
| Shop feed `/hooks/erp` | **404, not built yet** (shop session). A price/cost push endpoint is not in the spec either. |
| SKU + barcode against `caitem` (local copy) | 640 variants have a barcode; 623 agree. **15 Zepto pairs look shifted by one** (shop `FZ000067` has barcode `03-00090`; local `03-00090` is `FZ000223`). 2 barcodes (`03-00105`, `03-00106`) are not in the local copy. The local copy may be older than live: check there first. |
| Variants with no SKU | 82. Also `1527`, `1528` have a SKU that is not in `caitem`. |
| Fixit | 1,468 variants; 1,467 carry a Fixit code (`18-018`, `26-4323`); none has a barcode; none is in `caitem` (2 stray `04-` items). All 1,467 exist in the Fixit database view. |
| Price, shop vs `caitem.xstdprice` (640) | 546 equal, 46 shop lower, 13 shop higher, 35 no price. |
| Stock, shop vs source stock (640) | 365 equal, 269 differ (set by hand in the shop). The first stock push overwrites them. |
| Cost | 127 of 656 items were ever received in 100007; for all 127 the last receipt value equals the last PO rate. **529 have no cost yet.** |
| Fixit rule (`XX-` code, stock > 5, sales not 0) | **1,242 items qualify** with stock summed over Gulshan + Central + Ecommerce. 635 are already in the shop, 607 are not. 833 of the shop's Fixit variants do not qualify. |
| Local copy | `MR--000007` present. `CUS-000073` is not (created on live). |

## 3. The cycle: columns hit per step

### Step 1. Order comes in -> `opmob` (one row per line)  [every 5 min]

| Table | Column | Value | Example | By |
|---|---|---|---|---|
| `caitem` (read) | `xitem, xalias, xitemnew, xdesc` | find the ERP item from the shop SKU | 0718 -> 01-00135 | I |
| `opmob` | `zid, ztime, zutime, xsl` | 100007, now, now, new uuid | | I |
| | `invoiceno` | `UM-` + shop order number | UM-1000001 | I |
| | `invoicesl` | the same number | 1000001 | I |
| | `xemp`, `username` | fixed | MR--000007, utilitymart | I |
| | `xcus`, `xcusname`, `xcusadd` | fixed | CUS-000073, UtilityMart, UtilityMart online shop | I |
| | `xterminal` | fixed | UTM | I |
| | `xitem`, `xdesc` | ERP item and its name | 01-00135, Hook--SS02 62**2mm | I |
| | `xqty`, `xprice`, `xroword` | line quantity, shop unit price before discount, line number | 2, 80, 1 | I |
| | `xlinetotal` | qty x price, whole taka | 160 | I |
| | `xdate` | order date, Dhaka | 2026-11-02 | I |
| | `xstatusord` | `New` | New | I, then ERP |
| | `xordernum` | empty; the ERP writes the COMO number | COMO421005 | ERP |
| | `xtra1` | order discount %, whole number (exact in `xtra5`) | 5 | I |
| | `xtra2` | net total after discount, before delivery | 745.75 | I |
| | `xtra3` | payment trail: SSLCommerz transaction id, or `COD` | COD | I |
| | `xtra4` | courier delivery (consignment) id, filled when the parcel is booked | PATHAO-12345 | I |
| | `xtra5` | JSON: delivery charge, shop order id, shipping method, note, exact discount | {"delivery_charge":100,...} | I |
| `um_order_map`, `um_processed_events` | | order id -> invoice, POs; each event once | | I |

Cancel before staff act: delete the `opmob` rows and the still-open PO. After: a staff to-do (`um_attention`).

### Step 2. Purchase -> `poord` / `poodt` (open PO per order per source business)

| Table | Column | Value | Example | By |
|---|---|---|---|---|
| `caitem` (source company, read) | `xitem, xstdprice` | the source item (= `xalias`) and its price | 100001, 0718, 57 | I |
| `opspprc` (source company, read) | `xpricecat, xqty, xdisc` | wholesale tier price for the day's total quantity (rule to confirm) | | I |
| `imtrn`, `poodt` (read) | `xitem, xqty, xsign, xqtyord, xqtygrn` | 100007 stock and quantity already on open POs, to get the shortfall | | I |
| `poord` | `xpornum, xdate, xsup, xstatuspor` | next number, today, supplier by source, **`1-Open`** | PO--000351, SUP-000001 | I |
| | `xref, xsupref` | the order number | UM-1000001 | I |
| | `xrem` | order number + shop order id | Auto-created from online order UM-1000001 (order_01JBX3K8Q2) | I |
| | `xwh, xproj, xtotamt` (+ the other total columns) | warehouse, project, sum of lines | 114 | I |
| `poodt` | `xpornum, xrow, xitem, xcode` | the PO, line, item | PO--000351, 1, 01-00135 | I |
| | `xqtyord` | shortfall only; no shortfall, no line | 2 | I |
| | `xrate, xprice, xlineamt, xdtwotax` | source price (less the wholesale tier), x qty | 57, 114 | I |
| | `xdatesch` | `2999-12-31`, as real PO lines hold | | I |

Supplier by source: 100001 = SUP-000001, 100000 = SUP-000002, 100005 = SUP-000003, **Fixit (code shaped `XX-XXX(X)`) = SUP-000005, rate = Central's sales rate (`store_buy_price`)**.
POs are created **once a day at a fixed time**, so the wholesale tier is worked out from the day's total per item, but each order keeps its own PO(s) with the order id on them.

### Step 3. Product details to the shop  [fixed time daily; stock also during the day]

| Table | Column | Value | Example | By |
|---|---|---|---|---|
| `final_items_view` (read) | `zid, item_id, stock` | stock of the source item (HMBR, GI, Zepto) | 100001, 0718, 4190 | I |
| `um_fixit_item` (read, step 4) | `sku, stock, std_price, cost` | Fixit items | 18-018 | I |
| `caitem` 100007 (read) | `xitem, xalias, xstdprice, xgitem` | barcode, SKU, std price, item group | 01-00135, 0718, 80, Trading Item | I |
| `poodt` / `imtrn` (read) | `xrate` of the latest PO line, or latest receipt `xval / xqty` | **cost price** (the two agree for all 127 bought items) | 57 | I |
| `um_product_sent` (new) | `sku, on_hand, std_price, cost_price, sent_at` | what was last sent, so only changes go | | I |
| HTTP | `POST /hooks/erp/stock` exists (`sku, on_hand`). **`POST /hooks/erp/products` (barcode, std price, cost, item group) must be added by the shop session.** | | | |

### Step 4. Fixit -> 100007 (separate script `fixit_pull.py`, on the HMBR server)

| Table | Column | Value | Example | By |
|---|---|---|---|---|
| Fixit DB `final_items_view` (read-only) | `zid, item_id, item_name, item_group, std_price, stock, avg_monthly_sales, store_buy_price` | only the followed Fixit items (100007 `caitem` with `xitemnew` = Fixit) | 18-018 | I |
| `caitem` 100007 (created by the shop's item event, step 5) | `xitem` = next `04-nnnnn`, `xalias` = Fixit code, `xitemnew` = Fixit, `xgitem` = Fix iT Item, `xwh`, rest from a template | new items only | 04-00001, 18-018 | I |
| `xcodes` | `xtype, xcode, xcodealt, xdescdet, zactive` | Price Category row | | I |
| `um_fixit_item` (new) | `sku, fixit_zid, stock, std_price, cost, item_group, sales, as_of` | where Fixit stock and cost "sit" in 100007 | | I |

### Step 5. Shop item changes -> `caitem`  [every 5 min, from the feed]

| Table | Column | Value | By |
|---|---|---|---|
| `caitem` | `xdesc, xlong, xnote, xlinks, xscode` | variant title, product title, options, handle, VG code (shop is master) | I |
| | `xstdprice` | **no longer written from the shop** (ERP is master) | - |
| | new item: `xitem` next code, `xalias`, `xitemnew`, `xgitem`, `xpricecat`, `xwh`, rest from template | | I |
| `xcodes` | Price Category row | | I |

### Step 6. Status back to the shop  [every 5 min, staff only]

| Table (read) | Column | Gives |
|---|---|---|
| `opmob` | `xstatusord, xordernum` | entered / order created |
| `opord`, `opdor`, `opcrn` | `xordernum -> xdornum -> xcrnnum` | confirmed (delivery order exists) / credited. The shop uses `confirmed` to stop holding the units back. |
| `um_order_map` (write) | `last_status, last_status_at` | |

## 4. Test and live-server plan

**Update:** every test below runs locally first (local `da` on PostgreSQL 14, local `fixit`, the live Medusa shop read-only). The live HMBR server (PostgreSQL 9.4) is the last step and repeats T1 and T2 there. I build my side against a local stand-in for `/hooks/erp` (the existing fake feed plus a `/products` call), write the contract from it, and hand that to the shop session; both sides then test against each other.

**Before the server**

| # | What | Who |
|---|---|---|
| A1 | Build `/hooks/erp` (feed, ack, order-status, stock) and the new `/products` push; generate the feed key and shared secret | shop session |
| A2 | Fix the shop data: 82 variants without SKU; the 15 + 2 Zepto pairs (check against live `caitem`); barcodes for Fixit | owner |
| A3 | My code changes (list below), local tests with the fake feed, a `--check` command, `--only-skus` and a Fixit puller | me |

**Server setup (once)**

| # | What |
|---|---|
| B1 | Python 3.6+ and `psycopg2` (already used for the October item upload). Folder with `erp_import.py`, `fixit_pull.py`, `erp_import.ini` (not in git). |
| B2 | `erp_import.ini`: shop base URL and the feed key + shared secret, database, `customer = CUS-000073`, `salesman = MR--000007`, terminal, warehouse, project, suppliers, Fixit DB read-only login. `allow_purge = false`, `allow_staging_feed = false`. **The Medusa admin key never goes on the server.** |
| B3 | Database rights: read on `caitem, final_items_view, imtrn, opspprc, opord, opdor, opcrn, casup, cacus, prmst`; insert/update/delete on `opmob, poord, poodt`; insert/update on `caitem, xcodes`; create the `um_*` tables. |
| B4 | `SHOW server_encoding;` is UTF8 and PostgreSQL is 9.4. Outbound HTTPS to `medusa.utilitymart.shop` and to the Fixit database works. |
| B5 | Staff create nothing: CUS-000073, MR--000007, SUP-000001..5 and the warehouse already exist on live. `--check` confirms. |

**Tests (in order; stop at the first failure)**

| # | Test | Pass when |
|---|---|---|
| T1 | `--check`: connects, version, encoding, CUS-000073 / MR--000007 / suppliers / warehouse exist, creates the `um_*` tables inside a rolled-back transaction, lists shop-vs-`caitem` mismatches | writes nothing; no errors |
| T2 | Fake feed + dry run on the server (`tests/fake_feed.py` serves 11 events built from live items) | every step runs on 9.4 and is rolled back |
| T3 | Dry run against the real feed | item events read; only warnings you accept |
| T4 | One real test order (cheap in-stock item, cash on delivery, note TEST): dry run, then `--commit` | `opmob` rows with `invoiceno UM-…`, `xtra1/3/4/5`; open PO with the order number; shop shows "entered". Then staff create the order: `opord.xref = Created From UM-…` |
| T5 | A second test order, cancelled in the shop right after | `opmob` rows and the open PO are gone; shop shows "canceled" |
| T6 | Product push for 5 SKUs (`--only-skus`) | stock, std price, cost, group match in the Medusa admin |
| T7 | Fixit pull for 20 SKUs, dry run then commit | new `04-` items, `um_fixit_item` rows match the Fixit report |

There is no staging shop (staging became live), so T4 and T5 are real orders in the live shop: tell whoever dispatches (Pathao) to ignore them.
**Go live:** Task Scheduler: importer every 5 min, product push daily, Fixit pull daily. **Rollback:** disable the tasks; test rows are removed by `invoiceno` and PO number (SQL provided with T4).

**Code changes after you approve this map**

1. Fixed customer and employee (no `cacus` matching or inserts); `invoiceno` / `invoicesl`; `xtra1` discount %; drop shopper data from `xtra5`.
2. PO: Fixit supplier, wholesale tier rate, daily timing, `xdatesch`, `xcodes.xdescdet`.
3. Stop writing shop price into `caitem.xstdprice`; add the product push and `um_product_sent`; add `--check`, `--only-skus`.
4. New `fixit_pull.py` and `um_fixit_item`.

## 5. Status and what is still open

Built and tested locally on 2 Nov (`tests/run_local_test.py`, 71 checks, plus a dry run of the whole real shop catalogue: 2,105 item events, no failures, 17 barcode warnings). Added 2 Nov (evening): price rule from the source `xsrate`, `erp_service.py` (NSSM clock), `sql/` prep scripts, `DEPLOY_HMBR_SERVER.md`. Not done: anything on the shop side; nothing run on PostgreSQL 9.4 yet.

1. **Shop session:** build `/hooks/erp` (feed, ack, order-status, stock) and the new `POST /products` from `SHOP_CONTRACT.md`.
2. **Price rule:** changed 2026-11-04 (the 2 Nov rule, source `xsrate` loaded from shop prices, was undone): 100007 price = Zepto's own `xsrate` (MRP), HMBR/GI `xstdprice` x 1.40, Fixit = Gulshan std price. Still to do: read `--job price` on the live data (check the `review` lines), then switch `[pricing] apply = true`.
3. **Live deletes and checks:** delete the two stray `04-00001` / `04-00002` items on the live server; run `--job check` there (PostgreSQL 9.4, `opmob.xtra1` type).
4. **Test orders** on the live shop once the hooks exist: you place them in the storefront, or I create them through the shop's store API?
5. **Reminder:** the rule for new Fixit items (prefix the shop has, stock above 5, sales not 0).

## Appendix: the 15 shop SKU / barcode pairs that disagree with the local `caitem`

| Shop SKU | Shop barcode | Local `caitem` SKU for that barcode |
|---|---|---|
| FZ000067 | 03-00090 | FZ000223 |
| FZ000223 | 03-00091 | FZ000224 |
| FZ000224 | 03-00092 | FZ000225 |
| FZ000225 | 03-00093 | FZ000222 |
| FZ000222 | 03-00094 | RZ000312 |
| RZ000312 | 03-00095 | RZ000315 |
| RZ000315 | 03-00096 | RZ000355 |
| RZ000355 | 03-00097 | RZ000440 |
| RZ000440 | 03-00098 | FZ000098 |
| FZ000098 | 03-00099 | FZ000099 |
| FZ000099 | 03-00100 | FZ000100 |
| FZ000100 | 03-00101 | FZ000166 |
| FZ000166 | 03-00102 | FZ000189 |
| FZ000189 | 03-00103 | RZ000157 |
| RZ000157 | 03-00104 | RZ000383 |
