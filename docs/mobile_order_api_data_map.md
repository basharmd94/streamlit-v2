# Mobile ERP API — Order-to-Cash Database Impact Map

Traced from source (`routers/*.py` → `controllers/db_controllers/*.py` → `models/*.py`, and raw SQL
where controllers bypass the ORM) and verified column-for-column against the local Postgres DB (`da` @
localhost:5432, per `database.py`). Covers the full order→delivery→payment/return→collection chain the
mobile app drives, not just order creation. A machine-readable copy of this map is at
[`mobile_order_api_data_map.json`](mobile_order_api_data_map.json) for use in Streamlit/other scripts.

Modules covered: **Orders** (`opmob`), **Delivery Orders / promised dates** (`opdor`/`opddt`),
**Sales Returns** (`opcrn`/`opcdt`), **Customer Payments / collection** (`glpmt`), **Customer
Receipts report** (`glheader`/`gldetail`), **Customer Balance/ledger** (`glheader`/`gldetail`). Other
modules exist in the API (items, customer master, manufacturing, feedback, RBAC, location) but aren't
part of this order-to-cash chain — ask if you want those mapped too.

## Endpoints covered

Mounted under prefix `/api/v1/order` (see `main.py`):

| Method | Path | Purpose | Writes? |
|---|---|---|---|
| POST | `/create-order` | Single order, one customer, N line items. Requires permission `order.create`. | Yes |
| POST | `/create-bulk-order` | Multiple orders (multiple customers) in one call, processed concurrently. Permission check currently commented out in code. | Yes |
| GET | `/get-pending-orders` | Last N orders with `xstatusord = 'New'` for the current user. | No (read-only) |
| GET | `/get-confirmed-orders` | Last N orders with `xstatusord = 'Order Created'`. | No (read-only) |
| GET | `/get-cancelled-orders` | Last N orders with `xstatusord = 'Not enough stock to create Order'`. | No (read-only) |

> **Correction to an earlier version of this doc:** all three "get-*-orders" endpoints read from
> **`opmob`** (via `get_orders_by_status()`, `SELECT ... FROM opmob WHERE xstatusord = :status`), not
> from `opord`/`opodt`. There is only one order table for the mobile app's own view of its orders —
> `opmob` — and its `xstatusord` column is what changes underneath it. Confirmed against live data: of
> 379k `opmob` rows, only 12 are still `'New'`, ~311k are `'Order Created'`, and ~68k are `'Not enough
> stock to create Order'` — i.e. almost every row's status has already been flipped by something other
> than this API (this API only ever *writes* `'New'`, it never updates `xstatusord` afterwards). That
> "something else" is presumably the same back-office/legacy process that separately populates the much
> larger `opord`/`opodt` tables — those remain a distinct, read-only-from-this-API pair of tables (see
> below), not the source of "confirmed orders."
| GET | `/health` | Liveness check. | No |

Both write endpoints funnel into `OrderDBController.create_order()`, so the table/column impact is
identical for a single order or each order inside a bulk batch.

Every request also passes through auth dependencies (`get_current_normal_user` → `get_current_user`)
and, for `/create-order`, the `@has_permission("order.create")` decorator — these touch several
auxiliary tables on **every** call regardless of order content.

## Tables written by an order submission

### `opmob` — the order/line-item table itself
One row is inserted **per line item** in the request (not per order). Primary key is `xsl` (a
generated UUID). Verified against Postgres — table has one extra column (`xappver`) that this API
never sets.

| Column | Type (DB) | Set by API? | Source | Notes |
|---|---|---|---|---|
| `zid` | integer | Yes | request body `zid` | Business/company ID |
| `ztime` | timestamp | Yes | `datetime.now()` at insert | Creation time |
| `zutime` | timestamp | Yes | `datetime.now()` at insert | Same as `ztime` on create |
| `invoiceno` | varchar(100) | Yes | `f"{current_user.terminal}-{invoiceno}"` | `invoiceno` is a random 12-digit number reformatted by `format_invoice_number()` |
| `invoicesl` | bigint | Yes | random 12-digit number (`generate_random_number(12)`) | Same value shared by all items in one order |
| `username` | varchar(100) | Yes | JWT-authenticated user | From `current_user.username` |
| `xemp` | varchar(100) | Yes | `current_user.employeeCode` | Looked up from `apiUsers` at login/token time, not re-queried here |
| `xcus` | varchar(100) | Yes | request body `xcus` | Customer code, **not validated** against a customer master table in this flow |
| `xcusname` | varchar(100) | Yes | request body `xcusname` | Free text from the app, not looked up server-side |
| `xcusadd` | varchar(100) | Yes | request body `xcusadd` | Free text |
| `xitem` | varchar(100) | Yes | item `xitem` | **Not validated** against an item master (e.g. `caitem`) — server trusts the app |
| `xdesc` | varchar(250) | Yes | item `xdesc` | Free text from the app |
| `xqty` | integer | Yes | item `xqty` | |
| `xprice` | double precision | Yes | item `xprice` | Client-supplied unit price, no server-side price lookup |
| `xstatusord` | varchar(100) | Yes | hardcoded `"New"` | Always `"New"` at creation; later transitions (`"Order Created"`, `"Not enough stock to create Order"`) happen **outside this API** |
| `xordernum` | varchar(100) | No (left NULL) | — | Populated later by whatever downstream process promotes `opmob` → `opord`/`opodt` |
| `xroword` | integer | Yes | item `xroword` | Line number within the order |
| `xterminal` | varchar(10) | Yes | `current_user.terminal` | Device/terminal code from the logged-in user |
| `xdate` | date | Yes | `datetime.now()` | Order date |
| `xsl` | varchar(100) (PK) | Yes | `str(uuid4())` | Unique per line item |
| `xlat` | double precision | Yes | item `xlat` | Optional GPS latitude captured on the item |
| `xlong` | double precision | Yes | item `xlong` | Optional GPS longitude |
| `xlinetotal` | integer | Yes | item `xlinetotal` | ⚠️ DB column is `integer` but the Pydantic schema/model treat it as `float` — decimals will be truncated by Postgres on insert |
| `xtra1`–`xtra5` | mixed | No (left NULL) | — | Spare/legacy columns, unused by this API |
| `xappver` | numeric(10,2) | No — **not in the SQLAlchemy model at all** | — | Exists in the live DB but this API can never populate it |

### `location_records` — optional GPS side-effect
Written only if **at least one item** in the order has a valid, non-zero `xlat`/`xlong` pair. Only the
**first** valid coordinate found among the order's items is used — it does not store one row per item.
This insert runs in its own commit, after the `opmob` rows are already committed, and failures here are
swallowed (logged as a warning) rather than failing the order.

| Column | Set by API? | Source |
|---|---|---|
| `username` | Yes | `current_user.username` |
| `latitude` / `longitude` | Yes | first valid item coordinate in the order |
| `timestamp` | Yes | `datetime.now()` |
| `xdate` | Yes | `timestamp` formatted `YYYY-MM-DD` |
| `business_id` | Yes | request `zid` |
| `formatted_address` | Yes, conditionally | reverse-geocoded via `utils/lat_long_converter.py` — **only if** the coordinate isn't `(0,0)`; this calls an external geocoding service |
| `altitude`, `accuracy`, `name`, `street`, `district`, `city`, `region`, `postal_code`, `country`, `maps_url`, `notes`, `device_info`, `is_check_in`, `shared_via`, `is_mock_location`, `dev_options_enabled` | No (left NULL/default) | Not populated by the order flow |
| `created_at` | No | DB default (`now()`) |

## Tables touched by auth/permissions on every call (read, occasional write)

These aren't order-specific but fire on **every** authenticated request to these endpoints, so any
tracing of "what does hitting this API do to the DB" must include them.

| Table | Access | When |
|---|---|---|
| `apiUsers` | Read | `get_current_user` looks up the user by username; `has_permission` re-queries it (with `roles`) to check `order.create` |
| `logged` | Read + **write** | `get_current_user` checks there's an active session row for the token; `session_activity_middleware` (global middleware, runs on nearly every request) updates that row's `zutime` to "now" on each call |
| `token_blacklist` | Read + occasional write | `is_token_blacklisted` checks the token isn't revoked, and opportunistically runs `cleanup_expired_tokens` (a `DELETE`) on every check |
| `roles`, `permissions`, `role_permission`, `user_role` | Read | Only for `/create-order` (permission-decorated); resolved via the `ApiUsers.roles` → `Role.permissions` relationship chain |

`order.create` in the local DB is currently granted to the `admin`, `manager`, and `sales` roles.

## Delivery Orders — where "promised payment date" actually lives

Mounted under `/api/v1/order` (`routers/delivery_route.py`, `controllers/db_controllers/delivery_db_controller.py`).
This is a different table pair from `opmob`/`opord` — a full, wide ERP delivery-order header/detail
(`opdor`/`opddt`, 594k / 1.66M rows locally), joined against `cacus` (customer) and `caitem` (item master).

| Method | Path | Purpose | Writes? |
|---|---|---|---|
| GET | `/get-delivery-orders` *(exact path per route file)* | List delivery orders for the logged-in salesperson (`xsp`), with line items, filterable by `xdornum`/`xcus`/`xdate`/`xitem`. | No |
| GET | `/get-delivery-order/{...}` | Single delivery order with full line-item detail. | No |
| PUT | `/delivery-date-update/{zid}/{xdornum}` | **The "promised payment date" endpoint.** | **Yes** |

`update_delivery_dates()` runs:
```sql
UPDATE opdor SET xdatedel = :delivery_date, xdatepay = :payment_date
WHERE zid = :zid AND xdornum = :xdornum
```

| Column | Table | Meaning | Written by this API? |
|---|---|---|---|
| `xdatepay` | `opdor` | **Promised/agreed payment date** for that delivery order | Yes — via `/delivery-date-update` |
| `xdatedel` | `opdor` | Promised/rescheduled delivery date | Yes — via `/delivery-date-update` |

⚠️ **Read/write asymmetry found:** neither the list endpoint (`get_delivery_orders`) nor the detail
endpoint (`get_delivery_order_detail`) selects `xdatedel` — they only ever `SELECT`
`o.xdate` (the original order date) and `o.xdatepay`. So `xdatedel` is **write-only from this API**:
the mobile app can set it via the PUT, but no GET response in this codebase ever returns it back.
If a Streamlit report wants the "current promised delivery date," it has to query `opdor.xdatedel`
directly — the API won't hand it back.

Other columns read (never written) from `opdor`/`opddt` on the GET endpoints: `xordernum`, `xcus`,
`xtotamt`/`xdiscamt`/`xdtdisc` (exposed as `grossamt`/`discamt`/`netamt`), `xstatusdor`, `xwh`, `xproj`,
plus line items from `opddt` (`xitem`, `xqty`, `xrate`, `xlineamt`) joined to `caitem` for `xdesc`/`xunitstk`
and to `cacus` for `xshort`/`xadd1`/`xorg`. None of these are ever written by this API — `opdor`/`opddt`
rows themselves are created by some other (non-mobile-API) process; this API only ever updates the two
date columns on an existing row.

The live `opdor` table also has a separate `xdatedue` column (a GL-style due date) — unrelated to the
mobile "promised payment date" feature; it's not referenced anywhere in this codebase.

## Sales Returns — "return registration" (`create-return`)

Mounted under Return tags (`routers/return_route.py`, `controllers/db_controllers/return_db_controller.py`).
This is the actual write-heavy "register a return" flow, separate from the read-only `Opcrn`/`Opcdt`
ORM stubs in `models/orders_model.py` — the controller uses raw parameterized SQL and writes far more
columns than the ORM models declare.

| Method | Path | Purpose | Writes? |
|---|---|---|---|
| GET | `/get-sales-returns` | List sales returns for the logged-in employee (`xemp`, taken from the JWT — not a query param), with items grouped by `xcrnnum`. Filters: `xdate`, `xcus`, `xcrnnum`, `xdornum`, `xitem`. | No |
| POST | `/create-return` | **Registers a new sales return** (one header + N line items), inside one transaction. No `@has_permission` check on this endpoint. | Yes |

**`opcrn` (return header)** — `create_return()` inserts 23 columns. Live table actually has ~48 columns
(`xglref`, `xinvnum`, `xteam`, `xtotamt`, `xtyperet`, `xretvstat`, tax fields, etc.) that this API never
sets:

| Column | Written by API? | Source |
|---|---|---|
| `ztime`, `zid`, `xcrnnum` | Yes | server time / request zid / voucher generated as `SR--NNNNNN` via `generate_voucher(table="opcrn", column="xcrnnum")` |
| `xdate`, `xcus`, `xdatecuspo`, `xordernum`, `xdornum`, `xemp` | Yes | from `return_header` in the request body |
| `xsec` | Yes | request or default `"Normal"` |
| `xproj` | Yes | request or default `""` |
| `xstatuscrn` | Yes | request or default `"1-Open"` |
| `xreason` | Yes | request or default `""` |
| `xappamt`, `xdisc`, `xdiscf` | Yes | hardcoded `0` |
| `xappcode`, `xmember` | Yes | current user's username (falls back to `"system@hmbrbd.com"`) |
| `xcur` | Yes | hardcoded `"BDT"` |
| `xexch` | Yes | hardcoded `1` |
| `xtrncrn` | Yes | hardcoded `"SR--"` |
| `xstr01` | Yes | duplicated from `header.xemp` |
| `xwh` | Yes | from request |
| everything else (`xglref`, `xinvnum`, `xteam`, `xmanager`, `xtotamt`, `xcounterno`, `xyear`, `xper`, `xdept`, `xrem`, `xpayccnum`, `xtype`, `xpflag`, `xref`, `xvoucher`, `xnote`, `xtyperet`, `xretvstat`, `xdttax`, `xdtcomm`, `xdtwotax`, `zutime`) | No | left NULL — including `xtotamt`, so the header itself never stores the return's total (the API computes and returns it in the response, but doesn't persist it on `opcrn`) |

**`opcdt` (return detail)** — one row **per returned line item**, keyed by `(zid, xcrnnum, xrow)`. The
INSERT explicitly lists 51 columns (matching most of the live table), but many are hardcoded
placeholders rather than real values:

| Column(s) | Written by API? | Source |
|---|---|---|
| `ztime`, `zid`, `xcrnnum`, `xrow`, `xitem`, `xdesc`, `xqty`, `xrate`, `xprice`, `xlineamt`, `xdornum`, `xwh` | Yes | from the request's return item / header |
| `xcode` | Yes | copied from `xitem` |
| `xunitsel` | Yes | request or default `"Pcs"` |
| `xcodebasis`, `xstype`, `xpricebasis`, `xtaxcat` | Yes | hardcoded constants (`"Our Code"`, `"Stock-N-Sell"`, `"Standard List"`, `"ANY"`) |
| `xstatusddt`, `xstatuscdt` | Yes | hardcoded `"1-Open"` |
| `xcur` | Yes | hardcoded `"BDT"` |
| `xwtunit` | Yes | hardcoded `0.0` |
| `xunitwt` | Yes | hardcoded string `"0.0"` (⚠️ column is `character varying` in the live DB, so this is intentional, not a type bug — but note `opddt.xwtunit`/`xunitwt` elsewhere in the schema have the *opposite* type pairing, so don't copy this convention blindly) |
| `xcfsel`, `xexch` | Yes | hardcoded `1` |
| `xqtydel`, `xqtyinv`, `xratesys`, `xmargin`, `xmarginsys`, `xdisc`, `xdiscsys`, `xdiscf`, `xcomm`, `xcommsys`, `xexchsell`, `xratesys`, `xtaxrate1`–`xtaxrate5`, `xlineamtsys`, `xfreight` | Yes | hardcoded `0` |
| `xtaxcode1`–`xtaxcode5`, `xlong`, `xlinks` | Yes | hardcoded `""` |
| `xcurprice` | Yes | hardcoded string `"0"` |
| everything else in the live table (`xbatch`, `xserialnum`, `xbin`, `xaltqty`, `xtypeserial`, `xstdcost`, `xcost`, `xsup`, `ximtrnnum`, `xdept`, `xdtwotax`, `xdttax`, `xdtdisc`, `xdtcomm`, `xtypecon`, `zutime`, `xordernum`) | No | not part of the INSERT at all |

**Practical implication:** a return created via this API has **no tax, discount, commission, or costing
data** — those columns are all zeroed/blanked regardless of what the return actually involves. If your
Streamlit reporting needs accurate return economics, don't trust `opcdt.xtaxrate*`/`xdisc*`/`xcomm*`/`xcost`
for rows created through this endpoint; compute from `xqty × xrate` (`xlineamt`) instead.

## Customer Payments — the mobile "register a collection" flow

Mounted at `/api/v1/customers/customer-payment` (`routers/customer_payment_route.py`,
`controllers/db_controllers/customer_payment_controller.py`, table `glpmt`). This is almost certainly
what the mobile app calls "collection" from a field rep's point of view: a rep logs that they collected
money from a customer, and it lands here — separate from the accounting-posted GL receipt voucher (see
next section).

| Method | Path | Purpose | Writes? |
|---|---|---|---|
| POST | `/{zid}/` | Register a new payment/collection from a customer. No `@has_permission` check. | Yes |
| GET | `/get-all-payments/` | List payments, filterable by `zid`/`xcus`/`xemp`/`xdate`, paginated. | No |
| PUT | `/{xpmtnum}/update-status/` | Change `xpaystatus` (e.g. `"Send"` → `"Received"`). | Yes (one column) |

`glpmt` columns (live table has 16; all but `xdornum`/`xtotamt` are used by this API):

| Column | Written by API? | Source |
|---|---|---|
| `xpmtnum` | Yes (create) | voucher generated as `PMT` + sequence via `generate_voucher(table="glpmt", column="xpmtnum")` |
| `zid`, `xcus`, `xshort`, `xemp`, `xname`, `xpaydate`, `xpayamt`, `xpaytype`, `xbankdetail`, `xpaystatus`, `xremarks` | Yes (create) | request body (`CreatePaymentRequest`) |
| `xpaystatus` | Yes (update) | `PUT /{xpmtnum}/update-status/` overwrites just this column |
| `ztime` | Yes | ORM column default (`datetime.utcnow`) at insert |
| `zutime` | Yes | ORM `onupdate` — refreshed on both create and the status-update |
| `xdornum`, `xtotamt` | No | exist on the live table, not referenced anywhere in this API |

Note `xpaydate` here is the **actual payment date being recorded** (a field the rep fills in when
logging the collection), not a promise — the promise/commitment date is `opdor.xdatepay` from the
Delivery Orders section above. Two different "payment date" concepts live in two different tables; don't
conflate them in a report.

Live data check: only 9 `glpmt` rows exist locally, and all 9 are still `xpaystatus = 'Send'` — i.e. in
this dataset, no collection submitted through this endpoint has yet been marked `'Received'`.

## Customer Receipts report & Customer Balance — the accounting side of "collection"

Two read-only endpoints sit on top of the general ledger (`glheader`/`gldetail`) rather than `glpmt`.
Neither has a corresponding write endpoint in this API — these vouchers are posted by some other
(back-office/desktop ERP) process, not the mobile app.

**`GET/POST /api/v1/customers/customer-rct/`** (`customer_rct_route.py` → `CustomerRCTController`) —
despite being a `POST`, it's a pure report with no side effects:
```sql
SELECT glheader.xdate, glheader.xvoucher, gldetail.xsp, prmst.xname, gldetail.xsub, cacus.xshort, gldetail.xprime
FROM glheader JOIN gldetail ... JOIN prmst ... JOIN cacus ...
WHERE glheader.xvoucher LIKE 'RCT-%' AND gldetail.xsp = :user_id ...
```
Reads: `glheader.xdate`/`xvoucher`, `gldetail.xsp`/`xsub`/`xprime`, plus `prmst.xname` (salesperson name)
and `cacus.xshort` (customer short name) for display. Locally there are **118,402** `RCT-`-prefixed
`glheader` rows — this is the real, accounting-confirmed receipt ledger, much larger than the 9 rows in
`glpmt`, confirming receipts are formally posted elsewhere and only surfaced here for read.

**`POST /api/v1/customers/customer-balance/`** (`customer_balance_route.py` → `CustomerBalanceController`)
reconstructs a running customer ledger, also entirely from `glheader`/`gldetail`, no writes:
- Opening balance: `SUM(gldetail.xprime)` for vouchers before the start date, excluding `%OB%` vouchers, filtered by `gldetail.xproj` (derived from `zid` via a hardcoded `{100000: "GI Corporation", 100001: "GULSHAN TRADING", 100005: "Zepto Chemicals"}` map — a business-ID-to-project lookup that lives in code, not a table).
- Payments in range: same `glheader`/`gldetail` join, filtered to voucher prefixes `RCT-`, `JV--`, `CRCT`, `STJV`, `BRCT`.
- Orders in range: same join, filtered to voucher prefix `INOP` (locally the single largest voucher-prefix group at ~574k rows across `INOP0`–`INOP4`).
- Running balance is computed in Python by merging and sorting both lists — not a DB computation.

## Related tables that this API does NOT write (context only)

`models/orders_model.py` also defines `Opord` (`opord`) and `Opodt` (`opodt`) — a *different* order
header/detail pair from both `opmob` and `opdor`/`opddt`. `sales_return_db_controller.py`
(`SalesReturnDBController.get_net_sales_with_all_returns`, called from **`GET
/api/v1/customers/customer-by-id/{zid}/{customer_id}`** in `customers_route.py`) reads `Opord`/`Opodt`
to sum gross sales, `Opcrn`/`Opcdt` to sum sales-return amounts, and — corrected from an earlier version
of this doc — **also reads `Imtemptrn`/`Imtemptdt`** (`imtemptrn`/`imtemptdt`) to sum damage-return
amounts (filtered to voucher-like codes `%RECA%`, `%SRE-%`, `%RECT-%`, `%DSR-%`). All four table pairs
are read-only here, feeding a customer "net sales vs. target, with an offer message" calculation — none
of them are written by this API.

So, counting every order-shaped table this codebase knows about, there are now **three** distinct
order records in play, and it's worth being precise about which is which:

| Table pair | Written by this API? | What it actually is |
|---|---|---|
| `opmob` | Yes (`/create-order`, `/create-bulk-order`) | The mobile app's own order log, one row per line item. `xstatusord` here is what "pending/confirmed/cancelled" filters on. |
| `opdor` / `opddt` | Partially — only `xdatedel`/`xdatepay` on `opdor`, via date-update | The "real" ERP delivery order (header/detail), including the promised payment/delivery dates. Rows themselves come from elsewhere. |
| `opord` / `opodt` | No — read-only, and only by the returns module | A third order table, read for return-eligibility totals; nothing in this API creates these rows either. |

None of the three are foreign-keyed to each other in a way this API relies on — there's no code path
here that joins `opmob` to `opdor`/`opddt` or to `opord`/`opodt` by voucher/order number. If your
Streamlit app needs to reconcile "the order the rep sent" with "the ERP's version of that order," you'll
need a join key that isn't visible in this API (likely `xordernum`, which is `NULL` on fresh `opmob`
rows and only gets backfilled by whatever external process also flips `xstatusord`).

## Practical notes for downstream use (Streamlit etc.)

- **One `opmob` row = one order line**, not one order. Group by `(zid, invoiceno, invoicesl)` or
  `(zid, xcus, ztime)` to reconstruct an "order" the way `get_orders_by_status()` does (it groups by
  `zid, invoiceno, xordernum, xdate, xcus, xcusname, xstatusord`).
- `xstatusord` values observed in code: `"New"`, `"Order Created"`, `"Not enough stock to create Order"`.
  Only `"New"` is ever set by this API; the other two are set by an out-of-band process.
- `xordernum` is `NULL` for freshly submitted mobile orders — don't rely on it to identify newly
  created orders; use `xsl`/`invoiceno`/`invoicesl` instead.
- There's no server-side validation that `xcus`/`xitem` exist in a customer/item master when the order
  is created — bad or stale codes from the app will land in `opmob` as-is.
- `xlinetotal` is stored as an **integer** in Postgres despite being typed as float end-to-end in the
  API — any cents/decimal portion sent by the app is silently truncated on insert.
- **"Promised payment date"** = `opdor.xdatepay`, set via `PUT /order/delivery-date-update/{zid}/{xdornum}`.
  Don't confuse it with `glpmt.xpaydate` (the *actual* collection date a rep logs when they record a
  payment) — they're unrelated columns on unrelated tables.
- **"Return registration"** = `POST /order/create-return`, writing `opcrn` (header) + `opcdt` (one row
  per item). Tax/discount/commission/cost columns on `opcdt` are always zeroed by this endpoint — don't
  use them for return-economics reporting; derive amounts from `xqty`/`xrate`/`xlineamt` instead.
- **"Collection registry"** is really two different things depending on which side you mean: the mobile
  app's own submission log is `glpmt` (writable, via `/customer-payment`), while the accounting-confirmed
  ledger of actually-posted receipts is `glheader`/`gldetail` filtered to `RCT-`-prefixed vouchers
  (read-only in this API, ~13,000x more rows locally than `glpmt` — confirming receipts get formally
  posted by a separate back-office process, not by this API).
- Several of these write endpoints (`create-return`, `customer-payment` create/update) have **no
  `@has_permission` decorator**, unlike `/create-order` which requires `order.create`. If you're using
  this map to reason about who can write what, that's a real asymmetry in the current code, not an
  oversight in this doc.
