# ERP integration guide (version 2.2: shop to ERP, with stock and order status coming back)

**Audience:** the Claude Code session that builds the ERP side (a scheduled importer on the ERP server, PostgreSQL database `da`), and the owner. It assumes the reader **knows the ERP's order entry (`opmob`) and its sales/return documents**; those table definitions are not in this repository, so this guide tells that session what data arrives and where it must end up, and the session maps it onto the real columns.
**Status:** 2026-10-26, version 2.2 (adds the **customer information**: section 6, the customer event in 3.1, the fuller customer block in 5.1, decisions D8 and D14). 2026-10-25, version 2.1: the owner had answered the first round of questions (section 12 shows which decisions are settled) and added two flows **back to the shop: stock and the ERP order status**. Version 1 of this file (two-way sync with webhooks on both sides) is replaced.

---

## 1. Direction and schedule (recommended)

**Everything an online customer or the shop's staff does flows one way: shop (Medusa) -> ERP** (customers and their addresses, orders, cancellations, returns, payments, parcels, new items, item edits, prices). Only **two things flow back, ERP -> shop: the consolidated stock of company 100007, and the ERP's status of each order** (informational; never shown to customers, never emailed). No conflict rules, no "which edit is newer", no loops: every fact has exactly one writer. The ERP is the accounting book; the shop is where the business happens, and the two processes are independent (section 5.0).

**The ERP pulls on a schedule; nothing on the ERP server is exposed to the internet.**

```
 SHOP (Medusa)                  ERP SERVER (Windows, PG 9.4)
 keeps an ordered feed          scheduled importer, every 5 min
 of everything for the ERP      (every call starts from the ERP)

           <---- HTTPS ----     1. GET the next events
                                2. write them to the ERP database
                                   (one transaction per event)
           ---- HTTPS ---->     3. POST acknowledgement
           ---- HTTPS ---->     4. POST order status changes
           ---- HTTPS ---->     5. POST stock of company 100007
                                   (changed SKUs, on a schedule)
```

Why this shape:
- You already run Python scripts with `psycopg2` on that server (the 2026-10 item upload). A scheduled importer needs **no inbound port, no public API, no webhook, no web framework** on a Windows Server 2016 / PostgreSQL 9.4 machine. The ERP only makes outbound HTTPS calls.
- The shop keeps the log, so if the ERP is off for a day, the next run simply catches up in order. Nothing is lost.
- One writer per fact means no conflicts to settle. A latency of up to 5 minutes between a sale and the ERP entry is the price; it is normal for accounting.
- The **payload contract is identical** if the owner later prefers "the shop pushes to an ERP endpoint" instead; only the transport changes. Section 11 says what would change.

**The two flows that come back (decided by the owner):** the shop cannot know the stock held by the three businesses behind company 100007, nor whether the ERP has accepted an order, so the ERP importer reports both. They are still **calls made by the ERP importer to the shop** (outbound from the ERP server), so nothing on the ERP is exposed. Section 8 (stock) and 5.5 (order status) define them.

**What is NOT synced back:** names, categories, descriptions and prices edited in the ERP. After an item exists, **the shop is the place to edit it** (name, category, description); an ERP-only edit of those is not read by the shop and is overwritten by the next shop edit. **Prices follow the "never lower" rule (D5):** the shop's price raises the ERP's, never lowers it.

---

## 2. What the shop (Medusa v2) holds

| Fact | Detail |
|---|---|
| Host | `https://medusa.utilitymart.shop` (staging today; the live shop gets its own host at go-live). |
| Money | BDT only, **VAT-inclusive** prices, numbers in taka (decimals allowed). |
| Product model | A **product** (what a shopper sees, "Padlock Fency") has **variants** (50 mm, 60 mm). **One variant = one ERP item.** The ERP groups variants with `caitem.xscode` ("VG" code). |
| `sku` | Variant SKU = the **source item code** = `caitem.xalias` on company 100007 (examples `SF000016`, `2050`). Unique. |
| `barcode` | Variant barcode = company 100007's own item code `caitem.xitem` (example `01-00010`). Unique. |
| Gaps today | 724 variants; 642 have a SKU, 640 a barcode (about 80 have none yet). An order line without a SKU **cannot be synced**: the shop holds that order back, shows it in an admin list, and sends it once the SKU is filled. |
| Customers | A shopper either has an **account** (signs in with email + password) or checks out as a **guest**; staff can also create a customer by hand in the Medusa admin (phone orders). Each has a name, a phone, an optional email and, for accounts, an **address book** (several delivery addresses, one default). Details and every field: section 6. |
| Orders | `display_id` 5, 6, ... plus a long id `order_01...`. Payment: cash on delivery now; SSLCOMMERZ (bKash, Nagad, cards) later. Delivery by Pathao (`consignment_id`, delivery fee). Customers may be guests. |
| Categories | The site's categories follow Shopify's open taxonomy (ids like `ha-9-4`), not `xgitem`. See section 7. |
| Stock | One stock location. `available = stocked - reserved`; web orders reserve stock until fulfilment. |

---

## 3. The shop feed (built by the shop session; the ERP session consumes it)

All calls below are made **by the ERP importer to the shop**. Base `https://medusa.utilitymart.shop/hooks/erp`.

**Authentication.** `Authorization: Bearer <ERP_FEED_KEY>` (a long random key the owner puts in the importer's config file on the ERP server; never in chat or in git) over HTTPS. Calls that carry a body (acknowledgements, stock snapshots) also send `X-UM-Timestamp` (Unix seconds, rejected if more than 5 minutes off) and `X-UM-Signature: sha256=<hex HMAC-SHA256 of "<timestamp>.<raw body>" with ERP_SHARED_SECRET>`. The shop may also restrict the key to the ERP server's public IP.

### 3.1 `GET /feed?after=<seq>&limit=100`

Returns the next events **in strict order**, oldest first. `seq` is an integer that only ever grows; `after=0` starts from the beginning. Repeating a call returns the same events.

```json
{ "events": [
    { "seq": 41, "id": "evt_01...", "type": "item.upserted",     "at": "2026-10-25T08:29:58Z", "version": 3, "data": { "...": "see 4.1" } },
    { "seq": 42, "id": "evt_01...", "type": "customer.upserted", "at": "2026-10-25T08:29:59Z", "version": 1, "data": { "...": "see 6.2" } },
    { "seq": 43, "id": "evt_01...", "type": "order.created",     "at": "2026-10-25T08:30:00Z", "version": 1, "data": { "...": "see 5.1" } },
    { "seq": 44, "id": "evt_01...", "type": "order.fulfilled",   "at": "2026-10-25T10:02:11Z", "version": 2, "data": { "...": "see 5.2" } } ],
  "last_seq": 44, "has_more": false, "server_time": "2026-10-25T10:05:00Z" }
```

Rules the importer relies on:
- **Items first.** Before the first event that mentions a SKU the ERP has not seen, the shop emits an `item.upserted` for it, so an order never references an unknown item.
- **Customers before their orders.** Before an order event, the shop emits a `customer.upserted` for the order's customer (account, guest or staff-created, section 6), so an order never references a customer the ERP has not seen. The order also carries its own copy of the customer and address (5.1), so the importer can work from the order alone if it prefers.
- **Every event is a complete snapshot** of the thing it describes at that moment (the whole order, the whole item), not a delta. The importer can therefore apply it as "make the ERP look like this", which is safe to repeat.
- `id` is unique and permanent. `version` grows per order/item/customer every time that record changes. Every response also carries `"environment": "staging"` or `"production"` so the importer always knows whether it is reading test data (the ERP side decides what to do with it; see 9.6b).

### 3.2 `POST /ack`

After processing (success **or** failure) the importer reports back, so the shop can show the sync state on each order and list problems:

```json
{ "results": [
    { "seq": 42, "id": "evt_01...", "status": "ok",     "erp_ref": { "customer_code": "C-004711" } },
    { "seq": 43, "id": "evt_01...", "status": "ok",     "erp_ref": { "order_no": "WO-000123", "document": "opmob", "id": "...", "customer_code": "C-004711" } },
    { "seq": 41, "id": "evt_01...", "status": "failed", "error": { "code": "unknown_group", "message": "No xgitem for ...", "retryable": false } } ] }
```
`status` is `ok` or `failed`; an `ok` result may carry `warnings` (for example `{ "code": "price_not_lowered", "erp_price": 650.00, "shop_price": 540.00, "sku": "2050" }`, or `{ "code": "field_truncated", "field": "address", "max": 100 }`, or `{ "code": "customer_matched_other_name", ... }` from 6.3), which the shop lists on its "ERP sync" page so staff can reconcile the prices. A `failed` event with `retryable: true` is re-offered by the feed on a later call with `?retry=1` (otherwise the shop shows it to staff). **A failed event must never block the events after it**: the importer records the failure, acknowledges it and continues (see 9.4).

### 3.3 `POST /stock`  (ERP -> shop)

The consolidated stock of company 100007 (section 8): `{ "as_of": "2026-10-25T10:00:00Z", "items": [ { "sku": "2050", "on_hand": 42 }, ... ], "full": false }`. Send **only the SKUs whose number changed** since the last call (`full: false`), and **every SKU once a night** (`full: true`; the shop then treats SKUs not listed as unknown, not zero). Up to 500 items per call. `on_hand` is a whole number >= 0.

### 3.3b `POST /order-status`  (ERP -> shop)

The ERP's view of each order, sent whenever it changes (section 5.5):
```json
{ "statuses": [ { "external_order_id": "order_01M5...", "display_id": 5, "erp_status": "confirmed", "at": "2026-10-25T11:20:00Z",
                  "refs": { "opmob": "WO-000123", "sale": "DO-004455", "credit_note": null } } ] }
```
The shop stores it and shows it to staff on the order. **It never emails or messages the customer and never changes the shop's own order or fulfilment state.**

### 3.4 `GET /health` returns `{ "ok": true, "server_time": "..." }` (use it first in each run).

**Status codes:** 200 OK; 401/403 bad key, signature or clock; 422 invalid body (message says why); 429 and 5xx: wait and retry on the next run. Calls are idempotent: repeating an ack, a status or a snapshot has no extra effect.

---

## 4. Items: shop -> `caitem` (company 100007)

### 4.1 Event `item.upserted` (data)

```json
{ "sku": "2050", "barcode": "01-00010", "name": "Pad Lock 50mm 6 Key", "variant_title": "50mm",
  "product_title": "Padlock Fency", "product_handle": "padlock-fency",
  "options": { "Size": "50mm", "Color": "Silver" },
  "category": { "shop_id": "ha-9-4", "name": "Locks & Latches", "path": "Hardware > Locks & Latches" },
  "price": 540.00, "active": true, "weight_g": null,
  "erp_source": "HMBR" }
```
`erp_source` is the business the item belongs to, chosen by staff when they create the product in the shop: **`GI`, `HMBR`, `ZEPTO`, `FIXIT` or `OUTSIDE`** (section 4.3); it is only used when the ERP has to **create** the item. `price` is the variant's **regular** price (not a temporary sale price). `barcode` is empty for an item created in the shop that the ERP has never seen. `name` is the ERP-style name (see decision D1: until the owner says otherwise, the variant's full name `product_title + " " + variant_title`). `active:false` means the shop has unpublished it.

### 4.2 What the importer writes (rules taken from the existing upload scripts, `update_100007_prices.py`)

Match first by `caitem.zid = '100007' AND caitem.xitem = barcode`; if no barcode, by `caitem.zid = '100007' AND caitem.xalias = sku`. Then:

| Case | Action |
|---|---|
| **Found** | `UPDATE caitem SET xdesc, xlong, xnote, xlinks, xscode?, xgitem?, zutime = now()` as below. **Never touch `xitem`, `xalias`, `xitemnew`, `xpricecat`.** |
| **Not found** | `INSERT` a new row (columns below) with the next item code (4.3) and also insert the `xcodes` row (4.4). Acknowledge with the new `xitem`; the shop stores it as the variant's `barcode`. |

Column mapping (all on `caitem`, key `(zid, xitem)`):

| `caitem` column | Value | Notes |
|---|---|---|
| `zid` | `'100007'` | HMBR Online Shop (D12). |
| `xitem` | existing barcode, or the next generated code (4.3) | the ERP's own item code (= the shop's `barcode`). |
| `xalias` | `sku` | the source item code (the item's own code in the business that holds it). For an item created in the shop, the shop's SKU is whatever staff entered; it must match the code in the source business. |
| `xitemnew` | source company id (`'100000'` GI, `'100001'` HMBR/Trading, `'100005'` Zepto; Fixit and Outside: **D13**) | disambiguates `xalias`. Insert only. |
| `xdesc` | `name` | length-limited (existing check: 250); truncate and log, do not fail. |
| `xlong` | `product_title` | the "base product" name (old content was discardable). |
| `xnote` | options as `Size:50mm,Color:Silver` (comma-separated pairs, blank pairs skipped) | overwritten. |
| `xlinks` | `product_handle` | the website product handle (links variants of one product). |
| `xscode` | the **VG code** of the product | If any `caitem` row in 100007 already has this `xlinks` handle, **reuse its `xscode`**; otherwise allocate the next `VG####` (highest existing + 1). A product with a single variant used to get a code too ("lone" items had none in the Master sheet; follow whatever the ERP uses today). |
| `xgitem` | group label | existing labels for 100007 rows: "Trading Item" (100001), "GI (Home Product) Item" (100000), "Zepto Item" (100005). For Fixit and Outside items: **D13**. Insert only. |
| `xpricecat` | `sku` (the source code) | links the item to the price tables (`opspprc.xpricecat`, `xcodes`). Insert only. |
| `xstdprice` | `price` | **Never lower (D5, as in the 2026-10 upload):** write the shop's regular price only if the row has **no price yet (NULL or 0) or the shop price is higher**. If the shop price is lower, leave the ERP price and return the warning `price_not_lowered` (3.2). It is the *regular* price; what a customer actually paid (a sale price, a discount) is on the order line. The owner will first bring the shop's prices up to the ERP's 100007 prices, so warnings should be rare. The importer's dry run must list, for the owner, every item where the shop price is **lower** than the ERP price (skipped) and every item where it is **higher** by more than 10% (would be raised). |
| `xcatful` | `'Auto'` | **Required on INSERT**: the column has a hidden CHECK (`NOT NULL AND <> ''`); every existing row uses the literal `'Auto'`. |
| `ztime`, `zutime` | `now()` | `ztime` on insert only. |

### 4.3 The next item code and the five businesses

Item codes on company 100007 look like `NN-NNNNN`. The prefix says which business the item belongs to. The existing upload script has: **`01` HMBR / Trading (company 100001), `02` GI (100000), `03` Zepto (100005)**; the owner adds **`04` Fixit** and **`05` Outside** (existing rows `04-00001`, `04-00002` already exist). For **04 and 05** the item row simply belongs to company 100007 itself (the owner: "mentioned within 100007 as far as the online shop is concerned"). The shop therefore asks staff for **`erp_source` = GI / HMBR / ZEPTO / FIXIT / OUTSIDE** whenever a product is created (a required field on the shop's product screen) and sends it in `item.upserted`:

| `erp_source` | prefix | `xitemnew` | `xgitem` label |
|---|---|---|---|
| `HMBR` | `01` | `'100001'` | "Trading Item" |
| `GI` | `02` | `'100000'` | "GI (Home Product) Item" |
| `ZEPTO` | `03` | `'100005'` | "Zepto Item" |
| `FIXIT` | `04` | `'100007'` (the item lives inside 100007 as far as the shop is concerned) | "Fixit Item" |
| `OUTSIDE` | `05` | `'100007'` (products bought from vendors when needed, not Fixit's) | "Outside Item" |

Allocation, as in the existing script: `SELECT substring(xitem from 1 for 2), max(xitem) FROM caitem WHERE zid = '100007' AND xitem ~ '^[0-9]{2}-[0-9]{5}$' GROUP BY 1`, take the number after the dash, add one, zero-pad to 5. Do the allocation **inside the same transaction as the insert and take an advisory lock** (`pg_advisory_xact_lock`) so two importer runs can never take the same code. Items that already exist in the ERP keep their code; `erp_source` is ignored for them.

### 4.4 The `xcodes` row ("Price Category" link)

For every item, ensure one row exists:

| `xcodes` column | Value |
|---|---|
| `zid` | `'100007'` |
| `xtype` | `'Price Category'` |
| `xcode` | the item's `xitem` (the barcode) |
| `xcodealt` | the item's `xalias` (the SKU) |
| `zactive` | `'1'` (`'0'` when the shop says `active:false`) |
| `ztime`, `zutime` | `now()` |

The earlier script **deleted all legacy `Price Category` rows for 100007 and re-inserted 619**; the importer must **not** delete; it only inserts a missing row or updates `zactive`. (Use SELECT-then-INSERT: PostgreSQL 9.4 has no `ON CONFLICT`.)

`opspprc` (`zid, xpricecat, xqty, xstdprice, xdisc, xdateeff, xdateexp, xqtypur, xtype`) holds quantity-break and special prices keyed by `xpricecat`; **the shop does not write it** unless the owner asks (D5).

---

## 5. Orders, payments, deliveries, returns: shop -> `opmob`

The order entry goes into **`opmob`** (and whatever line/detail table the ERP session knows it uses). Map the data below onto it. **Where a column choice is not obvious, the ERP session decides and writes the mapping into its hand-over note (section 10); do not guess silently.**

### 5.0 How the shop's order and the ERP's order relate (owner's rules)

The shop and the ERP each work on the order **independently**:
- The **shop** verifies the order, books the courier and ships it. The ERP does not drive any of that.
- The **ERP** receives the order as an entry in **`opmob`**. ERP staff check and verify it, and then **convert it to a sale** (the entry moves from `opmob` to **`opdor` / `ddt`**, the sale documents; the owner's wording). That conversion is the ERP's own step and has no effect on the shop's order processing. Its status is reported back to the shop (5.5), for staff only.
- If the shop **cancels** or the customer **returns** the order, the ERP follows:

| State of the document in the ERP | Shop cancels the order | Shop records a return (full or part) |
|---|---|---|
| **Still in `opmob`** (not yet converted) | the importer **deletes** the `opmob` entry (header and lines) | for a part return, change the entry's lines to what remains; a full return deletes it |
| **Converted** (in `opdor` / `ddt`) | create the matching **credit note in `opcrn` / `opcdt`** for the full order | create the **credit note in `opcrn` / `opcdt`** for the returned lines (and refunded delivery charge, if any) |

Every deletion and credit note is written by the importer, logged, and reported back through the order status (5.5). The importer must **check the document's state inside the transaction** (an ERP user may convert it a second before the cancel arrives) and then take the matching row of the table. If the ERP's own staff delete an entry that the shop has not cancelled, report `removed_in_erp` (5.5) and do not re-create it automatically.

### 5.1 Event `order.created` (and `order.updated`): data

```json
{ "external_order_id": "order_01M5PE3KDY0MR1574YEJCT9J4R", "display_id": 5, "placed_at": "2026-10-25T08:30:00Z",
  "status": "open",
  "customer": { "shop_id": "cus_01...", "kind": "guest", "first_name": "Rahim", "last_name": "Uddin", "name": "Rahim Uddin", "phone": "01712345678", "phone_verified": false,
                "email": "rahim@example.com", "company": null, "accepts_marketing": false, "erp_customer_code": null },
  "delivery_address": { "name": "Rahim Uddin", "phone": "01712345678", "line1": "House 5, Road 2", "line2": "", "area": "Mirpur 10", "zone": "Block A,B, Mirpur 6", "city": "Dhaka", "country": "BD", "postcode": "", "pathao": { "city_id": 1, "zone_id": 5, "area_id": 51 } },
  "billing_address": { "same_as_delivery": true },
  "payment": { "method": "cod", "status": "unpaid", "gateway": null, "transaction_id": null },
  "items": [ { "line": 1, "sku": "2050", "barcode": "01-00010", "erp_source": "HMBR", "name": "Pad Lock 50mm 6 Key", "quantity": 2, "unit_price": 370.00, "discount": 0.00, "line_total": 740.00 } ],
  "shipping": { "method": "Pathao home delivery", "charge": 100.00, "free_delivery_applied": false },
  "discounts": [ { "code": "TEST10", "amount": 0.00 } ],
  "totals": { "items": 740.00, "discount": 0.00, "shipping": 100.00, "total": 840.00, "currency": "BDT", "prices_include_vat": true },
  "notes": "Call at the gate", "channel": "web" }
```

| Shop field | What the ERP needs it for |
|---|---|
| `external_order_id`, `display_id` | The **idempotency key** and the number staff recognise. Store `external_order_id` in a column that can be looked up (a reference/remarks column, or a new indexed column or small side table `um_order_map(external_order_id, erp_key)`). Show `display_id` in the document reference (for example `WEB-5`). A second event for the same `external_order_id` **updates** the same ERP document, never creates another. |
| `customer`, `delivery_address`, `billing_address` | The **customer snapshot at the time of the order** (who it belongs to) and where it goes. Fields and rules: section 6 (6.1 lists every field, 6.3 how the ERP matches or creates its customer). `customer.kind` is `account`, `guest` or `staff_created`; `shop_id` can be `null` for a guest who gave no email. **The order keeps its own copy**: the name and address written on the ERP document are the ones in this event and never change when the customer later edits their account (6.4). Phones arrive as `01XXXXXXXXX` (6.1). The shop collects **one** address at checkout, so billing is the delivery address (`same_as_delivery:true`). Names and addresses may contain **Bangla**: check the database encoding (`SHOW server_encoding;`). If it is not UTF8 the importer must transliterate or store only what fits, and say so in its note. |
| `notes` | The customer's delivery note ("call at the gate"); store with the delivery address on the document. |
| `items[]` | Order lines. Resolve each line's ERP item by `caitem.xitem = barcode` (or `xalias = sku`) on company 100007. An unknown item is a `failed` event (`unknown_item`), not a skipped line. Each line carries `erp_source`; lines with **`OUTSIDE`** are goods that must be **bought from a vendor** for this order (no stock exists), so the ERP side may want to flag them on the entry for purchasing. |
| `unit_price`, `discount`, `line_total`, `totals` | Prices **include VAT**. **Write the amounts exactly as sent; do no tax maths** (the owner: VAT is adjusted on the accounts side inside the ERP). Keep `totals.total` as the amount the customer owes; check that lines + shipping - discount = total and report `total_mismatch` otherwise. |
| `payment.method` | `cod` (cash collected by the courier later) or `online` (already paid through the gateway: `gateway`, `transaction_id`). Maps to the ERP's payment/receipt type. |
| `shipping` | Delivery charge: a charge line, or the ERP's freight field, as `opmob` models it. `free_delivery_applied:true` means charge 0 but the shop still pays the courier. |
| `status` | `open` for a new order. |

### 5.2 Later events for the same order

All carry the full order snapshot plus a small block saying what happened. The importer applies each as "bring the ERP document to this state".

| `type` | Extra data | The ERP should |
|---|---|---|
| `order.updated` | the new snapshot (address, notes, lines, totals) | update the `opmob` entry while it is still there; if it has already been converted to a sale, **do not change the sale**: acknowledge `failed` with code `locked` so staff can decide (the shop shows it) |
| `payment.captured` | `{ amount, method, transaction_id, captured_at }` | record the receipt (COD: cash collected; online: gateway reference) |
| `order.fulfilled` | `{ consignments: [{ id, courier: "pathao", fee_estimate }], items: [{ sku, quantity }] }` | **information only**: keep the consignment ids and the delivery fee estimate on the document (remarks/reference). The conversion to a sale is the ERP's own step (5.0); do not convert because this arrived |
| `order.delivered` | `{ at }` | mark delivered |
| `order.canceled` | `{ reason, at }` | delete the `opmob` entry, or credit-note it if already converted (5.0) |
| `order.refunded` | `{ amount, method, reason, at }` | record the refund (see 5.3) |

### 5.3 Returns: events `return.created`, `return.received`, `return.refunded`

```json
{ "external_return_id": "ret_01...", "external_order_id": "order_01...", "status": "received",
  "requested_at": "...", "received_at": "...", "reason": "damaged",
  "items": [ { "sku": "2050", "quantity": 1, "restock": true } ],
  "refund": { "amount": 370.00, "shipping_refunded": 0.00, "method": "cash" } }
```
Follow the table in 5.0: edit or delete the `opmob` entry if it is still there, otherwise write the credit note into **`opcrn` / `opcdt`** against the converted sale. **Whether the returned goods go back into stock is the ERP's own handling of the credit note**; the shop sends `restock` as information. Idempotent by `external_return_id`.

### 5.4 Stock edits in the shop admin are not sent

Stock for company 100007 is the **consolidated stock of the three businesses**, owned by the ERP (section 8). A stock number typed into the shop admin is overwritten by the next ERP snapshot (except `OUTSIDE` items, which are untracked) and is **not** sent to the ERP (the shop admin shows the stock fields as read-only information once the sync is on).

### 5.5 Order status, ERP -> shop (`POST /order-status`)

The importer keeps, per order, the state of the document it created and reports changes. Vocabulary (the shop shows these to staff on the order, in a box "ERP", and nowhere else):

| `erp_status` | Meaning |
|---|---|
| `entered` | in `opmob`, waiting for ERP staff (`refs.opmob` = the entry number) |
| `confirmed` | verified and converted to a sale in `opdor` / `ddt` (`refs.sale` = the document number) |
| `credited` | a credit note exists in `opcrn` / `opcdt` (`refs.credit_note`) |
| `canceled` | removed because the shop canceled it while still in `opmob` |
| `removed_in_erp` | the `opmob` entry disappeared without a shop cancellation (needs a human) |
| `failed` | the importer could not write it (reason in the acknowledgement) |

Send a status when it **changes** (poll the document tables on each run for the orders still open), plus a full resend nightly. **Never mean "tell the customer":** it must not trigger any message.

---

## 6. Customers (D8, D14)

The ERP session must be told **who the customer is** on every order, and the owner wants the customer information to travel in full. So the shop sends it two ways: a **complete copy inside every order** (5.1) and a **customer event of its own** (6.2) when a customer is created or changes. Whether the ERP keeps a customer master or just writes the details on the order document is decided in D8 (6.6); the data is the same either way.

### 6.1 What the shop knows about a customer

| Field | Meaning and rules |
|---|---|
| `shop_id` | The shop's permanent customer id (`cus_01...`). Stored in the ERP as a reference. `null` only for a guest who gave neither email nor account. |
| `kind` | `account` (registered, signs in with email + password), `guest` (checked out without an account) or `staff_created` (made by staff in the Medusa admin, for phone or counter orders). |
| `first_name`, `last_name`, `name` | As typed by the customer (any script, **Bangla possible**). `name` is the two joined with a space. `last_name` may be empty (one-word names). |
| `phone` | Bangladeshi mobile. The shop stores what the customer typed (spaces and dashes removed; a `+88` or `88` prefix is allowed), and **the feed converts it to the 11-digit form `01XXXXXXXXX` before sending**, so the same person always arrives with the same number (the importer normalises again as a safeguard, 6.3). **Always present** for orders; it is the customer's main contact and the best matching key (6.3). |
| `phone_verified` | `true` once the customer has confirmed the number with a one-time code (planned: required before a customer's first cash-on-delivery order). `false` until then. A verified phone is a stronger match than an unverified one. |
| `email` | **Optional** (the owner's decision: order updates cannot rely on email alone). `null` when not given. **Never use email as the only key.** |
| `company` | Business name for a company customer. **Not collected yet** (always `null` today); reserved so the feed does not change when it is. |
| `accepts_marketing` | Whether the customer agreed to promotional messages. The ERP must not use the contact details for promotion when this is `false`; order and delivery messages are separate. |
| `created_at`, `updated_at` | When the customer first appeared in the shop and was last edited. |
| `erp_customer_code` | The customer's code in the ERP once the importer has reported it (`erp_ref.customer_code`, 3.2); `null` before. The shop stores it so staff can see it and so every later event carries it back. |
| `addresses[]` | The address book (accounts and staff-created customers; a guest has just the order's address). Each: `shop_id` (`caddr_01...`), `label` (for example "Home", "Office"), `is_default` (true for the delivery address pre-filled at checkout), `name`, `phone` (the receiver, can differ from the customer's), `line1` (house, road, landmark), `line2`, `area`, `zone`, `city`, `country` (`BD`), `postcode`, `pathao` (`city_id`, `zone_id`, `area_id`, which are Pathao's ids; **informational, never a key in the ERP**). `city` is what the shop calls the district. |

What the shop does **not** hold, so the ERP must not expect it: national ID, date of birth, credit limit, customer type, price category, payment terms, tax or BIN number, opening balance. Those are ERP-side attributes (6.3).

### 6.2 Event `customer.upserted` (data)

Sent when a customer is created and every time the customer or an address changes (new address, edited name or phone, changed default, address deleted, phone verified, marketing choice changed). A **complete snapshot**, like every event (3.1).

```json
{ "customer": { "shop_id": "cus_01...", "kind": "account", "first_name": "Rahim", "last_name": "Uddin", "name": "Rahim Uddin",
                "phone": "01712345678", "phone_verified": true, "email": "rahim@example.com", "company": null, "accepts_marketing": true,
                "created_at": "2026-10-20T06:12:00Z", "updated_at": "2026-10-25T08:29:59Z", "erp_customer_code": null,
                "addresses": [
                  { "shop_id": "caddr_01...", "label": "Home", "is_default": true, "name": "Rahim Uddin", "phone": "01712345678",
                    "line1": "House 5, Road 2", "line2": "", "area": "Mirpur 10", "zone": "Block A,B, Mirpur 6", "city": "Dhaka", "country": "BD", "postcode": "",
                    "pathao": { "city_id": 1, "zone_id": 5, "area_id": 51 } } ] } }
```

### 6.3 What the ERP does with it (matching rules, whichever way D8 goes)

1. **Match by phone.** Normalise to `01XXXXXXXXX` and look the customer up by phone (on the ERP's customer master, or among the importer's own `um_customer_map` rows). **Email is not a key** (it can be missing, shared or mistyped). Do not match by name.
2. **Found:** link to it. Do **not** overwrite fields staff maintain in the ERP (name spelling, credit limit, customer type, price category, terms, anything not in 6.1). Fill an ERP field from the shop only when it is empty. If the shop name differs from the ERP name for the same phone, keep the ERP's and return the warning `customer_matched_other_name` (with both names), so staff can look; this is common (a family sharing one phone).
3. **Not found:** create it (if D8 chose a master) with name, phone, email, address and the shop reference, in whatever columns and with whatever required defaults (customer type "online", price category) the ERP session finds right, and write that into its mapping note. If D8 chose "no master", keep the details on the document only.
4. **Acknowledge with `erp_ref.customer_code`** (3.2) so the shop can store it.
5. **Addresses:** the ERP keeps the default address on the customer (or all of them if its customer table has a place for several). The address written on a **document** is always the one in that order's event (6.4).
6. **Lengths and characters:** the ERP session writes down the maximum length of each customer and address column in its mapping note; a longer value is truncated and the ack carries `field_truncated` (field name and max). Never fail a whole order for a long address.
7. A customer the shop sends twice (replay, or an edit with nothing new for the ERP) changes nothing the second time.

### 6.4 Edits after an order: documents never change

The ERP document keeps the name, phone and address **exactly as they were on the order** (accounting record). When the customer later edits their account or address, the shop sends a new `customer.upserted`; that updates the ERP's customer record (6.3, only empty fields and the default address) but **never an existing `opmob` entry or sale**. A change to an **order's** delivery address is a different thing: it arrives as `order.updated` (5.2) and follows its rules (applied while the entry is still in `opmob`, `locked` once it is a sale).

### 6.5 Customer removal (D14)

A customer can ask the shop to delete their account. The shop then sends `customer.erased` with `{ shop_id, at }` and blanks the personal data on its side. The ERP's accounting documents stay as they are (they are business records and must be kept); on the ERP's customer record the importer replaces name, phone, email and address with a neutral text (for example "Deleted customer") **only if no unpaid balance or open document exists**, otherwise it acknowledges `failed` with code `customer_in_use` and staff decide. Default; owner may change it (D14).

### 6.6 Customer master or generic account (D8)

The ERP session first reports which of these `opmob` needs (section 10, item 3):
- **A customer code from a master table** -> **recommended: one master record per customer, matched by phone (6.3).** It gives the owner repeat-customer history, the customer's addresses and a ledger per customer.
- **Free text on the entry** (name, phone, address columns) -> no master is required; the importer writes the details from 6.1 on the document, and `um_customer_map` (9.5) is enough to recognise repeat customers.
- **Fallback if the ERP insists on a code but no per-customer master is wanted:** one generic "Online customer" account (the ERP session names it) with the real name, phone and address on the document. The customer data still arrives in full; it is simply not turned into master records.

---

## 7. Categories

The shop's categories (195, the Shopify taxonomy) and the ERP's groups (`caitem.xgitem`, also the finer ERP categories) are different lists. Direction is shop -> ERP only, so the shop sends, for every item, both its **taxonomy id and readable path** (4.1); the importer decides what to store:
- Minimum: keep `xgitem` as it is for existing items (the 2026-10 upload set it from the source company) and put the shop's category path **in `xnote` or a new column** only if the owner wants reporting by website category (D3).
- A mapping table `um_category_map(shop_category_id, erp_group)` on the ERP side is only needed if the owner wants the ERP group chosen automatically from the shop category; the importer then reads it and reports unmapped categories in its log.

---

## 8. Stock, ERP -> shop (D4: settled by the owner)

Company 100007 holds **no stock of its own**. Its items are "bought" from the businesses behind them, and those hold the stock. So the stock the shop must show for an item is the stock of its **source item** in its source business:

| Items | Where the stock is |
|---|---|
| `01` HMBR (100001), `02` GI (100000), `03` Zepto (100005) | in those companies of this ERP (`caitem.xitemnew` says which) |
| **`04` Fixit** | in **Fixit's own, separate ERP**. The consolidated view must **pull it from there** (the owner will work out how on the ERP side; this guide only requires that Fixit's stock ends up in the same view, one number per SKU). |
| **`05` Outside** | **nowhere: no stock exists.** These are products the business buys **directly from shops/vendors when needed** (owner). The ERP sends **no stock** for them. The shop does not track their quantity: variants whose product has `erp_source = OUTSIDE` are set to **"not tracked" (always orderable)** in the shop, and staff cancel an order if the vendor cannot supply. |

**The ERP session builds a consolidated stock view for zid `100007`** (a materialized view or table refreshed on a schedule, for example `mv_stock_100007`), one row per 100007 item:

| column | meaning |
|---|---|
| `sku` | `caitem.xalias` of the 100007 item |
| `xitem` | the 100007 item code (the shop's barcode) |
| `source_zid` | `caitem.xitemnew` for 01/02/03; a label such as `FIXIT_ERP` for 04; empty for 05 (the business or system that holds the stock) |
| `on_hand` | the **physical** stock of source item `xalias` in company `source_zid` (as the owner's current-stock report shows it) |
| `as_of` | when the number was computed |

Rules:
- **Validate the query against reality.** In 2026-09 `SUM(stockqty)` over `mv_stock_movement` disagreed with the owner's own stock report (`Current_stock_100001.xlsx`) for 1,161 of 2,355 Trading items (the view being higher in 1,113; item `1843`: view 4,333, real 0), and the view holds 4 rows with corrupt future dates (one in year 2102). Find the computation that reproduces the ERP's **own stock report**, check it on at least 20 items (zero-stock and large-stock items included), and only then build the view on it. Do not build on `final_items_view`.
- **`on_hand` is physical stock, not net of open web orders in `opmob`.** The shop subtracts its own not-yet-confirmed orders itself (below). If the ERP view cannot avoid netting them, tell the shop session so it does not subtract twice.
- Several 100007 items can share one source item; send each SKU its own row. `05` Outside items have no row in the view at all.
- **Push per SKU on a schedule:** the importer compares the view with what it last sent and calls `POST /stock` with **only changed SKUs**, as often as the view is refreshed (suggested every 15 minutes), and with every SKU once a night (`full: true`). If the view is older than 2 hours the importer says so in its log; the shop shows a banner when the last snapshot is older than 2 hours.

**What the shop does with it** (shop session): `stocked quantity := on_hand - units the shop has already shipped but the ERP has not yet confirmed as a sale`, so a parcel that left the shop is not counted as still in stock during the gap between "shipped" and "ERP converts to a sale"; units of orders not yet shipped stay reserved as today. When the ERP reports `confirmed` for an order (5.5), its units stop being subtracted separately because the ERP number has dropped. Shop-side manual stock edits are overwritten and not sent to the ERP. For `OUTSIDE` items the shop does not track stock at all (always orderable).

---

## 9. The importer: what the ERP session builds

**9.1 Form.** One Python 3 script (`erp_import.py`) using `requests` (or `urllib`) and `psycopg2`, configured by a small file next to it that is **not committed** (`ERP_BASE_URL`, `ERP_FEED_KEY`, `ERP_SHARED_SECRET`, database connection). Run by Windows Task Scheduler every **5 minutes**, with a lock so two runs never overlap. Target **PostgreSQL 9.4** and the Python version actually installed on the server (check; the earlier scripts avoided `ON CONFLICT` and non-stdlib extras where possible).

**9.2 Safe by default** (same convention as the earlier scripts): `--dry-run` is the default and rolls everything back; `--commit` writes. Logs to a dated file; one line per event with `seq`, type, key and result.

**9.3 One run:**
1. `GET /health`. Stop quietly if the shop is unreachable.
2. Read the saved cursor (`um_sync_state.last_seq`, a one-row table the script creates).
3. `GET /feed?after=<cursor>&limit=100`; loop while `has_more`.
4. For each event, in order, **in its own transaction**: look up `um_processed_events(event_id)`; if present, skip (already done); otherwise apply (sections 4 to 5), insert the event into `um_processed_events` with the result, commit, advance the cursor.
5. `POST /ack` with the results of the batch.
6. For orders not yet in a final state, read the state of their documents (`opmob`, `opdor` / `ddt`, `opcrn` / `opcdt`) and `POST /order-status` for the ones that changed (5.5).
7. Compare the consolidated stock view with the last values sent and `POST /stock` the changed SKUs (section 8); once a night send everything.

**9.4 Failures.** A bad event (unknown item, constraint violation, total mismatch, locked document) is **recorded as failed with a clear code and message, acknowledged, and skipped**; the cursor still advances so one bad order never blocks the rest. Network errors stop the run and the next run resumes from the same cursor. Never leave a half-written order: each event is a single transaction.

**9.5 Tables the importer creates** (small, in `da`): `um_sync_state(id, last_seq, updated_at)`, `um_processed_events(event_id primary key, seq, type, key, result, processed_at)`, optionally `um_order_map(external_order_id primary key, erp_key, version, updated_at)`, and `um_customer_map(shop_customer_id primary key, phone, erp_customer_code, version, updated_at)` (phone indexed; it is what 6.3 matches on).

**9.6b Where it runs while it is built.** The owner will run and test everything **locally on the Mac** (the local copy of `da`) against the staging shop's feed before the live server is touched. Test orders in `opmob` can simply be deleted; the ERP session decides how to mark and clean them. The importer should print `environment` from the feed at the start of every run and must never be started against the live database with a staging feed by accident (suggestion: a config switch `ALLOW_STAGING_FEED=false` on the live server).

**9.6 Database role.** A dedicated role with INSERT/UPDATE on the order-entry tables, the customer master table (if D8 uses one), `caitem`, `xcodes` and the `um_*` tables, SELECT on the stock ledger; nothing else.

---

## 10. What the ERP session hands back

1. `erp_import.py` + its config template + a one-page **run guide** (how to install, schedule, dry-run, commit, read the log, roll back).
2. The **mapping note**: for every field in sections 4 to 6, the exact table and column it was written to (especially `opmob` and its line table), **how the customer is stored** (6.6: customer code from a master, free text, or the generic account), the exact customer-master table and columns, the required defaults it had to choose, the maximum length of each customer and address column, and anything it could not follow.
3. The **database encoding** answer, the **definition of the consolidated stock view** (the query, how it was validated against the stock report, its refresh schedule), and, if used, the **name of the generic online customer**.
4. A **test database** (a copy, not live) and a **test run log** showing: a new item, an order with 3 lines, a cancellation, a fulfilment, a return with restock, a **new customer, a returning customer (same phone, no duplicate) and a customer who changed their address** (6.3, 6.4), and the same events replayed twice (no duplicates).
5. Written confirmation of which of `D1` to `D14` it assumed.

**Acceptance (both sessions, before anything touches live accounting):**
- Place a test order on staging -> within one run the ERP has exactly one `opmob` entry with the right lines, totals (VAT-inclusive, untouched), shipping, **the customer's name, phone and delivery address** and the shop's order number in its reference, and the shop shows ERP status `entered`.
- Convert that entry to a sale on the ERP side -> within one run the shop shows `confirmed` with the sale number, and no customer message was sent.
- Cancel an order that is still in `opmob` -> the entry is deleted and the shop shows `canceled`. Cancel (or return) an order already converted -> a credit note is in `opcrn` / `opcdt` and the shop shows `credited`.
- Customers: a second order from the same phone number (even with another name or no email) links to the **same** ERP customer, no duplicate; the ack carries `customer_code` and the shop shows it on the customer. Edit the account's address in the shop -> the ERP customer's default address follows, and the existing `opmob` entry and sale are **unchanged**. A Bangla name and address arrive intact (or the note says what was transliterated). A guest with no email imports fine. A 300-character address imports with `field_truncated`, not as a failed order.
- Replay the same feed twice -> still one document. Change the cursor back by 50 -> no duplicates.
- Fulfil it in the shop (parcel off on staging) -> the ERP document gets the consignment reference and is **not** converted by that.
- Stop the importer for an hour, place five orders -> the next run brings all five in order.
- Feed an order with an unknown SKU -> that one is `failed` in the shop's admin list, the other orders after it still import.
- Stock: the numbers equal the ERP's own current-stock report for a sample of 20 items (zero-stock and large-stock items included). Change source stock in the ERP -> the shop's available quantity follows within one view refresh; place an order on the shop -> available drops at once, and does not drop a second time when the ERP later converts it.

---

## 11. If the owner prefers "the shop pushes to an ERP endpoint"

The payloads in sections 4 to 5 stay exactly as they are (stock and order status would then be sent by the shop's own request to the ERP, or kept as importer calls). The shop would `POST` each event (with `X-UM-Timestamp`, `X-UM-Event-Id`, `X-UM-Signature`, 202/200 means done, 422 means do not retry, 5xx means retry with back-off), and the ERP would need a small always-on HTTPS service reachable from the shop server (public address with an allow-list, or a tunnel). That is more to build and secure on a Windows 2016 / PostgreSQL 9.4 server, which is why the scheduled importer is recommended. Decide before the ERP session starts coding.

---

## 12. Decisions (status after the owner's answers of 2026-10-25)

| # | Decision | Status / default |
|---|---|---|
| **D1** | Item **name** sent to the ERP (`xdesc`) | **Settled:** the full variant name (`product title + variant title`, for example "Padlock Fency 50mm"). The shop's curated title is never overwritten from the ERP. |
| **D2** | Direction | **Settled:** shop -> ERP for orders, returns, payments, parcels, items, prices; **ERP -> shop for stock and order status only**. |
| **D3** | Category reporting in the ERP | **Open, default:** keep `xgitem` as is; the shop category path is sent for information only. |
| **D4** | Stock | **Settled:** a consolidated stock view for 100007 built by the ERP session from the businesses' stock, pushed per SKU on a schedule (section 8). |
| **D5** | Prices | **Settled:** the shop's regular price is written to `xstdprice` with the **"never lower" rule kept** (only when the ERP has none or a lower one). `opspprc` untouched. The owner first updates the shop's prices to the ERP's 100007 prices. |
| **D6** | When an order becomes a sale | **Settled:** on the ERP side, when ERP staff verify the `opmob` entry and convert it to a sale; independent of the shop's own processing; status reported back (5.0, 5.5). |
| **D7** | VAT | **Settled:** amounts are VAT-inclusive and written untouched; the ERP's accounts side adjusts VAT. |
| **D8** | Customers | **Updated 2026-10-26** (the owner pointed out the guide lacked customer information): the feed now carries the **full customer and address book** (section 6). **Open, recommended default:** one ERP customer record per customer, matched by phone (6.3), if `opmob` takes a customer code; free text on the entry if it does not; the single generic "Online customer" account only as a fallback (6.6). The ERP session reports what `opmob` needs. |
| **D9** | Reachability and testing | **Settled:** built and tested locally on the owner's Mac against the staging feed; test entries in `opmob` are deleted by the ERP side; the ERP session handles test orders and cleanup. *Still needed before go-live:* the ERP server may call out to the live shop host. |
| **D10** | Items first seen in the ERP | **Default:** ignored by the shop; new products are created in the shop. |
| **D11** | Deletions | **Default:** the shop never deletes ERP items (an unpublished product becomes `zactive = '0'` in `xcodes`). Orders follow 5.0 (delete in `opmob` / credit note after conversion). |
| **D12** | Company | **Settled:** zid `100007`. |
| **D13** | Item source and codes | **Settled:** `erp_source` = GI (`02`), HMBR (`01`), ZEPTO (`03`), FIXIT (`04`), OUTSIDE (`05`); 04 and 05 items belong to company 100007 (`xitemnew = '100007'`), `xgitem` labels "Fixit Item" and "Outside Item" (owner: fine for now). Fixit's stock comes from Fixit's own ERP (section 8). Outside items are bought from vendors when needed: no stock, untracked in the shop. |
| **D14** | Customer removal | **Default:** `customer.erased` blanks the ERP customer's personal fields unless a balance or open document exists; accounting documents are never altered (6.5). |

**What the shop session builds in return:** the feed, acknowledgement, order-status and stock receivers (section 3); an outbox that guarantees nothing is lost; subscribers on customers and their addresses (one `customer.upserted` per change, `customer.erased` on removal), orders, payments, fulfilments, returns and item edits; the customer's ERP code (`erp_customer_code`) kept on the customer and shown in the Medusa customer page next to the order's "ERP" box; a required **`erp_source`** choice (GI / HMBR / ZEPTO / FIXIT / OUTSIDE) on the product screen (choosing OUTSIDE switches the product's variants to "stock not tracked"); an **"ERP" box on each order** (status from 5.5, references, sync result) and an admin page "ERP sync" (last successful run, orders waiting, failed events with the ERP's message, items without a SKU, last stock snapshot age, a "mark for retry" button); the stock rule of section 8. It is switched off (`ERP_SYNC_ENABLED=false`) until the owner turns it on.
