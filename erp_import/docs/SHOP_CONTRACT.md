# What the ERP importer expects from the shop (`/hooks/erp`)

For the shop session. Draft of 2026-11-02, written from the working ERP side. `tests/fake_feed.py` implements every call below
(it is the executable version of this page: same auth, same checks, same errors), so you can run it and compare.
The base design is `ERP_INTEGRATION_SPEC.md` (same folder) v2.2; **this page lists only what differs or is new.** Where they disagree, this page wins.

Base `https://medusa.utilitymart.shop/hooks/erp`. Every call is made BY the ERP importer (outbound only).

## Auth (unchanged)
`Authorization: Bearer <ERP_FEED_KEY>`. Calls with a body also send `X-UM-Timestamp` (unix seconds, reject if off by more than 5 minutes) and
`X-UM-Signature: sha256=<hex HMAC-SHA256 of "<timestamp>.<raw body>" with ERP_SHARED_SECRET>`. 401/403 on a bad key, signature or clock; 422 with `{"error": "<why>"}` on a bad body.

## Calls

| Call | Direction | Status |
|---|---|---|
| `GET /health` -> `{"ok":true,"server_time":"...","environment":"live"}` | ERP -> shop | as spec. The shop is live, so `environment` is `live` (the importer refuses `staging` unless told otherwise) |
| `GET /feed?after=<seq>&limit=100` | ERP -> shop | as spec |
| `POST /ack` `{"results":[{"seq","id","status":"ok"\|"failed","erp_ref":{},"warnings":[],"error":{"code","message","retryable"}}]}` | ERP -> shop | as spec |
| `POST /order-status` `{"statuses":[{"external_order_id","display_id","erp_status","at","refs":{}}]}` | ERP -> shop | as spec; `erp_status` is one of `entered`, `order_created`, `confirmed`, `credited`, `canceled`, `removed_in_erp` |
| `POST /stock` `{"as_of","full","items":[{"sku","on_hand"}]}` | ERP -> shop | as spec (whole number >= 0, at most 500 items). Sent every run for changed SKUs; `full: true` once a day when the list fits one call |
| **`POST /products`** | ERP -> shop | **new, see below** |

## New: `POST /products`
Sent once a day at a fixed time (and on demand). The ERP is the master of these fields; the shop applies them.
```json
{ "as_of": "2026-11-02T09:00:00Z", "full": false,
  "items": [ { "sku": "0718", "barcode": "01-00135", "price": 80.0, "cost_price": 57.0, "item_group": "Trading Item", "on_hand": 4190 } ] }
```
At most 500 items per call; `full: true` only when the call carries every followed item (the ERP chunks larger lists and sends `false`). Match the variant by **`sku`**. Every field except `sku` is optional.

| Field | The shop does |
|---|---|
| `barcode` | set the variant barcode (the ERP item code). This is how shifted or missing barcodes are fixed. If the barcode already belongs to another variant, skip that item and report it in `conflicts` |
| `price` | set the variant's BDT price (VAT inclusive, taka). The ERP price overwrites the shop price |
| `cost_price` | store as `metadata.cost_price` on the variant (the field the Fixit upload already uses) |
| `item_group` | store as `metadata.erp_item_group` on the variant (Fixit's product type already carries it; leave product types alone) |
| `on_hand` | same meaning as `/stock` (physical stock; the shop still subtracts its own reserved and shipped units) |

Response `200`: `{"ok": true, "applied": 118, "unknown_skus": ["..."], "conflicts": [{"sku": "...", "reason": "barcode in use"}]}`. Unknown SKUs are not an error. `422` only when the body itself is malformed (what `fake_feed.py` checks: `items` is a list of at most 500, each has a `sku`, `price` and `cost_price` are numbers >= 0, `on_hand` is a whole number >= 0). Repeating a call has no extra effect.

## What changes in the events the ERP reads

1. **Customers are no longer stored in the ERP.** All online sales go to one ERP customer. Keep sending `customer.upserted` / `customer.erased` or stop; the ERP acknowledges them and ignores them. The ERP does not check name, phone or address any more: that validation is the shop's. **Shopper details can be left out of every event.**
2. **`display_id` must be a whole number** on every order event. It becomes the ERP order number `UM-1000001` and the key that links every ERP document back to the order. An order without one fails with `missing_display_id`.
3. **`order.created` / `order.updated`** (shape as in the spec 5.1) needs, per order: `external_order_id`, `display_id`, `placed_at`, `items[]` (`sku`, `barcode`, `quantity`, `unit_price`, `line_total`, `name`), `shipping.charge`, `totals` (`items`, `discount`, `shipping`, `total`), `payment` (`method` = `cod` or the gateway, `gateway`, `transaction_id` when known), `notes`. `unit_price` is the price **before** discount; the order discount is in `totals.discount` (the ERP turns it into a percent). The importer checks `sum(line_total) = totals.items` and `items + shipping - discount = total` (within 1 taka) and fails the order with `total_mismatch` otherwise.
4. **Payment trail.** `payment.captured` must carry `transaction_id` (the SSLCommerz id for online payments). The ERP writes it onto the order's ERP line rows. An online order created before payment is stored as `ONLINE` until this event arrives; cash on delivery is stored as `COD`.
5. **Courier trail.** `order.fulfilled` must carry `consignments: [{"id": "<Pathao consignment id>", ...}]`. The ids are written onto the order's ERP line rows.
6. **Items.** `item.upserted` (spec 4.1) is unchanged, with two rules: the importer matches by **SKU first**, then barcode; the ERP's SKU-to-barcode pairing wins when they disagree (it warns, and `/products` then corrects the shop). `erp_source` must be one of `HMBR`, `GI`, `ZEPTO`, `FIXIT`, `OUTSIDE`. The shop price is used only as the starting price of an item the ERP has never seen; after that the ERP price flows the other way through `/products`.
7. **Returns and refunds** (`order.refunded`, `return.*`) are acknowledged as `failed` with code `return_manual`; staff handle them by hand in the ERP. Nothing for the shop to change.
8. **Bundles are sent as their parts (decided 2026-11-04).** The ERP has no bundle items, but every part of a bundle is an ordinary ERP item. A customer who buys a bundle gets all its parts, so the ERP must see each part: its stock goes down and the daily purchase order buys it from its source company. So an order event carries **one line per part, never a line for the bundle itself**:
   * `sku` / `barcode` / `name` = the part's own (the same SKU the part has as a normal product); `quantity` = bundle quantity x the part's quantity inside the bundle.
   * The bundle's price is split over its parts so the order still adds up: each part gets a share of (bundle price x bundle quantity) in proportion to its regular price x its quantity in the bundle; round every share to a whole taka and put the rounding difference on the largest line, so the part lines add up to the bundle price exactly. `unit_price = line_total / quantity` (decimals are fine). The checks in point 3 stay as they are.
   * Add `bundle_of` (the bundle's name or SKU) to each part line so staff can trace it. The ERP ignores it; it does not change any amount.
   * Order-level discounts stay in `totals.discount` as today. Bundle stock in the shop (the lowest of each part's stock divided by its quantity in the bundle) is the shop's own business; the ERP never receives a stock figure for a bundle.
   * A line that still arrives with `bundle: true` and no ERP item fails the order with code `bundle_not_expanded` (it shows on the ERP sync page). Fix the expansion and send `order.updated`; the ERP then enters it.

## Idempotency and order of events (unchanged)
Events are processed in `seq` order, each in its own transaction, and acknowledged with `ok` or `failed` (a failed one never blocks the rest). A replayed event is re-acknowledged without changing anything.

## To test against the stand-in
```
python tests/fake_feed.py           # TEST events built from real local items, http://127.0.0.1:8099/hooks/erp
python tests/fake_feed.py --snapshot logs/medusa_snapshot.json   # item events from the real catalogue (needs dev_tools/medusa_snapshot.py)
python tests/run_local_test.py      # drives the importer through order, fulfilment, payment, edit, PO, products, Fixit
```
