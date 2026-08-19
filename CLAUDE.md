# CLAUDE.md — Business Data Analysis App

Codebase guide for Claude Code. Keep this file up to date as the project evolves.

---

## Project Overview

Streamlit-based business analytics dashboard for a group of 4 entities:
- **100001** HMBR Tools & Chemicals Ltd. (parent importer)
- **100000** GI Corporation (manufacturing subsidiary)
- **100009** Gulshan Packaging Co. (internal captive packaging, no sales team)
- **100005** Zepto Chemicals (independent consumer brand)

All entities share back-office functions. 100001 and 100000 share the same field sales team.
100001 + 100009 share inventory (cross-ZID via `xdrawing` in `caitem`).

**Entry point:** `app.py` | **Python env:** `streamlitEnv3.10.13` (pyenv) | **DB:** PostgreSQL (`config/global_db.ini`)

---

## Directory Structure

```
app.py                  # BaseApp: page_config, session state, auth gate, sidebar nav/filters, routing
auth/                   # auth.py (login/session/page-access), setup_db.py (run via python auth/setup_db.py)
config/settings.py      # DB params from global_db.ini
core/
  analytics.py          # Analytics class — table name -> query function (query_map)
  db.py                 # ThreadedConnectionPool, get_data(), get_dataframe()
  queries.py            # All SQL builders, each returns (sql, params)
processing/             # Pure pandas transforms, no st.* calls (except @st.cache_data)
  common.py             # data_copy_add_columns, create_download_link, shared utils
  financial.py          # Level 0->S IS/BS/CFS builders, compute_mtd_is, _MTD_CODES
  overall_margin.py, target_management.py, consolidation.py, collection.py, ...
views/                  # One public display_*()/render_*() per file, UI only
  financial.py, financial_dashboard.py, target_management.py, inventory.py,
  margin.py, collection.py, sales.py, purchase.py, basket.py, ar_analysis.py,
  daily_sales.py, accounting.py, home.py
visualization/common_v.py  # plot_histogram, plot_bar_chart (Plotly wrappers)
data/                   # targets.json, public_holidays.json, warehouse_filters.json (gitignored runtime)
                        # hierarchy.json, ls_account_notes.json, labels.json (committed)
db_sync/                # Standalone DB sync scripts (separate from the app)
```

### Critical folder rules
- **Never rename `views/` to `pages/`** — Streamlit auto-lists anything in `pages/` in the sidebar before login.
- `core/` = infrastructure only (DB conn, query builders, Analytics). No business logic.
- `processing/` = business logic only, pure pandas, no `st.*` (except `@st.cache_data`).
- `data/` = all JSON data files.

---

## Data Flow Pipeline

```
PostgreSQL -> core/queries.py (sql, params) -> core/db.get_data() -> core/analytics.Analytics -> .data (DataFrame)
  -> processing/*.py (pure transforms) -> views/*.py (Streamlit UI only) -> app.py (BaseApp wires nav, no logic)
```

Global state (set in `BaseApp.navigation()`, read via `st.session_state.*`, never from a fresh widget):
`zid`, `proj`, `current_page`, `authenticated`, `username`, `user_role`.

---

## Analytics Class Pattern

```python
from core.analytics import Analytics
df = Analytics("table_name", zid=zid, filters={"year": [2026], "month": [6]}).data
```

- Registered in `query_map` dict in `core/analytics.py`. To add a table: write a query function in
  `core/queries.py` returning `(sql, params)`, then register it in `query_map`.
- `zid` may be str/list/tuple. `purchase` and `stock_movement` auto-expand to 2 ZIDs
  (100001 -> also adds 100009) since packaging items span both entities.
- Always check `if df is not None and not df.empty` before using `.data`.
- For data that doesn't change with filter widgets, wrap loads in module-level `@st.cache_data`.

---

## Key Data Sources (DB Tables / Views)

| Alias | Table | Notes |
|---|---|---|
| `sales` | `opdor` + `opddt` + `imtrn` | Sales orders -> line items -> inventory cost |
| `return` | `opcrn`/`opcdt` + `imtrn` | Customer returns |
| `stock` | `imtrn` | Cumulative stock balance (xqty*xsign, xval*xsign) |
| `cacus_directory` | `cacus` | Customer directory (cusid, cusname, cusmobile, whatsapp=xtaxnum, area) |
| `final_items_view` | DB view | Current stock: item_id, item_name, item_group, stock |
| `gldetail`/`glheader`/`glmst` | GL tables | Financial postings |
| `caitem` | Item master | xdesc=itemname, xabc=itemgroup, xdrawing=cross-ZID code mapping, packcode |

### Key column mappings
- `opddt.xdtwotax` -> `altsales` (gross revenue, maps to GL 08010001)
- `opddt.xdtdisc` -> `proddiscount` (GL 07080001 Discount Paid — **never subtract from Revenue in IS context**)
- `imtrn.xval` -> `cost` (COGS)
- `cacus.xtaxnum` -> `whatsapp`
- `gldetail.xprime` -> GL posting amount (Revenue = negative credit, Expense = positive debit)

### Sign convention
- **Level 0 / raw GL**: Revenue = negative, Expense = positive
- **Level S IS**: Revenue = positive, Expense = negative (flipped by `_ls_sum`)
- **`final_sales`** = `altsales - proddiscount` (margin analysis only, NOT in financial IS)

---

## SQL Rules (`core/queries.py`)

1. **No f-string interpolation of filter values** — always parameterized: `"WHERE zid = %s"`, return `(sql, (zid,))`.
2. Every query function returns `(sql_string, params_tuple)`. (A few legacy functions return only `sql`; `Analytics` handles both via isinstance — don't add new ones that way.)
3. IN clauses: use `_build_in_clause(list)` helper -> `(placeholders, params)`.
4. Tables joining `caitem` that need packcode resolution must apply this CASE in SQL (never replicate in Python — case-sensitivity bugs):
   ```sql
   CASE WHEN caitem.packcode IS NOT NULL AND caitem.packcode <> ''
        AND caitem.packcode != 'NO' AND LEFT(caitem.packcode, 2) != 'KH'
        THEN caitem.packcode ELSE table.itemcode END AS itemcode
   ```
5. `stock` table has data-entry errors (e.g. `year=2102`). When building year selectors:
   `valid_years = [y for y in years if 2000 <= y <= current_year + 1]`.

---

## Processing Layer Rules

- Pure pandas in, DataFrame out. No `st.*` calls.
- **Never use `ProcessPoolExecutor`** — macOS "spawn" re-imports `streamlit` in workers and silently returns empty results. Use sequential or `ThreadPoolExecutor`.
- Merge keys: use minimal stable code columns (e.g. `["warehouse","itemcode"]`). Never merge on `itemname`/`itemgroup` — string mismatches cause silent join failures.
- `applymap(fn).fillna(0)` on mixed DataFrames corrupts date columns. Format/`.fillna('')` date columns first, then run numeric cleanup only on numeric columns.
- Financial CFS: "Prior Period Net Profit/Loss" row must be zero for all months except January in monthly perspective (BS Net P/L is YTD cumulative; including it elsewhere double-counts).

---

## Views Layer Rules

- One public `display_*`/`render_*` entry point per file; private helpers prefixed `_`.
- Views call processing functions — no groupby/merge/complex pandas in views.
- Wrap multi-selectbox / data-dependent sections in try/except with a friendly `st.warning`.
- Use `st.download_button` for CSV downloads (not `create_download_link`, which base64-embeds and crashes browsers >~100k rows).
- Cap `st.dataframe` display at ~50,000 rows with an info notice; download for full data.
- Use `st.session_state.zid` for the active ZID; if a view needs its own ZID selector, default it to `st.session_state.zid`.

---

## Auth (`auth/auth.py`)

- bcrypt for password hashing; all DB access via `core.db.get_data()`.
- Passwords stored as `bytea` arrive as `memoryview` — cast with `bytes(hashed)` before `bcrypt.checkpw`.
- `check_page_access(page_name)` checks `page_permissions` table; page names must match the `menu` list in `app.py`.
- Sidebar hidden on login page via CSS (`[data-testid="stSidebar"] {display: none;}`); `initial_sidebar_state` must stay `"expanded"`.

---

## Financial Statements Architecture

### Levels
- **Level 0**: Raw GL detail (every xacc x period)
- **Level 1/2**: Aggregated buckets
- **Level S**: Management view — `build_pl_level_s()` in `processing/financial.py`
- **Level T**: Adjusted Level S (inter-company eliminations)
- **Level C/C2**: Consolidated across ZIDs

### Level S IS row order (key rows)
Revenue -> Others Revenue -> MRP Discount -> **Adjusted Revenue (Pending)** -> COGS -> **Gross Profit** -> SG&A sub-rows -> **Total SG&A** -> 0708-Discount Paid -> S&D Expenses -> **Total S&D** -> 0501-Others Direct -> **EBITDA** -> 0630-Bank Interest -> 0633-Interest Loan -> **Total Interest** -> VAT rows -> Income Tax -> **Net Income**

### MTD IS Dashboard (`views/financial.py` -> `_render_mtd_dashboard`)
- "📊 MTD Dashboard" radio inside Level S Monthly view
- Revenue/COGS from imtrn/opdor pipeline (`altsales` gross, not net of discount)
- All opex from `gldetail` SUM(xprime) for current month, negated to Level S sign
- 3M average from last 3 completed period columns of `pl_s`
- Toggle "Use 3M Averages for SG&A, Interest & Tax in Net Income": when ON, EBITDA/Net Income recalc using 3M avg for Total SG&A, Total Interest, VAT/Tax. Discount Paid and S&D always remain MTD actuals.
- `_MTD_CODES` dict in `processing/financial.py` mirrors the ac_code sets in `build_pl_level_s` — keep both in sync when adding codes.

---

## Target Management (`views/target_management.py`)

View mode radio: `["👤 Individual Salesman", "📊 All Salesmen Overview", "📈 Moving Average", "📦 Current Stock"]`

- **Individual Salesman**: full current-year (Jan–Dec) target entry, defaults to current month. Metric cards incl. Daily Avg Sales (3M) = `total_3mo / wd_3mo`. Requires sidebar to include ≥3 prior months, else `last3` is empty and a warning shows. Inventory Coverage section at bottom.
- **All Salesmen Overview**: per-salesman summary, Daily Required = `(target - mtd_sales) / remaining_wd`, Daily Avg (3M) = `total_3mo / wd_3mo`. Caption shows the exact 3M window + working-day count.
- **Current Stock**: source `final_items_view` (filtered by zid), columns Item ID/Name/Group/Stock, search filter, 1-hour TTL (`_load_final_items`).

### Working days & holidays
- `_is_working_day`: Mon–Thu and Sat–Sun are working days; **Friday (`weekday()==4`) is always off**, hardcoded — do not add Fridays to `data/public_holidays.json`. The holidays set is only for additional non-Friday off-days (Eid, national holidays falling Sat–Thu).
- Holidays stored in `data/public_holidays.json` under `"holidays"` (list of `"YYYY-MM-DD"`), managed via `_get_holidays`/`_prune_holidays` (keeps current + previous calendar year).
- Targets keyed as `{zid}_{spid}_{year}-{month:02d}` in `data/targets.json`.

---

## Inventory Analysis (`views/inventory.py`)

- `_DEFAULT_WAREHOUSES`: Finished Goods Store Packaging, HMBR -Main Store (4th Floor), Raw Material Store Packaging.
- `_DEFAULT_ITEMGROUPS`: 12 groups incl. "Import Item" — edit directly in file.
- Final Stock ZID toggle: OFF = totals combined across 100001+100009; ON = per-ZID split. Cross-ZID grouping uses `itemcode` only (no `itemname`); `_meta` lookup prefers primary ZID names.

---

## Cross-ZID Item Mapping (`views/purchase.py` → "🔗 Cross-ZID Mapping" mode)

Audit report for the 100001 ↔ 100009 packaging-item link — lives in Purchase Analysis (moved from Manufacturing Analysis, where it was added by mistake). Scope is fixed to Gulshan Packaging (100009) finished-goods (`xitem LIKE 'FH%'`) and raw-material (`xitem LIKE 'HPI%'`) items only — other 100009 prefixes (`RAW`, `FK0`, `KPI`, `KRI`) are out of scope and won't appear here even if they happen to carry a valid `xdrawing`.

**Related in Python, not via a SQL JOIN** — `core/queries.py::get_gulshan_fg_rm_items` fetches every FH/HPI item as-is (including blank/`'NO'`/`'KH*'` `xdrawing` values), `core/queries.py::get_hmbr_catalog_lookup` fetches the full 100001 catalog as a lookup table, and `processing/purchase_inventory.py::build_crosszid_item_mapping` relates them. This replaced an earlier single-query LEFT JOIN version whose WHERE clause filtered to *valid* `xdrawing` before the join even ran — which meant an item with a blank/invalid `xdrawing` was excluded from the result set entirely, never even reaching the "no duplicate" flag. Confirmed on real data: `HPI000008` (blank `xdrawing`) was silently missing from the old report; it now correctly surfaces as `❌ No Duplicate`. `build_crosszid_item_mapping` guarantees one output row per FG/RM item no matter what its `xdrawing` looks like — verified 256 items in, 256 rows out.

Three states are still surfaced, unchanged: `✅ Match`, `⚠️ Name Mismatch` (matched but names differ), `❌ No Duplicate` (either no `xdrawing` at all, or one that doesn't resolve — typo'd/stale value). The raw `xdrawing` value is now also shown as its own column (**"XDrawing (100009)"**) specifically so a "No Duplicate" row's cause is visible at a glance — blank vs `'NO'` vs `'KH...'` vs a mistyped/broken code all look different there, even though all four collapse to the same Status.

Only checks the 100009→100001 direction, since `xdrawing` is the only explicit linkage signal (100001's own `xdrawing` column means something unrelated — variant consolidation, not cross-ZID). The reverse direction (100001 items with no 100009 counterpart) isn't checked — most of 100001's catalog has nothing to do with 100009 and was never meant to have one.

---

## Customer Columns (all sales-derived tables)

| DB column | Alias | Display label |
|---|---|---|
| `cacus.xmobile` | `cusmobile` | Mobile |
| `cacus.xtaxnum` | `whatsapp` | WhatsApp Number |
| `cacus.xcity` | `area` | Area |

`whatsapp` flows through `build_customer_wise_monthly`/`build_customer_product_monthly` in `processing/target_management.py` — must be in `id_cols` for pivots.

---

## App Collections (`views/glpmt_shared.py` → "📲 App Collections" mode)

`glpmt` is an ERP table (synced, not app-owned) holding payments salesmen enter directly into a separate mobile Ordering app, staged pending reconciliation into the real GL ledger — not the same thing as `crm_call_log`/`marketing_leads` (those are app-owned tables this Streamlit app writes to; `glpmt` is read-only here, written by the Ordering app). `core/queries.py::get_glpmt_data` LEFT JOINs `prmst` on `xemp` for a properly-formatted salesman name (`glpmt.xname` itself is raw/informal, e.g. `"emon"` vs prmst's `"Md. Abdullah Al Mamun Emon"`), falling back to the raw value if the employee code isn't in `prmst`.

Sorted by `ztime` (when the entry was actually made) descending, latest first — **not** `xpaydate` (the payment's own date on the voucher, which can be back-dated and differ from when it was logged).

One shared panel (`views/glpmt_shared.py::render_glpmt_panel`) is mounted identically in both **Collection Analysis** (`views/collection.py`) and **Target Management** (`views/target_management.py`) — same filters (salesman/emp code, customer, date-of-entry range), same table, same sort. Edit the shared module, not either call site, to change behavior in both places at once.

### Feeding into "Latest Collection" (Salesman Due + Customer Support)
`processing/salesman_due.py::merge_latest_app_payment` folds a customer's latest glpmt entry into the existing "Latest Collection Date"/"Latest Collection Amount" columns — **whichever source's date is later wins**, per (Customer Code, ZID). Adds a `Collection Source` column (`"Ledger"` / `"App (Pending)"`) so viewers can tell a shown collection is still unreconciled — **still computed, but no longer shown**; see "Collection Source removed from display" further down. Deliberately does **not** touch `Current Balance` — an unreconciled app payment hasn't actually reduced the real ledger balance yet (could still be rejected before posting).

Two call sites, both already naturally per-ZID before this merge runs (no extra ZID-scoping needed in the merge itself):
- **Salesman Due** (`views/collection.py::_load_salesman_due_reports`/`_load_salesman_due_reports_any` → `processing/salesman_due.py::build_salesman_due_reports`) — each function call already handles exactly one ZID end-to-end (including the 100001+100000 combined-scope path, which runs the whole per-ZID pipeline twice and concats after).
- **Customer Support → Latest Sales & Collection** (`processing/customer_support.py::build_latest_sc_for_zid`) — receives the ALL-ZID `load_all_glpmt()` output and slices it to the current ZID internally, mirroring how `ar_df_cleaned`/`cacus_df` are already handled there.

Never merge glpmt data across ZIDs before calling either entry point — a customer code is only unique within one ZID.

---

## Returns Registry (`views/returns_registry.py` → Target Management "↩️ Returns Registry" mode)

Customer returns salesmen log directly into the mobile Ordering app, still pending approval — `opcrn.xstatuscrn = '1-Open'` only (other statuses: `2-Accepted`, `3-Issued` — not shown here). Same architectural family as `glpmt`/App Collections: an app-staged entity read from the ERP, not app-owned.

- `core/queries.py::get_returns_registry` — header rows (`opcrn`), one per return. **`opcrn.xtotamt` is blank on open returns** (not finalized yet) — the displayed total is `SUM(opcdt.xlineamt)` instead, joined in and grouped by `xcrnnum`. Sorted by `xdate` DESC.
- `core/queries.py::get_returns_registry_items` — product line items (`opcdt`), joined back to `opcrn` so only lines belonging to an open header are included. `opcdt.xdesc` is a reliable snapshot (confirmed matches `caitem.xdesc`) — no `caitem` join needed for item names.
- Real data has at least one legacy row with an out-of-range sentinel date (`2999-12-31`, exceeds pandas' `Timestamp` max) — `pd.to_datetime(..., errors="coerce")` is mandatory here, not optional; without it the whole page crashes with `OutOfBoundsDatetime`. The affected row degrades gracefully (blank date, sorts last).

**Table 1 → Table 2 relationship**: the "Customer" filter above Table 1 is a single-select (not multiselect, unlike the Salesman filter) — it does double duty, narrowing Table 1 **and** driving which customer's product lines populate Table 2 below. Table 2 stays empty with a prompt until a customer is chosen.

Spans all 3 ZIDs (100000/100001/100005 all have open returns) — `Analytics("returns_registry", zid=zid, ...)` is parameterized by whatever ZID is active in the sidebar, same as every other single-ZID-scoped table in this app; no cross-ZID merging needed here (unlike glpmt → Latest Collection above).

---

## Feedback (`views/feedback.py` → Target Management "💬 Feedback" mode)

Market-level feedback salesmen log via the mobile Ordering app — about a customer, a product, a delivery issue, or a collection issue. Same architectural family as `glpmt`/Returns Registry/promise dates: an app-staged entity read from the ERP `feedback` table, not app-owned.

- `core/queries.py::get_feedback_data` — one row per feedback entry, LEFT JOINed to `cacus`/`caitem`/`prmst` for display names (`customer_id`→`cacus.xcus`, `product_id`→`caitem.xitem`, `user_id`→`prmst.xemp`, all confirmed matching on real data). LEFT JOINs, not INNER — `user_id` is blank on ~10 legacy rows (predates the field being captured), and those must still surface with a blank Salesman/Emp Code rather than silently vanishing.
- **Four independent, non-exclusive tags per row**: `customer_id` set, `product_id` set, `is_delivery_issue`, `is_collection_issue`. Not mutually exclusive — confirmed on real data: 10 rows have both `customer_id` AND `product_id` set, 4 rows have both `is_delivery_issue` AND `is_collection_issue` true. So a single feedback entry can legitimately appear in more than one of the four category tables. About 70% of rows (145/204) carry none of the four tags (general feedback with no category) — those don't appear in any table, by design.
- `views/feedback.py::_CATEGORIES` drives all four tables off one shared render path (`_render_feedback`) — a dict of `{label: {mask, id_cols, id_rename, empty_msg}}`. Customer/Product tables show that category's identity columns (Cust Code/Customer, Item Code/Item Name); Delivery/Collection Issue tables show no extra identity column since the DB has none for those two (just the boolean flags) — only Date/Emp Code/Salesman/Feedback, per what was asked.
- Filters: Salesman (Emp Code) multiselect + Date range, both scoped to whichever category is currently selected (so the salesman dropdown only lists salesmen who actually have entries in that category).

Spans all 3 ZIDs (100000/100001/100005 all have entries) — same single-ZID-scoped `Analytics("feedback", zid=zid, ...)` pattern as Returns Registry, no cross-ZID merging.

---

## Promised Delivery / Payment Dates (`opdor.xdatedel` / `opdor.xdatepay`)

Salesmen log a promised delivery date and promised payment date on orders via the mobile Ordering app. `core/queries.py::get_cus_delivery_payment_promise` returns, per customer, the pair from their most recent order **that actually has both fields set** (`DISTINCT ON`, ordered by `xdate DESC`) — not simply their most recent order overall, which may predate or lack this field. Currently only populated for ZID 100001 (237 customers); other ZIDs degrade gracefully to empty.

Same `2999-12-31` "unset" sentinel as Returns Registry shows up here too — excluded at the SQL level (`<> '2999-12-31'`), not just coerced client-side, since a naive `pd.to_datetime()` on it crashes with `OutOfBoundsDatetime`.

Wired into `processing/customer_support.py::load_all_delivery_payment_promise()` (same `_ZID_PROJECT` loop pattern as `load_all_glpmt`/`load_all_cacus`) and merged into **three** places — both Customer Support views, plus Collection Analysis → Salesman Due:
- **90-Day Activity** (`build_7day_feed`, renamed from "14-Day" — the window itself changed from 13 to 89 days back, and is now user-adjustable up to 180, see "Time Range slider" below; `core/queries.py::get_sales_7day`'s DO-detail window changed to match its 180-day max. Function/table names kept as `7day`/`14day` in a few internal-only spots — not user-facing, left alone to limit blast radius) — customer-level attribute (comes from `opdor`, not tied to any one voucher), so after the merge it's zeroed out (`NaT`) on every row whose `txn_type != "Delivery"`, **and further** zeroed out on Delivery rows where `promised_delivery <= xdate` (that row's own transaction date) — a promise dated at/before a given delivery has already passed relative to it, so it's not a live promise worth surfacing on that row. Displayed as **"Delivered Date"** (label only — the underlying value is still the salesman's *promised* date, not a confirmed delivery) and **"Promised Payment"**. The Type filter (`cs_type_filter`) defaults to `"Delivery"` on page load rather than `"All Types"`.
- **Latest Sales & Collection** (Customer Support, `build_latest_sc_for_zid`) and **Latest Sale & Collection** (Collection Analysis → Salesman Due, `processing/salesman_due.py::merge_delivery_payment_promise`) — both customer-level tables (one row per customer, no `txn_type` to restrict against), so the gate here is `promised_delivery > last_sale_date` / `> "Sales Date"` instead — a promise dated at/before the customer's most recent actual sale is stale relative to it. Displayed as **"Delivery Date"** / **"Promised Payment"** in both. Internal column name in `build_latest_sc_for_zid`'s output is `delivery_date` (renamed from `promised_delivery` specifically because this table's label differs from 90-Day Activity's "Delivered Date").
  - **On current real data, this gate zeroes out nearly everything** in both customer-level tables — the `opdor` promise-date feature is sparsely populated and clustered around late-2023/early-2024, while most customers' actual sales are far more recent, so `promised_delivery > last_sale_date` is rarely true. This was verified as correct behavior, not a bug, before shipping.

**Sales role cannot see Latest Sales & Collection in Customer Support** — `display_customer_support` only offers that radio option when `st.session_state.user_role != "sales"`; sales users see 90-Day Activity only, with no second option in the radio at all (not just blocked after selection).

**Known pre-existing issue, not introduced by this feature**: `mv_ar_transactions` (backing the AR ledger / 90-Day Activity feed) has at least one garbage far-future date (`2102-10-11`-class, same family as the `stock` table's `year=2102` bug noted under Common Pitfalls) that always passes any `>= cutoff` date filter regardless of window size — it would have affected the old 14-day window too. Not fixed here since it's a shared MV touching many other pages.

### Returned Date (90-Day Activity only)

Same pattern as promised delivery/payment, for the "Return" `txn_type` instead of "Delivery": `core/queries.py::get_cus_return_entry_date` returns, per customer, the date the salesman entered their most recent return into the mobile Ordering app (`opcrn.xdate`) — the *app-logged* date, distinct from `xdate` on the AR ledger's own Return row (`SRT`/`SRJV`/`IMSA` voucher — when it was actually posted/reconciled).

Deliberately **not** restricted to `xstatuscrn = '1-Open'` like Returns Registry is — by the time a return shows up as a "Return" row in the AR-ledger-backed 90-Day feed, it has already been posted, meaning its `opcrn` status has moved on to `2-Accepted`/`3-Issued`. Confirmed on real data: 128,712 of 128,894 `opcrn` rows are `3-Issued`, only 141 are still `1-Open` — filtering to open-only here would return almost nothing. Same `2999-12-31` sentinel excluded at the SQL level, same reason as the promise-date query.

Wired via `processing/customer_support.py::load_all_return_entry_date()` (same `_ZID_PROJECT` loop pattern) into `build_7day_feed` as `return_entry_date`, zeroed out (`NaT`) on every row whose `txn_type != "Return"`. Displayed as **"Returned Date"**. Not merged into either Latest Sale & Collection table — those have no `txn_type` to restrict against, and it wasn't asked for there.

### 90-Day Activity Time Range slider

`_render_90day_activity` exposes a `st.slider("Time Range (days)", min_value=15, max_value=180, value=15, step=15, key="cs_activity_days")` — applies to the whole feed regardless of Type filter, replacing what used to be a fixed 90-day window. `build_7day_feed(..., days=days_selected)` just changes its cutoff calc (`today - Timedelta(days=days-1)`); since `_ar_data()`/`load_all_ar_ledgers()` already loads full unfiltered AR history, moving the slider is a pure in-memory re-filter — no new DB round trip. The DO-detail sub-table (`_render_do_detail`, backed by `_sales_14day_data()` / `get_sales_7day`) is loaded once at the widest possible window (180 days, cached) and then sliced client-side to the same `days_selected` cutoff, so it stays in sync with the slider without re-querying per change.

### Paid Date / Amount Paid (both Latest Sale & Collection locations)

Dedicated columns at the very end of the table (after `current_balance`/"Balance") showing the customer's **latest `glpmt` (mobile-app) payment** — date + amount — via `processing/salesman_due.py::latest_app_payment_lookup` / `merge_app_paid_columns`. Same "must be after the latest sale" gate as Delivery Date: `Paid Date > last_sale_date` / `> "Sales Date"`, otherwise `NaT`.

This is deliberately **separate** from `merge_latest_app_payment`'s existing "Latest Collection Date/Amount" folding (which still runs first, unchanged) — that function only surfaces an app payment when it *out-dates the ledger*, so a viewer can't tell from "Latest Collection Date" alone whether it came from the app or the ledger (this is also why the `Collection Source` column was removed from display — see below). Paid Date/Amount Paid always shows the app entry specifically, independent of whether it happened to win that comparison.

### Collection Source removed from display, Promised Payment highlighted when overdue

- `Collection Source` (`"Ledger"` / `"App (Pending)"`, added by `merge_latest_app_payment`) is still computed internally — needed there to decide whether an app payment out-dates the ledger — but is no longer surfaced to viewers in either Latest Sale & Collection table. Dropped via `.drop(columns=["Collection Source"], errors="ignore")` in `build_salesman_due_reports` (Collection Analysis) and simply left out of the `keep` whitelist in `build_latest_sc_for_zid` (Customer Support).
- `processing/common.py::highlight_overdue_date(df, col, ref_date=None)` returns a pandas `Styler` that flags any cell in `col` with `background-color: #ff4b4b; color: #ffffff` when its date is before today (or `ref_date`) — high-contrast red chosen to read on both light and dark Streamlit themes. Applied to the "Promised Payment" column in 90-Day Activity, both Customer Support Latest Sales & Collection tables, and Collection Analysis's Latest Sale & Collection. `st.dataframe` accepts a `Styler` in place of a plain DataFrame, and `column_config` still applies on top of it — confirmed no conflict between the two. Not applied to "Paid Date"/"Amount Paid" — those are historical facts (a payment that already happened), not a promise that can go "overdue".

---

## Marketing Leads CRM (`views/marketing.py` → "🎣 Leads" mode)

Facebook Lead Ads (or similar) CSV/Excel exports get uploaded here and tracked through to conversion.

### App-owned tables (NOT synced via `db_sync`)
Created once via `db/sql_scripts/create_marketing_leads_tables.sql`, same convention as `crm_call_log`/`users`/`page_permissions` — written directly by the app, not mirrored from the ERP.
- **`marketing_leads`** — one row per lead. `zid` is set at upload time from the uploader's active ZID (never present in the export). `fb_lead_id` (the export's own `id` column) is `UNIQUE (zid, fb_lead_id)` so re-uploading the same export is a no-op. Any CSV column outside the fixed schema (lead forms carry different custom questions — e.g. a Bengali institution-type question that won't recur on every form) is packed into `extra_fields` JSONB as `{question: answer}` instead of requiring a schema change per form.
- **`marketing_lead_call_log`** — call history against a lead, separate from `crm_call_log` (which is keyed on `cusid`, a real ERP customer code — a lead isn't one yet). Mirrors `crm_call_log`'s shape and adds `next_visit_date`.

### Conversion tracking — no `cus_code` column on `marketing_leads`
When staff convert a lead to a real customer in the ERP, they manually paste the lead's `fb_lead_id` into that new customer's `cacus.xurl` field. The app reads this live via `get_cacus_lead_links` (`cacus.xurl` join, registered in `Analytics` as `"cacus_lead_links"`) rather than storing a customer code on the lead row — conversion can happen at any time on the ERP side, so a stored value would go stale.

### Permissions (role-gated inside the view, not via `page_permissions`)
Both `crm` and `sales` already have page-level access to "Marketing Analysis" in `page_permissions`. Finer-grained access is enforced in `views/marketing.py::_show_leads` by `st.session_state.user_role`:
- **`crm`/`admin`**: top-level radio (`_show_leads`) — **"➕ Add Leads"** (bulk upload + single-lead form + edit-lead form, no tables) and **"📞 Call Log"** (Table 1 leads list → call-log entry panel → Table 2 all call logs, in that order). CSV downloads on both tables.
- **`sales`**: no radio — Table 1 (leads list) only, read-only. No upload, no edit, no call-log entry, no Table 2.

### Editing a lead after it's saved: `views/marketing.py::_show_edit_lead` (CRM/admin only)
Third tab under "➕ Add Leads" (`✏️ Edit Lead`, alongside Bulk Upload and Single Lead), since a lead saved either way — bulk import or the manual form — previously had no way to fix a typo or update its details afterward. Pick a lead from a selectbox, edit in a pre-filled `st.form`, save.

`core/queries.py::update_marketing_lead_sql` covers `full_name`, `company_name`, `work_phone_number`, `job_title`, `street_address`, `lead_stage` — nothing else. **`id` and `fb_lead_id` are deliberately never in the SET list and can't be changed from this form**: `id` is the FK target for `marketing_lead_call_log.lead_id` (changing it would orphan call-log history), and `fb_lead_id` is the join key staff paste into `cacus.xurl` to track conversion (changing it after the fact would break an already-recorded conversion link). `WHERE id = %s AND zid = %s` — the `zid` check is cheap defense-in-depth on top of `id` already being globally unique as the PK.

`lead_stage` uses a fixed dropdown (`_LEAD_STAGES` = New/Contacted/Qualified/Follow-up/Converted/Not Interested) with one safeguard: if the lead's current stage isn't in that list (a legacy/custom value), it's appended as an extra option so the field doesn't silently get overwritten by whatever the dropdown defaults to just because its current value wasn't a preset choice.

Executed via `core/db.py::execute_write` (single DML helper, same one `crm_call_log` deletes use) — no bulk/`execute_values` machinery needed here, it's always exactly one row.

### Shared call-log module: `views/lead_call_log_shared.py`
Mirrors `views/call_log_shared.py`'s panel styling exactly (imports `blue_header`/`BLUE_FOOTER` from it) but keyed on `lead_id` instead of `cusid`, and adds a `next_visit_date` field to both the entry form and the history badges. **Outcomes are lead-specific, not shared with Customer Support** — `LEAD_OUTCOMES`/`_LEAD_OUTCOME_BADGE` are defined locally in this file (Customer Support's `OUTCOMES` are order/AR/relationship states, meaningless before a lead converts): `Not Answered`, `Not Interested`, `B2C`, `Wrong Lead`, `Asked to Submit Sample`, `Sample Submitted`, `Still Using Sample – Will Contact After`, `Follow-up Requested`, `Promised to Order`, `Deal Completed`.

### `views/call_log_shared.py::OUTCOMES` is genuinely shared, not per-surface
Unlike the lead call log above, Customer Support and Marketing → Inactive Outreach read and write the exact same `crm_call_log` rows for a given `cusid` — Inactive Outreach's own UI says so directly ("Call logs are shared with Customer Support"). So `OUTCOMES` (`Promised`, `Paid`, `Not answered`, `Dispute`, `Delivered`, `Not Delivered`, `Returned`, `Switched Business`, `Price Issues`, `Other`) is one list used by both surfaces — adding an option here makes it selectable (and consistently displayed) everywhere this call log appears, not just wherever the request for it originated. `_OUTCOME_BADGE` only defines custom colors for 4 of the 10 outcomes; the rest render with a generic gray badge — that's the existing pattern, not a gap to fill in.

### Bulk insert + dedup: `core/db.py::execute_values_insert`
One round trip via `psycopg2.extras.execute_values`. **No `ON CONFLICT` clause** — the live server predates Postgres 9.5 (confirmed: both `CREATE INDEX IF NOT EXISTS` and `ON CONFLICT` throw syntax errors there), so dedup happens in Python instead: `get_existing_lead_fb_ids` fetches already-saved `fb_lead_id`s for the ZID and `_bulk_insert_leads` filters the upload batch against them before a plain `INSERT`. Returns `cur.rowcount`, or `-1` on a DB error — callers must NOT clamp this to 0 (a real bug: `-1` clamped via `max(n, 0)` once silently reported failed uploads as "0 new leads saved").

### Sample upload template + CSV/Excel `dtype=str` fix
`processing/marketing_leads.py::build_leads_upload_template` — a one-row example CSV in exactly `_ID_COL` + `_FIXED_COLS` order (English-only column names, one filled-in example row), downloadable from an expander above the file uploader in `_show_lead_upload`. Exists because a real Facebook Lead Ads export was confusing a CRM manager filling leads by hand — its custom per-form questions (e.g. a Bengali institution-type question) aren't part of the fixed schema and silently land in `extra_fields` instead of the visible leads table, with no indication in the raw CSV that that's what would happen. The template sidesteps this: fill in only the known columns, upload through the same `_show_lead_upload` path unchanged.

While verifying the round trip, found and fixed a real pre-existing bug in the same function: `pd.read_csv(uploaded)` / `pd.read_excel(uploaded)` had no `dtype` hint, so pandas infers a numeric-looking column (`work_phone_number`, `id`) as `int64` and **silently drops the leading zero** — every Bangladeshi phone number (`01711234567` → `1711234567`) and any leading-zero `id` gets corrupted on upload, template or not. Fixed by reading with `dtype=str` throughout; confirmed blank cells still parse as real `NaN` under `dtype=str` (doesn't change any of `parse_leads_upload`'s existing `pd.isna()`/`errors="coerce"` handling).

### `area` and `lead_cost` columns (2026-08-19 addition)
Neither is platform-sourced. **`area`** — the lead's exact area/division, entered by whoever compiles the upload (template column right after `street_address`). **`lead_cost`** — hand-calculated by the CRM manager, template's last column, `NUMERIC(12,2)`; `parse_leads_upload` runs it through `pd.to_numeric(errors="coerce")` so one bad cell (`"1,000"`, `"500/-"`) becomes `NULL` for that row instead of an invalid-numeric-literal error aborting the entire batch insert (`execute_values` is one round trip for the whole file).

**Column order is safety-critical here, not just cosmetic.** `_bulk_insert_leads` builds each row tuple positionally — `(zid,) + tuple(r) + (uploaded_by,)` off `processing/marketing_leads.py::_FIXED_COLS`'s column order — and hands it straight to `core/queries.py::insert_marketing_leads_sql`'s `INSERT` column list. The two lists must stay in lockstep, or values silently land in the wrong columns with no error at all. `area` sits right after `street_address` and `lead_cost` is the last entry in `_FIXED_COLS`, matching both the `INSERT` statement and the DDL. `build_manual_lead_row` and `build_leads_upload_template` were both updated in the same change since they also construct rows against `_FIXED_COLS` — verified end-to-end (bulk-upload path and manual-entry path) against a temp table before shipping.

Two DB scripts, not one: `db/sql_scripts/create_marketing_leads_tables.sql` (the `CREATE TABLE IF NOT EXISTS` — updated in place for a *fresh* setup) and `db/sql_scripts/add_marketing_leads_area_lead_cost_columns.sql` (new — a non-destructive `ALTER TABLE` for a server that already has the *old* schema and real lead data on it). The `ALTER` script deliberately doesn't use `ADD COLUMN IF NOT EXISTS` — that's a Postgres 9.6+ feature, and this server predates 9.5 per the `ON CONFLICT`/`CREATE INDEX IF NOT EXISTS` incompatibilities already documented above; safe to run once, errors (not silently no-ops) if run twice.

### Backup/restore pair for a DROP + CREATE migration path
`db/sql_scripts/backup_marketing_leads_to_csv.sql` / `restore_marketing_leads_from_csv.sql` — for whoever prefers a clean drop-and-recreate over the `ALTER TABLE` script above (e.g. a full schema reset) but still needs to keep existing lead + call-log data. Both use `\copy`, a psql meta-command that runs client-side (wherever `psql` is invoked from), not server-side — no server filesystem access needed, and the CSVs land in the current working directory.

**`id` preservation is the whole point of the restore script, not incidental.** `marketing_lead_call_log.lead_id` is a hard FK to `marketing_leads.id`; a plain re-INSERT that lets `SERIAL` renumber rows from 1 would silently repoint every restored call log at the wrong lead (or a nonexistent one). The restore script's `\copy ... (id, zid, ...)` explicitly lists `id` in the column list, which inserts the literal backed-up value instead of invoking the `SERIAL` default — then `setval(pg_get_serial_sequence(...), MAX(id))` on both tables afterward, so the *next* app-created lead/call-log doesn't collide with a restored id. Verified end-to-end against real Postgres: original ids preserved exactly (including a table with deliberate id gaps, mimicking rows deleted over time), call logs still correctly linked to the right lead by content after restore, a JSONB `extra_fields` value containing both an embedded comma and an escaped quote round-tripped byte-for-byte through the CSV, and a fresh insert after restore got a non-colliding id with no manual sequence bookkeeping needed.

### Python alternative: `db/sql_scripts/restore_marketing_leads_from_csv.py`
Same restore job as `restore_marketing_leads_from_csv.sql`, for when the CSV comes from pgAdmin's own Export Data feature (or any manual export) instead of `\copy` — no need to also run the matching backup script, since export is on the user. Same `id`-preservation + sequence-reset logic as the SQL version.

Two things this needed that the SQL version didn't, both found by testing against a deliberately adversarial fake pgAdmin export (UTF-8 BOM prefix, shuffled column order, spelled-out `true`/`false` booleans) rather than just reasoning through it:
- **`encoding="utf-8-sig"`** on the `pd.read_csv` call — pgAdmin (running against a Windows server here) commonly prepends a UTF-8 BOM to CSV exports; plain `utf-8` leaves it stuck on the first header, turning `id` into `"﻿id"` and making the required `id` column look missing.
- **Real bug caught by the test, not shipped**: the first `_load_csv` draft used `df.where(~df.isin(["NULL","null",""]), None)` to convert blanks to `NULL` — but `pd.read_csv(dtype=str)` already turns a blank cell into pandas' float `NaN`, which never equals the literal string `""`, so that check silently missed every genuinely-blank cell. Those `NaN`s then flowed straight into `execute_values`, and psycopg2 renders a bare float `NaN` as the SQL literal `'NaN'::float` — which fails immediately against every non-numeric column (`jsonb`, `text`, `timestamp`, ...). Fixed by converting real `NaN` to Python `None` via `df.astype(object).where(pd.notna(df), None)` **first**, then handling the literal text `"NULL"` as a separate pass. This is the same category of bug as `dtype=str`'s leading-zero fix in `parse_leads_upload` earlier — pandas' default type inference doesn't match what a DB insert needs, and it only surfaces when you actually try the round trip.

Column-flexible by design (matches columns by name against a fixed allowlist per table, not position or count), so it doesn't care whether the export happens to include `area`/`lead_cost` or not, or what order the columns come in.

Backup runs against the *old* schema (no `area`/`lead_cost` columns exist yet) — its restore counterpart's explicit column list omits both, so every restored (old) lead correctly gets `NULL` for both rather than erroring on a column-count mismatch.

---

## Git / Deployment

- **Main branch**: `main` — always deployable. Feature branches merged to main when approved.
- **Server**: Windows, `git pull origin main` to update.
- Runtime JSON files (`data/targets.json`, `data/public_holidays.json`, `data/warehouse_filters.json`) are gitignored — server users manage independently.
- Merge conflict on `auth/setup_db.py`: `git checkout --theirs auth/setup_db.py && git add auth/setup_db.py && git commit --no-edit`.
- **Never** commit `*.ini` files (DB credentials, gitignored by design).

### Running locally
```bash
pyenv activate streamlitEnv3.10.13
streamlit run app.py
python auth/setup_db.py   # auth table setup; sys.path fix included, works from any dir
```

---

## Common Pitfalls / Known Bugs

1. **Revenue in IS context**: use `altsales` (gross), never `altsales - proddiscount` — discount is a separate GL line (07080001).
2. **Cross-ZID inventory**: group by `itemcode` only — same code has different names in 100001 vs 100009 `caitem`.
3. **"Blank" item group**: stored as NULL/empty, not the string `"Blank"`. Check `isna() | str.strip() == ""`.
4. **MTD 3M averages**: sidebar must include ≥3 prior months for Daily Avg (3M)/Monthly Avg (3M) to be nonzero.
5. **`_ls_sum` negates**: raw GL Revenue is a credit (negative xprime) -> Level S flips to positive. Always negate gldetail MTD sums too.
6. **Styler `_calc` column**: drop internal helper columns before applying Pandas Styler; use `row.name` for row-level logic.
7. **`pages/` folder name** triggers Streamlit's built-in multi-page nav — keep `views/`.
8. **`stock_flow` itemcode mismatches**: apply the packcode CASE in SQL and merge on `["warehouse","itemcode"]` only.
9. **`database.ini`/`global_db.ini` not present in git worktrees** (gitignored) — copy manually after checkout, or login fails silently with no error.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
