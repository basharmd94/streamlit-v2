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
whatsapp_webhook/       # Standalone FastAPI service (separate from the app, see below)
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

View mode radio: `["👤 Individual Salesman", "📊 All Salesmen Overview", "🎯 Salesman Score", "📊 3 Month Averages", "🧾 SR Trn", "📦 Current Stock", "🔮 Next Month Target", "🗺️ Field Tracking", "📲 App Collections", "↩️ Returns Registry", "💬 Feedback"]` — the old standalone "Moving Average" tab is folded into "📊 3 Month Averages".

- **Individual Salesman**: full current-year (Jan–Dec) target entry, defaults to current month. Metric cards incl. Daily Avg Sales (3M) = `total_3mo / wd_3mo`. Requires sidebar to include ≥3 prior months, else `last3` is empty and a warning shows. Inventory Coverage section at bottom.
- **All Salesmen Overview**: per-salesman summary, Daily Required = `(target - mtd_sales) / remaining_wd`, Daily Avg (3M) = `total_3mo / wd_3mo`. Caption shows the exact 3M window + working-day count. **`% Collection`** (both the current-month table in `_render_overview` and every prior-month expander table in `_render_prior_month_section`) = `MTD Collection / Net Sales × 100` — straight percentage, denominator is Net Sales (sales − returns), not gross Sales. Previously divided by `(1.02 × Sales)` (gross, with a `1.02` inflation factor) — explicit correction, both occurrences fixed identically. Distinct from Individual Salesman's `% Collection vs Target` metric card (`MTD Collection / Monthly Target`), which was already on a different formula and untouched by this fix.
- **Salesman Score** (`views/salesman_score.py` + `processing/salesman_score.py`, dispatched from `🎯 Salesman Score`): same `% Collection` fix applied here too, found via an explicit follow-up sweep of the whole Target Management page for this exact pattern. Two spots: the displayed preview column in `views/salesman_score.py` (was already missing the `1.02` factor from an earlier partial fix, but still divided by gross `sales`) and — more consequentially — **`processing/salesman_score.py::compute_salesman_scores`**'s own `score_collection` component, a real 45%-weight slice of the composite 0–100 score, which had the identical gross-`sales`-denominator bug independently (the view's displayed column and the score engine compute their own `pct_coll` separately, so both needed the fix). Both now use `coll / net_sales`. Swept the rest of the page (Returns Registry, Feedback, App Collections, Field Tracking, Next Month Target, 3 Month Averages, SR Trn) for the same pattern — no other occurrences found.
- **`"Collection Gap"`** (`_render_overview`, `= target * 1.02 - mtd_coll`) is a **different** metric — an absolute gap vs. Target with its own `1.02` buffer, not a Collection-vs-Sales ratio — and was deliberately left untouched by the sweep above; flag if this should be normalized too.
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

### Statistical Analysis mode (`display_inventory_analysis_main` → `_render_statistical_analysis`)

A top-level `st.radio("Analysis Mode", [...])` right under the page title switches between the existing Stock & Movement content (default) and a separate "📊 Statistical Analysis" mode — chosen deliberately light: it reuses `Analytics("final_items_view", zid=zid)` as-is (already registered, already used elsewhere by Target Management/Marketing), no new SQL, no joins to `stock_value`/`stock_movement`/`caitem`. Scoped to the currently selected ZID only (not combined across ZIDs — mixing sales velocity across unrelated businesses, e.g. HMBR vs Zepto, wouldn't be meaningful).

Two metrics, radio-selected, **defaults to "Days to Clear"** per what was asked:
- **Days to Clear** = `stock / avg_monthly_sales × 30`, reusing `final_items_view`'s own `avg_monthly_sales` column (a 3-month trailing average already baked into the view — same formula Purchase Analysis's Total Inventory Overview uses, so the number stays consistent app-wide). Items with `avg_monthly_sales = 0` are excluded (undefined), with an explicit count shown rather than silently dropped. **Verified on real data this exclusion is large and real** (322 of 2,353 100001 items had any DO/SRE/RECA activity in the trailing window) — confirmed via `imtrn` that this reflects genuinely non-moving catalog items, not stale local data (latest real transaction date was 2026-08-15, i.e. current).
- **Sales Value** = `stock × std_price`, using `final_items_view`'s own `std_price` column directly — **this is `caitem.xstdprice`, the sales/list price, not `caitem.xstdcost`** (a real but separate cost field on `caitem`, deliberately not joined in — explicit user choice to keep this feature light and to use the sales-price valuation). Items with `std_price = 0` are excluded the same way (no price set → value undefined), not silently zeroed.

Both metrics show mean/median/std/min/max via `st.metric` columns. **"Mode" is deliberately not a literal statistical mode** — continuous values (especially Sales Value, effectively all-distinct) rarely repeat exactly, so a raw mode is usually meaningless. Reported instead as the **modal histogram bucket** (most-populated bin) in a caption below the metrics.

**Min/Max/Bins** — same three `st.number_input` widgets as 📈 Order Analytics (`views/sales.py`/`views/collection.py`'s own Order Size Distribution), not a slider (explicit ask: "I dont need the slider for choosing the number of bins"). Matches that mechanism exactly, including scope: Min/Max is a genuine population filter (`stat_df[col] >= value_min` / `<= value_max`), applied **before** summary stats are computed — mean/median/std/min/max, the chart, and the bucket totals table all reflect the filtered range, not just the chart, same as Order Analytics. Replaced an earlier 95th-percentile auto-clip + slider design (checkbox + `st.slider`) that predated this request.

**Drill-down**: a `st.selectbox` listing every bucket with its item count (`"50.0 – 70.0 (23 items)"`) — picking one renders a table of the matching items (Item Code/Name/Group/Stock/Avg Monthly Sales/Std Price/Days to Clear/Sales Value) below, sortable by the active metric descending, with a CSV download.

**Bucket Totals table**: always rendered at the bottom (independent of whether a bucket is drilled into above) — one row per histogram bucket, columns Item Count / **Avg Days to Clear** / **Total Sales Value**, regardless of which metric currently defines the bucket edges (so switching the metric radio still shows both figures side by side per bucket — e.g. "for items priced $X–$Y in Sales Value, what's their average Days to Clear?"). A shared `_bucket_mask(idx)` closure (same half-open-except-last-bin logic as the drill-down) is reused for both the drill-down table and every Bucket Totals row, so the two can never disagree on which items fall in which bucket. `.mean()`/`.sum()` on pandas Series auto-skip the `NaN`s that show up when a bucket's items include some excluded from the *other* metric's own population (e.g. an item with `avg_monthly_sales = 0` has `days_to_clear = NaN` even when it appears in a Sales-Value-defined bucket).

---

## Total Inventory Overview (`views/purchase.py` → `_render_total_inventory`)

Combined 100001 + 100009 stock table in Purchase Analysis. A `st.radio("Stock column", ["Total Stock", "Break down by ZID (100001 / 100009)"])` (`key="total_inv_stock_mode"`) gates everything below — **default is "Total Stock"** (just the existing single combined column, renamed "Stock" → "Total Stock"); switching to the breakdown option adds **`100001 Stock`** and **`100009 Stock`** to its left. The extra query only runs when the radio is actually switched — `show_breakdown` guards both the query call and the per-`resolved_code` aggregation, so the default view pays zero extra DB cost.

**`final_items_view` has no `zid=100009` branch of its own** (confirmed via `pg_get_viewdef` on the live server — it's a 4-way `UNION ALL` over zid 100001/100000/100005 only). For a 100001 item cross-ZID-linked to a 100009 packaging item (via `caitem.xdrawing`), the view's first branch already sums `stk_100001.stock + stk_100009.stock` into that single 100001 row — so `Analytics("inventory_overview", zid="100009")` (`inv_109` in the code) always returns **zero rows**, not just a zero-stock column.

Two approaches were tried and abandoned first: summing `inv_101`/`inv_109`'s own `stock` column per ZID (broken — `inv_109` is always empty, so "100009 Stock" silently showed 0); and adding a 5th `zid=100009` branch directly to the materialized view (worked, verified zero-diff, but **reverted** at the user's request before shipping — that MV DDL isn't kept anywhere in the repo).

**Landed on**: `core/queries.py::get_inventory_zid_stock_split` (registered as `"inventory_zid_stock_split"` in `Analytics`), which queries **only** the 100009 half live from `imtrn`/`caitem` — mirroring the view's own `stk_100009` subquery exactly (`'Finished Goods Store Packaging'`/`'Raw Material Store Packaging'` warehouses, non-blank `xdrawing`, **`GROUP BY ca9.xdrawing`** — load-bearing: an earlier version of this same check without the `GROUP BY` undercounted any 100001 item linked from *multiple* 100009 items, e.g. an `HPI` raw-material item and an `FH`/`FZ` finished-good item both sharing one `xdrawing`). **100001 Stock is deliberately not queried at all** — `views/purchase.py::_render_total_inventory` derives it as `Total Stock − 100009 Stock` (existing combined value minus the live-queried 100009 figure), so the two halves always add back up to the total by construction, not by two independently-computed numbers happening to agree. Verified against real Postgres: reconciles to zero diff across all 2,353 100001 items, no negative values, and item `1122` (the dual-link case) correctly sums to 4500 (1900 + 2600) rather than reporting either half alone.

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
- Real data has at least one legacy row with the `2999-12-31` sentinel date (see Common Pitfall #11) — degrades gracefully (blank date, sorts last) once coerced.

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

Same `2999-12-31` sentinel as Returns Registry (Common Pitfall #11) — excluded at the SQL level here (`<> '2999-12-31'`) rather than coerced client-side.

Wired into `processing/customer_support.py::load_all_delivery_payment_promise()` (same `_ZID_PROJECT` loop pattern as `load_all_glpmt`/`load_all_cacus`) and merged into **three** places — both Customer Support views, plus Collection Analysis → Salesman Due:
- **90-Day Activity** (`build_7day_feed`, renamed from "14-Day" — the window itself changed from 13 to 89 days back, and is now user-adjustable up to 180, see "Time Range slider" below; `core/queries.py::get_sales_7day`'s DO-detail window changed to match its 180-day max. Function/table names kept as `7day`/`14day` in a few internal-only spots — not user-facing, left alone to limit blast radius) — customer-level attribute (comes from `opdor`, not tied to any one voucher), so after the merge it's zeroed out (`NaT`) on every row whose `txn_type != "Delivery"`, **and further** zeroed out on Delivery rows where `promised_delivery <= xdate` (that row's own transaction date) — a promise dated at/before a given delivery has already passed relative to it, so it's not a live promise worth surfacing on that row. Displayed as **"Delivered Date"** (label only — the underlying value is still the salesman's *promised* date, not a confirmed delivery) and **"Promised Payment"**. The Type filter (`cs_type_filter`) defaults to `"Delivery"` on page load rather than `"All Types"`.
- **Latest Sales & Collection** (Customer Support, `build_latest_sc_for_zid`) and **Latest Sale & Collection** (Collection Analysis → Salesman Due, `processing/salesman_due.py::merge_delivery_payment_promise`) — both customer-level tables (one row per customer, no `txn_type` to restrict against), so the gate here is `promised_delivery > last_sale_date` / `> "Sales Date"` instead — a promise dated at/before the customer's most recent actual sale is stale relative to it. Displayed as **"Delivery Date"** / **"Promised Payment"** in both. Internal column name in `build_latest_sc_for_zid`'s output is `delivery_date` (renamed from `promised_delivery` specifically because this table's label differs from 90-Day Activity's "Delivered Date").
  - **On current real data, this gate zeroes out nearly everything** in both customer-level tables — the `opdor` promise-date feature is sparsely populated and clustered around late-2023/early-2024, while most customers' actual sales are far more recent, so `promised_delivery > last_sale_date` is rarely true. This was verified as correct behavior, not a bug, before shipping.

**Sales role cannot see Latest Sales & Collection in Customer Support** — `display_customer_support` only offers that radio option when `st.session_state.user_role != "sales"`; sales users see 90-Day Activity only, with no second option in the radio at all (not just blocked after selection).

**Known pre-existing issue, not introduced by this feature**: `mv_ar_transactions` has a garbage far-future date that always passes any `>= cutoff` filter (Common Pitfall #11) — would have affected the old 14-day window too. Not fixed here since it's a shared MV touching many other pages.

### Returned Date (90-Day Activity only)

Same pattern as promised delivery/payment, for the "Return" `txn_type` instead of "Delivery": `core/queries.py::get_cus_return_entry_date` returns, per customer, the date the salesman entered their most recent return into the mobile Ordering app (`opcrn.xdate`) — the *app-logged* date, distinct from `xdate` on the AR ledger's own Return row (`SRT`/`SRJV`/`IMSA` voucher — when it was actually posted/reconciled).

Deliberately **not** restricted to `xstatuscrn = '1-Open'` like Returns Registry is — by the time a return shows up as a "Return" row in the AR-ledger-backed 90-Day feed, it has already been posted, meaning its `opcrn` status has moved on to `2-Accepted`/`3-Issued`. Confirmed on real data: 128,712 of 128,894 `opcrn` rows are `3-Issued`, only 141 are still `1-Open` — filtering to open-only here would return almost nothing. Same `2999-12-31` sentinel excluded at the SQL level (Common Pitfall #11).

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
Unlike the lead call log above, Customer Support and Marketing → Inactive Outreach read and write the exact same `crm_call_log` rows for a given `cusid` — Inactive Outreach's own UI says so directly ("Call logs are shared with Customer Support"). So `OUTCOMES` (`Promised`, `Paid`, `Not Paid`, `Not answered`, `Dispute`, `Delivered`, `Not Delivered`, `Returned`, `Switched Business`, `Price Issues`, `Other`) is one list used by both surfaces — adding an option here makes it selectable (and consistently displayed) everywhere this call log appears, not just wherever the request for it originated. `_OUTCOME_BADGE` only defines custom colors for 4 of the 11 outcomes; the rest render with a generic gray badge — that's the existing pattern, not a gap to fill in.

### Last 2 Deliveries button (Latest Sale & Collection call-log panel)
`views/customer_support.py::_render_last_do_button`, mounted right after `_render_call_log_panel` in both `_render_merged_sc_table` and `_render_sc_table_zepto` (NOT in 90-Day Activity — that page already has its own always-visible DO-detail expander, unrelated to this). On-click only, via `st.session_state` — the product line items are never fetched for a customer until the button is actually pressed, so selecting a customer to view/log calls doesn't also trigger this extra query every time.

`core/queries.py::get_customer_last_do_items` — one customer's most recent 2 distinct DO vouchers (`GROUP BY voucher ORDER BY MAX(date) DESC LIMIT 2`) and their line items, from `mv_sales_line_items`. Deliberately **not** date-windowed like `get_sales_7day`'s fixed 180-day cap — a slow-moving customer's last delivery could be older than that, and since this is scoped to one customer via `WHERE zid = %s AND cusid = %s` (indexed) plus `LIMIT 2` vouchers, it stays cheap regardless of how far back it has to look.

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

### Single Lead / Edit Lead now expose every marketing_leads column
`views/marketing.py::_render_lead_fields(prefix, defaults=None, show_stage=False)` is a shared form-field renderer used by both `_show_manual_lead_entry` (blank form, `show_stage=False` — a brand-new lead defaults to `lead_stage='New'` via the DB default, no reason to override it at creation) and `_show_edit_lead` (`defaults=row.to_dict()`, `show_stage=True`). Only Full Name and Phone Number are required; everything else in the table — `area`, `platform`, `lead_status`, `is_organic` (tri-state selectbox: Unknown/Yes/No, since the DB column is a nullable boolean), `lead_cost`, `created_time`, and the `ad_*`/`adset_*`/`campaign_*`/`form_*`/`inbox_url` platform-metadata fields — is present as an optional input. The platform-metadata fields are visually grouped under a "(optional)" markdown divider since they're normally only meaningful for a real platform-sourced lead, not a hand-entered one, but they're still directly editable, not hidden behind an expander (Streamlit forms and expanders don't reliably nest well together, and there was no need to risk it).

`_parse_lead_fields(raw)` converts the widget output into typed values — `is_organic` label → `True`/`False`/`None`, `lead_cost` text → `float`/`None` (raises `ValueError` with a user-facing message on a non-numeric entry rather than crashing), `created_date` → a UTC `Timestamp` or `None`. `extra_fields` is deliberately not exposed anywhere in either form — it's per-form custom-question JSON with no natural single-value UI, and editing raw JSON by hand invites more breakage than it's worth.

**Column-order safety, same pattern as everywhere else this session touched INSERT/UPDATE column lists**: `views/marketing.py::_LEAD_UPDATE_COLS` must match `core/queries.py::update_marketing_lead_sql`'s `SET` list exactly, and `_update_lead` builds its params tuple positionally off that list, not by name. Verified end-to-end against real Postgres (both the INSERT path via `build_manual_lead_row` and the UPDATE path via `_update_lead`) before shipping. `_update_lead` also blank-to-`None`s the same set of optional text columns (`_LEAD_BLANK_TO_NONE_COLS`) that `build_manual_lead_row`'s `_blank_to_none` helper already used on the INSERT side, so clearing a field in Edit behaves the same as leaving it blank when creating a lead — `platform`/`lead_status` are the one exception, falling back to `"manual"` instead of `NULL` on both paths (there's no meaningful "unset" state for those beyond that default).

### Call Log page: Area column/filter, "Next Follow Up" rename, one-row filter layout
- **Table 1 (Leads)**: `area` added to the displayed columns and to the search box (now "Search name / company / phone / area"). `build_lead_summary_table` needed no change — `area` was already flowing through from `leads_df` since it starts from a full `.copy()` of the `_LEAD_COLS` SELECT.
- **Table 2 (All Call Logs)**: `core/queries.py::get_all_lead_call_logs` now also joins `l.area` in (previously only `full_name`/`company_name`/`work_phone_number`) — `processing/marketing_leads.py::build_lead_call_log_table` needed no change either, it's a passthrough. `area` is both a new filter (multiselect) and a new displayed column.
- **"Next Visit" → "Next Follow Up"**, renamed everywhere it's user-facing: Table 1's column, Table 2's column, Table 2's filter, and — for consistency, since leaving some instances renamed and others not would read as a mistake — `views/lead_call_log_shared.py`'s call-log entry form field and history badge text (`📅 Next visit: ...` → `📅 Follow Up: ...`). The underlying DB column (`next_visit_date`) and internal Python variable/key names are untouched, matching every other display-only rename this session.
- **Next Follow Up filter redesign**: replaced the old checkbox ("Filter by next visit date") gating a conditional date-range picker with a plain always-visible multiselect of the distinct follow-up dates that are actually scheduled — mirroring how the Outcome filter already worked. This sidesteps the exact problem the checkbox existed to prevent (most calls have no follow-up date, so a date-range filter shown by default would hide nearly everything) without needing a toggle: an empty multiselect selection means "no filter," same as Outcome.
- **All four Table 2 filters (Date Called, Outcome, Area, Next Follow Up) now render in one row** via `st.columns(4)`, in that order.

---

## Collection Analysis — Overview: Salesman Collections & Returns (`views/collection.py`, `analysis_mode == "Overview"`)

Sits directly below the existing "📊 Pivot Table Analysis" expander (`collection.display_entity_metric_pivot`). Salesman selectbox ("Code - Name", built from the union of salesmen appearing in `filtered_data_c`/`filtered_data_r` so a salesman who only has returns, or only collections, in the chosen range still shows up) + a `st.date_input` date range, side by side, then an explicit **"📥 Load Data"** button — nothing queries or renders until it's clicked, per how this was asked to work. Click again after changing the salesman or range to refresh; the previous result stays on screen (via `st.session_state["_ca_overview_sp_result"]`) until then, it isn't cleared just because a filter widget changed.

`processing/collection.py::build_salesman_range_transactions(collection_df, return_df, spid, start_date, end_date)` returns **only** that salesman's Collection + Return rows within the chosen range (both ends inclusive) — Sales/DOs and mobile orders are deliberately excluded, matching what was asked for. Originally built month-locked to "this calendar month" (`build_salesman_month_transactions`, `year`/`month` params) before being generalized to an arbitrary date range in the same session.

- Collections come from `mv_collection_vouchers` (`glvoucher`, one row per voucher already) — `area`/`cusname`/`spname` are already embedded on each row by the query itself.
- Returns come from `get_return_data` (`revoucher`) — **one row per return *line item***, not per voucher, so they're grouped (`sum(treturnamt)`) down to one row per voucher first, same pattern as `views/target_management.py`'s "SR Trn" day-book table.
- Output columns: Date, Type (`Collection`/`Return`), Voucher, Cust Code, Customer, Area, Amount — the Voucher column is literally the RCT/CRCT/BRCT-style collection voucher or SRT-style return voucher code, sorted newest first.
- Date range defaults to the 1st of the current month through today, but is fully user-adjustable — verified against real data with a custom range (Aug 1–20) reproducing the exact same 55 collections + 36 returns the original "this month" version found, and a narrower 3-day sub-range correctly returning fewer rows.

---

## Collection Analysis — Salesman Due → "📊 Statistical Analysis" (`views/collection.py::_render_salesman_due_ar_stats`)

A 5th `sub_report` option alongside Main Due Report / Latest Sale & Collection / Customer Credit Trickle-down / Missing Customers, by customer, reusing the exact same `reports`/`reports2`/`_sd_scope` loading and dual-ZID (`100001 + 100000`) combine logic already on this page — no new data-loading mechanism.

Two metrics, radio-selected, **not from the same underlying customer population**:
- **AR Balance** — `report_cc_with_names.total_due` (Customer Credit Trickle-down), summed per customer via `groupby(["xsub","customer_name"])` — a customer can span >1 salesman row within the trailing 4-month FIFO window (verified: 8 of 696 100001 customers). Verified this sum reconciles exactly to Latest Sale & Collection's own `Current Balance` column for the same customer (e.g. `CUS-005596`: 1794.67 + 5988.33 = 7783.00, both sides).
- **Days Since Last Sale** — the trickle-down table carries **no date column at all**, so this is sourced from Latest Sale & Collection (`report_df["Sales Date"]`, `today − Sales Date`) instead, then **restricted to the AR Balance customer population** — Latest Sale & Collection covers every customer with any sale history (1,281 locally), a much broader set than the trickle-down table's near-zero-balance filter keeps (696 locally); without the restriction the two metrics would silently describe different customer sets. Restriction is a left-merge of the two per-metric frames on `Customer Code` (`["ZID","Customer Code"]` in combined scope, since customer codes are only unique within one ZID — kept as two separately-tagged rows rather than merged across businesses).

Same design as Inventory Analysis's Statistical Analysis mode (mean/median/std/min/max via `st.metric`; modal *histogram bucket* instead of a literal statistical mode; Min/Max/Bins via the same three `st.number_input` widgets as 📈 Order Analytics — not a slider — applied as a genuine population filter before stats/chart/bucket-totals are computed, replacing an earlier 95th-percentile-clip design; a shared `_bucket_mask(idx)` closure keeps the drill-down table and Bucket Totals table's per-bucket item sets identical by construction) — **built standalone here, not as a shared component** (explicit choice over factoring out a shared helper, to avoid touching the already-shipped `views/inventory.py` code as a side effect of this feature).

**Bucket Totals table**: always rendered at the bottom, one row per bucket, columns Customer Count / **Total Balance** / **Avg Days Since Last Sale** — both figures shown together regardless of which metric currently defines the bucket edges, mirroring Inventory's Bucket Totals design exactly (same rationale: e.g. "for customers with a balance of $X–$Y, what's their average days since last sale?").

---

## Sales Analysis — Overall Sales Analysis → Overview → Legacy Sales Report

A second summary-stats table, same visual layout as the existing top-of-page one (`overall_sales.display_summary_statistics_body` — factored out of `display_summary_statistics` specifically so this second table doesn't re-emit the page's `st.sidebar.title` a second time), rendered at the very bottom of "Overview" mode, right after the pivot tables. Exists to let the user directly compare this app's numbers against a separate, standalone family of monthly email-report scripts (`HM_15`/`H_15_1`/`H_15_2`, one per business: HMBR/GI/Zepto) that predate this app and are still emailed out independently — the two had never been reconciled before, and turned out to disagree by a real, non-trivial amount (HMBR: ~4.8% for August 2026).

**`core/queries.py::get_legacy_sales_summary`** (registered `"legacy_sales_summary"`) reproduces those scripts' own sale/return netting logic **in SQL, faithfully, bugs and all** — this is deliberately not a "corrected" number, it's "what would the legacy script's own logic say" so the two can be compared like-for-like:
- Revenue = raw `opddt.xlineamt` (not `altsales − proddiscount`) — confirmed empirically near-identical to `final_sales` for all 3 businesses this pass, so not itself a source of gap, just faithfully matching what the legacy script actually sums.
- A return line is matched to a sale line by **`(xordernum, xitem)` alone** and nets against whichever **month the original sale fell in** — not the return's own date. A SQL `LEFT JOIN` on that same key naturally reproduces the legacy script's own real fan-out bug too: if one `(xordernum, xitem)` has more than one matching return line (e.g. two partial credit notes), the sale row — and its full sale amount — gets duplicated once per match. Confirmed on real August 2026 data this inflates the total: GI +8,360, Zepto +134,820, HMBR +16,760.
- RECT-type returns (`imtemptrn`/`imtemptdt`, the mobile-app return path) are excluded entirely, matching the legacy scripts' own behavior of reporting those in a separate sheet rather than netting them in.
- **ZID 100005 (Zepto) exception**: the legacy Zepto script reads from `opord`/`opodt`, not `opdor`/`opddt` (what every other business's script, and the rest of this app for every ZID including Zepto, uses) — `get_legacy_sales_summary` branches on `zid == "100005"` to match. Confirmed these two table-pairs hold the **exact same order set** (zero orders in either but not the other) but store **different `xtotamt` header values** for the same order (e.g. one real order: 63,199 in `opdor` vs 114,139 in `opord`) — `opdor.xappamt` (this session's own earlier-established AR/GL-reconciled ground truth for Zepto) sits much closer to `opord.xtotamt` than to `opdor`'s own `xtotamt` column, so `opdor.xtotamt` specifically is unreliable for Zepto. Doesn't corrupt this feature's numbers though, since the line-level `opodt.xlineamt` actually summed here was separately confirmed to match `final_sales` closely regardless — flagged as a landmine for anything else built off `opdor.xtotamt` for Zepto.
- Full grand-total reconciliation (all 3 businesses, exact numbers, further decomposition of what the residual gap is once fan-out and RECT are accounted for) is in this session's history, not repeated here — the finding that survives as code is just the faithful reproduction above.

`processing/overall_sales.py::calculate_legacy_summary_statistics` maps the query's one-row aggregate into the exact same 12-key dict shape as `calculate_summary_statistics`, so `display_summary_statistics_body` renders both identically. **Only `Net Sales` is a fair like-for-like comparison** between the two tables — `Total Sales`/`Total Returns` individually are not, since this table's "return" is only counted at all when it matched a sale line in the requested scope (order-matched), unlike the normal pipeline's period-actual return total — captioned explicitly in the UI. Scope (year/month) is whatever's already loaded for the rest of Overall Sales Analysis (`filtered_data["year"/"month"].unique()`), not a separate picker — the two tables are always describing the same period by construction.

**Behind an explicit "📥 Load Legacy Sales Report" button**, not loaded on every page render — same `st.session_state["_os_legacy_stats"]`-persisted pattern as Collection Analysis → Overview's "Load Data" button: the result stays on screen across unrelated reruns until the button is clicked again (e.g. after changing the sidebar's year/month), rather than firing an extra DB round trip on every Overview page load.

### Order/Return Detail Audit (nested under the Legacy Sales Report, once loaded)

Row-level drill-down for auditing *why* the two totals differ, not just *that* they differ — Year + Month selectors (constrained to whatever's currently loaded in `filtered_data`, i.e. "within the sidebar's timeline") + a "🔍 Generate Comparison" button, producing two side-by-side New-vs-Legacy tables for exactly one month.

**Matching key is `xordernum`, not `opddt.xdornum`** (the "DO--..." number `mv_sales_line_items`/the rest of the app calls "voucher") — a real identity split discovered while building this: `opddt` carries both fields on the same row, but the legacy scripts' own join logic only ever uses `xordernum`, and Zepto's legacy table (`opodt`) has no `xdornum` column at all. Since `xordernum` is the only key both systems can be compared on, that's the row grain for the Order Comparison sheet; `do_numbers` (`string_agg` of `opddt.xdornum` per order, comma-joined for the rare 1-order-to-many-DO case, confirmed 1:1 in practice for HMBR August 2026) rides along as a cross-reference column only. Returns need no such bridging — both systems key off `opcrn.xcrnnum` directly.

Four new queries in `core/queries.py`, each scoped to ONE year+month (not lists, unlike `get_legacy_sales_summary` above — this is a single-month drill-down):
- `get_new_order_detail` / `get_legacy_order_detail` — per-`xordernum` totals, "new" always from `opdor`/`opddt` (uniform across ZIDs, matching how the rest of the app treats Zepto) vs "legacy" branching to `opord`/`opodt` for ZID 100005 via the shared `_legacy_tables()` helper (also now used by `get_legacy_sales_summary`, DRY'd up in the same pass).
- `get_new_return_detail` — mirrors `get_return_data`'s real population (`opcdt`/`opcrn` UNION `imtemptdt`/`imtemptrn`, tagged with a `source` column) filtered by the **return's own date**, grouped per voucher instead of returning line-level rows.
- `get_legacy_return_detail` — mirrors `get_legacy_sales_summary`'s `ret_lines`/`sale_lines` CTEs (return matched to a sale via `xordernum`+`xitem`, filtered by the **sale's month**), grouped per return voucher. RECT returns are structurally impossible here (only ever draws from `opcdt`/`opcrn`), so any return voucher present only on the legacy side is guaranteed to be a real credit-note return, not a RECT one — `build_legacy_audit_tables` backfills `Source = "Credit Note"` for those rows.
- All four `SUM(...)` calls are `COALESCE`-wrapped after finding real NULL `imtemptdt.xlineamt` rows for some RECT lines (qty/rate populated, amount genuinely blank in the source data — a real data-quality quirk, not something introduced by this feature) — bare `SUM` over all-NULL rows returns SQL `NULL`, which would've rendered as a blank instead of `0`.

`processing/overall_sales.py::build_legacy_audit_tables` outer-joins New against Legacy per sheet (Order Comparison on `Order Number`, Return Comparison on `Return Voucher`), adds `Delta` (`New Amount − Legacy Amount`, treating a missing side as 0) and `Present In` (`Both` / `New Only` / `Legacy Only`), sorted by `|Delta|` descending so the biggest discrepancies surface first.

**Verified against real HMBR August 2026 data**: all 2,184 orders present in **Both** with only sub-cent rounding deltas (confirms the revenue-formula match established above extends to the transaction level) — but Returns tells the real story: only 449 of 783 return vouchers are `Both`, **319 are `New Only`** (a return genuinely dated in August whose matched sale wasn't — the return-timing-mismatch mechanism, now visible at the exact-voucher level instead of just estimated in aggregate; one single example, `SR--111495`, alone accounts for 104,940 of the gap), **15 are `Legacy Only`**, and all 17 RECT/Mobile-App returns are (structurally, always) `New Only`. Sum of each column reconciles exactly to the corresponding aggregate figures from `get_legacy_sales_summary`/the normal Overview total.

Two inline `st.dataframe` tables (numeric columns formatted, `na_rep="—"`) plus mismatch-count `st.metric` cards, and an in-memory `.xlsx` (`_build_audit_excel`, `pd.ExcelWriter(engine="openpyxl")` into a `BytesIO`, two sheets — "Order Comparison" / "Return Comparison") via `st.download_button`, so the row-level detail can be audited offline too. Result persisted in `st.session_state["_os_audit_tables"]`, same stays-on-screen-until-regenerated pattern as the rest of this page.

---

## Sales Analysis — Order Analytics → "Product Orders" (`views/sales.py`, `sub_mode == "Product Orders"`)

A 4th `oa_sub` option alongside Order Size Distribution / Return Size Distribution / Rolling Average. **Needs zero new SQL** — built entirely from `_oa_data` (the `mv_sales_line_items` pull already loaded for the rest of Order Analytics), via plain pandas. Self-contained: does not participate in the page's separate "Filter by" (Area/Salesman/Product Group/Customer) radio.

Originally shipped as a line-level order table + a grouped-by-voucher table (raw drill-down). **Replaced** with a deal-pattern characterization once the underlying free-goods/MRP-discount mechanics were understood well enough that raw drill-down was no longer needed — this is the actual groundwork for a 100005 (Zepto) project moving from a "buy X get Y free" discount model to quantity-tier pricing (1–12 units @ price A, 12+ @ price B, 24+ @ price C, etc.), and needed to answer: *what free-goods deals are actually being given today, and what's the resulting effective price per unit?*

**Revenue bug found and fixed first**: `totalsales`/`opddt.xlineamt` is wrong for **100005 specifically** — confirmed against a real ERP data pull (order `DO--058487`, item `FZ000030`, a 120-paid + 30-free deal) that `xlineamt = xdtwotax − xdtdisc − xdisval`, where `xdisval` is the line's share of the item-wise **MRP Discount** (GL `07080001`, `xdtcomm_line × xqty`) — but that MRP gap is *already* implicitly priced into `xdtwotax` (wholesale rate `xrate`) vs `xprice` (MRP), so subtracting `xdisval` again double-counts it, sometimes understating revenue by ~50%. True per-order revenue is `opdor.xappamt` (= `xdtwotax − xdtdisc`, no `xdisval` term), confirmed to equal Accounts Receivable in `gldetail` (acct `1030001`) to the cent, and matching what the GL separately labels "MRP Discount" (`07080001`) vs "Honour Discount"/`proddiscount` (`07080002`) as two distinct, additive contra-revenue lines. `opdor.xtotamt` (`= SUM(xlineamt)`) does **not** match AR for 100005. Fix: everywhere in this feature, use `final_sales` (= `altsales − proddiscount`, already computed by `common.data_copy_add_columns`) instead of `totalsales`/`xlineamt` — mathematically identical to `totalsales` for every other ZID (verified against 100001), so this is a safe universal formula, not a ZID-conditional branch. **This same bug likely affects other pages that read `totalsales` directly for 100005** — flagged as a separate follow-up (`task_1e913f5a`), not fixed everywhere in this pass.

**Current design**: product selectbox + a `st.slider("Time Range (months)", 1, 12, value=1)` — trailing N months anchored to the max date actually present in the already-loaded (sidebar-scoped) sales data, not `pd.Timestamp.today()` (a pure in-memory re-slice, same pattern as Customer Support's 90-Day Activity slider). A line is classified as a **free-goods line** when `proddiscount >= altsales * 0.99` — `mv_sales_line_items` doesn't expose `opddt.xdisc` directly, so this is a proxy for the confirmed real signal (`xdisc = 100.00` exactly marks a free line; a partial "Honour Discount" line shows `xdisc` between 0 and 100). Lines are collapsed to one row per **order** (`voucher`) for the selected product — the same product legitimately appears on 2 lines within one order (a paid line + a separate 100%-discounted free line) — splitting `quantity` into `Paid Qty`/`Free Qty` via `np.where(is_free_line, ...)` before the `groupby("voucher").agg(...)`.

Summary metrics (`st.metric`): Orders, orders With Free Goods (count + %), Avg Price — No Free, Avg Price — With Free, and the resulting % price reduction. **Deal Pattern Distribution table**: one row per distinct `(Paid Qty, Free Qty)` combination actually observed (including plain full-price orders at `Free Qty = 0`) — Orders / Total Qty Sold / Total Revenue / Avg Eff Price/Unit / Free % of Qty, sorted by Orders descending. Verified against real Postgres (100005, `FZ000005`, 3-month window): 863 orders, 53.8% with free goods, dominant pattern `12 paid + 2 free` (243 orders) reconciling to an avg effective price of exactly `170 × 12/14 = 145.71` BDT/unit, matching the manual math this whole feature started from; pattern-table totals reconcile exactly to the underlying per-order totals with zero residual.

### Discount Characteristics — MRP Discount vs. Honour Discount (`core/queries.py::get_sales_discount_detail`)

`mv_sales_line_items` doesn't expose `opddt.xdisc`/`xdisval`/`xdtcomm`, so this is the **one genuinely new query** in the whole Product Orders feature — registered as `"sales_discount_detail"` in `Analytics`, loaded via `views/sales.py::_load_discount_detail` (full ZID history, no date filter, same in-memory-reslice pattern as everything else here — cached once, sliced per `_po_months` window change). Returns raw `voucher, date, itemcode, quantity, disc_pct (opddt.xdisc), altsales (xdtwotax), mrp_disc_amt (xdisval), honour_disc_amt (xdtdisc)` per line.

An earlier attempt characterized "Honour Discount" using the `mv_sales_line_items` proxy (`0.5% < proddiscount/altsales < 99%`) and found it nearly flat (~17–21%) across order-value buckets — **this was actually re-discovering the MRP Discount pattern under the wrong name**, a real mislabeling risk given both `07080001` (MRP Discount) and `07080002` (Honour Discount, which also carries the free-goods amount) are collapsed together at the `proddiscount`/`opdor.xdtdisc` level. Redone with the real `opddt.xdisc` field:

- **Free goods and Honour Discount are overwhelmingly independent mechanisms**, not paired the way the original worked example (`FZ000030`: 120 paid @ 8% Honour Discount + 30 free) suggested — verified on real data (100005, 3-month window): only **24 of 1,444** (voucher, item) combos with a free-goods line also carry an Honour Discount on the paid line; 2,666 combos have an Honour Discount with *no* free goods at all.
- **Both MRP Discount and Honour Discount behave as near-fixed, per-product rates**, not order-value- or customer-negotiated ones — e.g. item `FZ000037`: 705 Honour-Discount lines, mean 27.86%, **std only 2.68**; tighter than even within-customer variation (std 9.12 vs. overall std 11.06 across all products/customers). A few products (e.g. `FZ000045`, mean 53.05%, std 10.07) clearly don't fit this pattern and need individual attention.
- **MRP Discount %** (`mrp_disc_amt / altsales * 100`) applies per-line regardless of free/paid status — it's a fixed BDT/unit gap between `xprice` (MRP) and `xrate` (wholesale) for that item, so both the paid row and the free row of the same deal contribute to (and agree with) the same estimate.

Displayed as 4 `st.metric` cards under the Deal Pattern Distribution table (MRP Discount % mean/std, count+% of lines carrying an Honour Discount, Honour Discount % mean±std) — summary-level by design, not a full stats/histogram treatment like Inventory/AR Statistical Analysis, since the point here is characterizing existing per-product rates, not building a general-purpose distribution explorer.

**Not yet built**: revenue-neutral quantity-tier simulation (apply candidate tier prices to historical total qty, compare to actual `final_sales`) — the planned next phase now that free-goods, MRP Discount, and Honour Discount are all characterized per product.

### Quantity-Tier Revenue Simulation (bottom of Product Orders, below Discount Characteristics)

5-tier input (`st.columns(5)` for Qty, `st.columns(5)` for Price — a 6th input row would look cramped, and 5 was the explicit ask) — leave a tier's Qty at 0 to disable it. Tiers are deduped/sorted ascending by threshold; `_tier_price_for_qty(q, tiers_asc)` walks the sorted list and keeps the last (i.e. highest) threshold `<= q`, so a tier's price applies at-and-above its threshold (`Tier 2 Qty=12` prices exactly `12`, not `13+`) — verified against real data at every boundary (`11→170`, `12→156`, `23→156`, `24→145`).

Re-prices **every row of the already-computed `_po_patterns` table** (Deal Pattern Distribution), not just the free-goods ones — a plain full-price order also gets re-priced at its own tier, since under the new model there's no reason its price should stay pinned to the old flat rate. The qty a tier price applies to is **`Paid Qty + Free Qty` (total quantity the customer walked away with under the old system)** — the explicit assumption is that total historical volume moved is held constant; a customer who used to get "12 + 2 free" is assumed to still receive/order 14 units under the new system, now paying tier price for all 14, rather than assumed to cut back to ordering only their old paid quantity. This is a modeling choice, not a certainty — flagged in the discussion that led here, not re-litigated in the UI.

Orders whose total quantity falls below every defined tier (e.g. a 1-tier setup starting at `Qty=5`, order of 3 units) have no defined new price — **their old-system revenue is carried through unchanged** in the comparison rather than silently dropped, with a caption naming how many orders this affected. `Simulated Revenue (New) = Tier Price × Total Qty Sold` per pattern row (summed across all occurrences of that exact Paid/Free combo, matching `Total Qty Sold`'s existing definition). Verified end-to-end against real Postgres (100005, `FZ000005`, 3-month window, tiers `1@170 / 12@156 / 24@145`): dominant pattern `12+2` (243 orders, 14 units/order) — old revenue 495,720 (145.71/unit blended) vs. new 530,712 (156/unit flat) — aggregate actual **1,633,070.32** vs. simulated **1,661,473.00** (**+1.74%**), a directly actionable number for iterating tier prices against.

**Grouped by Order table** (`_po_grouped`), rendered below the line-level table: collapses to one row per `Voucher` via `.groupby("Voucher").agg(...)`, since the same product can appear on >1 line within one order (verified: 3,243 voucher+product combos locally, e.g. order `DO--019260` has item `12933` on 2 separate lines of 100 units each). Aggregation split by column meaning, not uniformly summed:
- `Date`/`Customer Code`/`Customer`/`Area` — `"first"`, since these are identical across every line of the same voucher, not a real aggregation choice.
- `Quantity`/`Altsales`/`Discount (Product)`/`Final Line Amount` — `"sum"`, the selected product's own totals across its line(s) within that order.
- `Discount (Order Total)`/`Total Order Amount` — `"first"`, **not** `"sum"` — these are already order-wide figures merged onto every line of the line-level table above, so summing them across a product's multiple lines in the same order would multiply-count them. Verified on the real dual-line example above that both values are identical across the product's two lines before grouping, confirming `"first"` is safe.

---

## Manufacturing Analysis — "🔄 Warehouse Flow" (`views/manufacturing.py::_render_warehouse_flow`)

An 8th `mfg_view_mode` radio option alongside FG Costing / FG Cost History / RM Rate Trend / RM Requirement / RM Stock Coverage / BOM Variance / MO Detail — for the same 3 entities (`_MANUFACTURING_ZIDS` = 100000/100005/100009). **Per-product** flow: `Raw Material → (MO) → Finished Goods warehouse → (transfer) → Sales Store → (DO) → market`. Independent of MO header/detail data, so it runs *before* the page's MO-empty early-return, not after. Branches into an inner `mfg_flow_mode` radio, both sharing the one cached `flow_raw` load and the `same_wh` (100009) detection at the top of `_render_warehouse_flow`:
- **📅 Choice Timeline** (`_render_flow_choice_timeline`) — a user-chosen `st.date_input` date range, described below.
- **📦 7-Day Stock Target** (`_render_flow_seven_day_target`) — a fixed trailing-3-months policy tool, described further down.

One new query, `core/queries.py::get_manufacturing_flow_detail` (registered `"manufacturing_flow_detail"`) — raw `imtrn` rows, grouped by `(warehouse, doctype, date, item)` and left-joined to `caitem` for name/group (no packcode CASE — this stays within one entity's own warehouses, not a cross-ZID 100001/100009 merge, matching the sibling `get_mo_header_data`/`get_mo_detail_data` queries in the same file). Scoped to a fixed warehouse list passed via `filters["warehouses"]`. Full history, no date filter in SQL, ~30k–150k rows per entity — `views/manufacturing.py::_load_manufacturing_flow` caches it once per ZID; `processing/manufacturing.py::compute_warehouse_flow_by_product` slices an arbitrary date range out of it in Python (opening = sum before start, closing = sum through end, grouped further by `itemcode`) so changing the date picker never re-queries. Originally shipped as a single entity-wide aggregate row — **replaced with one row per product** once it became clear an aggregate hid exactly the kind of item-level detail this feature exists to show; `compute_warehouse_flow` (the old aggregate function) is kept only for the BDT value one-off figures below, which are still intentionally aggregate.

**Real warehouse names and movement doctypes** (`processing/manufacturing.py::WAREHOUSE_GROUPS`), confirmed against live Postgres `imtrn` — not guessed:
- `RE--` = MO receipt into the FG warehouse — confirmed `xdocnum` on these rows literally equals the MO number (e.g. `MO--004946`).
- `TO--` = inter-warehouse transfer, both directions (a warehouse can show `TO--` inflow *and* outflow in the same window).
- `DO--` = delivery order (sale to market), always outbound from the Sales Store.
- **100000**: RM = `Raw Material Store`; FG = `Finished Goods Store` **+ `Manufacturing Store`** combined as one pool — ~17% of MO receipts land in `Manufacturing Store` instead of the main FG store; combining them nets out the internal `TO--` transfers between the two automatically (verified: `fg_other_qty` comes out to exactly `0.00`). Sales = `Sales Warehouse GI`.
- **100005**: RM = `Raw Metrial Warehouse Zepto` — **"Metrial" is the real (misspelled) name in the ERP**, not a typo to fix. FG = `Finished Goods Warehouse Zepto`. Sales = `Sales Warehouse(Zepto)`.
- **100009**: RM = `Raw Material Store Packaging`. FG = Sales = `Finished Goods Store Packaging` — **100009 has no separate sales warehouse at all** (captive packaging entity, no sales team per the Project Overview above) — its `DO--` sales draw directly out of the FG store. The view detects `fg == sales` and shows an explanatory `st.info`; **Transferred is always `0` and both warehouses' opening/closing figures are identical by construction**, not a bug.

**"Transferred Out" (FG side) and "Transferred In" (Sales side) are measured independently, not derived from one another** — found and fixed a real modeling mistake during verification: the two legs of a `TO--` transfer voucher don't necessarily post within the same window. Confirmed on real 100000 data, a 3-month window: **473,174** units left the FG side via `TO--` but only **238,572** had arrived at the Sales side by the window's end — a genuine timing lag between the source and destination legs, not a bug. Using only one "Transferred" figure (either side) would make the *other* section's arithmetic fail to reconcile.

**"Sales: Returns" (`SR--`) is broken out as its own column**, not folded into "Other" — added per an explicit follow-up question ("do returns increase inventory?"). Confirmed against live Postgres this doctype is `xsign = +1` (inventory-increasing) with zero exceptions across all three entities (`100000`: 6,158 rows; `100005`: 17,069 rows; `100009`: 3 rows — all positive), i.e. genuinely a return-driven stock increase, never a decrease.

**"Other" columns** (FG and Sales, each) cover every doctype besides `RE--`/`TO--`/`DO--`/`SR--` — e.g. `ISS-` (issues), `RECA`, and (at 100005's Sales warehouse specifically) dozens of small legacy numeric doctypes (`0001`–`0029`) tied to now-inactive regional Depot warehouses from ~2020–2024. Both sides' exclusion set is the **same full superset of all four known doctypes**, not just the doctypes that side "normally" sees — required for 100009's merged FG=Sales warehouse, where a per-side-only exclusion double-counted: an item's `RE--` receipt was appearing in both `FG: MO Added` *and* `Sales: Other` before this fix, since `RE--` was never excluded from the sales-side "other" filter. For the two entities with a genuinely separate FG/Sales warehouse this wider exclusion is a no-op (`RE--`/`SR--` simply never occur on the "wrong" side there). Added so `Opening + inflows − outflows + Other == Closing` reconciles **exactly** — verified **per item** against real Postgres for all three entities with zero residual across every row (`Closing` itself is always computed independently as a true cumulative balance, not derived arithmetically, so this is a genuine correctness proof, not a tautology). The TOTAL row at the bottom of the table is summed directly from the (possibly search-filtered) displayed rows, not recomputed separately, so it can never silently disagree with what's shown above it.

**Value figures (BDT)** (Choice Timeline mode) are inventory-cost basis throughout (`imtrn.xval`, i.e. `stockvalue`-equivalent), **not sales revenue**, and remain **entity-wide aggregate** (not per-product, per what was asked) — RM value start/end, FG warehouse value start/end, Sales Store value start/end, and "Total Sold in Period" (the COGS value of everything that left via `DO--` in the window, not the amount billed to customers) all use the same cost basis for internal consistency, explicitly captioned as such.

### 📦 7-Day Stock Target mode (`processing/manufacturing.py::compute_seven_day_stock_target`)

The actual policy question this whole feature exists to answer: **how much finished goods should sit in the Sales Store per product to always cover 7 days of demand, and how far is the business from that today** — sized so a stockout can be avoided. Fixed trailing 3 months (not user-adjustable, unlike Choice Timeline — a deliberately narrower, single-purpose tool), split into **non-overlapping 7-day segments walking backward from `pd.Timestamp.today()`** (13 segments = 91 days for a 3-month window; any leftover days short of a full 7-day segment are dropped, not padded). Every metric is computed **independently per segment** (via the same already-verified `_bal`/window-sum logic as `compute_warehouse_flow_by_product`, not rolled forward incrementally — a deliberate choice to reuse proven logic over a cleverer-but-riskier approach) and then **averaged across segments** — verified against real Postgres that this exactly reproduces a manual per-segment calculation (item `CF000003` at 100000: 13 segment values `[11108, 1890, 7340, 3310, 1053, 8620, 3050, 12390, 1445, 7440, 2123, 0, 0]` → mean `4597.615...`, matching the function's output to the full decimal).

**Stock Target (Qty)** = average 7-day `Sales: Sold (DO)` — the number the whole feature is built around. **Est. Unit Cost** = `SUM(net_val) / SUM(net_qty)` for MO receipts (`RE--`) in the FG warehouse across the *whole* 3-month window (one stable ratio, not averaged per-segment — a single window-long ratio is less noisy than averaging 13 individual per-segment ratios, and only the totals matter for a ratio). This is the **same `imtrn.xval` cost basis used throughout this feature** — deliberately *not* `compute_mo_cost`'s `cost_per_unit` (a different, pre-existing costing methodology elsewhere on this page: raw BOM material cost only, no overhead) — mixing two different cost bases into one feature would be confusing. Items with zero MO receipts in the window (real case found: 100005's `FZ000024` Draino Powder — sold 652 units/week on average but produced 0 in this window) get `NaN` cost/value, not silently `0` — they still show a real Stock Target (Qty), just no BDT figure, with the excluded count captioned.

**"New/Target" is the same number answering two different questions** — `SUM(Stock Target Qty × Est. Unit Cost)` across every product with a cost basis. Under a constant-7-day-buffer steady-state assumption, "how much should be *produced* per week to keep up with demand" and "how much should be *standing* in the Sales Store as a buffer" are mathematically the same quantity (one week of demand, valued at cost) — so both comparisons below share this one target figure, applied against two different **Current** baselines:
- **Production (MO) value**: Current = average, across segments, of that segment's *total* MO-receipt value summed over every product. New = the shared target. A positive Difference means current production is running behind what's needed to sustain the buffer.
- **Sales Store FG value**: Current = average, across segments, of that segment's *total* Sales Closing balance value summed over every product. New = the same shared target. A positive Difference means the business is currently under-stocked relative to the 7-day target; negative means overstocked.

Verified against real Postgres for all three entities: 100005 shows both differences positive (+36,901 MO, +67,719 FG) — currently under-producing *and* under-stocked relative to its own 7-day target, a concrete, actionable finding. 100000 and 100009 both show negative FG differences (currently holding well above 7 days of buffer).

---

## WhatsFly Messaging (`views/marketing.py` → "💬 WhatsFly Messaging" mode)

Single-message test panel, per `Whatsfly_Integration_docs/whatsfly-integration-guide.md`. **Current build phase only**: send one message to one hand-entered number using an approved template (or plain session text), see the raw response. **No receive-side/webhook handling in this app** — that's a separate FastAPI service, a later phase.

- **Credentials**: `config/whatsfly.ini` (gitignored, section `[whatsfly]` with `api_token`/`phone_number_id`) via `config/settings.py::get_whatsfly_params()` — returns `None` (never raises) if missing, so the view shows a setup `st.warning` instead of crashing.
- **`core/whatsfly.py`** — thin client: `send_text`, `get_templates`, `send_template`, `upload_media`. All raise `requests` exceptions up to the view, caught and shown via `st.error` (exploratory phase, not a hardened wrapper).
- **Template list is defensively parsed, never assumed one fixed schema** — this account's real shape is `{"status": "1", "message": [...]}` (a genuinely surprising wrapper key — `"message"`, not `"data"`/`"templates"`). `_wf_normalize_templates`/`_wf_extract_components` try several wrapper/body-text key guesses with a recursive `{{`-scan fallback, and always show the raw JSON in an expander so a wrong guess is visible, not silent. Confirmed real per-template fields: `id` (WhatsFly's own short internal id), `template_id` (a much longer, different, Meta-style id — two separate table columns, not collapsed into one), `whatsapp_business_id`, `template_name`, `template_type`, `template_category`.
- **Beautified preview + live variable substitution**: `_wf_render_bubble` renders header/body/footer as a WhatsApp-style chat bubble (`_wf_format_whatsapp_markup` converts `*bold*`/`_italic_`/`~strike~` to HTML **after** `html.escape()`-ing the source — verified this neutralizes an injected `<script>` tag rather than rendering it). `_wf_substitute_preview` re-renders the body live as `{{token}}` variable inputs are filled in (highlighted yellow, dimmed `[token]` placeholder while blank). Variable detection is scoped to the **body** only, via `_wf_extract_variable_tokens` — Meta templates support two placeholder formats chosen at template-creation time, never mixed within one template: **positional** (`{{1}}`, `{{2}}`) or **named** (`{{cusname}}`, `{{cuscode}}`); the shared helper returns the raw tokens either way (digit strings vs names) so both panels (this one and Direct WhatsApp below) detect variables correctly regardless of which format a given template uses.
- **Send contract — confirmed via two real dashboard-generated examples**, not guessed (no published contract exists anywhere — checked the guide word-for-word and WhatsFly's own public docs site, both dashboard/UI-only, no REST reference). **Standing workflow, saved to session memory**: the user pastes each new template's real dashboard example specifically so this stays evidence-based — a different header type (video/document) or a buttons component should get the same treatment, not extrapolation from the pattern below.
  - Endpoint: `POST /whatsapp/send/template`, flat params, no nesting.
  - **Naming trap**: the send param is called `template_id` but wants WhatsFly's **short internal `id`** (e.g. `435966`) — *not* the longer `template_id` field the template-*list* response returns for the same template. Two API surfaces reuse one field name for two different values.
  - Variables: **`templateVariable-<NAME>-<n>`** per variable, keyed by a NAME — not a numbered/generic array. Default for that Name field: if the template body itself uses **named** placeholders (`{{cusname}}`), the token *is* the name, used directly; if it uses **positional** placeholders (`{{1}}`, `{{2}}`), there's no name in the text itself, so `_WF_DEFAULT_VAR_NAMES = ["CUSNAME", "CUSCODE"]` pre-fills `{{1}}`/`{{2}}`'s Name field since both real dashboard examples used this pair — a 3rd+ positional variable has no confirmed name and stays blank. Either way the Name field stays user-editable.
  - Header image: **`template_header_media_url`** — a plain hosted URL (not `media_id`/`media_url`/`media_type`, both real wrong guesses along the way). `upload_media` (`POST /whatsapp/upload/media`, multipart, field `media_file`) uploads automatically on file select — triggers as soon as a file is chosen, keyed by `(filename, size)` in `st.session_state` so it doesn't re-upload on unrelated reruns — and its hosted URL feeds this field directly; a manual-URL text input is the fallback if skipping upload. `st.image(uploaded_file, width=200)` shows a local thumbnail immediately regardless of upload success.
  - `template_name`/`language_code` are absent from both real examples — dropped from the flat default entirely.
  - A **"Meta Cloud API style"** nested payload shape (`template: {name, language: {code}, components: [...]}`) is kept in the UI only as a documented fallback — WhatsFly's actual endpoint wants the flat shape above; that option's own `template_name`/`language_code` inputs render only when it's selected.
- **Response envelope handled both ways** (string `"1"`/`"0"` vs boolean, per the guide's own documented inconsistency) — `_render_wf_response` also pattern-matches Meta's generic "does not exist... graph-api" Graph API rejection (a real account/WABA-permission issue on WhatsFly's or Meta's side — invalid/deleted phone number ID, a token without permission on that WABA, incomplete Business Verification, or a disconnected integration partner — not a request-shape bug) and points at Meta Business Manager instead of the request shape.
- **Account-wide, not per-ZID** — one WhatsApp Business number across the whole group, so `_show_whatsfly_messaging()` takes no `zid` argument.

### Direct WhatsApp (`views/marketing.py` → "📨 Direct WhatsApp" mode)

Same single-message test panel as WhatsFly Messaging above, but calls Meta's own WhatsApp Cloud API directly (`graph.facebook.com`) — no WhatsFly in between. Test-number-only, against a separate Meta test WABA + test number so it never touches the real WhatsFly-routed production number.

- **Credentials**: `config/direct_whatsapp.ini` (gitignored, section `[direct_whatsapp]` with `access_token`/`phone_number_id`/`waba_id`, optional `graph_api_version` defaulting to `v21.0`) via `config/settings.py::get_direct_whatsapp_params()` — same never-raises-returns-`None` pattern as `get_whatsfly_params()`.
- **`core/direct_whatsapp.py`** — thin client: `send_text`, `get_templates`, `send_template`, `upload_media`. Unlike `core/whatsfly.py`, this follows Meta's own **published, documented** Cloud API contract directly — no defensive multi-key guessing needed, since the shape isn't reverse-engineered.
- **Simpler than the WhatsFly panel in one real way, because Meta's actual contract removes a WhatsFly-specific quirk**: there's exactly **one** request shape (no "flat vs Meta Cloud API style" guessing radio) — `POST /{phone_number_id}/messages` with the nested `template: {name, language: {code}, components: [...]}` body.
- **Two Meta placeholder formats, both supported** — chosen at template-creation time in Meta's own template editor, never mixed within one template: **positional** (`{{1}}`, `{{2}}`) or **named** (`{{cusname}}`, `{{cuscode}}`). `_wf_extract_variable_tokens` (shared with the WhatsFly panel) returns the raw `{{...}}` tokens in order — digit strings for positional, names for named — and `is_named_format` (first token non-numeric) decides how the send payload is built: positional sends plain `{"type": "text", "text": v}` parameters matched by array order; named sends `{"type": "text", "parameter_name": tok, "text": v}` per parameter, matched by name. No free-form naming UI here (unlike WhatsFly's editable Name field) — for a named template the parameter name comes straight from the template body's own token, since Meta ties the name to the approved template itself, not to metadata chosen at send time.
- **Reuses the WhatsFly panel's generic rendering helpers** (`_wf_format_whatsapp_markup`, `_wf_render_bubble`, `_wf_substitute_preview`, `_wf_extract_variable_tokens`, `_wf_extract_components`) rather than duplicating them — those are plain WhatsApp-template markup/preview helpers, not WhatsFly-specific, and Meta's own template shape (`components: [{type, text}]`) is exactly what `_wf_extract_components` already parses. `_wf_substitute_preview` matches by token text (not by casting to int), so it works for both placeholder formats.
- Header image: upload via Meta's own `/{phone_number_id}/media` endpoint → `image: {id: <media_id>}`, or a plain hosted URL fallback → `image: {link: ...}`.
- **Response handling**: success = 2xx with no top-level `"error"` key (surfaces the `wamid...` message id); failure = a nested `error: {message, type, code, ...}` object, rendered directly rather than guessed at.
- **Account-wide, not per-ZID**, same as WhatsFly Messaging.

### WhatsApp Webhook Receiver (`whatsapp_webhook/`, standalone FastAPI service)

The receive-side counterpart to Direct WhatsApp above — that panel only sends; a webhook is the only way delivery/read/failure status and inbound replies ever arrive (a send's own 200 response only means "Meta accepted the request"). A fully separate service, not part of the Streamlit app process: own FastAPI app, own `.env` credentials (`WHATSAPP_VERIFY_TOKEN`, `META_APP_SECRET`, distinct from `config/direct_whatsapp.ini`), and its **own Postgres database** (`schema.sql`, isolated from `da`) — though it shares the `streamlitEnv3.10.13` Python env rather than a dedicated venv. Build reference: `WhatsApp_Integration_docs/whatsapp-webhook-build.md`; run instructions: `whatsapp_webhook/README.md`.

**Current phase**: local dev only, tunneled with ngrok, registered against Meta's sandbox app + test number. Production deployment (Windows Server 2016, reverse proxy, TLS, persistent service) is explicitly deferred.

- **`main.py`**: GET `/webhook/whatsapp` echoes `hub.challenge` back once `hub.verify_token` matches (Meta's one-time verification handshake); POST verifies `X-Hub-Signature-256` (HMAC-SHA256 over the **raw** body, constant-time compare) before anything else — reject 403 on mismatch, this is the entire trust boundary. A verified payload gets logged to `webhook_events` synchronously (idempotency/audit backbone), then routed via a `BackgroundTask` so Meta gets its `200` immediately ("respond fast, process later"). Malformed JSON after a valid signature → `400` (Meta won't retry 4xx); a DB/processing failure inside the background task is caught and recorded as `processing_status='failed'`, never raised (the response already went out).
- **`handlers.py`**: routes each `changes[].field` — `messages` (both shapes: inbound `messages[]` array and outbound `statuses[]` array in one field) gets full handling; the four `message_template_*`/`template_category_update` fields upsert into `templates`; everything else (`phone_number_quality_update`, `account_update`, etc.) lands in the generic `account_alerts` audit table rather than being dropped. Status events are deduped on `(wamid, status)` and only move `messages.current_status` forward (a rank guard: `sent < delivered < read`, `failed` terminal) since Meta's at-least-once delivery can replay an earlier status after a later one already landed.
- **Known gap, documented in the README**: outbound sends via `core/direct_whatsapp.py` don't write into this database at send time (the two services aren't wired together yet) — so a status callback for one of those can arrive before this service has ever seen the message. `db.ensure_outbound_stub` creates a minimal placeholder `messages` row on first sight of an unrecognized `wamid` (`ON CONFLICT DO NOTHING`, so a future real send-side integration would just be a no-op here) — keeps the `message_status_events` FK intact without requiring that integration yet.
- Verified via `fastapi.testclient.TestClient` against a mocked DB layer (no real Postgres needed) — verify handshake (correct/wrong token), signature rejection (including that a DB failure while logging the rejection still returns 403 rather than 500), the happy path, and malformed JSON all confirmed before shipping.

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
10. **TOTAL row + `.style.format()`**: a manually-built `dict`-based TOTAL row (`{c: "" for c in df.columns}`, then overwriting some cells) crashes at render time (`ValueError: Unknown format code 'f' for object of type 'str'`) if any column left at its `""` default is ALSO covered by a numeric format spec — e.g. a column deliberately excluded from being summed (like a per-unit cost, where "total" is meaningless). The exception surfaces deep in Streamlit/pandas Styler internals (`_translate_body`), not at the `.style.format()` call itself, so a bare `try/except` around that call won't catch it. Fix: use `np.nan` (not `""`) for any TOTAL-row cell in a numerically-formatted column that has no real value — `na_rep` in `.style.format(fmt, na_rep=...)` renders it cleanly instead.
11. **Garbage sentinel dates in several ERP tables** — `opcrn.xdate` can be `2999-12-31` (an "unset" placeholder, seen in Returns Registry, the promise-date queries, and the return-entry-date query); `stock.year` can be `2102` (see SQL Rule #5 above); `mv_ar_transactions` has at least one `2102-10-11`-class garbage date that always passes a `>= cutoff` filter regardless of window size. Always either `pd.to_datetime(..., errors="coerce")` client-side or exclude at the SQL level (`<> '2999-12-31'`) — a naive `pd.to_datetime()` on these crashes the whole page with `OutOfBoundsDatetime`.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
