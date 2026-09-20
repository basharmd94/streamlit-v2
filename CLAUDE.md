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
data/                   # targets.json, public_holidays.json, warehouse_filters.json, hierarchy.json,
                        #   level_s_mapping.json, labels.json (gitignored runtime — the last three moved
                        #   here 2026-09-14: all editable via the in-app "⚙️ Config Editor"
                        #   (views/financial.py), so all three get hand-edited directly on the server
                        #   and are synced between environments by copy/paste, not git)
                        # ls_account_notes.json (committed — read-only, no in-app edit path)
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
- `prmst.xstatusemp` -> employee status. `'A-Active'` = payroll is currently issued to that
  person — the confirmed way to check if someone is still active (other real values seen:
  `R-Resigned`, `T-Terminated`, `H-Hold`, `D-Dismissed`, blank/`NULL`). Not yet wired into any
  view as of 2026-09-20 — see session memory for why (real rollout of this field's *data* is
  in progress on the live server, don't trust it against the local Postgres mirror yet).
- `prmst.xdisease` -> despite the name, this is NOT medical data — it holds the salesman's
  **current working area(s)**, comma-separated free text (e.g. `"Ibrahimpur, Askona"`). Take
  the value exactly as stored, don't reformat. **As of 2026-09-20 this is mid-rollout on the
  live server** — the local Postgres mirror still has old placeholder junk in most rows
  (`"Ok"`, `"OK."`, etc.), not real area data — confirm against the live server, not local,
  before building anything that reads this column. Wired into
  `processing/commission_campaigns.py::derive_spid_group_map` (see B.1/3/4 below) — correctly
  returns nothing locally until the live rollout finishes.
- `cacus.xstate` -> despite the "state" name, this is NOT a geographic state — it's a
  sales-zone/channel classification per customer area (`xcity`), confirmed real values (both
  100001 and 100000): `Dhaka retail`, `District`, `Dhaka General`, `Nawab pur`,
  `Kawranbazar`, `Alubazar`, `Imamgonj`, `Rahima enterprise`, `Sylhet Retail`. Already used
  elsewhere as a Retail/District split (`processing/ar_analysis.py`'s "Market" column: `xstate`
  contains "retail" case-insensitive -> "Retail", else "District"). Per `xcity` the value is
  highly (not 100%) consistent — a handful of stray mistagged rows per area don't change the
  majority. Also used by `processing/commission_campaigns.py::build_area_group_map` (see
  B.1/3/4 below) to derive salesman payout-group membership live, matched against
  `prmst.xdisease`.
- **A person is a real, currently-active salesman if they appear in the `op*` tables**
  (`opdor`/`opddt`/etc., i.e. `Analytics("sales", ...)`'s own `spid`) — **not** by checking
  `xemp`'s prefix. `SA--` is the common prefix but real exceptions exist (confirmed on real
  100001 data: `FI--000003`, `AD--000028`, `AD--000099` all post real sales). Any salesman
  picker that already sources its options from actual sales/collection rows (rather than
  filtering `prmst` by an `SA--` pattern) is already correct on this — e.g.
  `processing/commission_campaigns.py::derive_spid_group_map`'s `valid_spids` filter, which
  restricts prmst-derived payout groups to spids that actually appear in `Analytics("sales")`
  (otherwise non-salesman prmst rows with an area on file would show up too).

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

### Redesigned into a per-customer reconciliation table (2026-09-16)

The original table was one row per `glpmt` entry — useful as a raw log, but it didn't actually answer the question it exists for. Real business story, clarified by explicit follow-up: after a delivery (DO), a salesman gets the customer to promise a payment date and logs it into the mobile app (`glpmt.xpaydate`) on the customer's behalf; the point of this table is to check whether that promise actually panned out — did a real collection (RCT) come in, and does the whole chain (DO → promised payment → actual collection) hold together. **Now one row per customer instead of per entry** — their latest DO, latest RCT, latest `glpmt` entry, and current AR balance, side by side, so staff can see the whole story at a glance instead of piecing it together from a flat log.

**`processing/glpmt_reconciliation.py`** (new) does the actual per-customer aggregation, called from `render_glpmt_panel` after the *existing* filters (salesman/customer/date-of-entry) are applied to the raw `glpmt` rows exactly as before — filter first, then collapse to latest-per-customer, so a salesman/date filter still narrows things the same intuitive way:
- **`_latest_glpmt_entry_per_customer`** — the customer's most recently-**entered** row (by `entry_time`/`ztime`, the latest *action* the salesman took), carrying that one row's own `paydate`/`payamt`/`paytype`/`bankdetail`/`paystatus`/`remarks` along with it. This is "the latest payment date entered by the salesman," per the explicit spec — not a separate MAX(paydate) across all of a customer's entries.
- **`_latest_do_per_customer`** — sourced directly from `Analytics("sales", ...)` (full history, `mv_sales_line_items`-backed, matching this app's `final_sales = altsales − proddiscount` convention throughout), grouped by `(cusid, voucher)` first (a DO can have several line items) then by customer to find the single latest DO — date, voucher number, and total amount.
- **`_latest_rct_per_customer`** — sourced directly from `Analytics("collection", ...)` (`mv_collection_vouchers`, already one row per voucher) grouped by customer for the single latest RCT — date, `glvoucher` as the voucher number, and amount.
- **Customer Balance reuses `build_latest_sale_collection_report`'s own AR balance calc UNCHANGED** (`processing/salesman_due.py`, the same one Collection Analysis → Salesman Due already uses) — including its existing `|balance| < 100` near-zero filter. The merge onto it is an **INNER join**, so a customer whose balance has already gone near-zero silently drops out of this report entirely, by construction — **exactly the "if the balance goes close to 0, remove that customer from the table" behavior asked for**, reusing already-proven logic rather than reimplementing a threshold here.
- **Two sanity checks**, both anchored on the DO date, per explicit ask — **"Payment After DO"** (is the salesman's promised payment date genuinely after the DO it's presumably for) and **"RCT After DO"** (is the actual collection date genuinely after that DO). Both render `✅`/`⚠️`/blank (`_after_flag` helper) — blank specifically means "nothing to compare yet" (one side has no date on file), never treated as a failed check. **Deliberately does *not* also check a Return-voucher date as a third signal** — the explicit ask said "RCT/return date," read here as "the collection event's date," not a separate return check; flagged as a possible follow-up if that reading was too narrow.

**Column order** (per explicit spec): Cust Code / Customer / Emp Code / Salesman → Latest DO / DO Number / DO Amount → Latest RCT / RCT Number / RCT Amount / RCT After DO → Customer Balance → Latest Payment Date / Payment After DO → Entered Date / Amount Entered / Type / Bank Detail / Status / Remark. `RCT Number` was added beyond the literal spec (not explicitly asked for, unlike `DO Number`) since it's free — already pulled from `mv_collection_vouchers.glvoucher` — and directly mirrors `DO Number`'s own purpose (cross-referencing a specific voucher in the ERP).

Verified end-to-end against real Postgres (100001): 7 distinct customers had a `glpmt` entry, 5 survived the balance-based inner join (2 correctly dropped as near-zero); every `Payment After DO`/`RCT After DO` flag across all 5 rows was independently hand-verified correct against the raw dates (e.g. a customer with `paydate` 2026-05-27 against `do_date` 2026-09-01 correctly flagged `⚠️`; one with `paydate` 2026-09-02 against `do_date` 2026-08-03 correctly flagged `✅`). Confirmed live in a scratch preview — full 19-column table renders correctly end to end, values matching the independently-verified data exactly.

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

### Call Coverage Matrix condensed into aging buckets, axis swapped (2026-09-16)

The matrix (`views/customer_support.py::_render_coverage_matrix`, `processing/customer_support.py::build_callcoverage_matrix`) — mounted in both 90-Day Activity and Latest Sales & Collection's per-salesman tables — had one row (or, for Type, column) per **exact** `days_since_sale` integer value. Confirmed against real Postgres this genuinely was unusable, not just theoretically large: for ZID 100001's all-time "Latest Sales & Collection" data alone, customers spanned **13 to 4,525 days since last sale (12+ years) — 2,403 distinct day values**, i.e. 2,403 rows. The Salesman/City column axis was separately unbounded too: **321 distinct salesmen and 119 distinct areas**, all-time, most no longer relevant (ex-employees, historical one-off sales).

**Two changes, both from explicit follow-up once the row-explosion was diagnosed:**

1. **`_DAY_BUCKETS`** (`processing/customer_support.py`) — `days_since_sale` collapsed into 8 coarse ranges instead of one bucket per exact day: `0-7 / 8-14 / 15-30 / 31-60 / 61-90 / 91-180 / 181-365 / 365+` — finer near the top where day-level precision actually helps triage (a customer who just went quiet vs. one who's been quiet a week), coarser once an account is genuinely stale (a 2-year-dormant customer and a 3-year-dormant one don't need different rows). `_bucket_label(days)` maps a raw day count to its bucket; an ordered `pd.Categorical` keeps column order as the natural sequence, not alphabetical.
2. **Axis swapped** — buckets are now ALWAYS the column axis (only 8, reads fine as columns); the chosen "Group by" dimension (Salesman/City/Outcomes/Type) is always the row axis, since Salesman/City can still run to dozens-or-hundreds of values, which scroll naturally as rows but not as columns. This flips the original day-as-rows layout for every dimension, including Type (kept consistent even though Type's own column set — a handful of txn_types — was never the problem).

**`_ACTIVE_WITHIN_DAYS = 90`** (~3 months, exact value from explicit ask) — Salesman/City rows are additionally scoped to whoever has had **any** sale within that window, checked against the full pre-dedup input (`df.loc[df["days_since_sale"] <= 90, dim_col]`) rather than the already-deduped-to-one-row-per-customer `base` — so a salesman/area counts as active if ANY of their customers bought recently, even if the specific customer row shown for a given (dimension, bucket) cell is an old one. Verified against real Postgres (100001): Salesman rows dropped from 321 (all-time) to **37** (active in 90d) — a large cut, since salesmen genuinely leave/change roles. City rows only dropped from 119 to **111** — areas don't "quit" the way salesmen do, so the 3-month filter is far less impactful there; still a real improvement (2,403 exact-day rows → 111 scrollable ones), just a smaller one than for Salesman, worth knowing so 111 isn't mistaken for a bug.

**"Detect which area a filtered salesman sells in" needed no new code at all.** Both call sites (`_render_merged_sc_table`, `_render_sc_table_zepto`) already pre-filter their `df` to one salesman *before* calling `_render_coverage_matrix`, whenever the page's own "Salesman" selectbox has a value picked — this was already true before this change. Since the new active-scoping check (`_ACTIVE_WITHIN_DAYS`) runs against that same already-filtered `df`, picking "City" as the dimension automatically shows only the areas that filtered salesman actually covers — the salesman-narrows-area cascade falls straight out of the existing filter-then-aggregate architecture. Verified directly: filtering to a real salesman (`Muhammad Zakir Hossain`, 532 rows, covering 20 real areas total) and choosing City dim produced exactly 4 rows, and every one of those 4 areas was confirmed to be a subset of that salesman's own real area set — no leakage, no special-case logic needed.

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

## Customer Data View (`views/sales.py::display_customer_data_view_page`)

Two radio modes: **🔍 Individual DO/SR Check** (the original page — customer-mandatory, salesman/product/area-optional transaction lookup, unchanged, just moved into its own `_render_individual_do_sr_check`) and **⚖️ Rate Mismatch Audit** (new).

### Rate Mismatch Audit

Mobile orders land in `opord` (header) / `opodt` (line) — a **separate table pair from the main `opdor`/`opddt` sales flow**, not just a differently-named view of the same data. `opord.xordernum` only ever has one of two real prefixes (confirmed against live data): `COMO` (open mobile order) or `CO--` (confirmed, still pre-delivery) — it becomes a `DO--` voucher in the main flow once delivery is actually confirmed, which is out of scope here since the point is catching a bad rate *before* that happens.

`opodt.xrate` is the real, confirmed rate column (what prints on the invoice) — checked per line against two reference prices, both already established elsewhere in this app:
- `caitem.xstdprice` — the item's standard price.
- Wholesale special price — `GREATEST(caitem.xstdprice - opspprc.xdisc, 0)` at the item's lowest-`xqty` tier, the exact same formula `get_opspprc_data`/`get_inventory_overview` (Purchase Analysis's Total Inventory Overview) already use for "WH Price".

**Only genuine mismatches are returned** — `core/queries.py::get_rate_mismatch_audit` (registered `"rate_mismatch_audit"`) filters to `xrate IS DISTINCT FROM` both reference prices at the SQL level, per explicit ask ("the ones that match don't need to be shown"); matching *either* reference price is considered correct.

UI: a date picker (today, back to 30 days) + an explicit "🔍 Run Rate Audit" button — nothing queried until clicked, same stays-on-screen-until-rerun pattern as every other on-demand report in this app (`st.session_state["_cdv_audit_result"]`). One row per mismatched line: Voucher, Date, Cust Code, Customer (`cacus.xshort`), Item Code, Item Name, Qty, Invoiced Rate, Std Price, WH Price, plus a CSV download.

Verified against real data: a real September 2026 mismatch found (`COMO418641`, item `1304` "Spirit Level 20 Inch" — invoiced at 265.00 against a std price of 280.00 and WH price of 275.00, matching neither) — confirmed by direct SQL first, then the exact same single row surfaced through the UI end to end.

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

## WhatsFly Messaging (`views/marketing.py` → "💬 WhatsFly — Send Single Message" mode)

The primary WhatsApp send surface (Direct WhatsApp below is shut off — see its section). Single-message send panel, per `Whatsfly_Integration_docs/whatsfly-integration-guide.md`: pick a customer or enter a number by hand, send an approved template or plain session text, and see that number's own conversation history (received via the `whatsapp_webhook` service, fully wired — see its own section further down) right in the same panel.

- **Credentials**: `config/whatsfly.ini` (gitignored, section `[whatsfly]` with `api_token`/`phone_number_id`) via `config/settings.py::get_whatsfly_params()` — returns `None` (never raises) if missing, so the view shows a setup `st.warning` instead of crashing.
- **`core/whatsfly.py`** — thin client: `send_text`, `get_templates`, `send_template`, `upload_media`. All raise `requests` exceptions up to the view, caught and shown via `st.error` (exploratory phase, not a hardened wrapper).
- **Template list is defensively parsed, never assumed one fixed schema** — this account's real shape is `{"status": "1", "message": [...]}` (a genuinely surprising wrapper key — `"message"`, not `"data"`/`"templates"`). `_wf_normalize_templates` tries several wrapper keys with a fallback, and always shows the raw JSON in an admin-only expander so a wrong guess is visible, not silent. Confirmed real per-template fields: `id` (WhatsFly's own short internal id), `template_id` (a much longer, different, Meta-style id — two separate table columns, not collapsed into one), `whatsapp_business_id`, `template_name`, `template_type`, `template_category`, `locale` (the real language field — not `language`/`language_code`/`lang`, all wrong guesses before this was confirmed live).
- **`_wf_extract_components` — WhatsFly's REAL confirmed shape, fetched and inspected directly against the live account, not guessed**: the template list response is NOT the Meta-style `components: [{type, text}]` shape at the top level — that only exists nested inside a separate stringified `template_json` field (kept for admin raw-data debug only). The real top-level fields are `body_content`/`header_content`/`footer_content` — plain, already-unicode-decoded text, ready to display as-is. **This was a real, shipped bug fixed mid-session**: the old fallback (scanning every string for `'{{'`) found none in `body_content` (WhatsFly's own placeholders are `#NAME#`/`#!NAME!#`-style, not `{{n}}`) and instead matched the whole `template_json` blob (which does contain literal `{{1}}` inside its own still-JSON-escaped text), rendering that raw blob as if it were the message body. The old Meta-style `components`-list path is kept as a fallback only, for a template that somehow lacks `body_content` — not needed for any of this account's real templates.
- **`_wf_extract_variable_map`** — ground truth for what each positional variable is actually called, straight from WhatsFly's own `variable_map` field (e.g. `{"header":[],"body":{"1":"#!CUSNAME!#","2":"#!CUSCODE!#"},"button":[]}`) — no more guessing via `_WF_DEFAULT_VAR_NAMES` for any template that has one (every real template on this account today). **Confirmed inconsistent placeholder wrapper syntax even within one template** — `system_abandoned_cart_reminder_new`'s own `variable_map` has both `"#LEAD_USER_FIRST_NAME#"` (bare) and `"#!system-cart-product-list!#"` (with `!`) side by side for variables 1 and 2 — `_WF_PLACEHOLDER_RE` (`#!?([A-Za-z0-9_-]+)!?#`) matches both forms. `_WF_DEFAULT_VAR_NAMES = ["CUSNAME", "CUSCODE"]` survives only as the last-resort fallback for a template with no `variable_map` at all.
- **Image header auto-detected, no manual checkbox** — `header_type == "media" and header_subtype == "image"` on the template itself, confirmed live. A non-native (fallback-shape) template has no such field and still asks via checkbox.
- **Dummy-phone compose layout** (`_render_wf_template_send`): left column = inputs (header image at the TOP so it's already in the preview by the time you reach variables, then variables), right column = `_wf_render_phone_preview` — a single WhatsApp-chat-colored frame rendered as ONE html string (two separate `st.markdown` calls would render as siblings, not nested), showing the image + bubble together, live-substituted via `_wf_substitute_positional_preview` (position-ordered, matches `variable_map`'s "1"/"2"/... regardless of what each placeholder's own text says). Body shown exactly once (the old design showed it twice — once unfilled right after template selection, once filled in the variables section — collapsed into this one live preview). `_wf_render_bubble`/`_wf_substitute_preview` (the older `{{...}}`-matching versions) are kept **unchanged** — Direct WhatsApp below still uses them as-is.
- **CUSNAME/CUSCODE auto-fill from the picked customer** — when the recipient came from the customer picker (not typed manually), a variable whose `variable_map` name is `CUSNAME`/`CUSCODE` auto-fills from that customer's name/code and shows as a caption, not an editable input; every other variable (no data source, e.g. "cart total") always asks. Switching to manual phone entry makes CUSNAME/CUSCODE editable again.
- **No "Rebuild payload" step** — `_wf_build_template_payload` is a pure function of current widget state, called fresh every rerun (both for the live send and for the admin preview) — there is no separate editable/persisted JSON copy that can go stale.
- **`⚙️ Advanced / Debug` is admin-only** (`st.session_state.user_role == "admin"`) — template ID/endpoint overrides, the "Meta Cloud API style" fallback toggle, the live payload JSON, and the raw template/list dump are all invisible to non-admin users; they just get the plain compose UI.
- **Send contract — confirmed via two real dashboard-generated examples**, not guessed (no published contract exists anywhere — checked the guide word-for-word and WhatsFly's own public docs site, both dashboard/UI-only, no REST reference). **Standing workflow, saved to session memory**: the user pastes each new template's real dashboard example specifically so this stays evidence-based — a different header type (video/document) or a buttons component should get the same treatment, not extrapolation from the pattern below.
  - Endpoint: `POST /whatsapp/send/template`, flat params, no nesting.
  - **Naming trap**: the send param is called `template_id` but wants WhatsFly's **short internal `id`** (e.g. `435966`) — *not* the longer `template_id` field the template-*list* response returns for the same template. Two API surfaces reuse one field name for two different values.
  - Variables: **`templateVariable-<NAME>-<n>`** per variable, keyed by the NAME from `variable_map` (see above) — not a numbered/generic array.
  - Header image: **`template_header_media_url`** — a plain hosted URL (not `media_id`/`media_url`/`media_type`, both real wrong guesses along the way). `upload_media` (`POST /whatsapp/upload/media`, multipart, field `media_file`) uploads automatically on file select — triggers as soon as a file is chosen, keyed by `(filename, size)` in `st.session_state` so it doesn't re-upload on unrelated reruns — and its hosted URL feeds this field directly; a manual-URL text input is the fallback if skipping upload.
  - `template_name`/`language_code` are absent from both real examples — dropped from the flat default entirely.
  - A **"Meta Cloud API style"** nested payload shape (`template: {name, language: {code}, components: [...]}`) is kept in admin Debug only as a documented fallback — WhatsFly's actual endpoint wants the flat shape above; that option's own `template_name`/`language_code` inputs render only when it's selected.
- **Response envelope handled both ways** (string `"1"`/`"0"` vs boolean, per the guide's own documented inconsistency) — `_render_wf_response` also pattern-matches Meta's generic "does not exist... graph-api" Graph API rejection (a real account/WABA-permission issue on WhatsFly's or Meta's side — invalid/deleted phone number ID, a token without permission on that WABA, incomplete Business Verification, or a disconnected integration partner — not a request-shape bug) and points at Meta Business Manager instead of the request shape.
- **Recipient: customer picker or manual entry** (`_show_whatsfly_messaging(zid)`, ZID-scoped for the customer lookup — one WhatsApp Business number account-wide, but customers live per-ZID). Picking a customer resolves a WhatsApp-ready number via `processing/common.py::customer_whatsapp_numbers(cusmobile, whatsapp)`: prefers `cacus_directory`'s `whatsapp` column (`cacus.xtaxnum`) when populated — confirmed against real data it's already in 880-format 91% of the time — falling back to `cusmobile` (`xmobile`, local format) only when `whatsapp` is blank. Either field can hold multiple comma-separated numbers in one cell — first becomes primary, second (if present) becomes secondary, selectable via a checkbox. `to_whatsapp_number()` normalizes any raw value to 880-format. A UI-level sanity check (expects exactly 13 digits) flags — doesn't block — an obviously malformed result.
- **Conversation history + reply live in the same panel** — `_wf_load_conversation_history`/`_wf_render_chat_history` (split loader/renderer; the loader returns `[]` on any error, including a missing `config/whatsapp_webhook_db.ini` — the UI just says "No conversation history yet," never leaks Postgres setup details). Matched purely by phone number, taking a **list** (primary + secondary) since the `whatsapp_webhooks` database has no customer-code concept at all.
- **Real 24h session-window logic, not a placeholder "any history" check** — `_wf_last_inbound_timestamp` finds the customer's own most recent inbound message; `_wf_session_window_status` compares it to now (naive timestamps treated as UTC, matching this account's own confirmed convention) and renders one of: 🟢 `"Session open — Xh Ym left..."`, 🔒 `"Session expired — their last message was Xh ago..."`, or 🔒 `"No active session..."`. The plain-text reply (`_render_wf_text_send`, an `st.chat_input` — an actual chat box, not a text area + button) is rendered **inside** the Conversation History container, right below the bubbles, and only when the session is open. **Templates are always available below, unconditionally** — no more forced either/or toggle, since a template can be sent regardless of session state (that's what templates are for); the old `send_as_template` toggle was removed for this reason.

### Direct WhatsApp (`views/marketing.py` → "📨 Direct WhatsApp" mode)

**Currently shut off, not deleted** — the `"📨 Direct WhatsApp"` string was removed from the mode radio's options list (so it's unreachable), while all of this section's code stays in place; re-adding that one string brings it straight back. WhatsFly is the integration being developed going forward.

Same single-message test panel as WhatsFly Messaging above, but calls Meta's own WhatsApp Cloud API directly (`graph.facebook.com`) — no WhatsFly in between. Test-number-only, against a separate Meta test WABA + test number so it never touches the real WhatsFly-routed production number.

- **Credentials**: `config/direct_whatsapp.ini` (gitignored, section `[direct_whatsapp]` with `access_token`/`phone_number_id`/`waba_id`, optional `graph_api_version` defaulting to `v21.0`) via `config/settings.py::get_direct_whatsapp_params()` — same never-raises-returns-`None` pattern as `get_whatsfly_params()`.
- **`core/direct_whatsapp.py`** — thin client: `send_text`, `get_templates`, `send_template`, `upload_media`. Unlike `core/whatsfly.py`, this follows Meta's own **published, documented** Cloud API contract directly — no defensive multi-key guessing needed, since the shape isn't reverse-engineered.
- **Simpler than the WhatsFly panel in one real way, because Meta's actual contract removes a WhatsFly-specific quirk**: there's exactly **one** request shape (no "flat vs Meta Cloud API style" guessing radio) — `POST /{phone_number_id}/messages` with the nested `template: {name, language: {code}, components: [...]}` body.
- **Two Meta placeholder formats, both supported** — chosen at template-creation time in Meta's own template editor, never mixed within one template: **positional** (`{{1}}`, `{{2}}`) or **named** (`{{cusname}}`, `{{cuscode}}`). `_wf_extract_variable_tokens` (shared with the WhatsFly panel) returns the raw `{{...}}` tokens in order — digit strings for positional, names for named — and `is_named_format` (first token non-numeric) decides how the send payload is built: positional sends plain `{"type": "text", "text": v}` parameters matched by array order; named sends `{"type": "text", "parameter_name": tok, "text": v}` per parameter, matched by name. No free-form naming UI here (unlike WhatsFly's editable Name field) — for a named template the parameter name comes straight from the template body's own token, since Meta ties the name to the approved template itself, not to metadata chosen at send time.
- **Reuses the WhatsFly panel's generic rendering helpers** (`_wf_format_whatsapp_markup`, `_wf_render_bubble`, `_wf_substitute_preview`, `_wf_extract_variable_tokens`, `_wf_extract_components`) rather than duplicating them — those are plain WhatsApp-template markup/preview helpers, not WhatsFly-specific, and Meta's own template shape (`components: [{type, text}]`) is exactly what `_wf_extract_components` already parses. `_wf_substitute_preview` matches by token text (not by casting to int), so it works for both placeholder formats.
- Header image: upload via Meta's own `/{phone_number_id}/media` endpoint → `image: {id: <media_id>}`, or a plain hosted URL fallback → `image: {link: ...}`.
- **Response handling**: success = 2xx with no top-level `"error"` key (surfaces the `wamid...` message id); failure = a nested `error: {message, type, code, ...}` object, rendered directly rather than guessed at.
- **Account-wide, not per-ZID**, same as WhatsFly Messaging.

### Bulk Messaging (`views/marketing.py` → "📢 Bulk Messaging" mode)

Campaign audience builder, built in front of the same WhatsFly template picker used by the single-message panel. Page order is **audience filters first, template picker second**, per explicit ask ("build who, then pick what"). Nested under **📢 Bulk Messaging** as an inner "🧰 Filtered List" / "📋 Curated List" radio (moved out of being two separate top-level tabs, per explicit ask, to avoid the confusion of two entries for one feature).

The template picker dispatches to `_WF_BULK_TEMPLATE_VIEWS.get(template_name, _wf_bulk_default_view)` — **`_wf_bulk_default_view` is now the real, generic send flow, not a placeholder.** The original plan (one bespoke Python handler per campaign, `_WF_BULK_TEMPLATE_VIEWS`) was superseded by Phase 6's self-service Template Mapping tool (see below) once templates started getting created ad hoc — `_wf_bulk_default_view` resolves the template's own `variable_map`, looks up its saved mapping via `processing/wf_template_mapping.py`, and if anything is unmapped, stops with a pointer to **🔧 Template Mapping** instead of showing a send UI. Once fully mapped: same header-image auto-detect + upload/paste-URL pattern as the single-message panel and the Test send panel, flat-value variables asked once per campaign, each recipient's own variables resolved via `resolve_recipient_variables`, then handed to the shared `_render_campaign_confirm_and_send` engine — the real Confirm & Send button, opt-out/cooldown exclusion counts, and progress bar all live there. `_WF_BULK_TEMPLATE_VIEWS` stays available for a genuine one-off exception a saved mapping can't express, but stays empty in the normal case. Verified live against a real template (`whatsapp_support_contact`, image header + 2 variables): unmapped state correctly blocks with the Template Mapping pointer; once mapped, header-image upload appears, both variables resolve correctly against real customer data, and Confirm & Send appears with the right exclusion counts.

**Item-attribute variables** (`processing/wf_template_mapping.py::ITEM_ATTRIBUTES` — Item Code / Item Name / Standard Price / Wholesale-Discounted Price / **Product List**) — a 3rd mapping source alongside Customer attribute and Flat value, for a template that announces specific product(s) (e.g. a price-list/promo blast). Same "asked once per campaign, same value for every recipient" semantics as Flat value — not per-recipient data — except the value comes from picking real catalog item(s) (`views/marketing.py::_wf_item_price_catalog`, options shown as `itemname (itemcode)`) instead of typing it in. `_wf_item_price_catalog` reuses the already-registered `inventory_overview` Analytics pull (`_load_inv_overview`, itself shared with Purchase Analysis's Total Inventory Overview and Inventory Analysis) — **no new SQL** — computing `wh_price = GREATEST(std_price - min_disc_amt, 0)` in pandas, the identical formula `get_opspprc_data`/`get_inventory_overview`/the Rate Mismatch Audit already use in SQL for "WH Price" elsewhere in this app. `_wf_item_row_for_mapping` formats both price fields to plain 2-decimal strings (`"156.00"`, not Python's `"156.0"`) before they reach `wtm.build_item_values`/`resolve_recipient_variables`. Scoped to `final_items_view`'s own item population (items with a current stock record), same as every other `inventory_overview` consumer.

**Single product vs. Product List** — the first four item attributes (Item Code/Name/Std Price/WH Price) resolve against ONE picked product, shown via a plain `st.selectbox`. **Product List** (`source_key = "product_list"`, tracked in `wtm.MULTI_ITEM_KEYS`) is a genuinely different shape: it resolves against MULTIPLE picked products (`st.multiselect`), joined by `views/marketing.py::_wf_format_product_list` into one string — `"Item A (1234), Item B (5678), Item C (9012)"` — for a template announcing several products at once (e.g. new arrivals). Both `_show_wf_template_mapping`'s mapping dropdown and `_wf_bulk_default_view`'s send flow split a template's item variables into `single_item_vars`/`multi_item_vars` by checking each one's `source_key` against `MULTI_ITEM_KEYS`, and show exactly the picker(s) the mapped variables need — a template could in principle mix both, though in practice it's expected to be one or the other. This product choice is **deliberately separate from the audience-filter Product filter** in Bulk Messaging's Filtered List builder (`processing/wf_bulk_audience.py`) — that filter narrows *who* receives the campaign (customers who bought a given product); this picks *what product(s)* the message itself is about. In `_wf_bulk_default_view` the product picker is shown **right under the Header Image section**, before the flat-value inputs — choosing what's being promoted is the natural next step after the header image, ahead of any other per-campaign text fields. Sending is blocked (`item_pick_incomplete`) if a Product List variable is mapped but nothing's been picked yet, or if the item catalog is empty.

Verified against real Postgres (100001): 2,353 items resolve cleanly, 1,105 of them carry a real wholesale-tier discount (the rest have `wh_price == std_price`, i.e. no tier on file); a single-product scenario (`ITEMNAME`/`PRICE` mapped to `item_name`/`wh_price`, `CUSNAME` to a customer attribute) and a Product List scenario (one variable mapped to `product_list`, 3 items picked) both resolved correctly end-to-end through `build_item_values`/`resolve_recipient_variables`. The Template Mapping dropdown's index bucketing (`— not mapped yet — / Flat value / 8 customer attributes / 5 item attributes`) round-trips correctly both ways — saving a choice and reloading it back to the right selectbox default — confirmed by directly replaying the same index math the UI uses against real `CUSTOMER_ATTRIBUTES`/`ITEM_ATTRIBUTES` lists.

**Filter builder** (`processing/wf_bulk_audience.py` for all query/pandas logic, `views/marketing.py`'s `_wfb_*` functions for orchestration/widgets):
- **Free-form "➕ Add a filter"**, not a fixed pipeline — 13 filter types, any order, except: **Salesman only appears once Area is added** (they come off the same `opdor` rows — Salesman's own options need an Area to scope to), and **nothing can be added once Avg Collection Days is active** (it's the one expensive per-customer computation — a running-balance walk — so it always runs last, over whatever's already been narrowed down; confirmed no other hard cap exists anywhere in the builder — a "stuck at 3 filters" report turned out to be the Product filter silently failing, not a real limit, see below).
- **AND across filter types** — confirmed explicitly, not assumed. **Within a filter's own multiple values, it varies by filter**: Product's multiselect requires having bought *every* selected item (AND); **Area and Salesman are each OR-within-their-own-multiselect instead** (a customer qualifies if sold to in *any* of the selected areas / by *any* of the selected salesmen) — see the Area/Salesman entry below for the one exception to that OR, which is genuinely AND (which salesmen are even offered as options).
- **Removing a filter drops it and everything added after it** — later filters' own option lists were computed assuming it was still there.
- **Every rerun recomputes candidates from scratch by replaying the whole `filters` list against the CURRENT widget state** (`_wfb_compute_candidates`, no incremental caching) — this means changing the shared "Time window" slider *after* filters are already set does **re-evaluate** every window-dependent filter already in the list against the new window, it is not frozen at add-time. Verified live: an Area filter (`Abdulla Pur`) read 36 customers at a 6-month window and 87 customers after dragging the slider to 24 months, with no re-adding of the filter needed.
- **Area is deliberately `opdor.xdiv`** (order-level territory), **not `cacus.xcity`** used for "area" everywhere else in this app including the WhatsFly single-message panel — this filter's whole point is cascading into "which salesman actually sold here," which only `opdor`'s own `xdiv`+`xsp` pairing supports; `cacus.xcity` has no relationship to which salesman visited a customer. Blank/NULL `xdiv` (no territory assigned on that order) is excluded from the Area options, same as any other blank-bucket handling in this app.
- **Area and Salesman are both `st.multiselect`, not single-select** (changed from an earlier single-select version) — three distinct semantics layered together, per explicit spec:
  1. **Area's own multiselect is OR** — a customer qualifies if sold to in *any* of the selected areas (`processing/wf_bulk_audience.py::apply_area`, `isin`).
  2. **Which salesmen are even offered as options is a UNION (OR) across the selected areas** — `salesman_options(pool_df, areas)` returns any salesman who sold in *any* one of the selected areas. **This was originally AND (must cover every selected area) and got corrected after real-use testing**: with `[Abdulla Pur, Banani]` selected, the AND version reported zero eligible salesmen and was mistaken for a bug — turned out each area was worked by a *different* salesman (`Md. Suruj Mia` alone in Abdulla Pur; `Md. Aktaruzzaman`/`Md. Shahin alom` in Banani), so requiring one salesman to cover *every* selected area was too strict for how real sales territories actually split (mostly exclusive per salesman, not shared) — the explicit follow-up ask was "I was hoping to see all these salesman" (i.e. the full union), confirmed against a real 3-area case (`Abdulla Pur, Askona, Asulia`, 142 customers): the dropdown now correctly lists all 8 salesmen active across those three areas, not zero.
  - **Each option is annotated with which of the selected areas it actually covers** — `salesman_area_coverage(pool_df, areas)` maps `spid -> sorted areas-from-the-selected-set they sold in`, rendered as e.g. `SA--000137 - Ziaur Rahman (Asulia)` right in the dropdown label. This answers "will picking this salesman also pull in some other area's customers" *before* selecting them, not after.
  - **The row-level AND between area-membership and salesman-membership (point 3 above) is what actually keeps a selected salesman's result scoped to only the selected areas** — verified against real data: with `[Abdulla Pur, Askona, Asulia]` selected and `Ziaur Rahman` picked (whose own coverage among those three is Asulia-only), the result was 41 customers — confirmed via a standalone script to exactly equal both `apply_salesman(pool, [the 3 areas], [his spid])` and `apply_salesman(pool, ["Asulia"], [his spid])` alone, proving no leakage from an area he wasn't selected for (or one he might also sell in but wasn't part of this filter's area set at all).
  3. **Once specific salesmen are picked, that selection is OR** — `apply_salesman(pool_df, areas, spids)` is a row-level AND between area-membership and salesman-membership (the same `opdor` row must satisfy both), but OR *within* each of the two sets — a customer qualifies if any selected salesman sold to them in any selected area on the same order, not "any selected area at all" independently OR'd with "any selected salesman at all."
  - The Salesman filter's stored dict now carries `"area"` (the list of areas active when it was added) and `"value"`/`"labels"` as lists (mirroring the Product filter's own `value`/`labels` pattern) instead of single strings — `_wfb_describe_filter` joins both with `", "` for display.
- **One shared window slider** (1–24 months, default 6) governs Area/Salesman, Unique Products Bought, **Net Sales**, **Total Returns**, and the Product filter. Days Since Last Sale (all-time, same definition as elsewhere in this app), Customer Score, Current Balance, **Inactive**, Avg Collection Days, and the two exact-date filters (Ordered On / Collection Received On) are all independent of it — each computed on its own scope, not the shared one.
- **Net Sales (window)** — sum of `final_sales` (`altsales − proddiscount`) MINUS matching-window returns (`treturnamt`, same `opcdt`/`opcrn` ∪ `imtemptdt`/`imtemptrn` population as `get_return_data`) per customer, within the shared window. **Originally shipped counting gross sales only (no returns netted out) — a real bug, caught by explicit user follow-up ("make sure ... net sales after returns not overall sales")**, fixed via `processing/wf_bulk_audience.py::net_sales_by_customer` (`total_sales_by_customer(...).sub(total_returns_by_customer(...), fill_value=0)`), matching the "Net Sales" meaning already established elsewhere in this app (Common Pitfall #1; Target Management's `% Collection` formula). A customer whose returns exceed their sales in the window nets to a real negative value rather than being dropped — verified live (`-102,147` was the real minimum across all customers in one window). The always-shown audience-table column of the same name uses the identical calculation.
- **Total Returns (window)** — new filter added alongside Net Sales, sum of `treturnamt` per customer in the shared window, via `total_returns_by_customer`/`apply_total_returns_band` — a plain min/max band, same UI pattern as every other sales-lines-derived filter.
- **Current Balance** — AR balance from the same `build_customer_marketing_table` call Customer Score uses (see below); **not** scoped to the window, since a balance is a point-in-time snapshot, not something that happened "within" a period.
- **Inactive carries its own independent "Time window" slider** (1–24 months, defaults to whatever the shared slider currently reads but moves independently) — changed from an earlier version that reused the shared window, per explicit follow-up ask ("inactive needs its own timeline independent of the master timeline"). Verified live: a 19-month own-window found *fewer* inactive customers (5,623) than a 6-month own-window (6,954) — the correct direction, since a longer window gives a customer more chances to have made at least one purchase, so fewer qualify as having made none.
- **Customer Score and Current Balance both reuse `build_customer_marketing_table` exactly as Marketing → Customer Scoring computes it** ("whatever score/balance is currently showing") via `_wfb_customer_metrics` (one `@st.cache_data`-wrapped call returns both columns; `_wfb_map_from_metrics` extracts either as a `cusid -> value` dict) — neither is re-scoped to this feature's own window slider.
- **Avg Collection Days reuses `processing/collection.py::average_days_to_collection` unchanged** — confirmed via code trace this is genuinely "days since last sale to each collection event, averaged" (the function's `average_days_to_collection` return column), **not** "days between consecutive collections" (a separate, correctly-unused return value, `avg_days_between`). Carries its own independent "Time window" slider (1–36 months, same UI pattern later reused for Inactive above) — sales/returns/collections for this one filter are pulled only across that filter's own window, not the shared one, which is why a collection near the start of a narrow own-window can look like it has no matching sale (captioned in the UI). This is the one expensive per-customer computation — a running-balance walk — so it always runs last, over whatever's already been narrowed down.
- New queries in `core/queries.py` (all registered in `core/analytics.py`'s `query_map`): `get_bulk_area_salesman_orders`, `get_bulk_order_date_customers`, `get_bulk_sales_lines`, `get_bulk_returns_lines`, `get_bulk_collection_lines`, `get_bulk_last_sale_dates` — each accepts an optional `cusids` list to scope to already-narrowed candidates.
- **`get_bulk_sales_lines`'s `final_sales` and `get_bulk_returns_lines`'s `treturnamt` are both raw SQL arithmetic/columns on `NUMERIC` types**, so psycopg2 hands them back as Python `Decimal`, not `float` — a real bug hit live: an object-dtype column of `Decimal`s sums fine in pandas but crashes Streamlit's Arrow serialization the moment it's rendered or downloaded (`ArrowInvalid: Could not convert Decimal(...) ... to double`). Fixed once at each load boundary — `processing/wf_bulk_audience.py::load_sales_lines`/`load_returns_lines` cast to `float` immediately after the `Analytics` call — so every consumer (Net Sales filter, Total Returns filter, Product Count Band, the always-shown table column below) gets a clean type without each having to guard against it separately.
- **Live "Customers matching so far" count** after every add/remove — this reflects the raw filter-builder output only (before the two gates below).
- **Contactable-only gate, always applied, not a builder filter** — `processing/wf_bulk_audience.py::apply_contactable_only` requires **BOTH** `cusmobile` AND `whatsapp` non-blank on a row; missing either drops the customer from the audience entirely. Deliberately stricter than `processing/common.py::customer_whatsapp_numbers`' own "prefer whatsapp, fall back to cusmobile — either is enough" logic used by the single-message WhatsFly panel elsewhere in this app; here the explicit ask was both-or-nothing. `cacus_directory`'s own query already `COALESCE`s both columns to `''` (never `NaN`), so a stripped-empty-string check is sufficient. The dropped count is always shown as a caption (even when `0`), never silently absorbed.
- **Final manual review — a "remove these" multiselect, not an editable grid.** First shipped as an `st.data_editor` checkbox grid; **replaced** after explicit user feedback that Streamlit's canvas-rendered data-editor grid "comes with a lot of bugs" from prior experience — this session's own browser-automation tooling independently hit the same rough edge (pixel-coordinate clicks on the checkbox column were unreliable to drive), which corroborated the complaint rather than being the reason for it. Now: `st.multiselect("Remove from final list", options=<"cusid - cusname" labels>, default=<currently-excluded labels>)` — picking a customer removes them from the final table/CSV below; nothing to toggle per row, no grid.
  - Exclusions are persisted by **cusid** in `st.session_state["_wfb_excluded_cusids"]`, not by the widget's own state — the widget's `key` is fingerprinted off the current candidate set (`hash(tuple(sorted(cusids)))`) so that a filter/window change (different candidates) always gets a fresh widget instance instead of Streamlit trying to validate a stale `default` against options that no longer contain it (the exact failure mode `st.multiselect` raises on: a `default` value not present in the current `options`).
  - Verified live end-to-end: excluding `CUS-008441` dropped the final count by exactly 1 and removed it from the table; adding an unrelated Area filter (candidate set 9,959 → 62) made the multiselect correctly show no selection (that cusid isn't in the narrower set — no crash); removing the Area filter again (back to 9,959) brought `CUS-008441` back showing as excluded automatically, with the final count still down by 1 — confirming the exclusion survived a full round-trip through a candidate-set change it wasn't even visible during.
  - `st.session_state["_wfb_excluded_cusids"]` is reset on "🗑️ Clear all filters" too, so starting over starts clean.
- The three metric columns (`cusid`/`cusname`/`cusmobile`/`whatsapp`/`area` plus **Net Sales (window)**, **Current Balance**, **Current Score** — the same three always-shown columns as before) live in this same editable table now, ahead of the `Include` column. A **"Final list (after phone check + manual review)"** metric and the CSV download both read off the post-edit result (`edited_df[edited_df["Include"]]`), not the raw filter output — the CSV always matches exactly what's checked in the table at download time.
- **Product filter has explicit feedback, not silence** — a real bug found via user testing: picking "Product" showed no "Add" button and no explanation whether that was because nothing was selected yet or because a specific product genuinely wasn't sold in the window. Fixed with a caption stating how many products are available to the current candidates, an explicit "Select at least one product above to enable Add" when the multiselect is empty, and a "🔍 Check whether a specific product was sold in this window at all" search (queries the *unscoped* full window, independent of current candidates, and states plainly whether it was sold at all and to how many of the current candidates).
- **Edit-in-place for an already-added filter is not built** — only Remove exists (and Remove cascades, dropping everything added after it too, since later filters' own displayed option lists/ranges were computed assuming everything before them was still active). Given the recompute-from-scratch architecture above, an Edit control that reopens a filter's own input widget and replaces it in place — without cascading — would be low-risk to add (each filter's own stored value/range is self-contained; nothing downstream depends on *how* an earlier filter's value was chosen, only on the resulting candidate set, which gets recomputed fresh regardless). Flagged as a real, worthwhile follow-up, not yet built pending confirmation on scope (e.g. whether Edit should keep downstream filters intact, unlike Remove).

**Not yet built:**
- **Similar-customer / basket-analysis audience expansion** (idea only, not implemented) — starting from the Product filter: look at the customer profile (score, etc.) and *other* products bought by whoever bought the filtered product, then find *other* customers who bought those same associated products (even if they never bought the filtered product itself), to build a more robust "might buy this" candidate list than a strict "bought exactly this product" filter gives. A market-basket/collaborative-filtering-style expansion, not implemented — noted here for whenever it's picked up.

### WhatsApp Webhook Receiver (`whatsapp_webhook/`, standalone FastAPI service)

The receive-side counterpart to Direct WhatsApp above — that panel only sends; a webhook is the only way delivery/read/failure status and inbound replies ever arrive (a send's own 200 response only means "Meta accepted the request"). A fully separate service, not part of the Streamlit app process: own FastAPI app, own `.env` credentials (`WHATSAPP_VERIFY_TOKEN`, `META_APP_SECRET`, distinct from `config/direct_whatsapp.ini`), and its **own Postgres database** (`schema.sql`, isolated from `da`) — though it shares the `streamlitEnv3.10.13` Python env rather than a dedicated venv. Build reference: `WhatsApp_Integration_docs/whatsapp-webhook-build.md`; run instructions: `whatsapp_webhook/README.md`.

**Live in production** — `https://webhook.hmbr.com.bd/webhook/whatsapp`, Caddy reverse-proxying to the FastAPI service (NSSM service `WhatsAppWebhook`) on the Windows Server 2016 box. See `whatsapp_webhook/HANDOFF.md`/`WHATSAPP_SCREENCAST_PREP.md` for deployment specifics.

**The server's own venv runs Python 3.9**, not this repo's 3.10+ — confirmed by a real crash (`X | None` union syntax is 3.10-only; use `typing.Optional`/`Union` here, and `from __future__ import annotations` does NOT fix it since FastAPI's `get_type_hints()` re-evaluates the annotation regardless of deferral). Always verify changes to this folder against a real 3.9 interpreter before shipping (`pyenv install 3.9.21` + a throwaway venv), not just against this repo's own env — see saved session memory for the full incident.

- **`main.py`**: GET `/webhook/whatsapp` echoes `hub.challenge` back once `hub.verify_token` matches (Meta's one-time verification handshake); POST verifies `X-Hub-Signature-256` (HMAC-SHA256 over the **raw** body, constant-time compare) before anything else — reject 403 on mismatch, this is the entire trust boundary. A verified payload gets logged to `webhook_events` synchronously (idempotency/audit backbone), then routed via a `BackgroundTask` so Meta gets its `200` immediately ("respond fast, process later"). Malformed JSON after a valid signature → `400` (Meta won't retry 4xx); a DB/processing failure inside the background task is caught and recorded as `processing_status='failed'`, never raised (the response already went out).
- **`handlers.py`** (Meta-direct events): routes each `changes[].field` — `messages` (both shapes: inbound `messages[]` array and outbound `statuses[]` array in one field) gets full handling; the four `message_template_*`/`template_category_update` fields upsert into `templates`; everything else lands in the generic `account_alerts` audit table. Status events are deduped on `(wamid, status)` and only move `messages.current_status` forward (a rank guard: `sent < delivered < read`, `failed` terminal).
- **WhatsFly's own webhook delivery** — 4 separate routes, one per WhatsFly trigger (confirmed against the real dashboard: each trigger gets its **own** URL field, unlike Meta's single endpoint): `/webhook/whatsfly/{token}` (Incoming Message), `/outgoing/{token}`, `/status/{token}`, `/conversation/{token}` — each with its own independent token env var, since WhatsFly documents no signing/secret mechanism of its own (trust = a random token embedded in the URL path, constant-time compared; 404 not 403 on mismatch). `whatsfly_handlers.py` has real parsers for Incoming/Outgoing/Status, all built against real captured payloads, not guessed:
  - Incoming Message: flat shape (`chat_id`, `first_name`, `user_message`, `wa_message_id`, `whatsapp_bot_id`/`name`/`username`), **no explicit event-type field** and no timestamp field at all. `whatsapp_bot_id` (WhatsFly's own per-number id) is stored in the same `phone_number_id` column Meta-direct rows use for Meta's own id — different id space, same column, same role.
  - Outgoing Message: same shape plus `agent_name`. Text is prefixed `#ATTACHMENT:<type>#` when the send had a media header (confirmed on a real image-header template send) — no media URL/id anywhere in the payload, so the attachment itself isn't recoverable from this event, only that one existed.
  - Message Status Change: the one trigger with an explicit `"webhook_type": "message_status_change"` field. `message_status` values confirmed so far (`delivered`, `read`) match `db._STATUS_RANK`'s vocabulary exactly. `status_time` is a naive string, confirmed UTC (cross-checked against that row's own `received_at`).
  - Conversation Status Change: **no confirmed shape yet** — stays raw-capture-only (logged to `webhook_events`, no parser) until a real one arrives.
  - **Confirmed in practice, not just theoretical**: WhatsFly's delivery isn't ordering-guaranteed either — a real "delivered" status event arrived at the webhook *before* that same message's own Outgoing Message event. `db.upsert_outbound_message` (UPDATE-then-INSERT-if-0-rows) upgrades an existing stub row's content in place so either arrival order ends up at the same correct final state.
- **Known gap**: outbound sends via `core/direct_whatsapp.py` (Direct WhatsApp, currently shut off) don't write into this database at send time — `db.ensure_outbound_stub` creates a minimal placeholder `messages` row on first sight of an unrecognized `wamid` so the FK stays intact regardless.
- **Pre-9.5 Postgres server** — same constraint as `marketing_leads` elsewhere in this app: no `ON CONFLICT`. Every upsert in `db.py` is UPDATE-then-INSERT-if-0-rows instead.
- Verified via `fastapi.testclient.TestClient` against a mocked DB layer (no real Postgres needed), including the WhatsFly parsers tested against the exact real payloads captured in production — not synthetic examples.
- **Streamlit-side read access**: `core/whatsapp_webhook_db.py` — a dedicated, low-privilege SELECT-only role (never the webhook service's own write role), used by the WhatsFly Messaging panel's inline conversation history (`get_messages_for_contact`, matched by phone number — see the WhatsFly Messaging section above). Marketing → "📥 WhatsApp Message Log" (the standalone browse-everything view over the same data, `_show_whatsapp_message_log`) is **shut off, not deleted** — same treatment as Direct WhatsApp — since conversation history now shows inline per-customer in WhatsFly Messaging itself, making the standalone view redundant for day-to-day use; the code and its `_PRODUCT_ONLY_MODES` entry are untouched, just unreachable via the mode radio.

### Campaign History webhook staleness indicator (`views/marketing.py::_show_wf_campaign_history`)

A real production incident (2026-09-15): a real, successfully-sent 37-recipient campaign (`single_product_message`, ZID 100001) showed **Delivered: 0, Read: 0, Delivery Failed: 0** in Campaign History — not because nothing had actually delivered, but because **WhatsFly had silently stopped calling the webhook entirely, ~39h before the campaign was even sent**, and there was nothing in the UI to tell "confirmed zero" apart from "we have no idea, the pipeline's been silent for days." Diagnosed from a pgAdmin `.backup` of the live `whatsapp_webhooks` database (restored locally via `pg_restore --no-owner --no-privileges`, since the dump's own `CREATE DATABASE` locale — `English_United States.1252` — doesn't exist on a non-Windows Postgres) plus the webhook service's own `stdout.log`/`stderr.log`, both pulled from the server:
- `webhook_events` (every webhook this server has ever received, any type) had its **last row at `2026-09-13 20:53:08`** — 39h before the campaign's `12:02–12:05` send window on `2026-09-15`, and nothing since.
- `stdout.log` showed **28 total hits on WhatsFly's 4 token routes, all `200 OK`** — and `SELECT count(*) FROM webhook_events WHERE raw_payload ? 'wa_message_id'` also returned exactly **28**, with the same max `received_at` (`2026-09-13 20:53:08`). Every WhatsFly webhook call this server has ever received is already accounted for — none of the 28 are new. A text search of `raw_payload` for all 37 of the campaign's wamids found **zero matches**, not even a garbled fragment.
- This ruled out a "burst of 37 sends overwhelmed the receiver" theory (an initially plausible hypothesis, since the campaign's sends are spaced ~5-7s apart — a slow drip, not a burst, and nowhere near burst load anyway) — the receiver has a **clean 28/28 success rate** on every request it's ever actually gotten. The gap is entirely upstream, on WhatsFly's side (account/webhook-subscription issue), not a delivery or concurrency failure on this server.

**`core/whatsapp_webhook_db.py::get_last_webhook_received_at()`** — `MAX(webhook_events.received_at)` across every webhook type (WhatsFly's 4 routes + Meta-direct alike), account-wide, not scoped to any campaign/wamid — the cheapest possible pipeline-health signal. `_show_wf_campaign_history` calls this once per render (not per campaign) and compares it against `now()` and against each campaign's own `created_at`:
- **`_WF_WEBHOOK_STALE_THRESHOLD = timedelta(hours=6)`** — how long with zero webhook activity before the pipeline is flagged as possibly stalled. Generous enough that a normal quiet evening/weekend doesn't false-alarm, tight enough to catch a real outage (the incident above was a 39h gap).
- A campaign's own Delivered/Read/Delivery Failed figures are marked **`webhook_unconfirmed`** when that campaign's `created_at` is AFTER the last recorded webhook activity — meaning there's been zero opportunity for ANY status to report back on it yet, of any kind. This is a per-campaign flag, not a blanket "everything's stale" — a campaign sent *before* the pipeline went quiet (like the Sept 13 test campaign) still shows its real, already-confirmed delivered/read/failed numbers and is correctly left unflagged.
- **Three-tier display**, computed once in the Overview section: (1) `pipeline_stale AND any campaign unconfirmed` → a prominent `st.warning` naming the specific affected campaign id(s), stating plainly this is "not confirmed zeros" and pointing at WhatsFly's dashboard rather than this app as the likely cause (validated against the real incident: correctly fires for campaign #2, correctly stays silent for campaign #1). (2) `pipeline_stale` but no campaign sent since → a calm `st.caption` ("quiet a while, but no campaign has been sent since, so this may just be normal low traffic") — deliberately not alarming, since an idle pipeline with no recent sends is unremarkable. (3) not stale → a plain "last webhook activity: Xh Ym ago" caption, always visible as a low-key health readout. `last_webhook_at is None` (webhook_events table genuinely empty) gets its own distinct warning, since "never confirmed anything, ever" is a different, worse case than "was fine, now quiet."
- The **Past Campaigns table** gets a `Webhook` column showing `🕐 Unconfirmed` on affected rows, blank otherwise — so the flag travels with the data, not just the banner. The **per-campaign detail** drill-down shows the same caption when the selected campaign is itself unconfirmed.
- `_wf_format_timedelta_ago(delta)` — coarse `"2d 3h ago"` / `"5h 12m ago"` / `"3m ago"` formatting, not precision-sensitive, matching the existing `hrs`/`mins` style already used by `_wf_session_window_status`'s 24h-session-window caption elsewhere in this file.

Verified end-to-end against the restored real production data (not synthetic): `get_last_webhook_received_at()` correctly returns the tz-aware `2026-09-13 20:53:08+06` timestamp; the full per-campaign loop correctly flags campaign #2 (`webhook_unconfirmed=True`) and correctly leaves campaign #1 unflagged (`webhook_unconfirmed=False`, sent well before the gap); the warning banner text renders correctly naming `#2` specifically.

### Manual reconciliation — "🔄 Reconcile Unconfirmed with WhatsFly" (campaign detail, admin-only)

Follow-up to the staleness incident above: the user checked WhatsFly's own dashboard directly and confirmed the 37 messages genuinely sent, with real status (some read, one delivery failure) already known **on WhatsFly's side** — their webhook just never called this server to report it. So the ground truth exists, just not in our DB. First shipped as a one-recipient-at-a-time checker; the user then asked to reconcile a whole campaign in one go, skipping recipients already done — the version below.

**`core/whatsfly.py::get_message_status(wa_message_id)`** — `GET/POST /whatsapp/get/message-status` (params `apiToken`, `wa_message_id`, `whatsapp_bot_id`), documented in `Whatsfly_Integration_docs/whatsfly-integration-guide.md` with its own explicit note: *"for real-time tracking at scale, prefer the webhook over polling this"* — i.e. it's meant as exactly this fallback, not a webhook replacement. `whatsapp_bot_id` needs no new credential — confirmed it's the same value already configured as `phone_number_id` in `config/whatsfly.ini`. **Response shape confirmed against the live account** (2026-09-15, a real failed send): `{"status": "1", "message": {"message_status": "failed", "delivery_status_updated_at": null}}` — `message_status` (the first key `_wf_parse_message_status` tries) is correct on the first guess. **Confirmed: this endpoint has no failure-reason field at all** — unlike the webhook's own `message_status_change` event (which does carry `failed_reason`), a "failed" pulled from here will always have a blank failure reason in Campaign History. That's this endpoint's genuine limitation, not a parsing gap — explicitly accepted by the user ("I dont think you will get the failed reason, either ways just make sure its still wired to the campaign history").

**Persistence — `campaign_recipients.whatsfly_status`/`whatsfly_checked_at`/`whatsfly_error_detail`/`whatsfly_raw`** (`whatsapp_webhook/add_campaign_recipients_whatsfly_reconciliation_columns.sql`, non-destructive `ALTER TABLE`, no new grants needed — `streamlit_campaign_writer` already has `UPDATE` on this table). `processing/wf_bulk_campaign.py::update_recipient_whatsfly_reconciliation` writes all four; `whatsfly_raw` is **always** stored regardless of whether the parsed `status`/`error_detail` guess succeeded, so nothing is lost to a wrong parse — it can be re-parsed from the stored JSONB later with no new live call. `get_campaign_recipients`'s `SELECT *` picks the new columns up automatically.

**`views/marketing.py::_wf_parse_message_status(raw)`** — best-effort `(status, error_detail)` extraction, reusing the `_wf_guess` multi-key-candidate pattern already used for templates elsewhere in this file. **Deliberately does NOT read a bare top-level `"status"` key** — that's this API's own generic request-envelope indicator (`"1"`/`"0"`/`true`/`false`, see `_wf_normalize_templates`), not a per-message delivery state; checks the `"message"`/`"data"` wrapper keys (matching the confirmed template-list wrapper) for the real payload first, then tries `message_status`/`delivery_status`/`read_status`/`wa_status` for status and `failed_reason`/`error_title`/`error_message`/`error` for the failure reason. Verified against fake payloads shaped like the confirmed webhook vocabulary (nested-dict wrapper, list-wrapper, the top-level-envelope trap, and a genuinely-unrecognized shape returning `(None, None)` without crashing) — all four cases parse correctly.

**Two-tier UI**, both gated `user_role == "admin"`, appearing below the recipient table/download button in Campaign History's per-campaign detail (inside the existing "Select a campaign" dropdown's section, not a separate picker — kept there deliberately per explicit ask):
- **"🔄 Reconcile All Unconfirmed (N)"** — the primary action. Candidates are `wamid.notna() & ~webhook_confirmed & ~reconciled` (`reconciled` = `whatsfly_checked_at.notna()`) — i.e. recipients with no webhook data AND not already manually reconciled, so a full campaign's worth of already-done recipients are skipped on a re-run, exactly as asked. Loops with a live `st.progress()` bar, 1s pacing between calls (a read-only status check, lighter pacing than the 2s send loop). **Only a genuinely successful API response gets persisted/marked done** — a request-level failure (timeout, rate limit) is counted but leaves that recipient as a candidate for the next run, rather than silently giving up on it forever by marking it "checked" with no real data.
- **"Check one recipient individually"** (a collapsed `st.expander`) — the original single-recipient flow, kept as a secondary option for one-off re-checks (e.g. retrying a recipient that failed, or re-verifying an already-reconciled one), showing the raw JSON via `st.json` for inspection.

**`recipients["status_source"]`** ("Webhook" / "WhatsFly (manual)" / "Send only") is a new displayed column making it visible at a glance where each row's `delivery_status` actually came from — the fallback chain itself (`status_map` → `whatsfly_status` → raw `status`) now has three tiers instead of two, and `_failure_reason` gained the same third tier (`whatsfly_error_detail`, checked after the webhook-sourced reason comes up empty).

Verified end-to-end against the real restored production data (still no live WhatsFly call made from here, per the standing no-live-testing rule): the parser's 4 fake-payload cases all pass; a real persistence round-trip against campaign #2's actual recipients (3 rows manually reconciled with fake data, re-read back correctly, `still_unconfirmed` correctly drops from 37 to 34 with zero overlap between the two sets) confirmed the skip-already-done logic works exactly as intended, then cleaned up (reset to `NULL`) so the restored local DB wasn't left with fake data. For campaign #1 (the healthy Sept 13 test send, all 5 recipients already webhook-confirmed) the reconciliation section correctly doesn't appear at all (`still_unconfirmed` empty and `already_reconciled == 0`).

### Three same-day follow-ups: Past Campaigns not updating, failed-recipient export, missing-WhatsApp export

- **Past Campaigns table wasn't picking up reconciled data — a real gap.** The per-campaign summary loop (that builds the `📋 Past Campaigns` table's Delivered/Read/Delivery Failed columns) only ever read `status_map` (webhook data) — it never looked at `campaign_recipients.whatsfly_status` at all, so a manually-reconciled campaign kept showing 0/0/0 at the top of the page even after "🔄 Reconcile All Unconfirmed" had genuinely saved real data (visible correctly only in the per-campaign detail table below, which *did* already fold `whatsfly_status` in). Fixed by building a `combined_status` dict per campaign — `status_map` first, then `whatsfly_status` for any wamid the webhook never covered — mirroring the detail view's own fallback priority exactly, so the two can never disagree. **`webhook_unconfirmed` was also redefined** in the same fix: previously a pure timing signal ("sent after the last webhook activity"), it's now `still_unresolved > 0` — at least one wamid with neither webhook data nor a reconciliation yet. This means reconciling a campaign now correctly clears its 🕐 marker even when the webhook itself never reported anything at all, which the old timing-only definition could never do. Verified against real data: before reconciling, campaign #2 read `{delivered: 0, read: 0, delivery_failed: 0}`; after reconciling 3 recipients (2 read, 1 failed, matching a real captured response), it correctly read `{delivered: 2, read: 2, delivery_failed: 1, still_unresolved: 34}`; campaign #1 (already fully webhook-confirmed) was unaffected by the change.

- **Per-campaign "Failed Recipients" CSV** — a dedicated `st.download_button` in the per-campaign detail view (right after the full recipients CSV), filtered to `delivery_status == "failed"` (webhook- or reconciliation-sourced either way, via the same `delivery_status` column), columns `cusid`/`cusname`/`phone_number`/`wamid`/`delivery_status`/`status_source`/`failure_reason`. Built per explicit ask — *"so that we can collect their whatsapp number just in case ... they have a number that does not have whatsapp in it"* — a failed send often means the number on file isn't actually WhatsApp-capable, so this is the list to go re-verify.

- **"📵 Customers Missing a WhatsApp Number"** — a new standalone section at the very bottom of Campaign History (`views/marketing.py::_show_wf_campaign_history`, after the per-campaign detail block), placed there rather than inside Bulk Messaging's audience builder per explicit ask. **`processing/wf_bulk_audience.py::get_no_whatsapp_customers(df)`** (new) returns the exact population `apply_contactable_only` drops — rows missing `cusmobile` *or* `whatsapp`, not just both — as a real DataFrame rather than just a count, so it's downloadable. Kept as a separate function rather than changing `apply_contactable_only`'s own `(kept_df, dropped_count)` return signature (would risk breaking its existing callers) — same mask logic duplicated deliberately, not extracted, to keep that signature stable. Not tied to any specific past campaign or filter run — `campaign_recipients` only ever persisted who *was* included, never who got excluded, so this is a standing, always-current lookup (whoever's missing a number *today*) scoped to whichever ZID is active, sourced from `cacus_directory` via the already-cached `_load_cacus`. Verified against real Postgres (100001): 881 of 9,959 customers are missing `cusmobile` or `whatsapp` (or both) — including real cases with a populated `cusmobile` but blank `whatsapp`, confirming the "either, not both" criterion is what's actually applied, matching `apply_contactable_only` exactly.

---

## Usage Stats (`views/usage_stats.py` → "Usage Stats" menu item, admin-only)

App-owned table logging which page + primary radio-level mode each user opens, and when — designed and discussed with the user before building anything, per explicit ask. Same convention as `crm_call_log`/`marketing_leads`/`page_permissions`: written directly by this app via `core/db.py`, no separate role/grants needed (this app's single main-DB connection already has full access to its own database). Setup: `db/sql_scripts/create_view_usage_log_table.sql` — creates `view_usage_log` **and** grants the new "Usage Stats" menu item to `admin` via `page_permissions` (without that row, even an admin's sidebar menu won't show it — `app.py::navigation` filters the menu through `auth.check_page_access`, same as every other page).

**No real "page exit" event exists in Streamlit** — the whole script reruns on every interaction, there's no unload hook. "Time spent on a view" is therefore *derived*, not logged directly: `processing/usage_log.py::compute_durations` takes the gap between one view's `entered_at` and the *next* view's `entered_at` within the same `session_id`. The last view of a session gets no duration at all — there's genuinely no way to know when someone closed the tab, so it's left blank rather than guessed. This is the standard practical approach for this kind of lightweight analytics in a rerun-based framework, discussed and agreed with the user upfront rather than presented as a surprise limitation later.

**`processing/usage_log.py::log_view(page, view_path=None)`** — the single instrumentation point, called once per page (see the per-file list below). Only inserts a row when `(page, view_path)` differs from `st.session_state["_usage_last_logged"]` — critical, since Streamlit reruns the whole script on *any* interaction (typing in an unrelated filter, clicking an unrelated button on the same page), so without this guard every such rerun would insert a duplicate row for a view the user hasn't actually navigated away from. `session_id` is one UUID generated once per browser session (`ensure_session_id`, stored in `st.session_state`), letting events be grouped/sequenced for duration math. Never raises — a logging failure must never break the page it's instrumenting, so the whole body is wrapped in a bare `try/except: pass`.

**Retention: 1-year rolling window, per explicit ask.** `_cleanup_old_rows()` (`DELETE ... WHERE entered_at < now() - interval '365 days'`) runs probabilistically — a ~1-in-500 chance on any given `log_view` call — rather than via a scheduled job, since this app has no background scheduler at all (every write happens inside a normal page request/response cycle). A day or two of "late" cleanup doesn't matter for a retention window that isn't compliance-driven.

**Instrumentation — one line per page, added once, not per-widget.** "Radio level is enough," per explicit ask — only each page's *primary* mode-selecting control is tracked, not every secondary filter/toggle on the page:
- **Overall Sales Analysis / Overall Margin Analysis / Collection Analysis / Purchase Analysis / Target Management / Inventory Analysis / Manufacturing Analysis / Marketing Analysis / Customer Support / Customer Data View** — hooked right after that page's own primary `st.radio`, passing its value as `view_path`.
- **Financial Statements** — its top-level nav is an `st.sidebar.selectbox("Timeframe", [...])`, not an `st.radio` — treated as the same "primary mode control" role anyway (Yearly/Monthly/Quarterly/Daily/Lifetime/Config Editor), since the point is tracking which top-level view gets picked, not the exact widget type.
- **Accounting Analysis** — **page-level only, no radio-level detail.** Its three sections are `st.tabs()`, which has no server-readable "which tab is active" state — all three tab bodies execute on every rerun regardless of which one the user is actually looking at, so which tab is "in view" can't be determined from Python state without a custom JS component (out of scope here). Documented inline as a real, deliberate limitation, not an oversight.
- **Basket Analysis** — page-level only. The page is currently disabled (`display_basket_analysis_page` just shows "currently unavailable" — the real logic sits in an unreachable `_display_basket_analysis_page_disabled`), so there's no mode to log yet.
- **Home / Usage Stats itself** — page-level only, no radio (Home has no controls at all; Usage Stats logs its own visits for consistency).

**Reports** (`views/usage_stats.py`, date-range picker at top, default last 30 days, capped to the 1-year retention window): Overview metrics (total views / active users / sessions), Most-Used Pages (bar chart + table), Most-Used Pages by Mode, **Usage by User** (per explicit follow-up ask — views, distinct pages touched, total time, last active, with a CSV download), Usage by Role, Usage Over Time (daily trend), and Least-Used Pages — the last one lists **every** page in `usage_log.ALL_PAGES` (kept in sync with `app.py`'s own `menu` list by hand, documented as such), including ones with zero views in the selected window, so a genuinely unused feature stays visible instead of silently vanishing from a plain `groupby`.

Verified end-to-end against the real local Postgres mirror (not synthetic assumptions): direct DB round-trip confirmed `compute_durations`' gap math to the second (three consecutive views spaced 120s/480s/900s apart computed exactly those durations, with the session's last view correctly landing on `NaN`); all six report-building functions produced correctly-aggregated output; `build_least_used_pages` correctly listed all 14 real menu pages including zero-view ones; retention cleanup correctly removed a synthetic 400-day-old row while leaving recent rows untouched. Confirmed live in a real Streamlit rerun cycle too (not just a script): clicking through Home → Marketing Analysis/Leads → Collection Analysis/Salesman Due as a real admin session correctly logged 4 genuine transitions (plus the Usage Stats page's own self-logging interleaved between them) under one stable `session_id`, and the live Usage Stats page picked up and displayed that real data correctly (Total Views: 9, Active Users: 1, Sessions: 1 — exactly matching the underlying rows).

---

## Sales Commissions & Incentive Tracking (`views/commissions.py`, "Commissions" menu item)

Full design (rankings, campaigns, the pooled-FIFO/gate mechanics) lives in **`commission_tracking_design.md`**
(repo root, status "agreed design") — that file is the source of truth, not this section. Built one piece at a
time, discussed with the user before each piece, per that doc's own "discuss before building" rule.

- **Page structure**: a single top-level `st.selectbox("Commission Campaign", ...)` dropdown routes to one
  section-renderer per doc item (`_SECTIONS` dict in `views/commissions.py`). Every design-doc section
  (A.1 Best Performer, A.2 Highest Product Sales, A.4 Best App User, B.1/3/4 Campaign, B.2 Individual Target,
  B.5 New Customer) already has a dropdown entry — sections not yet built show a `st.info` placeholder pointing
  back at the relevant doc section instead of a blank/missing option, so the dropdown's shape doesn't need to
  change as each piece gets picked up.
- **Access**: `page_permissions` grants `'admin'` only for now (`db/sql_scripts/grant_commissions_page_permission.sql`,
  same non-destructive-INSERT convention as `create_view_usage_log_table.sql`'s "Usage Stats" grant) — a new,
  compensation-sensitive page defaults to admin-only; widen to other roles later if actually wanted.
- **Instrumented** in Usage Stats the same way every other page is — `usage_log.log_view("Commissions", section)`
  right after the dropdown, `"Commissions"` added to `usage_log.ALL_PAGES`.

### Product Tracking (only section built so far)

Redesigned 2026-09-19 from an initial plain-MTD version (replaced before it was ever
committed) into a **per-product Before/After comparison** — see
`commission_tracking_design.md`'s "Product Tracking" section for the full spec.

- Admin-edited JSON watchlist, **scoped per ZID** (products differ per business) —
  `data/commission_product_watchlist.json` (gitignored, same convention as
  `targets.json`/`public_holidays.json`) — but each entry now carries its own date pair:
  `{zid: {itemcode: {"cutoff": "YYYY-MM-DD", "back_to": "YYYY-MM-DD"}}}`
  (`processing/commissions.py::load_watchlist`/`save_watchlist`/`add_product`/
  `remove_product`/`update_product_dates`). Capped at **20 items per ZID** — "more
  wouldn't make sense" per the user.
- **Before/After windows, not one shared "current month"**: `After = [cutoff, today]`,
  `Before = [back_to, cutoff)` — cutoff belongs to After only, so the two windows never
  double-count it (`processing/commissions.py::build_product_comparisons`, boundary
  verified against real Postgres sales rows). `back_to` is capped at **6 months before
  cutoff** (`comm.min_back_to`) — enforced both by validation in `add_product`/
  `update_product_dates` and by the date-input widgets' own `min_value`/`max_value`, so an
  invalid combination is hard to even enter, not just rejected after the fact.
- **Each watched product has its own cutoff/back_to** — rendered as its own separate
  table (`_render_comparison_table`: Before row, After row, Change row = After − Before,
  signed), not one shared grid across products.
- Editing (add product with its own dates, edit an existing product's dates, remove) is
  gated to `st.session_state.user_role == "admin"` in-page, on top of the page-level
  admin-only gate. A non-admin with an empty watchlist sees a plain "ask an admin" message.
- Metrics (qty sold, sales revenue, qty returned, net qty) sourced from
  `sales_daily_item`/`returns_daily_item` (`mv_sales_daily_item`/`mv_returns_daily_item`,
  full history per ZID, cached `ttl=3600`, sliced per-product per-window in pandas) joined
  to `final_items_view` for name/group/current-stock (current stock shown as point-in-time
  context above each table, not per-window). **DB `NUMERIC` columns arrive as
  `Decimal`/object-dtype** (same class of bug as WhatsFly Bulk Messaging's Decimal
  Arrow-serialization crash) — explicitly `pd.to_numeric(...)`'d right after merging.
- **Two real Streamlit widget-state bugs hit and fixed while building this**: (1) a
  widget's `session_state` key cannot be assigned in the same script run where that widget
  was already instantiated — resetting the "Add a product" picker after a successful add
  needs a pending-reset flag checked at the *top* of the next run (`_comm_wl_add_reset`),
  not an immediate post-add assignment, which raises `StreamlitAPIException`. (2) a
  dependent widget's stored value (Back To) can fall outside newly-computed min/max bounds
  when the widget it depends on (Cutoff) changes on the same rerun — `_clamp_session_date`
  pre-clamps the stored value before the Back To widget is instantiated, and the call site
  skips passing `value=` on the run a clamp just happened (passing both raises/warns).
- Verified end-to-end against real Postgres (100001): add/remove/save round-tripped
  correctly through the live UI; the cutoff-date boundary was checked against a real sale
  record (`before`/`after` sums matched a manual split at that exact date); the 6-month
  clamp was exercised live (moving Cutoff back to July correctly snapped Back To to
  `cutoff − 1 day` with no crash and no stray warning); Change-row arithmetic and signed
  formatting (`+`/`-`) matched hand computation.

### Product / Stock Clearance / Slow-Moving Campaign (B.1/3/4)

`processing/commission_campaigns.py` (engine) + `views/commission_campaigns_view.py` (UI).
Full spec: `commission_tracking_design.md` §B.1/3/4. Rate per unit sold, paid only if the
customer's DO is fully collected by a deadline, pooled across 100001+100000, gated on both
ZIDs hitting their own sales target.

- **`recipient_type` ("salesman" or "customer") — confirmed 2026-09-19/20: this campaign
  type's commission can be paid to either the salesman OR the customer on the DO.** The
  eligibility logic (DO fully collected, by which deadline) is completely unchanged either
  way and is always keyed off the DO's own **salesman's** payout group — the deadline is a
  logistics/territory concept, not a recipient concept. Only the final aggregation step
  differs: group by `spid` (+`spname`) or by `cusid` (+`cusname`).
  `processing/commission_campaigns.py::build_do_totals` carries both identities (and both
  names) on every DO row for exactly this reason, regardless of which type a given
  campaign uses.
- **Rate AND cap vary PER PRODUCT, not campaign-wide — confirmed 2026-09-20** (this
  replaced an earlier single-campaign-wide rate/cap; see git history if resurrecting the
  old shape). `commission_campaigns.product_rates` (JSONB) is
  `{itemcode: {"rate": float, "cap": float|null}}` — **`rate` is the per-unit
  incentive/discount BDT amount being offered (e.g. 5 or 3), NOT the product's sales
  price** — this ambiguity was explicitly raised and resolved by the user; don't
  re-litigate it. `cap` (optional) caps how much ONE recipient (whichever `recipient_type`
  applies) can earn from ONE product across the whole campaign.
- **One Postgres table, `commission_campaigns`** (`db/sql_scripts/create_commission_campaigns_table.sql`)
  holds campaign *definitions* only (name, type, recipient_type, product_rates, sales
  window, and a `payout_groups` JSONB of `{group: deadline}` for that campaign) — no
  salesman/customer/DO/payout numbers are stored. Every payout figure is recomputed live
  whenever a campaign is opened. Schema was revised 2026-09-20 (table had zero real rows
  at the time — a straight `DROP`+recreate, not an `ALTER` migration).
- **Salesman → payout-group membership is derived LIVE, not a manual roster — confirmed
  2026-09-20 (replaced a same-day-earlier hand-maintained
  `data/commission_payout_groups.json` roster + editor UI, both removed outright).**
  `processing/commission_campaigns.py::build_area_group_map` maps `{xcity -> group}` from
  pooled 100001+100000 `cacus.xstate` (majority value per area; `"Sylhet Retail"` folds
  into `"District"`, confirmed by the user — everything else keeps its real `xstate` name
  as its own group, e.g. `"Dhaka retail"`, `"Nawab pur"`); `derive_spid_group_map` matches
  each salesman's `prmst.xdisease` area list against that map (majority vote across their
  listed areas, tie broken by whichever tied group's area is listed first), filtered by
  `valid_spids` to real salesmen only (per the op-tables-not-prefix rule above — otherwise
  non-salesman `prmst` rows with an area on file would show up too).
  `views/commission_campaigns_view.py::_render_derived_groups` replaced the roster editor
  with a read-only view (group list + which salesmen resolved). This roster is always
  salesman-based (it drives DO eligibility, see above) regardless of `recipient_type`.
  **Correctly returns nothing locally** — `derive_spid_group_map` returns `{}` against the
  local Postgres mirror, verified, since `prmst.xdisease` is still mid-rollout there (see
  "Key column mappings" above); `build_area_group_map` itself has no such gap and returns
  the real 8 groups locally today.
- **`resolve_do_paid_dates`** — the pooled FIFO resolver: per customer, a chronological
  walk where each collection pays off the oldest open DO(s) first, in full, before moving
  on; leftover collection becomes a "prepaid credit" applied to that customer's next DO(s).
  **Pre-groups by customer before the walk** (`{cusid: sub_df}` built once) — an earlier
  version that re-filtered the full collection DataFrame per customer inside the loop
  never finished on real data (~10k customers × 1M+ combined sales/collection events).
  With the fix, full 100001+100000 history resolves in ~15-20s.
  Verified two ways: hand-computed against a real customer's full DO/collection history
  (a collection that split its payment across two DOs, matched exactly, including which
  DO the split landed on); plus synthetic tests for prepaid credit applied to a future DO,
  a large overpay with no further DOs (no crash), and a DO that never gets paid (stays NaT).
- **Payout** = for each DO in-window whose salesman is in a roster group and was paid by
  that group's deadline: qty of the picked product(s) on that DO × that product's own
  rate, capped per recipient per product, summed per recipient (salesman or customer,
  per `recipient_type`). DOs that don't qualify are kept with their exclusion reason (not
  yet collected / no payout group / group has no deadline set / collected after the
  deadline) and shown in the UI, not silently dropped. **Verified the grouping choice is
  capping-neutral in the way it should be**: summed *raw* (pre-cap) payout is identical
  whether grouped by salesman or by customer (confirmed against real data — 312 either
  way) since it's the same underlying line items either way; only how the cap binds
  differs, correctly, by recipient granularity.
- **Gate** = `zid_target_sum` (sum of `data/targets.json` entries for that ZID across the
  window's calendar months) vs. `zid_actual_sales` (`final_sales` summed in-window,
  reusing `processing.common.data_copy_add_columns`) — if either 100001 or 100000 misses,
  `total_payout` is zeroed but the computed (pre-gate) figure still shows, captioned, so
  the gate failing doesn't look like "nothing sold."
- **Uptick** (companion metric, not a payout gate) — campaign-window qty/revenue for the
  picked product(s) vs. a trailing `uptick_baseline_months` average ending the day before
  the window starts. Returns `None` (not a crash or a misleading number) when the
  baseline period has zero sales — hit for real in testing (a genuine multi-month gap in
  one test product's sales history) and confirmed correct, not a bug to chase.
- **Caching**: the FIFO resolution + pooled sales/collection loads are
  `st.cache_data(ttl=3600)`-wrapped in `views/commission_campaigns_view.py` and shared
  across every campaign view — opening/switching campaigns doesn't re-pay the ~15-20s cost
  each time, only once per hour app-wide.
- **Two real bugs fixed**: (1) `core/db.py::get_data` is SELECT-only and never commits —
  using it for `INSERT ... RETURNING id` (to create a campaign and get its new id back)
  silently discarded the insert. Added `core/db.py::execute_write_returning` (commits AND
  fetches the RETURNING row) for this and any future write that needs its own id back.
  (2) A missing per-product `cap` (`None`/`NaN`) rendered as the literal string `"None"` in
  the Per-Product breakdown table instead of `.style.format(na_rep=...)`'s intended
  `"—"` — `st.dataframe` doesn't reliably respect Styler `na_rep` for this case (same
  class of issue as the TOTAL-row/Styler pitfall already documented above). Fixed the same
  way that pitfall recommends: pre-format the Cap column to a display string (with `"—"`
  for missing) before it ever reaches `.style.format()`, rather than leaving it numeric.
- Verified end-to-end live: created a real campaign (2 real products with their own
  rate/cap, salesman recipient type), got a real payout that matched a standalone
  hand-computation exactly, confirmed the gate-passed banner, the per-product breakdown
  (including the Cap fix), the excluded-DO reasons table, and campaign deletion, all
  against real Postgres data — no stubbed/mocked data anywhere in this feature. The
  customer-recipient path and the raw-sum-invariant check above were verified standalone
  (not re-driven through the live UI in the same pass — the UI code path for
  `recipient_type="customer"` is the same rendering code with a different string, already
  exercised for "salesman").
- **Not yet built**: B.2 (Individual Target Achievement, fully specified) and B.5 (New
  Customer Creation, needs design work) — see `commission_tracking_design.md`. Also see
  that doc for which of A/B's *other* not-yet-built sections are salesman-only,
  customer-or-salesman, or salesman-only-for-a-different-reason (New Customer Creation) —
  noted for whenever each is picked up, not yet relevant to what's built.

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
