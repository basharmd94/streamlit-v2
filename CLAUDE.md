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

## Customer Columns (all sales-derived tables)

| DB column | Alias | Display label |
|---|---|---|
| `cacus.xmobile` | `cusmobile` | Mobile |
| `cacus.xtaxnum` | `whatsapp` | WhatsApp Number |
| `cacus.xcity` | `area` | Area |

`whatsapp` flows through `build_customer_wise_monthly`/`build_customer_product_monthly` in `processing/target_management.py` — must be in `id_cols` for pivots.

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
