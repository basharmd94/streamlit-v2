# Agent Development Guide — streamlit-v2

This document is the authoritative reference for any AI agent or developer working on this codebase. Read it in full before making any changes. Every rule here was established either by design intent or by a hard-won bug fix.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Folder Structure](#2-folder-structure)
3. [The Data Flow Pipeline](#3-the-data-flow-pipeline)
4. [SQL & Query Rules](#4-sql--query-rules)
5. [The Analytics Class](#5-the-analytics-class)
6. [Processing Layer Rules](#6-processing-layer-rules)
7. [Views Layer Rules](#7-views-layer-rules)
8. [app.py & Navigation Rules](#8-apppy--navigation-rules)
9. [Authentication & Session State](#9-authentication--session-state)
10. [Visualization Rules](#10-visualization-rules)
11. [Download & Large Data Rules](#11-download--large-data-rules)
12. [Logging Rules](#12-logging-rules)
13. [Configuration Rules](#13-configuration-rules)
14. [Known Bugs & Hard-Won Fixes](#14-known-bugs--hard-won-fixes)
15. [Adding a New Page — Checklist](#15-adding-a-new-page--checklist)
16. [Adding a New SQL Table — Checklist](#16-adding-a-new-sql-table--checklist)
17. [What Not To Do](#17-what-not-to-do)

---

## 1. Project Overview

A multi-business Streamlit dashboard backed by a PostgreSQL database. It supports several business entities identified by a **ZID** (e.g. `100001`, `100000`, `100005`). Pages include sales, margin, collection, purchase, basket analysis, financial statements, accounting, and inventory.

**Entry point:** `app.py`
**Run command:** `streamlit run app.py`
**Python version:** 3.10
**Key dependencies:** `streamlit`, `psycopg2`, `pandas`, `plotly`, `bcrypt`, `extra_streamlit_components`

---

## 2. Folder Structure

```
streamilt-analysis_v2/
│
├── app.py                      # Entry point. BaseApp class only. No business logic.
│
├── auth/
│   ├── auth.py                 # Login, logout, session init, page access checks
│   └── create_user.py          # One-off user creation script (not part of app)
│
├── config/
│   ├── settings.py             # Reads database.ini → get_db_params()
│   ├── database.ini            # GITIGNORED. Must be present locally.
│   └── loggin_config.ini       # GITIGNORED. Optional — app falls back to basicConfig.
│
├── core/
│   ├── db.py                   # ThreadedConnectionPool, get_data(), get_dataframe()
│   ├── queries.py              # ALL SQL lives here. Every function returns (sql, params).
│   └── analytics.py            # Analytics class: maps table name → query → DataFrame
│
├── data/
│   ├── businesses.json         # ZID → business name mapping for financial statements
│   ├── hierarchy.json          # GL account hierarchy for purchase/financial analysis
│   ├── labels.json             # Display labels for financial statements
│   └── warehouse_filters.json  # Warehouse groupings for purchase analysis
│
├── processing/
│   ├── common.py               # Shared utilities (filters, pivots, download link, etc.)
│   ├── financial.py            # Cash flow, P&L, Balance Sheet builders
│   ├── overall_sales.py        # Sales aggregation and charting logic
│   ├── overall_margin.py       # Margin aggregation and charting logic
│   ├── collection.py           # Collection/AR analysis logic
│   ├── purchase.py             # Purchase cohort, batch profitability engine
│   ├── basket.py               # Market basket analysis logic
│   ├── descriptive_stats.py    # Summary statistics
│   ├── histogram.py            # Histogram helpers
│   └── yoy.py                  # Year-over-year comparison logic
│
├── views/
│   ├── home.py                 # display_home_page()
│   ├── sales.py                # display_overall_sales_analysis_page(), display_customer_data_view_page()
│   ├── margin.py               # display_margin_analysis_page()
│   ├── collection.py           # display_collection_analysis_page()
│   ├── purchase.py             # display_purchase_analysis_page()
│   ├── basket.py               # display_basket_analysis_page()
│   ├── financial.py            # display_financial_statements()
│   ├── accounting.py           # display_accounting_analysis_main()
│   └── inventory.py            # display_inventory_analysis_main()
│
├── visualization/
│   └── common_v.py             # plot_histogram(), plot_bar_chart() — Plotly wrappers
│
├── models/                     # Reserved for future ML models. Currently empty.
│
├── utils/
│   ├── loggin_config.py        # LogManager class with robust fallback initialisation
│   └── utils.py                # @timed decorator
│
└── db_sync/                    # Standalone database sync scripts. Do not modify.
```

### ⚠️ Critical Folder Name Rules

- **`views/`** — NEVER rename this to `pages/`. Streamlit reserves `pages/` for its built-in multi-page navigation, which would auto-list every file in the sidebar before login.
- **`data/`** — All JSON data files live here. Never put them in `processing/` or root.
- **`core/`** — Only infrastructure: DB connection, query builders, the Analytics class. No business logic.
- **`processing/`** — Business logic only. No Streamlit widgets, no `st.*` calls (except `st.cache_data` decorators).

---

## 3. The Data Flow Pipeline

Every piece of data in the app follows this exact pipeline. Never skip a layer.

```
PostgreSQL DB
     │
     ▼
core/queries.py          ← SQL string + params tuple. No execution.
     │
     ▼
core/db.py               ← Executes via ThreadedConnectionPool. Returns (records, cols) or DataFrame.
     │
     ▼
core/analytics.py        ← Analytics("table_name", zid=..., filters=...).data → pd.DataFrame
     │
     ▼
processing/*.py          ← Pure transformation: filter, group, aggregate, compute. Returns DataFrames.
     │
     ▼
views/*.py               ← Streamlit UI only: widgets, st.dataframe, st.plotly_chart, download buttons.
     │
     ▼
app.py (BaseApp)         ← Wires views to navigation. No business logic here.
```

### Global State — ZID and Page

- `st.session_state.zid` — the currently selected business ZID. Set in `BaseApp.navigation()`.
- `st.session_state.proj` — project name corresponding to the ZID.
- `st.session_state.current_page` — currently selected page name.
- `st.session_state.authenticated` — login state.
- `st.session_state.username` — logged-in username.
- `st.session_state.user_role` — role used for page access control.

**Never** read ZID directly from a widget inside a view. Always read `st.session_state.zid`.

---

## 4. SQL & Query Rules

**File:** `core/queries.py`

### Rule 1 — No f-string interpolation of user values. Ever.

```python
# ✅ CORRECT — parameterised
query = "SELECT * FROM sales WHERE zid = %s AND year = %s"
return query, (zid, year)

# ❌ WRONG — SQL injection risk
query = f"SELECT * FROM sales WHERE zid = {zid} AND year = {year}"
```

### Rule 2 — Every query function returns `(sql_string, params_tuple)`

```python
def get_my_data(filters: dict) -> Tuple[str, tuple]:
    zid = filters["zid"][0]
    sql = "SELECT col1, col2 FROM mytable WHERE zid = %s"
    return sql, (zid,)
```

**Exception:** A small number of legacy functions return only the SQL string (no tuple). The `Analytics` class handles this via isinstance check. Do not add new functions this way — always return a tuple.

### Rule 3 — IN clauses use the `_build_in_clause` helper

```python
from core.queries import _build_in_clause

placeholders, params = _build_in_clause(zid_list)
sql = f"SELECT * FROM table WHERE zid IN ({placeholders})"
return sql, params
```

### Rule 4 — Register every new query in `core/analytics.py`

After adding a function to `queries.py`, add it to the `query_map` dict in `Analytics.__init__`:

```python
query_map = {
    ...
    "my_table": queries.get_my_data,
}
```

### Rule 5 — Tables that join `caitem` must apply the packcode CASE

When a table stores raw `itemcode` but the rest of the app uses packcode-resolved codes, apply this CASE in SQL:

```sql
CASE
    WHEN caitem.packcode IS NOT NULL
     AND caitem.packcode <> ''
     AND caitem.packcode != 'NO'
     AND LEFT(caitem.packcode, 2) != 'KH' THEN caitem.packcode
    ELSE table.itemcode
END AS itemcode
```

**Never** try to replicate this logic in Python — the Python-side remapping has case-sensitivity bugs and is fragile. Put the CASE in SQL and LEFT JOIN caitem.

### Rule 6 — `purchase` and `stock_movement` tables take two ZIDs

The `Analytics` class auto-appends `100009` when `zid = 100001` for these two tables because packaging items span both entities. When writing their queries, always use `WHERE zid IN (%s, %s)`.

### Rule 7 — Validate year data from stock tables

The `stock` table has known data-entry errors where `year` contains garbage values (e.g. `2102`). When building year-based selectors, always filter:

```python
current_year = pd.Timestamp.today().year
valid_years = [y for y in years if 2000 <= y <= current_year + 1]
```

---

## 5. The Analytics Class

**File:** `core/analytics.py`

```python
df = Analytics("table_name", zid="100001", filters={}).data
```

- `zid` can be a string, list, or tuple.
- `filters` is a dict. `zid` and `project` keys are injected automatically — do not pass them manually unless overriding.
- `.data` is a `pd.DataFrame` or `None` on failure.
- Always check `if df is not None and not df.empty` before using.

### ZID auto-expansion rules (built into Analytics)

| table_name | Behaviour |
|---|---|
| `purchase` | If zid=100001, auto-adds 100009. Always needs exactly 2 ZIDs. |
| `stock_movement` | Same as purchase. |
| All others | Uses only the primary ZID. |

### Caching

For data loaded inside views that doesn't change with filter widgets, wrap the load call in a `@st.cache_data` function:

```python
@st.cache_data(show_spinner=False)
def _load_my_data(zid: str) -> pd.DataFrame:
    df = Analytics("my_table", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()
```

Cache functions must be module-level (not inside a class or another function). Arguments must be hashable.

---

## 6. Processing Layer Rules

**Folder:** `processing/`

### Rule 1 — No Streamlit in processing modules

Processing functions must be pure Python/pandas. They take DataFrames in, return DataFrames out. The only exception is `@st.cache_data` decorators on helper functions.

```python
# ✅ CORRECT
def compute_net_sales(sales_df, returns_df):
    ...
    return result_df

# ❌ WRONG
def compute_net_sales(sales_df, returns_df):
    st.write("Computing...")   # No st.* calls in processing
    ...
```

### Rule 2 — Never use `ProcessPoolExecutor` in processing modules

Streamlit runs inside its own event loop. On macOS (Python 3.8+), `ProcessPoolExecutor` uses "spawn", which re-imports all modules in every worker process — including `import streamlit as st`. This causes workers to crash silently and return empty results. Use sequential processing or `ThreadPoolExecutor` if concurrency is needed.

```python
# ❌ WRONG — breaks silently in Streamlit
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_chunk, chunks))

# ✅ CORRECT
results = process_chunk(full_df, sales_df)
```

### Rule 3 — Merge keys: use minimal stable keys

When merging two DataFrames where one comes from SQL (with packcode-resolved itemcodes) and another from a different source, merge only on `["warehouse", "itemcode"]` or similar stable keys. Never merge on string columns like `itemname` or `itemgroup` — whitespace or encoding differences will cause silent join failures.

### Rule 4 — `applymap` + `fillna(0)` corrupts date columns

If a DataFrame has both numeric and date/string columns, `applymap(fn).fillna(0)` will turn `NaT` date strings into `0`. Always format date columns first, then run numeric cleanup only on numeric columns:

```python
# Format date column first
df['combinedate'] = pd.to_datetime(df['combinedate'], errors='coerce').dt.strftime('%Y-%m-%d').fillna('')

# Then clean up numerics only
numeric_cols = [c for c in df.columns if c not in ('itemcode', 'itemname', 'combinedate')]
df[numeric_cols] = df[numeric_cols].apply(lambda col: col.map(handle_infinity_and_round)).fillna(0)
```

### Rule 5 — Financial CF statement: Prior Period NP monthly rule

In `processing/financial.py`, the "Prior Period Net Profit/Loss" row in Cash from Financing must be zero for all months except January. For monthly perspective, the BS Net Profit/Loss is a YTD cumulative figure — shifting it into financing for Feb–Dec double-counts the P/L already captured in the top-line Net Profit/Loss row.

```python
if selected_perspective.lower() == "monthly":
    for col in prior_np_series.index:
        key = _period_key(col)
        month = key[1] if isinstance(key, tuple) and len(key) == 2 else None
        if month != 1:
            prior_np_series[col] = 0.0
```

---

## 7. Views Layer Rules

**Folder:** `views/`

### Rule 1 — One public `display_*` function per file

Each view file exposes exactly one public entry-point function named `display_<page_name>(...)`. Private helpers are prefixed with `_`.

```python
# views/sales.py

def _build_options(df):          # private helper
    ...

@timed
def display_overall_sales_analysis_page(current_page, zid, data_dict):  # public entry point
    ...
```

### Rule 2 — Views call processing functions; they do not compute

Views must not contain groupby, merge, or complex pandas operations. Those belong in `processing/`. Views call processing functions and render their output.

### Rule 3 — Always wrap CP / data-heavy sections in try/except

Sections with multiple selectboxes that depend on data availability must have error handling:

```python
try:
    # selectbox and compute operations
    if df.empty:
        st.warning("⚠️ No data for the selected filters. Please reselect your options.")
        st.stop()
    ...
except Exception as e:
    st.warning("⚠️ Unable to generate report. Please reselect your options — the current combination may lack sufficient data for valid reporting.")
    st.caption(f"Details: {e}")
```

### Rule 4 — Use `st.download_button` for all downloads in views

Never use `create_download_link` (base64 HTML anchor) inside views for DataFrames that could be large. Use `st.download_button` instead — it does not embed data in the HTML and does not crash the browser:

```python
st.download_button(
    label=f"⬇ Download CSV ({len(df):,} rows)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="my_report.csv",
    mime="text/csv",
    key="dl_unique_key",   # must be unique per page
)
```

`create_download_link` in `processing/common.py` is still valid for small DataFrames in other views but auto-falls back to CSV for DataFrames > 1,048,576 rows.

### Rule 5 — Cap `st.dataframe` for large DataFrames

`st.dataframe` sends ALL rows to the browser via WebSocket. More than ~100,000 rows will freeze or crash Chrome. Always cap display with a notice:

```python
_LIMIT = 50_000
if len(df) > _LIMIT:
    st.info(f"Showing first {_LIMIT:,} of {len(df):,} rows. Download for full data.")
    st.dataframe(df.head(_LIMIT), use_container_width=True)
else:
    st.dataframe(df, use_container_width=True)
```

### Rule 6 — Use `st.session_state.zid` for ZID, not widget values

The ZID is set globally in `BaseApp.navigation()`. Views receive it as a parameter. Never create a second ZID selectbox inside a view except where explicitly needed (e.g. Financial Statements), and when you do, default it to `st.session_state.zid`:

```python
global_zid = st.session_state.get("zid", None)
default_idx = next((i for i, k in enumerate(options) if str(k[0]) == str(global_zid)), 0)
selected = st.selectbox("Select Business", options, index=default_idx)
```

---

## 8. app.py & Navigation Rules

**File:** `app.py`

### BaseApp structure

```
BaseApp.__init__()     ← set_page_config, session state init, auth.init_auth()
BaseApp.run()          ← auth gate: login page OR navigation()
BaseApp.navigation()   ← sidebar ZID selector + page selector + filters + routing
BaseApp.home()         ← delegates to views/home.py::display_home_page()
BaseApp.<page>()       ← thin wrapper that calls views/<page>.display_*()
```

### Navigation sidebar order

1. `st.sidebar.selectbox` — Select Business (ZID)
2. `st.sidebar.selectbox` — Menu (page selector)
3. Page-specific filters (year, month, salesman, etc.)
4. Load Data button
5. `st.sidebar.markdown("---")` + Logout button

### Sidebar visibility

The sidebar is hidden on the login page via CSS injected in `auth/auth.py::render_login_page()`:

```css
[data-testid="stSidebar"] {display: none;}
[data-testid="collapsedControl"] {display: none;}
```

Do not set `initial_sidebar_state="collapsed"` globally — it must remain `"expanded"` for post-login use.

### Page access control

Every page goes through `auth.check_page_access(page_name)`. Page names in the `menu` list must match the `page_name` values stored in the `page_permissions` table in PostgreSQL.

---

## 9. Authentication & Session State

**File:** `auth/auth.py`

- Uses `bcrypt` for password hashing.
- Uses `core.db.get_data()` for all DB access (no direct psycopg2 connections).
- `login()` fetches hashed password, calls `check_password()`, and sets `st.session_state` on success.
- `check_page_access()` queries `page_permissions` table with `(role, page_name)`.
- `init_auth()` must be called once in `BaseApp.__init__()`.

### memoryview handling

Passwords stored in PostgreSQL as `bytea` arrive as `memoryview` objects. Always cast before passing to bcrypt:

```python
if isinstance(hashed, memoryview):
    hashed = bytes(hashed)
return bcrypt.checkpw(password.encode('utf-8'), hashed)
```

---

## 10. Visualization Rules

**Folder:** `visualization/`

Only `common_v.py` is active. It contains two functions:

| Function | Purpose | Used In |
|---|---|---|
| `plot_histogram(data_dict, y_axis_title)` | Bar chart over timeline | `views/financial.py` |
| `plot_bar_chart(data, x_axis, y_axis, color, title)` | Grouped bar chart | `processing/descriptive_stats.py` |

All other charting is done with Plotly Express / Graph Objects directly inside processing modules. Do not add new files to `visualization/` unless they contain reusable chart primitives used in multiple places. One-off charts belong in the relevant processing module.

---

## 11. Download & Large Data Rules

### `processing/common.py::create_download_link(df, filename)`

- Generates a base64-encoded `<a>` tag for downloading as Excel.
- Automatically falls back to CSV if `len(df) > 1,048,576` (Excel row limit).
- **Only use for small-to-medium DataFrames** (< 100,000 rows) embedded in HTML.
- For views with potentially large data (e.g. inventory), use `st.download_button` directly.

### Excel size limit

Excel's hard limit is **1,048,576 rows × 16,384 columns**. Any DataFrame exceeding this will raise `ValueError` if written with `to_excel`. The `create_download_link` function handles this, but views must also cap display.

---

## 12. Logging Rules

**File:** `utils/loggin_config.py`

Uses `LogManager` class with a robust initialisation sequence:

1. Tries `config/loggin_config.ini`
2. Tries `config/logging.ini`
3. Falls back to `logging.basicConfig`

**Critical rule:** `config/logging.ini` and `config/loggin_config.ini` are gitignored. The app must start even if neither exists. The fallback is mandatory — never remove it.

Usage:

```python
from utils.loggin_config import LogManager
LogManager.logger.info("Something happened")
LogManager.logger.error(f"Something broke: {e}")
```

The `@timed` decorator in `utils/utils.py` automatically logs execution time for decorated functions using `LogManager`.

---

## 13. Configuration Rules

**File:** `config/settings.py`

Reads `config/database.ini` via `ConfigParser`. Returns a dict of connection params:

```python
from config.settings import get_db_params
params = get_db_params(section="postgresql")
```

`database.ini` is **gitignored**. It must be present locally for the app to connect to PostgreSQL. When setting up a new environment, copy it manually from an existing environment — it is never committed to git.

**`database.ini` format:**
```ini
[postgresql]
host=localhost
database=stream2
user=postgres
password=postgres
port=5432
```

### Connection Pool

`core/db.py` uses `psycopg2.pool.ThreadedConnectionPool(minconn=1, maxconn=10)`. The pool is initialised lazily on first request. Do not create direct `psycopg2.connect()` calls anywhere in the app — always go through `core.db.get_data()` or `core.db.get_dataframe()`.

---

## 14. Known Bugs & Hard-Won Fixes

These are real bugs that were found and fixed. Do not reintroduce them.

### Bug 1 — LogManager crash at import time
**Symptom:** App crashes before rendering the login page. Auth page never appears.
**Cause:** `logging.config.fileConfig()` called at class definition time on a file that doesn't exist.
**Fix:** Wrap in try/except with `basicConfig` fallback. Already done in `utils/loggin_config.py`.

### Bug 2 — `pages/` folder name triggers Streamlit auto-navigation
**Symptom:** All page module names appear as navigation links in the sidebar before login.
**Cause:** Streamlit automatically treats any folder named `pages/` as multi-page app pages.
**Fix:** The folder is named `views/`. Never rename it to `pages/`.

### Bug 3 — `ProcessPoolExecutor` silently returns zeros
**Symptom:** Purchase cohort shows cost=0, avg_price=0, week columns 1–4=0.
**Cause:** macOS Python 3.8+ uses "spawn" for multiprocessing. Child processes re-import `streamlit` and crash silently, returning empty result lists.
**Fix:** Replace with sequential `process_chunk()` call. Already done in `processing/purchase.py`.

### Bug 4 — `applymap().fillna(0)` corrupts `combinedate`
**Symptom:** `combinedate` column shows `0`, `NaT`, or epoch dates.
**Cause:** `applymap(handle_infinity_and_round).fillna(0)` ran after date formatting, converting NaT strings to 0.
**Fix:** Format date column with `.fillna('')` before numeric cleanup. Apply numeric cleanup only to numeric columns. Already done in `processing/purchase.py::generate_cohort()`.

### Bug 5 — Monthly Cash Flow check shows excess equal to Prior Period NP
**Symptom:** Cash-flow Check row is non-zero for every month except January. The excess equals the Prior Period Net Profit/Loss exactly.
**Cause:** BS "Net Profit/Loss" in monthly mode is a YTD cumulative. Shifting it into financing for Feb–Dec double-counts P/L already in the top-line row.
**Fix:** Zero out `prior_np_series` for all months != 1 in monthly perspective. Already done in `processing/financial.py::make_cashflow_statement_level0()`.

### Bug 6 — `stock_flow` itemcodes don't match `stock` itemcodes
**Symptom:** Movement Analysis columns (`abs_qty_K`, `qty_in_K`, etc.) all show 0.
**Cause 1:** `get_stock_flow_data()` returned raw itemcodes without packcode CASE logic, so merge with `inv_df` (which uses packcode-resolved codes) failed.
**Cause 2:** Groupby used `["warehouse","itemcode","itemname","itemgroup"]` — string mismatches in `itemname`/`itemgroup` caused silent join failure.
**Fix:** Applied packcode CASE in `get_stock_flow_data()` SQL. Reduced groupby/merge keys to `["warehouse","itemcode"]` only. Already done.

### Bug 7 — Movement Analysis window always empty (year=2102 in DB)
**Symptom:** Movement Analysis shows all-zero movement columns despite data existing.
**Cause:** `stock` table has at least one row with `year=2102` (data-entry error). The year selectbox defaults to the highest value, making `cutoff_mi = 2102*12+4 = 25228`. All real flow data has `mi ≈ 24300`, below the window threshold.
**Fix:** Filter year list to `2000 <= year <= current_year + 1` before building selectbox. Already done in `views/inventory.py`.

### Bug 8 — `database.ini` not present in git worktrees
**Symptom:** Login fails silently. DB connection pool never initialises. Every login attempt returns False with no error shown.
**Cause:** `*.ini` is gitignored. Git worktrees do not inherit untracked files from the main working tree.
**Fix:** Manually copy `config/database.ini` from the main repo to the worktree after checkout.

---

## 15. Adding a New Page — Checklist

1. **Create** `views/my_page.py` with a single public `display_my_page(current_page, zid, ...)` function.
2. **Add** the display function import in `app.py`: `from views import ..., my_page`
3. **Add** a method to `BaseApp`: `def my_page_analysis(self): my_page.display_my_page(...)`
4. **Add** the page name to `menu` list in `BaseApp.navigation()`.
5. **Add** the routing branch in `navigation()`:
   ```python
   elif self.current_page == "My Page Name":
       self.my_page_analysis()
   ```
6. **Insert** a row into the `page_permissions` PostgreSQL table for each role that should access it.
7. **Add** any new SQL queries to `core/queries.py` (returning `(sql, params)` tuple).
8. **Register** new query functions in the `query_map` in `core/analytics.py`.
9. **Add** any new processing logic to a file in `processing/` (no `st.*` calls).

---

## 16. Adding a New SQL Table — Checklist

1. **Write** the query function in `core/queries.py`:
   ```python
   def get_my_table_data(filters: Dict[str, Any]) -> Tuple[str, tuple]:
       zid = filters["zid"][0]
       sql = "SELECT col1, col2 FROM my_table WHERE zid = %s"
       return sql, (zid,)
   ```
2. **Register** it in `Analytics.query_map` in `core/analytics.py`:
   ```python
   "my_table": queries.get_my_table_data,
   ```
3. **If the table uses raw `itemcode`**, apply the packcode CASE by LEFT JOINing `caitem` in the SQL — do not remap in Python.
4. **If the table needs two ZIDs** (like purchase), add it to the special-case block in `Analytics.__init__`:
   ```python
   if table_name in ("purchase", "stock_movement", "my_table"):
   ```

---

## 17. What Not To Do

| Action | Why |
|---|---|
| Name a folder `pages/` | Streamlit reserves it for auto-navigation — will show all modules in sidebar before login |
| Use f-strings to inject filter values into SQL | SQL injection vulnerability |
| Call `psycopg2.connect()` directly | Bypasses the connection pool; causes connection leaks |
| Put `st.*` calls inside `processing/` modules | Breaks the separation of concerns; processing must be pure |
| Use `ProcessPoolExecutor` | Crashes silently in Streamlit on macOS due to "spawn" multiprocessing |
| Use `applymap().fillna(0)` on mixed DataFrames | Corrupts date and string columns |
| Merge DataFrames on `itemname` or `itemgroup` | String mismatches cause silent join failures — use code columns only |
| Use `create_download_link` for large DataFrames | Base64-encodes the whole DataFrame into HTML — crashes the browser above ~100k rows |
| Call `st.dataframe()` on 1M+ row DataFrames | Sends all data over WebSocket — crashes Chrome |
| Add new query functions that return only a SQL string | All new functions must return `(sql, params)` tuple |
| Store business logic in `app.py` or `views/` | Business logic belongs in `processing/` |
| Use `st.session_state` keys without `init_auth()` | Session keys may not exist; always check with `.get()` |
| Trust raw year values from `stock` table without validation | DB contains rows with garbage years (e.g. 2102) |
| Commit `*.ini` files to git | They contain credentials; they are gitignored by design |

#
Error handling rule: If any API call returns a 529 Overloaded error, 
retry automatically with exponential backoff (2^attempt seconds, starting 
at attempt=0), up to 5 retries. Print a message like 
"[Retry N/5] 529 Overloaded — waiting Xs..." before each wait. 
Do not treat 529 as a fatal error unless all retries fail.