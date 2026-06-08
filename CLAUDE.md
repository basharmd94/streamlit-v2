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

**Entry point:** `app.py`  
**Python env:** `streamlitEnv3.10.13` (pyenv)  
**DB:** PostgreSQL (credentials in `config/global_db.ini`)

---

## Directory Structure

```
app.py                        # Main Streamlit entry point, sidebar filters, routing
auth/                         # Auth tables setup and user management
  setup_db.py                 # Run with: python auth/setup_db.py (sys.path fix included)
config/
  settings.py                 # DB params from global_db.ini
  logging.ini / global_db.ini
core/
  analytics.py                # Analytics class — maps table names → query functions
  db.py                       # DB connection helpers (get_data, get_dataframe)
  queries.py                  # All SQL query builder functions
processing/
  common.py                   # data_copy_add_columns, shared utilities
  financial.py                # Level 0→S IS/BS/CFS builders, compute_mtd_is
  overall_margin.py           # Margin analysis helpers
  target_management.py        # build_customer_wise_monthly, build_customer_product_monthly
  consolidation.py            # Multi-ZID consolidation engine
  collection.py               # Collection analysis
  ...
views/
  financial.py                # Financial Statements page (Yearly / Monthly / Level S)
  target_management.py        # Target Management page
  inventory.py                # Inventory Analysis page
  margin.py                   # Overall Margin Analysis
  collection.py               # Collection Analysis
  ...
data/
  targets.json                # Runtime — gitignored
  public_holidays.json        # Runtime — gitignored
  warehouse_filters.json      # Runtime — gitignored
  hierarchy.json              # GL account hierarchy
  ls_account_notes.json       # Level S IS/BS/CFS account descriptions
  labels.json                 # income_statement_label, balance_sheet_label mappings
```

---

## Key Data Sources (DB Tables / Views)

| Alias | Table | Notes |
|---|---|---|
| `sales` | `opdor` + `opddt` + `imtrn` | Sales orders → line items → inventory cost |
| `return` | `opcrn`/`opcdt` + `imtrn` | Customer returns |
| `stock` | `imtrn` | Cumulative stock balance (xqty*xsign, xval*xsign) |
| `cacus_directory` | `cacus` | Customer directory (cusid, cusname, cusmobile, whatsapp=xtaxnum, area) |
| `final_items_view` | DB view | Current stock: item_id, item_name, item_group, stock |
| `gldetail` / `glheader` / `glmst` | GL tables | Financial postings |
| `caitem` | Item master | xdesc=itemname, xabc=itemgroup, xdrawing=cross-ZID code mapping |

### Key column mappings
- `opddt.xdtwotax` → `altsales` (gross revenue, maps to GL 08010001)
- `opddt.xdtdisc` → `proddiscount` (maps to GL 07080001 Discount Paid — **never subtract from Revenue in IS context**)
- `imtrn.xval` → `cost` (COGS)
- `cacus.xtaxnum` → `whatsapp` (WhatsApp Number)
- `gldetail.xprime` → GL posting amount (Revenue = negative credit, Expense = positive debit)

### Sign convention
- **Level 0 / raw GL**: Revenue = negative, Expense = positive
- **Level S IS**: Revenue = positive, Expense = negative (sign flipped by `_ls_sum`)
- **`final_sales`** = `altsales - proddiscount` (used in margin analysis only, NOT in financial IS)

---

## Analytics Class Pattern

```python
from core.analytics import Analytics

df = Analytics("table_name", zid=zid, filters={"year": [2026], "month": [6]}).data
```

Registered table names → query functions in `core/analytics.py` `query_map`.  
To add a new table: add to `query_map` AND add a query function in `core/queries.py`.

---

## Financial Statements Architecture

### Levels
- **Level 0**: Raw GL detail (every xacc × period)
- **Level 1/2**: Aggregated buckets
- **Level S**: Management view — `build_pl_level_s()` in `processing/financial.py`
- **Level T**: Adjusted Level S (inter-company eliminations)
- **Level C/C2**: Consolidated across ZIDs

### Level S IS row order (key rows)
Revenue → Others Revenue → MRP Discount → **Adjusted Revenue (Pending)** → COGS → **Gross Profit** → SG&A sub-rows → **Total SG&A** → 0708-Discount Paid → S&D Expenses → **Total S&D** → 0501-Others Direct → **EBITDA** → 0630-Bank Interest → 0633-Interest Loan → **Total Interest** → VAT rows → Income Tax → **Net Income**

### MTD IS Dashboard (`views/financial.py` → `_render_mtd_dashboard`)
- Appears as **"📊 MTD Dashboard"** radio inside Level S Monthly view
- Revenue/COGS from imtrn/opdor pipeline (`altsales` gross, not net of discount)
- All opex from `gldetail` SUM(xprime) for current month, negated to Level S sign
- 3M average from last 3 completed period columns of `pl_s`
- Toggle: "Use 3M Averages for SG&A, Interest & Tax in Net Income"
  - When ON: EBITDA and Net Income recalculate using 3M avg for Total SG&A, Total Interest, VAT/Tax
  - Discount Paid and S&D always remain MTD actuals
- Per-section breakdown expanders (SG&A, S&D, Interest) with per-xacc detail

### `_MTD_CODES` dict in `processing/financial.py`
Mirrors the ac_code sets in `build_pl_level_s`. Keep both in sync if adding new codes.

---

## Target Management Architecture (`views/target_management.py`)

### View mode radio
`["👤 Individual Salesman", "📊 All Salesmen Overview", "📈 Moving Average", "📦 Current Stock"]`

### Individual Salesman page
- Target entry: full current year (Jan–Dec), defaults to current month
- Metric cards: Daily Avg Sales (3M), Monthly Avg Sales (3M), etc.
- **Daily Avg (3M)** = `total_3mo / wd_3mo` (working days in 3-month window, ≈79)
- **⚠️ Requires sidebar to include ≥3 prior months** — if `last3` is empty, shows warning
- Inventory Coverage section at bottom (green/red/purple/blue status vs prior-month stock)

### All Salesmen Overview
- Table 1: per-salesman summary with Daily Required vs Daily Avg (3M)
- Caption shows the exact 3M window and working-day count for transparency
- **Daily Required** = `(target - mtd_sales) / remaining_wd`
- **Daily Avg (3M)** = `total_3mo / wd_3mo` — must load 3+ prior months in sidebar

### Current Stock tab
- Source: `final_items_view` DB view, filtered by zid
- Columns: Item ID, Item Name, Item Group, Stock
- Search filter (Item Name or Group)
- 1-hour cache TTL (`_load_final_items`)

### Working day rule
Mon–Thu + Sat–Sun are working days; **Friday and public holidays are off** (`_is_working_day`).

### Target / Holiday storage
- `data/targets.json` — gitignored runtime file
- `data/public_holidays.json` — gitignored runtime file
- `data/warehouse_filters.json` — gitignored runtime file
- Targets keyed as `{zid}_{spid}_{year}-{month:02d}`

---

## Inventory Analysis (`views/inventory.py`)

### Default warehouses (`_DEFAULT_WAREHOUSES`)
```python
_DEFAULT_WAREHOUSES = [
    "Finished Goods Store Packaging",
    "HMBR -Main Store (4th Floor)",
    "Raw Material Store Packaging",
]
```

### Default item groups (`_DEFAULT_ITEMGROUPS`)
12 groups including "Import Item". Modify this list directly in the file.

### ZID toggle (Final Stock table)
- Toggle OFF: totals combined across 100001 + 100009
- Toggle ON (ZID column): shows per-ZID split
- Cross-ZID: itemcode groupby only (no itemname), `_meta` lookup prefers primary ZID names

---

## Customer Columns (all sales-derived tables)

| DB column | Alias | Display label |
|---|---|---|
| `cacus.xmobile` | `cusmobile` | Mobile |
| `cacus.xtaxnum` | `whatsapp` | WhatsApp Number |
| `cacus.xcity` | `area` | Area |

`whatsapp` flows through `build_customer_wise_monthly` and `build_customer_product_monthly` in `processing/target_management.py` — must be in `id_cols` for pivot tables.

---

## Git / Deployment

- **Main branch**: `main` — always deployable
- **Feature branches**: created for larger features, merged to main when approved
- **Server**: Windows, `git pull origin main` to update
- **Runtime JSON files** are gitignored — server users manage them independently
- If merge conflict on `auth/setup_db.py`: `git checkout --theirs auth/setup_db.py && git add auth/setup_db.py && git commit --no-edit`

### Running locally
```bash
pyenv activate streamlitEnv3.10.13
streamlit run app.py
```

### Running `setup_db.py` (auth table setup)
```bash
python auth/setup_db.py
```
(Works from any directory — `sys.path` fix included at top of file.)

---

## Common Pitfalls

1. **Revenue in IS context**: use `altsales` (gross), never `altsales - proddiscount` — discount is a separate GL line (07080001).
2. **Cross-ZID inventory**: group by `itemcode` only, not `(itemcode, itemname)` — same code has different names in 100001 vs 100009 caitem.
3. **"Blank" item group**: stored as NULL/empty in DB, not the string `"Blank"`. Check with `isna() | str.strip() == ""`.
4. **MTD 3M averages need historical data**: sidebar must include ≥3 prior months for Daily Avg (3M) / Monthly Avg (3M) to show nonzero values.
5. **`_ls_sum` negates**: raw GL Revenue is a credit (negative xprime) → `_ls_sum` flips to positive for Level S. Always negate gldetail MTD sums too.
6. **Styler `_calc` column**: drop internal helper columns before applying Pandas Styler; use `row.name` (the index) for row-level logic instead.
