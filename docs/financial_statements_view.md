# Financial Statements View — Code Reference

## Overview

The Financial Statements page is defined in `views/financial.py` and rendered via the `display_financial_statements(current_page, zid)` function (decorated with `@timed`).

All data processing is handled by `processing/financial.py`. The view file is purely concerned with sidebar controls, calling processing functions, and rendering the results.

---

## Sidebar Controls

All controls live in `st.sidebar`:

| Control | Type | Values | Default |
|---|---|---|---|
| **Timeframe** | Selectbox | `Yearly`, `Monthly` | `Yearly` |
| **Select End Year** | Selectbox | Last 10 years descending | Current year |
| **Select Start Month** | Selectbox | `1–12` | `1` |
| **Select End Month** | Selectbox | `1–12` | `12` |

- **Yearly mode** builds a `year_list` of 5 consecutive years ending at `selected_year`, e.g. `[2021, 2022, 2023, 2024, 2025]`.
- **Monthly mode** uses only the last year from `year_list` (i.e. `selected_year`) and iterates months `start_month` through `end_month`.

---

## Main Page Controls (inline)

Two controls rendered in a 2-column layout at the top of the page:

| Column | Control | Purpose |
|---|---|---|
| Left | **View Statements For** | Selectbox of `(zid, project)` tuples — selects which business unit's data to display. Defaults to the session's active `zid`. |
| Right | **Select Level** | Selectbox — controls which detail level is shown (see Level Options below). |

---

## Level Options

```
Level 0 - Most Detail
Level 1 - Moderate Detail
Level 2 - Least Detail
Level S - Customised Detail
```

Only the selected level's tables are rendered. Levels 0–2 are fully implemented. Level S shows `st.info("Coming soon.")`.

---

## Data Loading

Business units and projects are read from `data/businesses.json`. Label mappings are read from `data/labels.json`:
- `income_statement_label` → maps `ac_lv4` codes to Income Statement category names
- `balance_sheet_label` → maps `ac_lv4` codes to Balance Sheet category names

All business units and their projects are processed upfront into two dictionaries keyed by `(zid, project)`:
- `main_data_dict_pl` — Profit & Loss data
- `main_data_dict_bs` — Balance Sheet data

---

## Processing Pipeline

The processing pipeline is identical for both Yearly and Monthly perspectives (different function calls are used per perspective). After the data is loaded and the user selects a business unit, the following sequence runs:

### Step 1 — Strip hierarchy columns
```python
drop_cols = ['ac_type','ac_lv1','ac_lv2','ac_lv3','ac_lv4','ac_lv5']
pl_lv0 = pl_raw.drop(columns=drop_cols)
bs_lv0 = bs_raw.drop(columns=drop_cols)
```

### Step 2 — Level 0 base computation
```python
pl_sorted, net_profit, net_profit_m, dep_row = financial.sort_pl_level0(pl_lv0, selected_perspective=...)
```
- `pl_sorted` — full Income Statement rows sorted in IFRS order, with a **Net Profit/Loss** row appended at the bottom.
- `net_profit` — Series of net profit totals per period (used to inject NP into Balance Sheet).
- `net_profit_m` — Monthly-only: cumulative net profit per month within each fiscal year.
- `dep_row` — Single-row DataFrame for Depreciation (ac_code `06360001`), used downstream for cash flow construction.

```python
bs_lv0 = financial.append_net_profit_to_bs_level0(bs_lv0, net_profit)
```
- Appends the Net Profit row into the Balance Sheet so Assets = Liabilities + Equity + NP.

```python
coc_lv0 = financial.cash_open_close(bs_lv0)
```
- Extracts opening and closing cash positions from the Balance Sheet for cash flow reconciliation.

```python
cfs_df, summary_df = financial.make_cashflow_statement_level0(pl_lv0, bs_lv0, coc_lv0, selected_perspective=...)
```
- `cfs_df` — Full IFRS-structured Cash Flow Statement with: Net Profit/Loss, Working Capital rows → WC subtotal, Cash from Operations, CapEx rows → CapEx subtotal, Financing rows → Financing subtotal, Net ΔCash.
- `summary_df` — A compacted summary of the cash flow (used in the "Cash Flow Summary" expander).

### Step 3 — Level 1 & 2 consolidation
```python
pl_lv1, pl_lv2 = financial.level_builder(pl_sorted, "IS")
bs_lv1, bs_lv2 = financial.level_builder(bs_lv0, "BS")
```
- Groups Level 0 rows into L1 and L2 buckets using the IS/BS lookup maps from `data/` JSON config, summing numeric columns.

```python
pl_lv1, bs_lv1, net_profitlv1, dep_rowlv1 = financial.add_np_and_balance_lv1(pl_lv1, bs_lv1, selected_perspective=...)
cfs_lv1 = financial.consolidate_cfs(cfs_df, level=1, debug=True)
cfs_lv1, summary_df1 = financial.build_cfs_level1_summary_df(cfs_lv1, net_profitlv1, dep_rowlv1, coc_lv0)
```

```python
pl_lv2, bs_lv2, net_profitlv2 = financial.add_np_and_balance_lv2(pl_lv2, bs_lv2, selected_perspective=...)
cfs_lv2 = financial.consolidate_cfs(cfs_df, level=2, debug=True)
cfs_lv2, summary_df2 = financial.build_cfs_level2_summary(cfs_lv2, net_profitlv2, dep_rowlv1, coc_lv0)
```

> **Note:** All three levels are always computed regardless of which level is selected. Only the rendering is conditional.

---

## Rendering Structure

Each level renders 4 expanders. The first 3 are **expanded by default**; the Cash Flow Summary is **collapsed by default**.

### Level 0 — Most Detail
| Expander | Content | Default State |
|---|---|---|
| Income Statement | `pl_sorted` | Open |
| Balance Sheet | `bs_lv0` | Open |
| Cash Flow Statement | `cfs_df` | Open |
| Cash Flow Summary | `summary_df` | Closed |

### Level 1 — Moderate Detail
| Expander | Content | Default State |
|---|---|---|
| Income Statement | `pl_lv1` | Open |
| Balance Sheet | `bs_lv1` | Open |
| Cash Flow Statement | `cfs_lv1` | Open |
| Cash Flow Summary | `summary_df1` | Closed |

### Level 2 — Least Detail
| Expander | Content | Default State |
|---|---|---|
| Income Statement | `pl_lv2` | Open |
| Balance Sheet | `bs_lv2` | Open |
| Cash Flow Statement | `cfs_lv2` | Open |
| Cash Flow Summary | `summary_df2` | Closed |

### Level S — Customised Detail
```
st.info("Coming soon.")
```

---

## Variable Reference

| Variable | Type | Description |
|---|---|---|
| `pl_sorted` | DataFrame | Level 0 Income Statement, IFRS-ordered + Net Profit row |
| `bs_lv0` | DataFrame | Level 0 Balance Sheet + Net Profit injected |
| `cfs_df` | DataFrame | Level 0 Cash Flow Statement (full IFRS detail) |
| `summary_df` | DataFrame | Level 0 Cash Flow Summary (compacted) |
| `pl_lv1` | DataFrame | Level 1 Income Statement (grouped by L1 bucket) |
| `bs_lv1` | DataFrame | Level 1 Balance Sheet (grouped by L1 bucket) |
| `cfs_lv1` | DataFrame | Level 1 Cash Flow Statement |
| `summary_df1` | DataFrame | Level 1 Cash Flow Summary |
| `pl_lv2` | DataFrame | Level 2 Income Statement (grouped by L2 bucket) |
| `bs_lv2` | DataFrame | Level 2 Balance Sheet (grouped by L2 bucket) |
| `cfs_lv2` | DataFrame | Level 2 Cash Flow Statement |
| `summary_df2` | DataFrame | Level 2 Cash Flow Summary |
| `net_profit` | Series | Net profit totals per period (Yearly) |
| `net_profit_m` | Series | Cumulative monthly net profit (Monthly only) |
| `dep_row` | DataFrame | Depreciation row (ac_code `06360001`) for CFS |
| `coc_lv0` | DataFrame | Opening/closing cash positions from Balance Sheet |

---

## Key Constraints / Notes

1. **No logic changes permitted in the view** — `views/financial.py` only controls display. All calculation changes must go in `processing/financial.py`.
2. **All levels are always computed** — the `if selected_level == ...` block only controls what is rendered, not what is processed.
3. **Yearly vs Monthly** — the two perspectives use different processing functions (`process_data` vs `process_data_month`, `net_profit` vs `net_profit_m` passed to BS) but are otherwise structurally identical in rendering.
4. **`zid` shadowing** — the outer function parameter `zid` is shadowed by the loop variable in the data loading loop (`for zid, details in businesses.items()`). The loop intentionally overwrites it to iterate all businesses.
5. **Level S** is a placeholder — no data computation is associated with it.
