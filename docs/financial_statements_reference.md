# Financial Statements — Complete Reference

> **Purpose of this document:** Full description of how the Financial Statements page works —
> data flow, statement levels, available perspectives, ratios, and all supporting panels.
> Intended as context for designing a companion graphical/dashboard tab.

---

## 1. Entry Points & Sidebar Controls

The page is rendered by `display_financial_statements()` in `views/financial.py`.

### 1.1 Timeframe (Perspective)

The sidebar **Timeframe** selector drives everything downstream.
Four options are available:

| Option | What it shows | Extra sidebar controls |
|---|---|---|
| **Yearly - Custom Range** | 5 years ending at *End Year*, bounded by Start/End Month | End Year, Start Month, End Month |
| **Yearly - Full Year vs YTD** | 4 prior full years + current year up to a chosen month | End Year, Up-to Month |
| **Monthly** | Same 5-year window but columns split by month (year × month) | End Year, Start Month, End Month |
| **Lifetime** | All years from earliest choice to current year | Earliest Year, Current Year Up-to Month |

The month range controls only apply to the IS/CFS.
The Balance Sheet always reads balance-sheet accounts at each period boundary.

### 1.2 Entity selector ("View Statements For")

A dropdown lists every individual business unit (ZID + project) **plus** a special
`consolidated / All Businesses - Consolidated` option at the top.

Default selection is pre-populated from the global `zid` stored in session state.

### 1.3 Level selector

The available levels depend on which entity and perspective are selected:

**Individual entity (any ZID, any perspective):**
- Level 0 — Most Detail
- Level 1 — Moderate Detail
- Level 2 — Least Detail
- Level S — Customised Detail

**Individual entity ZID 100001, Full Year vs YTD or Lifetime only:**
- Level T — Trading Adjustments *(additional option)*

**Individual entity ZID 100000, Full Year vs YTD or Lifetime only:**
- Level T — GI Adjustments *(additional option)*

**Consolidated view:**
- Level C — Raw Consolidation
- Level C2 — Consolidated Detail
- Level 1 — Moderate Detail
- Level 2 — Least Detail
- Level S — Customised Detail

---

## 2. Data Pipeline

### 2.1 Raw data loading

For every entity (ZID × project), two raw datasets are loaded via `financial.process_data()`:

- **Income Statement** raw data (excludes Asset & Liability account types)
- **Balance Sheet** raw data (excludes Income & Expenditure account types)

Both are pivoted by `(year, month)` into period columns.
Column labels are integers (Yearly) or `(year, month)` tuples (Monthly).

For **Full Year vs YTD** and **Lifetime**, prior years and the current year are loaded
separately and merged: prior years use full-year (months 1–12), the current year is
bounded by the selected up-to month.

### 2.2 Consolidated path

When *All Businesses - Consolidated* is selected:

1. **Level C** — raw per-ZID frames are concatenated (`build_level_c`).
   Retains the `zid` column. No eliminations.

2. **Level C2** — intercompany eliminations are applied (`build_level_c2_is`,
   `build_level_c2_bs`) using rules from `data/consolidation_rules.json`:
   - **InternalLoans** — paired interco loans are eliminated if they net to zero
     across all years; otherwise both rows are kept.
   - **SalesCOGS** — internal-ZID sales are netted out of COGS; only external sales
     are retained.
   - **External BS / ARAP2 BS / Allother2 BS** — external AR/AP and remaining BS
     items are summed by code+name.
   
   Level C2 becomes the *Level 0 equivalent* for all downstream levels in the
   consolidated view.

3. A `force_zero_periods` rule in the JSON can zero specific ZID × year cells
   (used to prevent double-counting when internal entities transferred NP during
   a model change year).

### 2.3 Hierarchy levels (Level 0 → Level 2)

Built entirely in `processing/financial.py`:

| Level | IS produced by | BS produced by |
|---|---|---|
| **Level 0** | `sort_pl_level0()` — sorts accounts by prefix code, appends Net Profit/Loss | `append_net_profit_to_bs_level0()` — appends Net P/L and Balance Check rows |
| **Level 1** | `level_builder(pl_sorted, "IS")` then `add_np_and_balance_lv1()` | `level_builder(bs_lv0, "BS")` then `add_np_and_balance_lv1()` |
| **Level 2** | `level_builder(pl_lv1, "IS")` then `add_np_and_balance_lv2()` | Same pattern |

The `level_builder()` function aggregates account codes by hierarchy prefix,
reducing the row count at each level.

The **Cash Flow Statement** is computed once at Level 0 (`make_cashflow_statement_level0()`)
then rolled up to Level 1 and Level 2 via `consolidate_cfs()`.

### 2.4 Sign convention

| Context | Positive = |
|---|---|
| Level 0 / 1 / 2 IS | Loss (raw accounting convention) |
| Level S IS | Profit (management view — signs are flipped) |
| Level S BS | Asset/Liability as stored |
| CFS | Inflow |

The **Sanity Check** panel accounts for the Level S sign flip when comparing Net
Profit/Loss across levels.

---

## 3. Statement Levels — What Each Shows

### 3.1 Level 0 — Most Detail / Level C2 — Consolidated Detail

Displays every individual account code that passes the type filter.
Four expandable panels:
- **Income Statement** — all IS lines sorted by prefix hierarchy
- **Balance Sheet** — all BS lines
- **Cash Flow Statement** — detailed CFS (working capital movements per account)
- **Cash Flow Summary** — bucketed summary (Operations / Investing / Financing)

Excel download available.

### 3.2 Level 1 — Moderate Detail

Same four panels, but IS and BS accounts are aggregated into ~15–25 named groups
(e.g. "08-Revenue", "07-SG&A Expenses").

### 3.3 Level 2 — Least Detail

Further aggregated into the broadest categories
(e.g. "Revenue", "COGS", "Net Profit/Loss" for IS;
"Current Assets", "Fixed Assets" for BS).

### 3.4 Level S — Customised Detail (Management View)

The most analytically rich level. Built by `build_pl_level_s()`, `build_bs_level_s()`,
`build_cfs_level_s()`.

#### Level S Income Statement rows

| Row | Description |
|---|---|
| Revenue | Raw revenue (all IS income lines summed) |
| Others Revenue | Non-sales income lines |
| MRP Discount | Discount embedded in listed prices |
| **Adjusted Revenue (Pending)** | Revenue − MRP Discount — primary revenue measure |
| COGS | Cost of goods sold |
| **Gross Profit** | Adjusted Revenue − COGS |
| SG&A | Header |
| 0612-Salary Expenses | |
| 0613-Employee Bonus | |
| 0614-Overtime | |
| 0615-Director Remuneration | |
| **Total SG&A** | |
| 0708-Discount Paid | Cash discounts given to customers |
| Sales & Distribution Expenses | All distribution/logistics sub-lines |
| **Total Sales & Distribution** | |
| 0501-Others Direct Expenses | |
| **EBITDA** | Gross Profit − Total SG&A − Total S&D − Others Direct |
| 0630-Bank Interest & Charges | |
| 0633-Interest-Loan | |
| **Total Interest & Charges** | |
| VAT Expenses from Rebate (A) | VAT rebate, sourced from GL — see §3.4.1 |
| VAT through Cash (i) | |
| Others Company Tax (ii) | |
| VAT Expenses Office (iii) | |
| **Net VAT Expenses Cash (B)** | |
| 0629-Income Tax Expenses (C) | |
| **0629-VAT & Tax Total (A+B+C)** | |
| **Net Income** | EBITDA − Interest − VAT & Tax Total |

#### Level S Balance Sheet rows

**Current Assets**
`Cash & CE` · `Bank Balance` · `AR (External)` · `AR (Internal)` · `AR (Agent/Dealers)` ·
`AR (Local)` · `Defaulted Receivables` · `AR Total` · `Prepaid Expenses` ·
`Advance Accounts` · `Stock in Hand` → **Total Current Assets (A)**

**Other Assets**
`Deferred CapEx` · `Gift Items` · `Loan to Hospital` · `Loan to Others (Surma)` ·
`Security Deposit` · `Loan to Others Concern` · `Other Investment` → **Total Other Assets (B)**

**Fixed Assets**
`Office Equipment` · `Corporate Office Equipment` · `Furniture & Fixture` · `Trading Vehicles` ·
`Private Vehicles` · `Plants & Machinery` · `Intangible Asset` · `Land & Building` → **Total Fixed Assets (C)**

→ **Total Assets (A+B+C)**

**Current Liabilities**
`Accrued Expenses` · `AP (Local)` · `AP (Internal)` · `AP (International)` · `AP Total` ·
`Money Agent Liability` · `Reconciliation Liability` · `C&F Liability` · `Others Liability`
→ **Current Liability (A)**

**Short Term Liabilities**
`Short Term Bank Loan (B)` · `Short Term Loan (Related Parties)` → **Total Short Term Liability (C)**

**Long Term Liabilities**
`Long Term Bank Loan (E)`

**Reserve & Funds**
`Employee Fund` · `Directors Award Fund` · `Office Rent Tax Fund` · `Employee Educational Fund` ·
`Security Deposit Fund` → **Total Reserve & Funds (D)**

→ **Total Liabilities (A+B+C+D+E)**

**Equity**
`Share Capital` · `Non-Cash Capital (Retained Earning)` · `Error Adjustment For Retained Earning` ·
`Net Profit/Loss` → **Total Equity** → **Total Liabilities & Equity** → **Balance Check** (= 0)

#### Level S Cash Flow Statement sections

| Section | Contents |
|---|---|
| **Cash from Operations** | Net Profit (negated for CFS convention) + Δ Working Capital items (AR, AP, Stock, Prepaid, Advances, Accrued, Agents, etc.) |
| **Cash from Investing** | Δ Fixed Assets, Δ Deferred CapEx, Δ Other Assets |
| **Cash from Financing** | Δ Short-term bank loan, Δ Long-term loan, Δ ST related-party loan, Δ Equity rows |
| **Opening / Closing Cash** | Reconciliation using Cash & CE + Bank Balance; **Cash-flow Check** = 0 |

#### 3.4.1 VAT rows — GL sourced

For Level S, the four VAT rows are fetched separately from the GL
(`queries.get_vat_breakdown_gl()`) and inserted via `compute_vat_is_rows()`.
In the consolidated view, VAT is summed across all ZIDs.
For Full Year vs YTD, prior years use months 1–12; Custom Range uses the selected range.

#### 3.4.2 Monthly perspective

When **Monthly** is selected, period columns become `(year, month)` tuples.
The BS uses a YTD cumulative Net Income (accumulated within each fiscal year)
via `_monthly_to_ytd()` so the Balance Check holds every month.

### 3.5 Level T — Trading Adjustments (ZID 100001 only)

Available only for Full Year vs YTD and Lifetime perspectives.

**What it does:** Removes the Industrial & Household (I&H) product segment that was
split out prior to 2025, to show the continuing business on a like-for-like basis.
Only adjusts years ≤ 2024.

**IS adjustments:**
- Revenue: I&H net sales removed
- COGS: I&H net cost removed
- SG&A + S&D: proportional reduction (8% of I&H revenue distributed across sub-lines)
- Pre-2022: VAT Rebate reclassified from COGS to its own line

**BS adjustments:**
- Stock in Hand: I&H closing inventory removed (data from `data/level_t_stock_inh.json`)
- Accounts Receivable: scaled by I&H revenue share
- Cash & Bank: scaled by I&H revenue share
- Equity: `1302-Error Adjustment For Retained Earning` absorbs the asset reduction gap; Balance Check = 0

**CFS:** Rebuilt entirely from the adjusted BS.

Displays:
- Income Statement (Level T)
- Balance Sheet (Level T)
- Cash Flow Statement (Level T)
- Cash Flow Summary
- **Level T — Adjustment Breakdown** expander: table of all adjustment line values by year

### 3.6 Level T — GI Adjustments (ZID 100000 only)

Mirror of Level T for GI Corporation — adds the I&H segment assets/revenues that
were previously housed in 100001 into the 100000 books.
Displays the same four statement panels plus the breakdown table.
The 100001 adjustment breakdown is shown as a reference alongside.

### 3.7 Level C — Raw Consolidation

Simple concatenation of all ZID Level-0 frames. Retains the `zid` column.
Appends Net Profit/Loss and Balance Check rows.
CFS is built from the Level C frames (`build_level_c_cfs`).
Displayed in three expandable panels: IS, BS, CFS.

---

## 4. Panels Rendered at Level S

### 4.1 Financial Ratios

An expandable panel (📊 Financial Ratios) rendered below the download link.
Computed by `_build_ls_ratios()` from the Level S IS, BS, and CFS.

**Profitability**

| Ratio | Formula |
|---|---|
| Markup on COGS (%) | Adjusted Revenue / COGS − 1 |
| Gross Profit Margin (%) | Gross Profit / (Adj. Revenue + Others Revenue) |
| Staff Cost / Revenue (%) | (Salary + Bonus + Overtime) / Revenue |
| Total SG&A / Revenue (%) | Total SG&A / Revenue |
| Discount Paid / Revenue (%) | Discount Paid / Revenue |
| Distribution / Revenue (%) | S&D Expenses / Revenue |
| Total S&D / Revenue (%) | Total S&D / Revenue |
| EBITDA Margin (%) | EBITDA / Revenue |
| Interest Coverage (×) | EBITDA / Total Interest & Charges |
| VAT Expense / Revenue (%) | Net VAT Cash / Revenue |
| Total Tax / Revenue (%) | VAT & Tax Total / Revenue |

**Liquidity** *(internal items excluded from both numerator and denominator)*

| Ratio | Formula |
|---|---|
| Current Ratio (×) | Adj. Current Assets / Adj. Current Liabilities |
| Quick Ratio (×) | (Adj. CA − Stock) / Adj. CL |

*Adj. CA = Total CA − AR Internal − AR Local;
Adj. CL = Total CL − AP Internal − Money Agent Liability*

**Working Capital Days**

| Ratio | Formula | Days Factor |
|---|---|---|
| DSO — Receivable Days | AR (External) × days / Revenue | 365 Yearly / 30 Monthly |
| DIO — Inventory Days | Stock × days / COGS | |
| DPO — Payable Days | (AP Local + AP International) × days / COGS | |
| Cash Conversion Cycle | DSO + DIO − DPO | |

**Leverage**

| Ratio | Formula |
|---|---|
| D/E 1 — Bank + MA + LT (×) | (ST Bank Loan + Money Agent + LT Loan) / Total Equity |
| D/E 2 — STL + LT (×) | (Total Short Term Liabilities + LT Loan) / Total Equity |

**Cash Flow**

| Ratio | Formula |
|---|---|
| OCF Coverage (×) | Cash from Operations / Total Current Liabilities |
| Free Cash Flow | Cash from Operations + Cash from Investing |

### 4.2 Cross-Level Sanity Checks

Rendered below the download link at Level S.
Shows three side-by-side tables comparing the same metric across Level 0 / 1 / 2 / S:
- **Net Profit / Net Income** — sign-normalised so all levels are comparable
- **BS Balance Check** — should be ~0 at all levels
- **CFS Cash-flow Check** — should be ~0 at all levels

### 4.3 Notes & Context / ZID Contribution Breakdown

A radio button toggles between two panels.
Available only at Level S (both individual and consolidated).

**Notes & Context panel**
Expandable (📋 Account Notes & Descriptions) with two inner tabs:
- *Account Notes* — description + contextual note for every IS/BS/CFS row
  (loaded from `data/ls_account_notes.json`)
- *Ratio Notes* — formula and interpretation for every financial ratio

**ZID Contribution Breakdown panel** (consolidated view only)

- **Account selector** — any Level S IS or BS row name
- **Period selector** — any available period column (defaults to most recent)
- **Table** — per-ZID value for the selected account and period
- **Donut pie chart** — absolute contribution by ZID (sized by `abs(value)`;
  flagged in title if mixed positive/negative values are present)
  Zero-value ZIDs are excluded from both table and chart.

---

## 5. Excel Downloads

Every level offers a combined Excel download containing IS + BS + CFS in separate sheets.
Generated by `common.create_combined_ls_download_link()`.

| Level | Filename |
|---|---|
| Level C | `LevelC_Consolidated_Financial_Statements.xlsx` |
| Level C2 / Level 0 | `LevelC2_Consolidated_Financial_Statements.xlsx` or `Level0_{zid}_Financial_Statements.xlsx` |
| Level 1 | `Level1_{zid}_Financial_Statements.xlsx` |
| Level 2 | `Level2_{zid}_Financial_Statements.xlsx` |
| Level S (individual) | `LevelS_Financial_Statements.xlsx` |
| Level S (consolidated) | `LevelS_Consolidated_Financial_Statements.xlsx` |
| Level T (Trading) | `LevelT_Financial_Statements.xlsx` |

---

## 6. Key Data Files

| File | Purpose |
|---|---|
| `data/businesses.json` | ZID list, names, projects, display labels |
| `data/labels.json` | Mapping of `ac_lv4` codes → IS/BS section labels |
| `data/consolidation_rules.json` | Interco elimination rules, force-zero periods, all_zids list |
| `data/ls_account_notes.json` | Level S row descriptions and ratio notes |
| `data/level_t_vat_prior.json` | Pre-2022 VAT rebate values for Level T IS adjustment |
| `data/level_t_stock_inh.json` | I&H closing inventory values (2016–2023) for Level T BS adjustment |

---

## 7. Summary Table — All Level × Statement Combinations

| Level | IS | BS | CFS | CFS Summary | Ratios | Sanity | Notes/ZID |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Level C | ✓ | ✓ | ✓ | ✓ | | | |
| Level C2 / Level 0 | ✓ | ✓ | ✓ | ✓ | | | |
| Level 1 | ✓ | ✓ | ✓ | ✓ | | | |
| Level 2 | ✓ | ✓ | ✓ | ✓ | | | |
| Level S | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Level T (Trading) | ✓ | ✓ | ✓ | ✓ | | | Adjustment breakdown |
| Level T (GI) | ✓ | ✓ | ✓ | ✓ | | | Adjustment breakdown |
