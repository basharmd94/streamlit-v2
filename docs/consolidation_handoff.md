# Consolidation Engine — Handoff Notes

> **Branch:** `claude/stupefied-payne`
> **Worktree:** `.claude/worktrees/stupefied-payne/`
> **Primary file:** `processing/consolidation.py`
> **Caller:** `views/financial.py` (two call sites: yearly Level 0 ~line 345, monthly Level 0 ~line 1110)

---

## 1. Business Structure

Ten businesses in the group, identified by ZID:

| ZID    | Name                    | Role        | Notes                        |
|--------|-------------------------|-------------|------------------------------|
| 100000 | GI Corporation          | Outward     | Separate entity from 2024 only |
| 100001 | HMBR T & C Ltd.         | Outward     | Main hub; buys from subs, sells to market |
| 100002 | Gulshan Chemical        | Subsidiary  | Sells exclusively to 100001  |
| 100003 | Gulshan Thread Tape     | Subsidiary  | Sells exclusively to 100001  |
| 100004 | Gulshan Plastic         | Subsidiary  | Sells exclusively to 100001  |
| 100005 | Zepto Chemicals         | Outward     | Sells to market              |
| 100006 | HMBR Grocery Shop       | Subsidiary  | Sells exclusively to 100001  |
| 100007 | HMBR Paint Roller Co.   | Subsidiary  | Sells exclusively to 100001; **zero GL data** |
| 100008 | HMBR Steel Scrubber Co. | Subsidiary  | Sells exclusively to 100001  |
| 100009 | Gulshan Packaging Co.   | Subsidiary  | Sells exclusively to 100001  |

**Key rules:**
- **100001 has NO internal AR** — it only has internal AP (to subs and to 100000/100005 where applicable).
- **100000 ↔ 100005** have no AR/AP interaction with each other.
- **100001 ↔ 100005** have no AR/AP interaction.
- Subsidiaries (100002–100009) sell exclusively to 100001; their `01030001` AR is 100% intercompany.
- 100007 has zero GL data — any asset on 100001's books pointing to it is a phantom.

---

## 2. Files Changed in This Session

| File | What Changed |
|------|-------------|
| `processing/consolidation.py` | **New file** — entire consolidation engine |
| `views/financial.py` | Added `render_consolidation_notes()` call at Level 0 for yearly and monthly views |
| `app.py` | Minor changes (prior sessions) |
| `core/db.py` | Minor changes (prior sessions) |
| `core/queries.py` | Minor changes (prior sessions) |
| `data/businesses.json` | Updated business list |
| `db_sync/schema_info.json` | DB sync metadata |
| `db_sync/sync_files.py` | DB sync utility |
| `processing/common.py` | Shared processing helpers |
| `processing/financial.py` | Financial statement processing |
| `data/level_s_mapping.json` | New — Level S mapping |
| `data/level_t_stock_inh.json` | New — Level T stock inheritance |
| `data/level_t_vat_prior.json` | New — Level T VAT prior |
| `docs/financial_statements_view.md` | Existing doc (from prior session) |

---

## 3. Key Constants in `processing/consolidation.py`

### `INTERCO_FORCE_ZERO`

Codes that are zeroed for a specific ZID **before summation** because their counterpart data does not exist or is unreliable:

```python
INTERCO_FORCE_ZERO: dict[str, set[str]] = {
    "100001": {
        "02050012",   # Loan to HMBR Paint Roller (100007) — no counterpart data
        "02050006",   # Loan to Karigor project — no counterpart data
        "02050001",   # Loan to HMBR Online Shop (100007) — 100007 has zero GL data → 151M phantom
    },
    "100000": {"02050012", "02050006"},
    "100005": {"02050003"},
}
```

### `INTERCO_BS_ELIMINATIONS`

Cross-code pairs (or same-code internal flows) zeroed unconditionally for all years.  
Both sides zeroed before summation so they don't inflate the consolidated BS.

```python
INTERCO_BS_ELIMINATIONS: dict[str, set[str]] = {
    "100001": {
        "02050010", "02050011", "02050013", "02050015",
        "02050017", "09060001", "10020017",
        "10020005", "10020006", "10020007", "10020008",
        "09030003",   # GROUP F — AP Internal ↔ 100000/01030002 and 100009/01030001
    },
    "100005": {
        "10020002", "02050002", "02050005", "10020005",
        "09030002",   # GROUP F — AP Internal ↔ 100000/01030002
    },
    "100000": {
        "01020003", "02050005", "10020003", "02050010", "02050015",
        "01030002",   # GROUP F — "AR from GTC & Zepto" interco AR
        "09030002",   # GROUP F — AP Internal ↔ 100009/01030001
    },
    "100002": {"02050005", "01030001"},   # GROUP G — sub AR
    "100003": {"02050005", "10020008", "01030001"},
    "100004": {"02050001", "10020003", "02050002", "01030001"},
    "100006": {"10020002", "01030001"},
    "100007": {"10020003", "01030001"},
    "100008": {"10020002", "01030001"},
    "100009": {"10020002", "02050001", "01030001"},
}
```

### `INTERCO_BS_YEAR_COND`

Year-range-conditional zeroing. Format: `list[tuple(year_from_incl_or_None, year_to_incl_or_None, {zid: set[codes]})]`

```python
INTERCO_BS_YEAR_COND: list[tuple] = [
    (None, 2022, {
        "100001": {"09030001"},
    }),
]
```

**Why this exists:**
- Before 2023 (≤2022): `09030001` on 100001 is **AP Local / Internal** — the counterpart to subs' `01030001` AR.
- From 2023 onward: `09030001` on 100001 is an **external AP code** (100001 switched to `09030003` for interco AP).
- So we zero `09030001` on 100001 for years ≤ 2022 only.
- For 2023+: `09030003` on 100001 is already in `INTERCO_BS_ELIMINATIONS` (unconditional).
- 100000's `01030002` always adjusts with `09030003` (100000 only exists from 2024).

---

## 4. AC Code Name Differences (Critical!)

The **same ac_code means different things in different businesses**. This caused confusion during analysis:

| ac_code  | 100001 meaning        | 100000 meaning          | 100005 meaning       |
|----------|-----------------------|-------------------------|----------------------|
| 09030002 | Construction Materials (EXTERNAL) | AP Internal (INTERCO) | AP Internal (INTERCO) |
| 09030003 | AP Internal (INTERCO) | AP International (EXTERNAL) | AP International (EXTERNAL) |
| 01030002 | Recognized Agent (EXTERNAL) | AR from GTC & Zepto (INTERCO) | Recognized Agent (EXTERNAL) |

**Treatment:**
- `09030002` in 100001 → **keep** (external, do not eliminate)
- `09030002` in 100000/100005 → **zero** (interco AP, in INTERCO_BS_ELIMINATIONS)
- `09030003` in 100001 → **zero** (interco AP ≥2023, in INTERCO_BS_ELIMINATIONS)
- `09030003` in 100000/100005 → **keep** (external AP)
- `01030002` in 100000 → **zero** (interco AR, in INTERCO_BS_ELIMINATIONS)
- `01030002` in 100001/100005 → **keep** (external AR from market)

---

## 5. Cross-Code Elimination Groups (from consolidation notes)

| Group | Description | ZID-Code (Asset/AR side) | ZID-Code (Liability/AP side) | Treatment |
|-------|-------------|--------------------------|------------------------------|-----------|
| A | 100001 loans to subs | 100001/02050010–02050017 | Sub equity/loan codes | Zero 100001 side |
| B | 100001 contra-equity to subs | 100001/10020005–10020008 | Sub equity codes | Zero 100001 side |
| C | 100001 other interco | 100001/09060001, 10020017 | Various | Zero 100001 side |
| D | 100005 interco balances | 100005/10020002,02050002,02050005,10020005 | Various | Zero 100005 side |
| E | 100000 interco balances | 100000/01020003,02050005,10020003,02050010,02050015 | Various | Zero 100000 side |
| F | Cross-code AR/AP pairs | 100000/01030002 ↔ 100001/09030003 | Both zeroed | Zero both |
| | | 100000/09030002 ↔ 100009/01030001 | Both zeroed | Zero both |
| | | 100005/09030002 ↔ (no matching AR found) | Zero 100005 side | |
| G | Sub AR (all sell to 100001) | 100002–100009/01030001 | 100001 AP (Option A — kept) | Zero sub AR only |

**Option A asymmetry (accepted):** Sub AR (`01030001`, total ~129.6M in 2022) is zeroed on sub side. The matching AP on 100001's side (`09030001` pre-2023, `09030003` post-2022) is:
- Pre-2023: zeroed via `INTERCO_BS_YEAR_COND`
- Post-2022: zeroed via `INTERCO_BS_ELIMINATIONS` (`09030003`)

---

## 6. Helper Functions

### `_year_in_range(year, yr_from, yr_to) -> bool`
Converts any year label to int, checks inclusive bounds. `None` = unbounded.

### `_stack_and_sum(data_by_zid, extra_zero=None, year_cond_zero=None) -> pd.DataFrame`
Stacks all ZID dataframes, applies force-zero and year-conditional zero, sums across ZIDs.

### `build_consolidated_raw() -> dict`
Main entry: loads BS and P&L for all 10 ZIDs, applies all eliminations, returns consolidated frames.

### `render_consolidation_notes()`
Public function called from `views/financial.py`. Renders a collapsed expander with full AR/AP interaction notes, group tables, and timeline exceptions.

---

## 7. Outward-Facing Focus Table

A second table in the year breakdown contribution report showing select ac_codes for **100000, 100001, 100005** only.

```python
_OUTWARD_FOCUS_ZID_CODES: dict[str, list[str]] = {
    "100000": [
        "08010001", "04010020", "03080001", "01020003", "01030002",
        "02050015", "02050005", "02050006", "02050010", "02050012",
        "01030003", "02050001", "01030001", "09010003", "10020008",
        "10020007", "10020006", "09030002", "10020003", "09030003", "09030001",
    ],
    "100001": [
        "08010001", "04010020", "01030002", "01030001", "02050012",
        "02050017", "01030003", "02050013", "02050011", "02050010",
        "02050015", "02050006", "10020008", "10020007", "10020006",
        "10020005", "10020004", "09060001", "10020017", "09030001",
        "09030003", "09030002", "09030004",
    ],
    "100005": [
        "08010001", "04010020", "01030002", "01030001", "02050005",
        "02050002", "02050001", "02050003", "10020005", "10020006",
        "09030003", "09030002", "09030001",
    ],
}
_PL_CODES: frozenset[str] = frozenset({"08010001", "04010020"})
_FOCUS_ZIDS: list[str] = ["100000", "100001", "100005"]
```

The table has 31 rows (deduplicated union, maintaining order). For each (ac_code, zid) combination:
- Uses P&L data if code is in `_PL_CODES`, BS data otherwise.
- Cell is `None` (blank) if that code is not in that entity's focus list.
- Applies the same elimination status logic as the main breakdown table.

---

## 8. Outstanding Issues / Future Work

### a) 01030003 naming in 100001
The consolidation sheet notes: *"both names exist for same ac_code, I need to keep both names — separate analysis required."* This code has not been handled yet.

### b) 09010003 (100000) ↔ 01050008 (100001)
Identified as a cross-code pair but both were zero in 2022. Not yet implemented. Check if they become non-zero in later years.

### c) 09030001 in 100001 post-2022
**Option A was accepted** — the AP Local code for 100001 in 2023+ may still embed some interco AP to subs. This is a known accepted asymmetry. If the user wants to revisit, Option B would be to surgically subtract only the interco portion based on sub AP lookup.

### d) BS balance verification
After all eliminations, the consolidated Balance Sheet should be verified for balance (Assets = Liabilities + Equity). This was mentioned as the next debugging step.

### e) 09030004 in 100001
Appears in the focus code list for 100001. Its interco nature has not been fully determined.

---

## 9. Where to Continue

1. Run the app on the worktree branch and verify the consolidated BS numbers make sense.
2. Check `render_consolidation_notes()` renders correctly in the UI under **Consolidated → Level 0**.
3. Verify the outward-facing focus table appears in the year breakdown modal.
4. Continue with BS balance debugging as mentioned by the user.
5. Address the outstanding issues above as directed.

---

## 10. Running the App

```bash
cd /Users/anilsaddatbijoy/Documents/streamlit_analysis_git_v3/streamilt-analysis/.claude/worktrees/stupefied-payne
streamlit run app.py
```

Or from the main repo (if the branch is checked out there):
```bash
cd /Users/anilsaddatbijoy/Documents/streamlit_analysis_git_v3/streamilt-analysis
git checkout claude/stupefied-payne
streamlit run app.py
```
