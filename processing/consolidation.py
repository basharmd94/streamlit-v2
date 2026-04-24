"""
Consolidation engine — aggregates all-business GL data into a single
consolidated set of raw financial data (pl_raw, bs_raw) that flows
unchanged through the existing Level 0 / 1 / 2 / S statement builders.

Key mechanics
-------------
Natural sign netting
    Level-0 assets are positive, liabilities negative.  Any intercompany pair
    recorded on both sides (e.g. 100000's loan asset 02050005 = +X and
    100001's matching liability 10020017 = −X) sums to zero automatically
    across businesses.  No explicit elimination code is required for these
    "matched" same-code pairs.

Cross-code AR / AP pairs
    Where businesses use *different* ac_codes for the same intercompany
    relationship (e.g. 100000's 01030002 AR ↔ 100001's 09030003 AP) both
    sides appear in the consolidated Level 0 detail but with opposite signs,
    so total assets and total liabilities are each inflated by the same
    amount and the BS balance check remains zero.

Code variants at Level S
    Economically identical accounts that use different ac_codes in different
    businesses (e.g. Recognised Agent = 01030002 in most ZIDs, 01030003 in
    100000) are merged into one bucket via the consolidated Level S routing
    defined in CONSOLIDATED_BS_ROUTES (and registered in
    processing/financial.py under _BS_LEVEL_S_CODE_ROUTES["consolidated"]).

Force-to-zero
    A small set of codes have a known data gap — one side of the intercompany
    pair is absent.  Management confirmed these should be treated as zero to
    avoid phantom consolidated balances.

P&L net-income correctness
    100001 acts as internal distributor to 100002/3/4/7/8/9.  Its revenue
    from those businesses (08010001) carries a positive sign; the matching
    COGS on the subsidiary side (04010020) carries a negative sign.  They
    cancel exactly in the consolidated sum, so *net income is correct*.
    Gross revenue and COGS are each overstated by the intercompany trading
    volume; the debug expander quantifies this.
"""

from __future__ import annotations

import ast
import re

import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CONSOLIDATION_ALL_ZIDS: list[str] = [
    "100000", "100001", "100002", "100003", "100004",
    "100005", "100006", "100007", "100008", "100009",
]

# Level S consolidated BS code routing.
# After intercompany netting:
#   ar_agent  — both 01030002 (most businesses) and 01030003 (100000)
#   ap_local  — 09030001 plus 09030002 (100001 Construction Materials AP)
#   ap_intl   — 09030003 plus 09030004 (100001 variant)
# Internal AR / AP buckets are empty because those intercompany amounts net
# within total assets / total liabilities at the group level.
CONSOLIDATED_BS_ROUTES: dict = {
    "ar_main":      ["01030001"],
    "ar_internal":  [],
    "ar_agent":     ["01030002", "01030003"],
    "ar_local":     [],
    "ar_defaulted": [],
    "ap_local":     ["09030001", "09030002"],
    "ap_internal":  [],
    "ap_intl":      ["09030003", "09030004"],
}

# Codes to force to zero — one side of the intercompany pair is absent.
# User confirmed these should be zero to avoid phantom consolidated balances.
INTERCO_FORCE_ZERO: dict[str, set[str]] = {
    "100001": {
        "02050012",   # Loan to HMBR Paint Roller (100007) — no counterpart data
        "02050006",   # Loan to Karigor project — no counterpart data
        "02050001",   # Loan to HMBR Online Shop (100007) — 100007 has zero GL data
    },
    "100000": {"02050012", "02050006"},   # same force-zeros on GI side
    "100005": {"02050003"},               # Loan to Paint Roller — no counterpart data
}

# ─────────────────────────────────────────────────────────────────────────────
# Cross-code BS intercompany eliminations
# ─────────────────────────────────────────────────────────────────────────────
#
# These are intercompany balances where the ASSET uses a different ac_code from
# the matching LIABILITY, so natural sign netting doesn't cancel them.  Without
# elimination both sides appear in the consolidated BS, inflating total assets
# and total liabilities equally (balance check stays 0 but gross figures overstate).
#
# Each entry zeros out that ac_code for that ZID BEFORE the groupby sum.
# Both sides of every pair must appear here so neither side survives in the total.
#
# ── AR / AP NOTE ─────────────────────────────────────────────────────────────
# Same ac_code carries DIFFERENT meanings across entities.  Critical cases:
#
#   09030002
#     100001 → "Construction Materials Suppliers MB" = EXTERNAL local AP
#              (added with 09030001; excluded from eliminations)
#     100000 → "AP Internal" = interco AP  ← ELIMINATED below (GROUP F)
#     100005 → "AP Internal" = interco AP  ← ELIMINATED below (GROUP F)
#
#   09030003
#     100001 → "AP Internal" = interco AP  ← ELIMINATED below (GROUP F)
#     100000 → "AP International" = external  (kept; merged with 100001's 09030004)
#     100005 → "AP International" = external  (kept; merged with 100001's 09030004)
#
#   01030002
#     100001 → "Recognized Agent" = external commission AR  (kept)
#     100005 → "Recognized Agent" = external commission AR  (kept)
#     100000 → "AR from GTC & Zepto" = interco AR owed by 100001/100005
#              ← ELIMINATED below (GROUP F)
#
#   01030001
#     100001 → Accounts Receivable = EXTERNAL only; 100001 has NO internal AR
#     100005 → Accounts Receivable = external
#     100000 → Accounts Receivable = external (entity created 2024; zero before)
#     100002/3/4/6/7/8/9 → all interco (subs sell exclusively to 100001)
#              ← ELIMINATED below (GROUP G)
#
# ── GROUPS ──────────────────────────────────────────────────────────────────
#
#  GROUP A — 100001 loans to subsidiaries (asset 02050xxx) / sub liability 10020002
#    02050010 (100001) ↔ 10020002 (100006)
#    02050011 (100001) ↔ 10020002 (100005)
#    02050013 (100001) ↔ 10020002 (100008)
#    02050015 (100001) ↔ 10020002 (100009)
#
#  GROUP B — 100001 ↔ 100000 cross-loans
#    02050017 (100001) ↔ 10020003 (100000)
#    09060001 (100001) ↔ 01020003 (100000)
#    10020017 (100001) ↔ 02050005 (100000)
#
#  GROUP C — 100005 intercompany loans
#    02050002 (100005) ↔ 10020008 (100003)
#    02050005 (100005) ↔ 10020003 (100004)
#    10020005 (100005) ↔ 02050002 (100004)
#
#  GROUP D — 100001 liabilities to subsidiaries
#    10020005 (100001) ↔ 02050005 (100002)
#    10020006 (100001) ↔ 02050005 (100003)
#    10020007 (100001) ↔ 02050001 (100009)
#    10020008 (100001) ↔ 02050001 (100004)
#
#  GROUP E — 100000 loans to subsidiaries
#    02050001 (100000) ↔ 10020003 (100007)
#    02050010 (100000) ↔ 10020002 (100006)  [if held by 100000 in other years]
#    02050015 (100000) ↔ 10020002 (100009)  [if held by 100000 in other years]
#
#  GROUP F — Outward-facing entity intercompany AR / AP
#    (All zero in 2022; coded here to auto-eliminate when data exists)
#    01030002 (100000) ↔ 09030003 (100001)   100000's "AR from GTC" / 100001's Internal AP
#    01030002 (100000) ↔ 09030002 (100005)   100000's "AR from GTC & Zepto" / 100005's Internal AP
#    01030001 (100009) ↔ 09030003 (100001)   100009's AR from 100001 / 100001's Internal AP
#    01030001 (100009) ↔ 09030002 (100000)   100009's AR from 100000 / 100000's Internal AP
#    NOTE: 100001 ↔ 100005 have NO AR/AP interaction.
#          100001's 09030002 is external (Construction Materials) — excluded here.
#
#  GROUP G — Subsidiary AR (01030001) — interco receivable from 100001
#    Subs (100002/3/4/6/7/8/9) sell exclusively to 100001.  Their 01030001 AR
#    is entirely intercompany and excluded from consolidated AR.
#    The matching AP on 100001's side sits inside 09030001 (mixed with external)
#    and is intentionally left in consolidated per the consolidation sheet
#    ("Add" for all businesses).  This results in a known ~129.6M asymmetry in
#    consolidated AP vs AR at Level 0 which reflects real outstanding interco
#    payables embedded in 100001's local AP.
#
# NOTE: 10020002 in 100001's own books (-1,050,000 in 2022) is an EXTERNAL
#       liability and is intentionally excluded from this dict.
#
INTERCO_BS_ELIMINATIONS: dict[str, set[str]] = {
    # ── GROUP A + D: 100001 asset and liability cross-codes ──────────────────
    "100001": {
        "02050010",   # Loan to HMBR Grocery Shop     ↔ 100006/10020002
        "02050011",   # Loan to Zepto Chemicals        ↔ 100005/10020002
        "02050013",   # Loan to HMBR Steel Scrubber    ↔ 100008/10020002
        "02050015",   # Loan to Gulshan Packaging       ↔ 100009/10020002
        "02050017",   # Loan to GI For Property        ↔ 100000/10020003
        "09060001",   # Bank balance Received in GI    ↔ 100000/01020003
        "10020017",   # Loan From GI Corporation       ↔ 100000/02050005
        "10020005",   # Loan From GCC                  ↔ 100002/02050005
        "10020006",   # Loan From GTT                  ↔ 100003/02050005
        "10020007",   # Loan From GPK                  ↔ 100009/02050001
        "10020008",   # Loan From Gulshan Plastic Co.  ↔ 100004/02050001
        # GROUP F — interco AP to outward-facing entities
        "09030003",   # AP Internal ↔ 100000/01030002 and 100009/01030001
    },
    # ── GROUP A: subsidiary liability side (all record as 10020002) ──────────
    "100005": {
        "10020002",   # Loan From GTC  ↔ 100001/02050011
        "02050002",   # Loan to Thread Tape            ↔ 100003/10020008  (GROUP C)
        "02050005",   # Loan to GTC (Gulshan Plastic)  ↔ 100004/10020003  (GROUP C)
        "10020005",   # Loan From Gulshan Plastic      ↔ 100004/02050002  (GROUP C)
        # GROUP F — interco AP to 100000
        "09030002",   # AP Internal ↔ 100000/01030002
    },
    "100006": {
        "10020002",   # Loan From GTC  ↔ 100001/02050010
        "01030001",   # AR from 100001 ↔ GROUP G sub interco AR
    },
    "100008": {
        "10020002",   # Loan From GTC  ↔ 100001/02050013
        "01030001",   # AR from 100001 ↔ GROUP G sub interco AR
    },
    "100009": {
        "10020002",   # Loan From GTC  ↔ 100001/02050015
        "02050001",   # asset side     ↔ 100001/10020007  (GROUP D)
        "01030001",   # AR from 100001/100000 ↔ GROUP F + GROUP G
    },
    # ── GROUP B + F: 100000 cross-codes ──────────────────────────────────────
    "100000": {
        "01020003",   # GI Bank balance in GTC         ↔ 100001/09060001
        "02050005",   # Loan to GTC                    ↔ 100001/10020017
        "10020003",   # Loan from GTC Against Factory  ↔ 100001/02050017
        "02050010",   # Loan to HMBR Grocery (if held) ↔ 100006/10020002
        "02050015",   # Loan to Gulshan Pkg (if held)  ↔ 100009/10020002
        # GROUP F — interco AR owed by 100001/100005 and interco AP to 100009
        "01030002",   # "AR from GTC & Zepto" ↔ 100001/09030003 and 100005/09030002
        "09030002",   # "AP Internal" ↔ 100009/01030001
    },
    # ── GROUP C + D: counterpart codes in subsidiaries ───────────────────────
    "100002": {
        "02050005",   # asset ↔ 100001/10020005  (GROUP D)
        "01030001",   # AR from 100001 ↔ GROUP G sub interco AR
    },
    "100003": {
        "02050005",   # asset ↔ 100001/10020006          (GROUP D)
        "10020008",   # liability ↔ 100005/02050002       (GROUP C)
        "01030001",   # AR from 100001 ↔ GROUP G sub interco AR
    },
    "100004": {
        "02050001",   # asset ↔ 100001/10020008           (GROUP D)
        "10020003",   # liability ↔ 100005/02050005        (GROUP C)
        "02050002",   # asset ↔ 100005/10020005            (GROUP C)
        "01030001",   # AR from 100001 ↔ GROUP G sub interco AR
    },
    # ── GROUP E: 100000 loan to 100007 ───────────────────────────────────────
    "100007": {
        "10020003",   # liability ↔ 100000/02050001 or 100001/02050001
        "01030001",   # AR from 100001 ↔ GROUP G (zero in 2022; future-proof)
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Year-conditional BS intercompany eliminations
# ─────────────────────────────────────────────────────────────────────────────
#
# Some intercompany pairs used DIFFERENT ac_codes in different eras because the
# chart of accounts evolved over time.  This list supplements
# INTERCO_BS_ELIMINATIONS with entries that only apply within a specific year
# range.
#
# Format:  list of (year_from_incl, year_to_incl, {zid: set[codes]})
#   year_from_incl = None  →  no lower bound
#   year_to_incl   = None  →  no upper bound
#   Both bounds are inclusive.
#
# ── Timeline: 100001 interco AP code change ──────────────────────────────────
#
#   ≤ 2022  100001 recorded AP owed to subsidiaries inside 09030001 (Local AP),
#           mixed with external supplier payables.  Matching subs' 01030001 AR
#           was already eliminated; zeroing 09030001 for 100001 in those years
#           removes the corresponding liability side so the BS stays balanced.
#
#   ≥ 2023  100001 moved interco AP to subs into 09030003 (Internal AP), which
#           is already in INTERCO_BS_ELIMINATIONS unconditionally.
#
# ── Timeline: 100000 AR code ─────────────────────────────────────────────────
#
#   100000 started existing as a separate legal entity in 2024, so no year
#   restriction is needed — 01030002 ("AR from GTC & Zepto") always adjusts
#   against 100001's 09030003 and 100005's 09030002, and those are already in
#   INTERCO_BS_ELIMINATIONS.
#
INTERCO_BS_YEAR_COND: list[tuple] = [
    # (year_from_incl, year_to_incl, {zid: set[codes]})
    (None, 2022, {
        # Pre-2023: 100001's interco AP to subs lived in 09030001
        "100001": {"09030001"},
    }),
    # 2023+: 09030003 in INTERCO_BS_ELIMINATIONS already covers it — no entry needed
]

# ─────────────────────────────────────────────────────────────────────────────
# Intercompany P&L elimination config
# ─────────────────────────────────────────────────────────────────────────────
#
# Business flow (confirmed):
#   100000 / 100001 / 100005  — outward-facing, no intercompany with each other
#   100002/3/4/6/7/8/9        — subsidiaries that trade *only* with 100001:
#       subs sell to / buy from 100001 exclusively
#
# Problem with simple summation:
#   • Subs' 08010001 revenue (selling back to 100001) inflates consolidated
#     gross Revenue.
#   • 100001's 04010020 COGS (buying from subs) inflates consolidated gross
#     COGS by the same amount.
#   • These two overstatements cancel → net income is correct, but gross
#     Revenue and gross COGS are both overstated by the intercompany volume.
#
# Elimination:
#   1. Remove subs' 08010001 from the consolidated Revenue row.
#   2. Add the same positive amount to the consolidated 04010020 row
#      (positive added to negative COGS = reduces absolute COGS by the
#      intercompany purchase cost sitting inside 100001's books).
#   Net income effect: −X (revenue removed) + X (COGS reduced) = 0 ✓
#
_INTERCO_SUBS: set[str] = {
    "100002", "100003", "100004", "100006", "100007", "100008", "100009",
}
_INTERCO_REVENUE_CODE: str = "08010001"   # sub revenue to eliminate
_INTERCO_COGS_CODE:    str = "04010020"   # 100001 COGS to net against

# Ordered metadata columns present in every process_data output
_META_COLS: list[str] = [
    "ac_code", "ac_name", "ac_type",
    "ac_lv1", "ac_lv2", "ac_lv3", "ac_lv4", "ac_lv5",
]

# Intercompany codes highlighted in the debug expander
_DEBUG_CODES: list[str] = [
    # Revenue / COGS
    "08010001", "04010020",
    # AR / AP variants
    "01030001", "01030002", "01030003",
    "09030001", "09030002", "09030003", "09030004",
    # Intercompany loans — asset side
    "02050001", "02050002", "02050003", "02050005",
    "02050006", "02050010", "02050011", "02050012",
    "02050013", "02050015", "02050017",
    # Intercompany loans — liability side
    "10020002", "10020003", "10020004", "10020005",
    "10020006", "10020007", "10020008", "10020017",
    # Special BS pairs
    "01020003", "09060001", "09010003", "01050008",
]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def render_consolidation_notes() -> None:
    """
    Render a collapsed expander with a concise summary of every consolidation
    adjustment applied to the group financial statements.

    Covers:
      • P&L intercompany revenue / COGS elimination
      • BS force-to-zero codes (data gaps)
      • BS cross-code intercompany eliminations (Groups A–E)
    """
    with st.expander("📋 Consolidation Adjustments — Summary Notes", expanded=False):
        st.markdown(
            """
**Outward-facing businesses** (external sales; no intercompany with each other):
100000 · GI Corporation *(entity from 2024)* &nbsp;|&nbsp; 100001 · HMBR T&C Ltd &nbsp;|&nbsp; 100005 · Zepto Chemicals

**Subsidiaries** (sell exclusively to 100001 — no external sales):
100002 · Gulshan Chemical &nbsp;|&nbsp; 100003 · Gulshan Thread Tape &nbsp;|&nbsp;
100004 · Gulshan Plastic &nbsp;|&nbsp; 100006 · HMBR Grocery &nbsp;|&nbsp;
100007 · HMBR Paint Roller &nbsp;|&nbsp; 100008 · HMBR Steel Scrubber &nbsp;|&nbsp;
100009 · Gulshan Packaging

> **100001 has no internal AR.** Its 01030001 is entirely from external customers.
> 100001 ↔ 100005 have no AR/AP interaction.

---

#### P&L — Intercompany Revenue & COGS Elimination

Subsidiaries sell exclusively to 100001 and record this as **08010001 Revenue**.
In a plain sum this inflates consolidated Revenue *and* consolidated COGS by the same
intercompany trading volume — net income is unaffected but gross figures are overstated.

| Adjustment | Code | Effect |
|---|---|---|
| Remove subs' intercompany sales | 08010001 | Consolidated Revenue reduced |
| Add back 100001's matching purchase cost | 04010020 | Consolidated COGS reduced by same amount |
| **Net income impact** | — | **Zero** |

---

#### BS — Force-to-Zero (confirmed data gaps)

One side of the intercompany pair is absent from the GL.
Management confirmed these should be treated as zero.

| Business | Code | Description |
|---|---|---|
| 100001 | 02050001 | Loan to HMBR Online Shop (100007 has no GL data) |
| 100001 | 02050012 | Loan to HMBR Paint Roller (no counterpart data) |
| 100001 | 02050006 | Karigor project advance (no counterpart data) |
| 100000 | 02050012 | Same — GI side |
| 100000 | 02050006 | Same — GI side |
| 100005 | 02050003 | Loan to Paint Roller (no counterpart data) |

---

#### BS — AR/AP Code Name Differences (same code, different meaning per entity)

The following codes carry **different names and meanings** across entities:

| Code | 100000 | 100001 | 100005 |
|---|---|---|---|
| **09030002** | "AP Internal" — interco AP *(eliminated)* | "Construction Materials Suppliers MB" — **external** *(kept)* | "AP Internal" — interco AP *(eliminated)* |
| **09030003** | "AP International" — **external** *(kept)* | "AP Internal" — interco AP *(eliminated)* | "AP International" — **external** *(kept)* |
| **01030002** | "AR from GTC & Zepto" — interco AR *(eliminated)* | "Recognized Agent" — **external** *(kept)* | "Recognized Agent" — **external** *(kept)* |
| **09030004** | not used | "AP International" — **external** *(kept; merged with others' 09030003)* | not used |

---

#### BS — Cross-Code Intercompany Eliminations

Where asset and liability use **different** ac_codes they do not cancel in a plain sum.
Both sides are zeroed before consolidation.

**Group A — 100001 loans to subsidiaries (02050xxx ↔ sub 10020002)**

| 100001 code | Sub ZID | Description |
|---|---|---|
| 02050010 | 100006 | Loan to HMBR Grocery |
| 02050011 | 100005 | Loan to Zepto Chemicals |
| 02050013 | 100008 | Loan to HMBR Steel Scrubber |
| 02050015 | 100009 | Loan to Gulshan Packaging |

**Group B — 100001 ↔ 100000 cross-loans**

| Code | ZID | Counterpart |
|---|---|---|
| 02050017 | 100001 | ↔ 10020003 · 100000 (Loan to GI for property) |
| 09060001 | 100001 | ↔ 01020003 · 100000 (Bank balance received in GI) |
| 10020017 | 100001 | ↔ 02050005 · 100000 (Loan from GI Corporation) |

**Group C — 100005 intercompany loans**

| Code | ZID | Counterpart |
|---|---|---|
| 02050002 | 100005 | ↔ 10020008 · 100003 (Loan to Thread Tape) |
| 02050005 | 100005 | ↔ 10020003 · 100004 (Loan to Gulshan Plastic) |
| 10020005 | 100005 | ↔ 02050002 · 100004 (Loan from Gulshan Plastic) |

**Group D — 100001 liabilities to subsidiaries**

| 100001 code | Sub ZID | Sub asset code |
|---|---|---|
| 10020005 | 100002 | 02050005 |
| 10020006 | 100003 | 02050005 |
| 10020007 | 100009 | 02050001 |
| 10020008 | 100004 | 02050001 |

**Group E — 100000 loan to 100007**

| Code | ZID | Counterpart |
|---|---|---|
| 02050001 | 100000 | ↔ 10020003 · 100007 (Loan to HMBR Paint Roller) |

**Group F — Outward-facing entity intercompany AR / AP** *(zero in 2022; coded for future years)*

| AR side | ZID | ↔ | AP side | ZID |
|---|---|---|---|---|
| 01030002 "AR from GTC & Zepto" | 100000 | ↔ | 09030003 "AP Internal" | 100001 |
| 01030002 "AR from GTC & Zepto" | 100000 | ↔ | 09030002 "AP Internal" | 100005 |
| 01030001 AR | 100009 | ↔ | 09030003 "AP Internal" | 100001 |
| 01030001 AR | 100009 | ↔ | 09030002 "AP Internal" | 100000 |

**Group G — Subsidiary AR (01030001) — interco receivable from 100001**

Subs (100002/3/4/6/7/8/9) sell exclusively to 100001. Their **01030001 AR is entirely
intercompany** and is excluded from consolidated AR. 100001's matching AP sits inside
**09030001** alongside external supplier payables and is kept in consolidated per the
consolidation sheet ("Add" for all businesses). This results in a known asymmetry in
consolidated AP vs AR, reflecting real outstanding intercompany payables embedded in
100001's local AP.

---

---

#### Timeline Exception — 100001 Intercompany AP Code Change

| Period | Code used by 100001 for interco AP to subs | Treatment |
|---|---|---|
| **≤ 2022** | **09030001** (Local AP) — interco AP was mixed in with external supplier payables | 09030001 zeroed for 100001 in pre-2023 year columns |
| **≥ 2023** | **09030003** (Internal AP) — dedicated code introduced | 09030003 zeroed (already in Group F above) |

> 100000 began operating as a separate legal entity in **2024** — no timeline
> restriction applies to its codes (01030002, 09030002).

---

> **External codes kept in full — not eliminated:**
> 10020002 in 100001 (−1,050,000) · 09030002 in 100001 (Construction Materials) ·
> 09030001 for all businesses in **2023+** · 09030003/09030004 in 100000 and 100005
"""
        )


def build_consolidated_raw(
    all_pl: dict[str, pd.DataFrame],
    all_bs: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine all businesses' raw GL data into consolidated IS and BS DataFrames.

    Parameters
    ----------
    all_pl : {zid_str: pl_raw DataFrame}
    all_bs : {zid_str: bs_raw DataFrame}

    Returns
    -------
    (cons_pl, cons_bs) — identical column structure to individual business
    raw data, ready to flow through the existing Level 0 / 1 / 2 / S builders.
    """
    cons_pl = _stack_and_sum(all_pl)
    cons_bs = _stack_and_sum(
        all_bs,
        extra_zero=INTERCO_BS_ELIMINATIONS,
        year_cond_zero=INTERCO_BS_YEAR_COND,
    )
    cons_pl = _apply_interco_eliminations(all_pl, cons_pl)
    return cons_pl, cons_bs


def render_consolidation_debug(
    all_pl: dict[str, pd.DataFrame],
    all_bs: dict[str, pd.DataFrame],
) -> None:
    """
    Render a Streamlit expander with per-business contribution tables for
    key intercompany codes.  Helps verify that intercompany pairs net to
    zero and flags any residual phantom balances.
    """
    with st.expander(
        "🔍 Consolidation Debug — Intercompany Code Breakdown",
        expanded=False,
    ):
        st.caption(
            "Each row shows the **sum of all period columns** for that ac_code "
            "in each business.  Same-code intercompany pairs (e.g. a loan asset "
            "and its matching liability) should sum to **~0** in the 'Total' column. "
            "Cross-code pairs (e.g. AR 01030001 ↔ AP 09030003) will appear in "
            "separate rows but still net to zero at the total assets / liabilities level."
        )

        for stmt_label, data_dict in [
            ("Income Statement (P&L)", all_pl),
            ("Balance Sheet (BS)",     all_bs),
        ]:
            rows: list[pd.DataFrame] = []
            for zid, df in data_dict.items():
                if df.empty:
                    continue
                ycols = list(df.select_dtypes("number").columns)
                mask  = df["ac_code"].astype(str).isin(_DEBUG_CODES)
                sub   = df.loc[mask, ["ac_code", "ac_name"] + ycols].copy()
                if sub.empty:
                    continue
                sub["_total"] = sub[ycols].sum(axis=1).round(1)
                sub["_zid"]   = str(zid)
                rows.append(sub[["_zid", "ac_code", "ac_name", "_total"]])

            if not rows:
                continue

            combined = pd.concat(rows, ignore_index=True)
            pivot = (
                combined
                .pivot_table(
                    index=["ac_code", "ac_name"],
                    columns="_zid",
                    values="_total",
                    aggfunc="sum",
                )
                .fillna(0.0)
            )
            pivot.columns.name = None
            pivot["Total"] = pivot.sum(axis=1).round(1)

            # Sort: highlight non-zero totals first
            pivot = pivot.sort_values("Total", key=abs, ascending=False)
            pivot = pivot.reset_index()

            st.markdown(f"**{stmt_label}**")
            st.dataframe(pivot, use_container_width=True)

        # ── Intercompany revenue / COGS elimination summary ──────────────────
        st.markdown("---")
        st.markdown("**P&L: Revenue & COGS elimination — 08010001 / 04010020**")
        st.caption(
            "Subsidiaries (100002/3/4/6/7/8/9) trade exclusively with 100001.  "
            "Their 08010001 revenue is intercompany and is **eliminated** from "
            "consolidated Revenue.  The same amount is added back to the "
            "04010020 COGS row to remove the matching intercompany cost from "
            "100001's books.  **Net income is unchanged.**  "
            "The 'Eliminated' column shows the total removed from Revenue "
            "(= total added back to COGS)."
        )
        _ic_rows: list[dict] = []
        _elim_total: float = 0.0
        for zid, df in all_pl.items():
            if df.empty:
                continue
            ycols = list(df.select_dtypes("number").columns)
            for code, label in [("08010001", "Revenue"), ("04010020", "COGS")]:
                mask = df["ac_code"].astype(str) == code
                sub  = df.loc[mask, ycols]
                if sub.empty:
                    continue
                total = round(float(sub.sum(axis=1).sum()), 1)
                is_sub = str(zid) in _INTERCO_SUBS
                _ic_rows.append({
                    "ZID":       str(zid),
                    "Code":      f"{code} {label}",
                    "Total (raw GL)": total,
                    "Role": (
                        "✓ outward-facing"
                        if str(zid) in {"100000", "100001", "100005"}
                        else "⚠ eliminated from Revenue"
                        if (is_sub and code == "08010001")
                        else "sub — COGS included"
                        if (is_sub and code == "04010020")
                        else ""
                    ),
                })
                if is_sub and code == "08010001":
                    _elim_total += total
        if _ic_rows:
            st.dataframe(pd.DataFrame(_ic_rows), use_container_width=True)
            st.info(
                f"**Total eliminated from consolidated Revenue (08010001):** "
                f"{_elim_total:,.1f}  |  "
                f"Same amount added back to consolidated COGS (04010020) — "
                f"net income effect: zero."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Public API (continued)
# ─────────────────────────────────────────────────────────────────────────────

def render_year_breakdown_report(
    all_pl: dict[str, pd.DataFrame],
    all_bs: dict[str, pd.DataFrame],
    key_suffix: str = "",
) -> None:
    """
    Render an interactive year-breakdown debug report inside a Streamlit
    expander.

    For every Level-0 ac_code that has at least one non-zero business, the
    report shows:
      • The original GL amount for each business (before any force-zero).
      • A **Consolidated** total (sum of *Added* amounts only).
      • Three status columns — **Added**, **Force-zero**, **No data** —
        listing which businesses fall into each category.
      • A human-readable **Summary** column such as
        "100000=500,000 + 100001=300,000 = 800,000 | zeroed: 100005 (50,000)"

    Parameters
    ----------
    all_pl       : {zid_str: pl_raw DataFrame}
    all_bs       : {zid_str: bs_raw DataFrame}
    key_suffix   : Unique suffix for Streamlit widget keys (prevents
                   duplicate-widget errors when the function is called from
                   both yearly and monthly branches).
    """
    with st.expander(
        "📊 Year-Breakdown Contribution Report",
        expanded=False,
    ):
        st.caption(
            "Select a year to see each Level-0 ac_code broken down by business.  "
            "**Added** = included in the consolidated sum.  "
            "**Force-zero** = GL data existed but was zeroed out (confirmed data gap).  "
            "**No data** = business had zero / no GL entry for this code."
        )

        # ── Collect unique year values from all raw DataFrames ────────────────
        all_years_set: set = set()
        for data_dict in (all_pl, all_bs):
            for df in data_dict.values():
                if df.empty:
                    continue
                for col in df.select_dtypes("number").columns:
                    all_years_set.add(_col_year(col))

        if not all_years_set:
            st.info("No period data available.")
            return

        all_years = sorted(all_years_set, reverse=True)
        sel_year = st.selectbox(
            "Select Year",
            options=all_years,
            format_func=str,
            key=f"_consol_yr_bkdn_{key_suffix}",
        )

        # ZID order for columns
        zids = [z for z in CONSOLIDATION_ALL_ZIDS if z in all_pl or z in all_bs]

        # Intercompany elimination amount for the selected year
        elim_for_year = _compute_elim_for_year(all_pl, sel_year)

        for stmt_label, data_dict in [
            ("Income Statement (P&L)", all_pl),
            ("Balance Sheet (BS)", all_bs),
        ]:
            is_pl = data_dict is all_pl
            rows = _build_breakdown_rows(
                data_dict, zids, sel_year,
                interco_subs=_INTERCO_SUBS              if is_pl else None,
                revenue_elim_code=_INTERCO_REVENUE_CODE if is_pl else None,
                cogs_adj_code=_INTERCO_COGS_CODE        if is_pl else None,
                cogs_adj_amount=elim_for_year           if is_pl else 0.0,
                bs_eliminations=None if is_pl else INTERCO_BS_ELIMINATIONS,
                bs_year_cond=None if is_pl else INTERCO_BS_YEAR_COND,
            )
            if not rows:
                st.markdown(f"**{stmt_label}** — no data for {sel_year}.")
                continue

            result_df = pd.DataFrame(rows)

            # Format numeric ZID columns + Consolidated
            num_cols_fmt = {
                z: "{:,.1f}"
                for z in zids
                if z in result_df.columns
            }
            num_cols_fmt["Consolidated"] = "{:,.1f}"

            st.markdown(f"**{stmt_label}** — {sel_year}")
            if is_pl and elim_for_year != 0.0:
                st.caption(
                    f"**Intercompany elimination active for {sel_year}:**  "
                    f"Subs' 08010001 Revenue is marked **Eliminated** and excluded "
                    f"from the Consolidated total.  "
                    f"The same amount ({elim_for_year:,.1f}) is shown as "
                    f"**COGS adj** on the 04010020 row — it reduces 100001's "
                    f"intercompany purchase cost so net income stays unchanged."
                )
            if not is_pl:
                _elim_rows = [r for r in rows if r.get("Eliminated", "—") != "—"]
                _yr_rule = (
                    "≤ 2022: 100001's 09030001 (Local AP) eliminated — "
                    "interco AP to subs was recorded there before 2023."
                    if _year_in_range(sel_year, None, 2022)
                    else "≥ 2023: 100001's 09030003 (Internal AP) eliminated — "
                    "interco AP to subs moved to this code from 2023 onwards."
                )
                if _elim_rows:
                    st.caption(
                        f"**Cross-code BS eliminations active for {sel_year}** "
                        f"({len(_elim_rows)} code(s) marked Eliminated).  "
                        f"Timeline rule — {_yr_rule}  "
                        f"Original GL amounts shown in ZID columns for transparency; "
                        f"Consolidated total excludes all eliminated codes."
                    )
            st.dataframe(
                result_df.style.format(num_cols_fmt, na_rep="0.0"),
                use_container_width=True,
            )

        # ── Outward-facing business focused intercompany check ────────────────
        st.markdown("---")
        st.markdown(
            "**Outward-Facing Business Focus — 100000 · 100001 · 100005**"
        )
        st.caption(
            "Shows only the codes listed on each entity's sheet in the "
            "consolidation workbook.  A blank cell means the code is not "
            "relevant for that entity.  Status reflects the same elimination "
            "rules as the full tables above."
        )
        focus_rows = _build_outward_focus_rows(
            all_pl, all_bs, sel_year,
            bs_eliminations=INTERCO_BS_ELIMINATIONS,
            bs_year_cond=INTERCO_BS_YEAR_COND,
        )
        if focus_rows:
            focus_df = pd.DataFrame(focus_rows)
            focus_num_fmt = {z: "{:,.1f}" for z in _FOCUS_ZIDS if z in focus_df.columns}
            focus_num_fmt["Consolidated"] = "{:,.1f}"
            st.dataframe(
                focus_df.style.format(focus_num_fmt, na_rep="—"),
                use_container_width=True,
            )
        else:
            st.info(f"No focus-code data available for {sel_year}.")


# ─────────────────────────────────────────────────────────────────────────────
# Outward-facing business focus codes (from consolidation sheet)
# ─────────────────────────────────────────────────────────────────────────────
# Ordered lists of ac_codes to highlight in the focused intercompany check
# table.  Only 100000, 100001, 100005 are included — these are the codes that
# appear on each entity's sheet in the consolidation workbook.
#
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

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_interco_eliminations(
    all_pl: dict[str, pd.DataFrame],
    cons_pl: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove intercompany revenue / COGS overstatement from the consolidated P&L.

    Subsidiaries (100002/3/4/6/7/8/9) sell exclusively to 100001.  Their
    08010001 revenue is intercompany; the matching cost sits in 100001's
    04010020.  After plain summation both rows are overstated by the same
    intercompany volume.

    Fix (net-income-neutral):
      • Subtract subs' 08010001 from the consolidated 08010001 Revenue row.
      • Add the same positive amount back to the consolidated 04010020 COGS
        row (positive added to a negative cost = reduces absolute COGS).

    The all_pl dict carries *original* per-business GL data (force-zero is
    applied inside _stack_and_sum, but it only touches 100000/1/5 codes that
    are unrelated to this elimination).
    """
    if cons_pl.empty:
        return cons_pl

    ycols: list = list(cons_pl.select_dtypes("number").columns)
    if not ycols:
        return cons_pl

    # ── Step 1: accumulate subs' 08010001 per year column ────────────────────
    elim = pd.Series(0.0, index=ycols, dtype=float)
    for zid, df in all_pl.items():
        if str(zid) not in _INTERCO_SUBS or df.empty:
            continue
        mask = df["ac_code"].astype(str) == _INTERCO_REVENUE_CODE
        if not mask.any():
            continue
        sub_df = df.loc[mask]
        for y in ycols:
            if y in sub_df.columns:
                elim[y] += float(sub_df[y].sum())

    if (elim == 0.0).all():
        return cons_pl   # nothing to eliminate (no sub revenue found)

    cons_pl = cons_pl.copy()

    # ── Step 2: subtract from consolidated Revenue row ───────────────────────
    rev_mask = cons_pl["ac_code"].astype(str) == _INTERCO_REVENUE_CODE
    if rev_mask.any():
        cons_pl.loc[rev_mask, ycols] = (
            cons_pl.loc[rev_mask, ycols].values - elim.values
        )

    # ── Step 3: add back to consolidated COGS row (reduces absolute COGS) ────
    cogs_mask = cons_pl["ac_code"].astype(str) == _INTERCO_COGS_CODE
    if cogs_mask.any():
        cons_pl.loc[cogs_mask, ycols] = (
            cons_pl.loc[cogs_mask, ycols].values + elim.values
        )

    return cons_pl


def _col_year(col) -> int | str:
    """
    Extract the year component from a period column label.

    Handles:
    • Integer / string integer  → e.g. 2024  or "2024"
    • Tuple                     → e.g. (2024, 3)
    • Stringified tuple         → e.g. "(2024, 3)"
    """
    if isinstance(col, tuple) and len(col) >= 1:
        try:
            return int(col[0])
        except (TypeError, ValueError):
            return col[0]
    if isinstance(col, str):
        col_s = col.strip()
        if col_s.startswith("("):
            try:
                parsed = ast.literal_eval(col_s)
                if isinstance(parsed, (tuple, list)) and len(parsed) >= 1:
                    return int(parsed[0])
            except (ValueError, SyntaxError):
                pass
        nums = re.findall(r"\d{4}", col_s)
        if nums:
            return int(nums[0])
        try:
            return int(col_s)
        except ValueError:
            return col
    try:
        return int(col)
    except (TypeError, ValueError):
        return col


def _compute_elim_for_year(
    all_pl: dict[str, pd.DataFrame],
    sel_year,
) -> float:
    """
    Return the total intercompany revenue elimination amount for one year.

    Sums subs' (100002/3/4/6/7/8/9) 08010001 values for all period columns
    that belong to sel_year (handles both yearly integer cols and monthly
    tuple/string cols).
    """
    total = 0.0
    for zid, df in all_pl.items():
        if str(zid) not in _INTERCO_SUBS or df.empty:
            continue
        mask = df["ac_code"].astype(str) == _INTERCO_REVENUE_CODE
        if not mask.any():
            continue
        sub_df = df.loc[mask]
        ycols = [
            c for c in sub_df.select_dtypes("number").columns
            if _col_year(c) == sel_year
        ]
        if ycols:
            total += float(sub_df[ycols].sum().sum())
    return total


def _build_breakdown_rows(
    data_dict: dict[str, pd.DataFrame],
    zids: list[str],
    sel_year,
    *,
    interco_subs: set[str] | None = None,
    revenue_elim_code: str | None = None,
    cogs_adj_code: str | None = None,
    cogs_adj_amount: float = 0.0,
    bs_eliminations: dict[str, set[str]] | None = None,
    bs_year_cond: list[tuple] | None = None,
) -> list[dict]:
    """
    Build the list of row-dicts for the year-breakdown report table.

    Returns one dict per ac_code that has at least one non-zero, non-nodata
    business entry.

    Optional intercompany elimination parameters (IS data only)
    ──────────────────────────────────────────────────────────
    interco_subs       : set of ZID strings whose revenue_elim_code entries
                         are intercompany and should be marked "Eliminated".
    revenue_elim_code  : ac_code where subs are eliminated (e.g. "08010001").
    cogs_adj_code      : ac_code that receives the add-back adjustment.
    cogs_adj_amount    : positive float — the total intercompany amount added
                         back to the COGS row.

    Optional BS elimination parameters
    ───────────────────────────────────
    bs_eliminations    : {zid: set_of_codes} — cross-code BS pairs zeroed for
                         ALL years.
    bs_year_cond       : list of (yr_from, yr_to, {zid: set_of_codes}) — BS
                         pairs zeroed only within the specified year range.
                         Applied conditionally based on sel_year.
    """
    # ── Step 1: collect metadata (ac_code → ac_name) ─────────────────────────
    code_meta: dict[str, str] = {}   # ac_code → ac_name
    for df in data_dict.values():
        if df.empty:
            continue
        for _, r in (
            df[["ac_code", "ac_name"]]
            .drop_duplicates("ac_code")
            .iterrows()
        ):
            c = str(r["ac_code"])
            if c and c not in code_meta:
                code_meta[c] = str(r["ac_name"])

    if not code_meta:
        return []

    # ── Step 2: per-ZID amount for the selected year ──────────────────────────
    # Sum all monthly columns that belong to sel_year (for yearly data there
    # is exactly one matching column; for monthly data there are up to 12).
    zid_amounts: dict[str, dict[str, float]] = {}   # zid → {ac_code → float}
    for zid, df in data_dict.items():
        if df.empty:
            zid_amounts[str(zid)] = {}
            continue
        ycols = [
            c for c in df.select_dtypes("number").columns
            if _col_year(c) == sel_year
        ]
        if not ycols:
            zid_amounts[str(zid)] = {}
            continue
        grp = (
            df[["ac_code"] + ycols]
            .groupby("ac_code")[ycols]
            .sum()
        )
        zid_amounts[str(zid)] = grp.sum(axis=1).to_dict()

    # ── Step 3: build one row per ac_code ────────────────────────────────────
    rows: list[dict] = []
    for ac_code in sorted(code_meta):
        ac_name  = code_meta[ac_code]
        zid_vals:    dict[str, float] = {}
        added:       list[str] = []
        eliminated:  list[str] = []   # intercompany — excluded from Consolidated
        force_zeroed: list[str] = []
        no_data:     list[str] = []
        consolidated = 0.0

        is_rev_code  = (ac_code == revenue_elim_code)
        is_cogs_code = (ac_code == cogs_adj_code)

        for zid in zids:
            raw_amt = zid_amounts.get(zid, {}).get(ac_code, 0.0)
            fz_set  = INTERCO_FORCE_ZERO.get(zid, set())
            is_fz   = ac_code in fz_set
            is_elim = (
                is_rev_code
                and interco_subs is not None
                and str(zid) in interco_subs
            )

            zid_vals[zid] = raw_amt   # always show original GL value

            # Check cross-code BS elimination (all-years)
            is_bs_elim = (
                bs_eliminations is not None
                and ac_code in bs_eliminations.get(str(zid), set())
            )

            # Check year-conditional BS elimination for this sel_year
            is_bs_year_elim = False
            if bs_year_cond is not None:
                for yr_from, yr_to, zcond_dict in bs_year_cond:
                    if (_year_in_range(sel_year, yr_from, yr_to)
                            and ac_code in zcond_dict.get(str(zid), set())):
                        is_bs_year_elim = True
                        break

            if is_fz:
                force_zeroed.append(zid)
            elif is_elim or is_bs_elim or is_bs_year_elim:
                # Intercompany — shown in per-ZID column but excluded from total
                if raw_amt != 0.0:
                    eliminated.append(zid)
                else:
                    no_data.append(zid)
            elif raw_amt != 0.0:
                added.append(zid)
                consolidated += raw_amt
            else:
                no_data.append(zid)

        # For the COGS code, the consolidated total is reduced by the
        # intercompany add-back (positive elim_amount added to negative COGS)
        cogs_adj_applied = 0.0
        if is_cogs_code and cogs_adj_amount != 0.0:
            cogs_adj_applied = cogs_adj_amount
            consolidated    += cogs_adj_amount   # makes COGS less negative

        # Skip rows where nothing happened
        if not added and not force_zeroed and not eliminated and cogs_adj_applied == 0.0:
            continue

        row: dict = {"ac_code": ac_code, "ac_name": ac_name}
        for zid in zids:
            row[zid] = zid_vals.get(zid, 0.0)
        row["Consolidated"] = round(consolidated, 1)

        # Status text columns
        row["Added"]      = ", ".join(added)       if added        else "—"
        row["Eliminated"] = ", ".join(eliminated)  if eliminated   else "—"
        row["Force-zero"] = ", ".join(force_zeroed) if force_zeroed else "—"
        row["No data"]    = ", ".join(no_data)     if no_data      else "—"

        # Human-readable summary
        if added:
            parts   = [f"{z}={zid_vals[z]:,.0f}" for z in added]
            summary = " + ".join(parts) + f"  =  {consolidated - cogs_adj_applied:,.0f}"
        else:
            summary = "0"

        if eliminated:
            elim_parts = [f"{z}({zid_vals[z]:,.0f})" for z in eliminated]
            summary += "  |  eliminated (interco): " + ", ".join(elim_parts)

        if cogs_adj_applied != 0.0:
            summary += (
                f"  |  interco add-back: +{cogs_adj_applied:,.0f}"
                f"  →  adj consolidated = {consolidated:,.0f}"
            )

        if force_zeroed:
            fz_parts = [f"{z}({zid_vals[z]:,.0f})" for z in force_zeroed]
            summary += "  |  zeroed: " + ", ".join(fz_parts)

        if no_data:
            summary += "  |  no data: " + "/".join(no_data)

        row["Summary"] = summary
        rows.append(row)

    return rows


def _build_outward_focus_rows(
    all_pl: dict[str, pd.DataFrame],
    all_bs: dict[str, pd.DataFrame],
    sel_year,
    bs_eliminations: dict[str, set[str]],
    bs_year_cond: list[tuple],
) -> list[dict]:
    """
    Build rows for the outward-facing business focused intercompany check table.

    Shows only the codes listed in _OUTWARD_FOCUS_ZID_CODES for 100000,
    100001, and 100005 in the order specified per entity.  Each row covers one
    ac_code; columns are those three ZIDs.  A cell is blank (None) when the
    code is not in that entity's focus list.
    """
    # ── Ordered deduplicated code list across all three ZIDs ─────────────────
    seen: set[str] = set()
    ordered_codes: list[str] = []
    for zid in _FOCUS_ZIDS:
        for code in _OUTWARD_FOCUS_ZID_CODES.get(zid, []):
            if code not in seen:
                seen.add(code)
                ordered_codes.append(code)

    # ── Collect ac_names and per-year amounts ─────────────────────────────────
    code_names: dict[str, str] = {}
    # {zid: {ac_code: float}}  — raw GL value for sel_year
    zid_amounts: dict[str, dict[str, float]] = {z: {} for z in _FOCUS_ZIDS}

    for zid in _FOCUS_ZIDS:
        for is_pl, data_dict in ((True, all_pl), (False, all_bs)):
            df = data_dict.get(zid, pd.DataFrame())
            if df.empty:
                continue
            ycols = [
                c for c in df.select_dtypes("number").columns
                if _col_year(c) == sel_year
            ]
            if not ycols:
                continue
            sub = df[["ac_code", "ac_name"] + ycols].copy()
            sub["_sum"] = sub[ycols].sum(axis=1)
            for _, r in sub.iterrows():
                code = str(r["ac_code"])
                if code and code not in code_names:
                    code_names[code] = str(r.get("ac_name", code))
                # Only store from the correct data source (P&L vs BS)
                if (code in _PL_CODES) == is_pl:
                    existing = zid_amounts[zid].get(code, 0.0)
                    zid_amounts[zid][code] = existing + float(r["_sum"])

    # ── Build one row per ac_code ─────────────────────────────────────────────
    rows: list[dict] = []
    for ac_code in ordered_codes:
        ac_name = code_names.get(ac_code, ac_code)
        row: dict = {"ac_code": ac_code, "ac_name": ac_name}

        is_pl_code = ac_code in _PL_CODES
        added:       list[str] = []
        eliminated:  list[str] = []
        force_zeroed: list[str] = []
        no_data_foc: list[str] = []
        not_in_focus: list[str] = []
        consolidated = 0.0

        for zid in _FOCUS_ZIDS:
            focus_set = set(_OUTWARD_FOCUS_ZID_CODES.get(zid, []))
            if ac_code not in focus_set:
                row[zid] = None          # blank — not relevant for this entity
                not_in_focus.append(zid)
                continue

            raw_amt = zid_amounts[zid].get(ac_code, 0.0)
            row[zid] = raw_amt

            # Status logic — mirrors _build_breakdown_rows
            fz_set = INTERCO_FORCE_ZERO.get(zid, set())
            is_fz = ac_code in fz_set

            is_bs_elim = (
                not is_pl_code
                and ac_code in bs_eliminations.get(zid, set())
            )
            is_bs_year_elim = False
            if not is_pl_code and bs_year_cond:
                for yr_from, yr_to, zcond_dict in bs_year_cond:
                    if (_year_in_range(sel_year, yr_from, yr_to)
                            and ac_code in zcond_dict.get(zid, set())):
                        is_bs_year_elim = True
                        break

            if is_fz:
                force_zeroed.append(zid)
            elif is_bs_elim or is_bs_year_elim:
                eliminated.append(zid)
            elif raw_amt != 0.0:
                added.append(zid)
                consolidated += raw_amt
            else:
                no_data_foc.append(zid)

        row["Consolidated"] = round(consolidated, 1)
        row["Added"]        = ", ".join(added)       if added        else "—"
        row["Eliminated"]   = ", ".join(eliminated)  if eliminated   else "—"
        row["Force-zero"]   = ", ".join(force_zeroed) if force_zeroed else "—"
        row["No data"]      = ", ".join(no_data_foc) if no_data_foc else "—"

        rows.append(row)

    return rows


def _year_cols(df: pd.DataFrame) -> list:
    """Return numeric (period / year) column labels from a raw data frame."""
    return list(df.select_dtypes("number").columns)


def _year_in_range(year, yr_from: int | None, yr_to: int | None) -> bool:
    """Return True if year (int or str) falls within [yr_from, yr_to] (inclusive, None = unbounded)."""
    try:
        y = int(year)
    except (TypeError, ValueError):
        return False
    if yr_from is not None and y < yr_from:
        return False
    if yr_to is not None and y > yr_to:
        return False
    return True


def _stack_and_sum(
    data_by_zid: dict[str, pd.DataFrame],
    extra_zero: dict[str, set[str]] | None = None,
    year_cond_zero: list[tuple] | None = None,
) -> pd.DataFrame:
    """
    Stack all businesses' DataFrames, sum numeric columns by ac_code,
    and rebuild the metadata block using the first available value per code.

    Parameters
    ----------
    data_by_zid    : {zid_str: raw DataFrame}
    extra_zero     : optional {zid_str: set_of_ac_codes} — codes to zero for
                     ALL year columns (INTERCO_BS_ELIMINATIONS).
    year_cond_zero : optional list of (year_from_incl, year_to_incl,
                     {zid_str: set_of_ac_codes}) — codes to zero only within
                     the specified year range (INTERCO_BS_YEAR_COND).
                     year_from/to = None means unbounded.
    """
    if not data_by_zid:
        return pd.DataFrame()

    # Canonical year columns — derived from the first non-empty df
    canonical_years: list = []
    for df in data_by_zid.values():
        if not df.empty:
            canonical_years = _year_cols(df)
            break

    if not canonical_years:
        return pd.DataFrame()

    num_frames:  list[pd.DataFrame] = []
    meta_frames: list[pd.DataFrame] = []

    for zid, df in data_by_zid.items():
        if df.empty:
            continue

        df = df.copy()
        ycols = _year_cols(df)

        # ── Force-to-zero (all years) ─────────────────────────────────────────
        zero_codes = INTERCO_FORCE_ZERO.get(str(zid), set())
        if extra_zero:
            zero_codes = zero_codes | extra_zero.get(str(zid), set())
        if zero_codes:
            zmask = df["ac_code"].astype(str).isin(zero_codes)
            df.loc[zmask, ycols] = 0.0

        # ── Year-conditional zero ─────────────────────────────────────────────
        if year_cond_zero:
            for yr_from, yr_to, zcond_dict in year_cond_zero:
                zcond_codes = zcond_dict.get(str(zid), set())
                if not zcond_codes:
                    continue
                # Select only the year columns that fall within the range
                in_range_cols = [
                    c for c in ycols
                    if _year_in_range(_col_year(c), yr_from, yr_to)
                ]
                if in_range_cols:
                    zmask = df["ac_code"].astype(str).isin(zcond_codes)
                    df.loc[zmask, in_range_cols] = 0.0

        # ── Numeric slice: ac_code + year columns only ────────────────────────
        num_df = df[["ac_code"] + ycols].copy()
        for yr in canonical_years:          # fill any missing year columns
            if yr not in num_df.columns:
                num_df[yr] = 0.0
        num_df = num_df[["ac_code"] + canonical_years]
        num_frames.append(num_df)

        # ── Metadata slice ────────────────────────────────────────────────────
        meta_cols_present = [c for c in _META_COLS if c in df.columns]
        meta_frames.append(df[meta_cols_present].copy())

    if not num_frames:
        return pd.DataFrame()

    # ── Sum numeric by ac_code ────────────────────────────────────────────────
    combined_num = pd.concat(num_frames, ignore_index=True)
    summed = (
        combined_num
        .groupby("ac_code", as_index=False)[canonical_years]
        .sum()
    )

    # ── Build consolidated metadata ───────────────────────────────────────────
    # For each ac_code, take the first non-null value from any business.
    # groupby.first() skips NaN, so the first real value wins.
    all_meta = pd.concat(meta_frames, ignore_index=True)
    all_meta  = all_meta.replace("", pd.NA)
    cons_meta = all_meta.groupby("ac_code", as_index=False).first()

    # ── Merge metadata + summed amounts ──────────────────────────────────────
    result = cons_meta.merge(summed, on="ac_code", how="outer")
    result[canonical_years] = result[canonical_years].fillna(0.0)
    result["ac_code"] = result["ac_code"].fillna("").astype(str)
    result["ac_name"] = result["ac_name"].fillna(result["ac_code"])
    result["ac_type"] = result["ac_type"].fillna("")

    # ── Restore column order: metadata first, then year columns ──────────────
    meta_out = [c for c in _META_COLS if c in result.columns]
    year_out = [c for c in canonical_years if c in result.columns]
    result   = result[meta_out + year_out]

    return result.reset_index(drop=True)
