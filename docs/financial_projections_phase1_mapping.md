# Phase 1, Step 1 — Cash Roll-Forward Account Header Mapping

Status: **draft, needs review — do not start Step 2 (real data sourcing) until the
flagged rows below are resolved.** Source doc: `financial_projections_design.md`.

Source: `CashAnalysis2023` (Google Sheet, id `1m7L4Or9nu2_CwKa0Th7QAEN80chMvEitic-i4hGyoYo`),
the three authoritative tabs confirmed in the design doc — `GTC Admin + Motiur` (100001),
`GI Test` (100000), `Zepto-M` (100005). Row labels pulled directly from the live sheet
(exported to xlsx and parsed), not retyped from memory. The sheet's parallel
`Trading-M`/`GI`/`Trading Test`/`Zepto test` tabs exist too but are explicitly NOT
authoritative per the design doc — ignored here.

Account codes below were looked up live against this app's own `glmst` (account master)
table, scoped per ZID (the same code number can mean a different account on a different
ZID's chart of accounts, confirmed while doing this lookup — e.g. `10020002` = "Loan From
Fix-iT" on 100001 but "Loan From GTC" on 100000).

Confidence key: **✅ confident** (direct account-master match) · **⚠️ needs your
confirmation** (plausible match, but more than one candidate or no exact match found) ·
**❓ open question** (couldn't resolve from the account master at all).

---

## GTC Admin + Motiur (ZID 100001)

Real formula for the row 27 cash balance (confirmed from the live sheet, matches the
design doc's citation): `HMBR Cash Balance = HMBR Collection + HMBR Additional Collection
+ HMBR Balance Forward − HMBR Payorder (Tax & VAT) − HMBR International Payment − HMBR
Total Payment` (`=B16+B17+B26-B20-B24-B25`), where `HMBR Total Payment` (row 25) is itself
`=B28` and `HMBR Total Payments` (row 28) is `=SUM(B29:B47)` — the full local-payments list.

| Row | Sheet Label | Proposed Mapping | Confidence |
|---|---|---|---|
| 3 | C&F and other charges | Import clearing/C&F cost — `09050010`/`09050019`/`09050021`/`09050024` (several named C&F-agent accounts exist on 100001) | ⚠️ multiple candidates, which one(s) actually feed this row? |
| 4 | Shipment Value | Import shipment value (goods-in-transit) — Purchase module import POs, not a single GL code | ⚠️ likely BS "Goods in Transit"/inventory-in-transit asset — confirm account |
| 5 | Pay order Value (Tax & VAT) | Import duty/VAT on the shipment — `04010008` Customs Tax & VAT on imports (already in this app's COGS bucket, `build_pl_level_s`) | ✅ |
| 6 | % Addition | Derived ratio (markup used to project Shipment Sales Value from Shipment Value) — not an account | — n/a |
| 7 | Shipment Sales Value | Projected sales value of the shipment (Shipment Value × (1+% Addition)) — feeds Revenue (`08010001`) downstream | ✅ conceptually, it's a projection input not a GL pull |
| 8 | Local Puchase Trading Items | Local (non-import) purchase — COGS local-purchase codes (`04010002`/`04010004`) | ✅ |
| 10 | HMBR Product Inventory (Opening) | BS Inventory opening balance | ✅ (Inventory module / `imtrn`) |
| 11 | HMBR Product Sales % | Derived ratio — not an account | — n/a |
| 12 | HMBR Product Discount | `0708-Discount Paid` (`07080001`) vs MRP Discount (`07080002`)? | ⚠️ needs your call — which discount concept this row represents |
| 13 | HMBR Product Sales Value | Revenue — `08010001`, sourced from Sales module (`mv_sales_line_items`/`altsales`) | ✅ |
| 14 | HMBR Product Inventory (Closing) | BS Inventory closing balance | ✅ |
| 16 | HMBR Collection | Collections from customers — Collection module (`mv_collection_vouchers`) | ✅ |
| 17 | HMBR Additional Collection | Ad-hoc/one-off receipts — same Collection-module family, or manual GL entries | ⚠️ what distinguishes this from row 16 operationally? |
| 18 | LC Open | Import Letter-of-Credit opening — a bank/Purchase-side commitment | ❓ no direct GL account found — may not exist in the ERP at all, could need manual tracking |
| 19 | BL withdrow | Bill of Lading withdrawal (import clearance step) | ❓ same as above — likely not a GL-tracked event |
| 20 | HMBR Payorder (Tax & VAT) | Import duty/tax cash payment — `04010008` (same code as row 5, different stage: estimate vs. actual cash-out) | ⚠️ confirm row 5 vs row 20 are really the same account at two pipeline stages |
| 21 | TT | Telegraphic transfer (international wire) — bank cash-outflow, not a P&L code | ⚠️ BS bank movement, which account? |
| 22 | Others International Payment | Misc international payment | ⚠️ candidate: `06220010` Spare Parts Purchase bill (Foreign), but likely a catch-all |
| 23 | Others ( Transport + Hasbolt ) | Local transport/misc — S&D expense bucket | ⚠️ candidate: `07030001-3` (Vehicle/Transport family in `sd_expenses`) |
| 24 | HMBR International Payment | Aggregate of intl payments (≈ TT + Others Intl) | — derived, not its own account |
| 25 | HMBR Total Payment | `=B28`, i.e. the full local-payments sum below | — derived |
| 26 | HMBR Balance Forward | BS Cash account, prior month closing | ✅ |
| 27 | HMBR Cash Balance | BS Cash account, this month closing — **the model's own output** | ✅ (computed, not sourced) |
| 28 | HMBR Total Payments | `=SUM(B29:B47)` | — derived |
| 29 | Tota supplier Local | Local supplier AP payments — Purchase module cash outflow | ✅ |
| 30 | Dhaka Bank Loan Payment | `10010004` Dhaka Bank Short Term Loan / `12010002` Dhaka Bank Building Loan (principal repayment) | ⚠️ which one — short-term or building loan? |
| 31 | UCB Loan Payment | `10010008` Short Term Loan UCB / `10010005` Loan Against Trust Received From UCB (LTR) | ⚠️ same ambiguity as above |
| 32 | Loan Payment Rahima | `02050008` Loan to Rahima Enterprise (repayment reducing this receivable) | ✅ |
| 33 | Staff Salary | `06120001`/`06120002` Salary Expense (Level S "0612-Salary Expenses") | ✅ |
| 34 | Malek Jobeda Loan Payment | `09040007` Malek Jobeda Foundation | ✅ |
| 35 | Interest & Loan Paid to Md Sir | `10020001` Loan From MD Sir (principal) + `06300003` Interest on Md Sir Loan (interest, in `bank_interest`) | ✅ (two accounts, one row — combined figure) |
| 36 | Loan to Bijoy sir | `02050004` Loan to Directors | ⚠️ generic "Directors" account, not named for Bijoy specifically — confirm this is the right one |
| 37 | Interest Paid to Dhaka Bank | `06350001` Interest on Term & Building Loan (DBL) | ✅ |
| 38 | Interest Paid to UCB Bank | `06350002` Interest on Term Loan (UCB) | ✅ |
| 39 | Repayment to DBL-CC A/C | `10010003` Bank Overdraft Dhaka Bank-CC-A/C | ✅ |
| 40 | Income Tax & Misc Expenses | `06290002` Income Tax Expenses, blended with a misc code | ⚠️ which misc code, or is this two figures combined into one row? |
| 41 | Sun Flower Loan Repayment | `10020011` Loan From Sun Flower | ✅ |
| 42 | Operating Expenses (sales, delivery, logistics charges) | S&D expense bucket (`sd_expenses` codes, `0701-07140xxx` family) | ⚠️ confirm scope vs. row 23's "Others (Transport+Hasbolt)" — do these overlap? |
| 43 | C & F bill | `09050024` C & F bill / Charge. | ✅ |
| 44 | Loan Payment Fix iT | `10020002` Loan From Fix-iT / `10020003` Loan From Fix it(G.R) | ⚠️ two Fix-iT-named accounts on 100001 — which one(s)? |
| 45 | Investment & Misc | — | ❓ no clear account-master match |
| 46 | Misc. Expenses (Unforseen) | General misc expense code | ❓ needs a specific code |
| 47 | Loan to Zepto | `02050011` Loan to Zepto Chemicals (asset, HMBR's side) — counterpart on Zepto's own books is `10020002` "Loan From HMBR" (100005) | ✅ — and this is the exact inter-company pair Level C2's elimination logic already nets |
| 48 | Loan to GI corporation for Int. Purchase | `02050017` Loan to GI For Property Share? — counterpart candidates on GI's books: `10020002` "Loan From GTC" / `10020003` "Loan from GTC Against Factory Floor" (100000) | ⚠️ the only "Loan to GI" account found is labeled "For Property Share," not "Int. Purchase" — is this the same loan under an old label, or does the Int. Purchase advance not have its own dedicated account? |

---

## GI Test (ZID 100000)

Real formula: `GI Cash Balance = GI Collection + GI Additional Collection + GI Balance
Forward − GI International Payment − GI Total Payment` (`=F5+F6+F11-F9-F10`) — note this
one does **not** subtract a Payorder line the way GTC's does (GI Payorder, row 7, isn't
in the formula at all). Flagging this as a real structural difference, not an omission —
confirm before assuming GI's formula should mirror GTC's exactly.

| Row | Sheet Label | Proposed Mapping | Confidence |
|---|---|---|---|
| 3 | GI Product Sales (Old) | Revenue — `08010001`, Sales module | ✅ |
| 4 | GI Product Sales (New) | Same as above — two columns for what looks like an old-vs-new sales-tracking method | ⚠️ what's the actual difference between "Old" and "New"? |
| 5 | GI Collection | Collection module | ✅ |
| 6 | GI Additional Collection | Same ambiguity as GTC row 17 | ⚠️ |
| 7 | GI Payorder | Import duty/tax payment, `04010008`-equivalent — **not used in the Cash Balance formula** | ⚠️ confirm this is intentionally excluded |
| 8 | BL withdrow | Same as GTC row 19 | ❓ |
| 9 | GI International Payment | Aggregate international payment | ✅ conceptually (used directly in the formula) |
| 10 | GI Total Payment | `=F13`, i.e. the local-payments sum below | — derived |
| 11 | GI Balance Forward | BS Cash opening | ✅ |
| 12 | GI Cash Balance | BS Cash closing — model output | ✅ |
| 13 | GI Total Payment | `=SUM(F14:F22)` | — derived |
| 14 | Total Local supplier payment | Purchase module cash outflow | ✅ |
| 15 | Loan Repayment to (GTC) | `10020002` Loan From GTC / `10020003` Loan from GTC Against Factory Floor (100000) — counterpart to GTC's `02050017`/`Loan to GI` asset | ⚠️ same "which loan account" ambiguity as GTC row 48 |
| 16 | Staff Salary with Bonus | `06120001/2` Salary + `06130001/2` Bonus (Level S "0612"+"0613") combined | ✅ |
| 17 | Operating Expenses | S&D/SG&A operating bucket | ⚠️ same scoping question as GTC row 42 |
| 18 | Accrued Sales Distribution & Operating 7% | `07120004` Sales Distribution & Operating Expenses / `09010003` Accrued Sales Distribution & Operating Expenses | ✅ |
| 19 | Loan Repayment to (Fix iT) | `10020005` Loan From Fix iT (100000) | ✅ |
| 20 | Loan repayment to MD Sir | `10020001` Loan From MD Sir. (100000) | ✅ |
| 21 | Mold and Machinaries | `03050001` Mold & Dice / `03060001` Plant & Machinary (capex, BS fixed asset) | ✅ |
| 22 | Misc. Expenses (unforseen) | Same open item as GTC row 46 | ❓ |
| 24 | C&F and other charges | Same as GTC row 3 | ⚠️ |
| 25 | Shipment Value | Same as GTC row 4 | ⚠️ |
| 26 | Pay order Value | Same as GTC row 5/GI row 7 | ⚠️ |
| 27 | % Addition | Derived ratio | — n/a |
| 28 | Shipment Sales Value | Same as GTC row 7 | ✅ conceptually |
| 30 | Trading+GI | Looks like a combined/rollup row, not its own account | ⚠️ confirm what this aggregates |

---

## Zepto-M (ZID 100005)

Real formula for row 18 "Balance": `=F13+F16-F17` = `Collection + B-F (Balance Forward) −
Total Payment Expenses` — note this **doesn't directly add row 14 "Additional collection"**
the way the design doc's generic formula implies; either it's baked into "Collection"
upstream or it's a real gap. Flagging for your confirmation, not assuming either way.

Zepto is a manufacturer (per this app's own `_MANUFACTURING_ZIDS`), so "MO" here almost
certainly means **Manufacturing Order** (`moord`/`moodt`, per the design doc's own Step 2
data-source list), not "Mobile Order" — flagging the assumption explicitly since the
sheet itself never spells it out.

| Row | Sheet Label | Proposed Mapping | Confidence |
|---|---|---|---|
| 2 | % of MO as Sales | Derived ratio | — n/a |
| 3 | MO Value of Products | Manufacturing Order output value — `moord`/`moodt` | ⚠️ confirm MO = Manufacturing Order, not something else |
| 4 | % Addition | Derived ratio | — n/a |
| 5 | Total Inventory Balance (Avg Sales Rate) | BS Inventory, valued at avg sales rate (not cost) — a label-only row in the sheet, no data in it | — n/a |
| 6 | Sales (Trade Value) | Revenue — `08010001`, Sales module | ✅ |
| 7 | Sales % Increase | Derived ratio | — n/a |
| 8 | Estimated % Inventory as Sales | Derived ratio, empty in the sheet | — n/a |
| 9 | MRP Discount | `07080002` MRP Discount (Level S row of the same name) | ✅ |
| 10 | Additional Discount | `07080001` Discount Paid? | ⚠️ or a genuinely separate discount bucket — confirm |
| 11 | Total Sales With MRP Price | Derived (Sales + MRP Discount + Additional Discount) | — n/a |
| 12 | After Sales Inventory | BS Inventory, post-sale — label-only row, no data | — n/a |
| 13 | Collection | Collection module | ✅ |
| 14 | Additional collection | Same ambiguity as GTC row 17 — and NOT used in the Balance formula (see above) | ⚠️ |
| 15 | % of Collection | Derived ratio | — n/a |
| 16 | B-F | BS Cash opening (Balance Forward) | ✅ |
| 17 | Total Payment Expenses | `=F20`, the local-payments sum below | — derived |
| 18 | Balance | BS Cash closing — model output | ✅ |
| 20 | Total Payment Details | Same as row 17 | — derived |
| 21 | Supplier Payment | Purchase module cash outflow | ✅ |
| 22 | Marketing Expenses | SG&A marketing codes | ⚠️ which specific code(s) |
| 24 | Loan Payment Md Sir | `10020001` Loan From Director (100005) | ⚠️ generic "Director" account on Zepto's books, same ambiguity as GTC row 36 |
| 25 | Staff Salary | `06120001/2` Salary Expense | ✅ |
| 27 | Brand promotion | Marketing/SG&A code | ⚠️ |
| 28 | Loan Repayment to HMBR (Salary) | `10020002` Loan From HMBR (100005) | ✅ |
| 29 | Loan Repayment to HMBR | Same account as row 28 — what's the "(Salary)" distinction? | ⚠️ two rows, one account — confirm they're really different things |
| 30 | Loan Repayment to GI Corporation | No "Loan From GI Corporation" account found on 100005's books | ❓ |
| 31 | Lab Equipment & Maintenance | Capex/BS fixed asset or SG&A maintenance code | ⚠️ |
| 32 | Operating Expenses (sales, delivery, logistics charges) | Same scoping question as GTC row 42 | ⚠️ |
| 33 | Increase % of Operating Expenses | Derived ratio | — n/a |
| 34 | Festival Bonuses | `06130001/2` Bonus? | ⚠️ |
| 35 | Loan Repayment to GI | Same open item as row 30 | ❓ |
| 36 | Machineries | Capex, BS fixed asset | ✅ conceptually |
| 37 | Misc. Expenses (Unforseen) | Same open item as GTC row 46 | ❓ |
| 38 | Director remuneration | `06150001-3` Director Remuneration (Level S "0615") | ✅ |

---

## Summary of what needs your input before Step 2 starts

1. **Import pipeline rows** (LC Open, BL withdrow, Shipment Value, C&F charges, TT,
   Payorder) — several have no clean 1:1 GL account and may need to be sourced from the
   Purchase module's import-PO data instead of `gldetail`, or may not be trackable at
   all inside this app today. Worth a dedicated discussion.
2. **Same-named accounts appearing more than once per ZID** (e.g. "Loan From Fix-iT" at
   two different codes on 100001; "Dhaka Bank"/"UCB" loan rows each having a short-term
   vs. LTR vs. building-loan candidate) — need your call on which one each sheet row
   actually tracks.
3. **Inter-company loan pairs** (rows 47/48 on GTC, row 15 on GI Test, rows 28-30/35 on
   Zepto-M) — mostly resolved, but "Loan to GI corporation for Int. Purchase" only has a
   candidate account labeled "For Property Share," and "Loan Repayment to GI Corporation"
   (Zepto) has no matching account on Zepto's books at all.
3. **Rows with no account-master match at all** (Investment & Misc, Misc. Expenses
   (Unforseen), Loan Repayment to GI Corporation from Zepto) — need either a real code
   pointed out, or a decision to source these from somewhere other than `gldetail`.
4. **GI Test's formula genuinely excludes Payorder** while GTC's includes it — confirm
   this is intentional before building a single shared formula function for both.
