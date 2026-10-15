# Financial Projections & Forecasting — Design Doc

Status: **discussion draft, agreed build order, not yet built.** Same discussion-first
convention as `commission_tracking_design.md`. Whoever picks this up: the build order
below is confirmed — start at Phase 1, in order, don't reshuffle without checking with
the user first. §E still has real unresolved items (WACC chief among them) — don't guess
those either.

---

## Build Order (confirmed) — merges every ask + every suggestion, deduplicated

The user's original five asks and Claude's seven suggestions overlapped heavily once
laid out side by side (e.g. "relational projections" and "integrated 3-statement model"
are the same thing; "margin/discount sensitivity" and "driver-based SKU revenue
modeling" are the same thing). This is the merged result, explicitly re-ordered per the
user's own priority: **the cash roll-forward rebuild goes first, because it's already in
active use today** (the Google Sheet gets maintained by hand right now) — everything
else follows, in the order below, "one after another."

### Phase 1 — Cash Roll-Forward Model Rebuild (do this first, real priority)

Replaces the Google Sheet (`GTC Admin + Motiur` = 100001, `GI Test` = 100000, `Zepto-M`
= 100005 — **confirmed authoritative**; the sheet's own `Trading-M`/`GI`/`Trading Test`
counterparts are NOT the source of truth, ignore them). The user gave an explicit
4-step process for this phase — follow it in this order, not shortcut:

1. **Account header mapping** — map every row label in the three sheets (`HMBR
   Collection`, `Staff Salary`, `Loan to Zepto`, `HMBR International Payment`, etc.) to
   the equivalent Level 0 / Level 1 / Level S account header already in this app's
   Financial Statements module. Every row needs a mapping before step 2 can start.
2. **Real data sourcing** — pull the actual monthly numbers from this app's own
   tables instead of hand-typing them: Sales (`mv_sales_line_items`), Purchase,
   Manufacturing (`moord`/`moodt`), Inventory (`imtrn`), and Expenses/Balance-Sheet
   payouts (`gldetail`/GL). This is the entire point of the rebuild — the data already
   exists in the ERP, it shouldn't need manual re-entry into a spreadsheet every month.
3. **Cross-check against the financial statements** — since step 2 pulls from several
   different source tables (sales module vs. GL vs. manufacturing) rather than one
   unified source, reconcile the computed numbers against the existing Level 0/S
   statements before trusting any of it. Do not skip this — this app's own convention
   all session has been "verify against real data before shipping," and it matters even
   more here since three independent data sources have to agree.
4. **Build the actual monthly cash roll-forward model**, replicating the sheet's real
   formula (`Collections + Additional Collections + Balance Forward − Import/Payorder
   outflows − Total Local Payments`, confirmed via the real GTC formula `=B16+B17+B26-B20-B24-B25`)
   — **including a proper projection mechanism this time**, i.e. the straight-line /
   manual-% mechanic from the user's original ask (#5 in the prior draft), built directly
   into this cash model rather than bolted on separately. The existing inter-company
   loan structure (GTC funds GI/Zepto's international purchases, they repay — see the
   real `Loan to Zepto` / `Loan Repayment to (GTC)` rows) must be preserved and netted
   properly when this rolls up to a consolidated view, mirroring how Level C2 already
   nets inter-company loans for actuals (`processing/consolidation.py`).

### Phase 2 — SKU-level driver-based revenue & discount modeling

Model revenue as `Volume × Price × (1 − Discount%)` **at the individual SKU level** — the
user confirmed full SKU granularity, not product-group/category. This is what makes the
original discount-sensitivity question ("what would net profit have been without that
10 lakh discount") a scenario toggle instead of a manual one-off calculation — and it's
also what feeds Phase 3 below instead of a single flat growth-rate assumption.

### Phase 3 — Full 3-statement integrated projection model

IS, BS, and CFS respond together, not just the IS (the existing Full-Year Projection
tool is IS-only — see §B.1). Revenue comes from Phase 2's SKU-level model instead of a
flat %. Working-capital movement uses **collection days, inventory days, and capex as
real levers** (confirmed) — AR moves with revenue via Collection Days
(`processing/collection.py::average_days_to_collection`), Inventory moves with COGS via
Days to Clear (Inventory Analysis → Statistical Analysis), AP similarly — those changes
feed the CFS, the CFS changes the BS's cash balance, and the BS's own debt balance feeds
back into next period's interest expense. Built on Level P/C2 per the user's original
ask, since that's what already consolidates across businesses. Inter-company elimination
(same requirement as Phase 1) applies here too, at the consolidated level.

### Phase 4 — Scenario framework

Named scenarios (Base/Upside/Downside or whatever naming is preferred), each carrying
its own lever values (revenue growth, gross margin %, discount %, collection days,
payment days, capex, inter-company loan draws), saved and comparable side by side.

### Phase 5 — Rolling monthly rebasing

Plan vs. Actual every month, re-forecast the remaining months rather than a single
static once-a-year projection — given how dynamic the real numbers already are (one
discount decision flipped a month from profit to loss), a forecast that's stale by month
3 isn't very useful.

### Phase 6 — Bridge / Variance Waterfall report

Decomposes the gap between two periods (or Plan vs. Actual) into Volume / Price /
Discount / Mix / Cost / Opex effects — the structured, repeatable version of the exact
question that started this whole conversation, runnable any month instead of computed
by hand once.

### Phase 7 — DCF valuation (last, and gated on an input the user hasn't supplied yet)

Per individual business and combined. Built last on purpose — a DCF is only as good as
the projected free cash flow feeding it, which doesn't exist in a modeled form until
Phase 3+. **Blocked on WACC** — the user is researching this separately (real
socio-economic context factors to weigh before settling on a rate) and will supply it;
do not assume or derive a WACC in the meantime.

---

## B. What already exists (confirmed by reading real code, not assumed)

### B.1 Level P and the current Financial Statements module

- **Level P ("Level P - Projection View")** already exists — `processing/financial.py::build_condensed_view()`, selectable in the Financial Statements page's Level dropdown (`views/financial.py:1661`), both per business and for **"All Businesses - Consolidated"**. It's a condensation of Level S (~28 IS rows → 10, ~30 BS rows → ~19), not itself a consolidation engine — for the consolidated view it sits on top of **Level C2** (`processing/consolidation.py`), which is where the real inter-company elimination logic already lives: inter-company loan pairs netted (dropped if they net to zero across every year, else kept both-sided), AR/AP routed into External/ARAP/Allother buckets, internal sales stripped out of COGS between 4 "external" sales ZIDs and 6 "internal" supply ZIDs.
- **A projection tool already exists, but it's narrower than Phase 3 above**: the **Full-Year Projection** calculator (`views/financial.py:2128-2305`, Yearly-Full-Year-vs-YTD timeframe only) already does the "straight-line vs. manual-%" mechanic — but **IS-only** (no BS, no CFS), **single annual snapshot** (not multi-month), and the "manual %" levers are Revenue / COGS-as-%-of-revenue / SG&A / S&D / Interest / VAT — whole-statement levers, not SKU- or discount-level. Persisted as flat JSON snapshots (`data/level_p_projections.json`, capped at 20). A separate single-month **Break-even calculator** also exists (Monthly timeframe only, `data/level_p_breakeven.json`).
- **The gap this whole doc exists to close**: the existing tool answers "if revenue/COGS/opex move by X%, what's the annual Net Income" — it does **not** trace that into cash flow or the balance sheet, and it has no concept of *why* revenue or margin might move (volume vs. price vs. discount, per SKU).

### B.2 The Google Sheet cash model, studied directly

Three sheets studied and **confirmed as the authoritative source** (`GTC Admin + Motiur` = 100001, `GI Test` = 100000, `Zepto-M` = 100005 — the sheet's own parallel `Trading-M`/`GI`/`Trading Test` tabs are not the source of truth), all the same recipe:

```
Cash Balance(month) = Collections + Additional Collections + Balance Forward(prior month's closing)
                       − Import/Payorder outflows − Total Local Payments(month)
```

Real formula, GTC row 27: `=B16+B17+B26-B20-B24-B25`. "Total Local Payments" is a flat `SUM()` of business-specific line items — loan repayments, staff salary, interest, tax, operating expenses, misc.

**A real inter-company lending structure is already modeled by hand** in this sheet, and it's the concrete version of "what affects what" for cash specifically: GTC's payment list includes `Loan to Zepto` and `Loan to GI corporation for Int. Purchase` (stored as % allocations); GI's payment list includes `Loan Repayment to (GTC)`; Zepto's includes `Loan Repayment to HMBR` and `Loan Repayment to GI Corporation`. GTC (the importer) funds the other two, they repay.

**Why rebuild this at all** (the user's own reasoning, worth keeping on record): tracking it by hand in a spreadsheet has become difficult to maintain — the data already exists in the ERP across several modules, so it shouldn't require manual monthly re-entry.

### B.3 Operational data this app already computes, relevant as forecast anchors

Rather than guessing assumptions for a forecast (e.g. "collections take 45 days" as a made-up number), these already exist and are real, verified figures:
- **Avg Collection Days** — `processing/collection.py::average_days_to_collection`.
- **Inventory Days to Clear** — Inventory Analysis → Statistical Analysis mode (`stock / avg_monthly_sales × 30`).
- **AR aging / current balance** — `processing/salesman_due.py` (Customer Credit Trickle-down, Latest Sale & Collection).
- **Sales targets already set per salesman** — `data/targets.json`, usable as a bottom-up revenue-growth input instead of (or alongside) a single top-line %.

---

## C. Decisions already confirmed (not open anymore)

- **Source-of-truth sheets**: `GTC Admin + Motiur`, `GI Test` — explicitly confirmed. `Zepto-M` was the only Zepto sheet ever named by the user (no parallel "Zepto test" was ever put forward as an alternative), so it's treated as authoritative by the same pattern — flag if that's wrong, it was never separately re-confirmed the way the other two were.
- **Scenario levers**: revenue growth, gross margin %, discount %, collection days, inventory days, payment days, capex — all confirmed in scope, not just revenue growth.
- **UI placement**: a **new standalone page**, added to the nav bar as **"Financial Projections"** — not embedded inside the existing Level P section of Financial Statements. Code should still follow the existing `views/financial.py`/`processing/financial.py` conventions (Analytics pattern, SQL-for-fetch/pandas-for-transform, Level S sign convention, etc.) rather than inventing new patterns.
- **Persistence**: DB tables are fine, but **kept minimal** — only for data that would be genuinely expensive to recompute on every page load (matching the "one table, not four" instinct from Commission Tracking). Prefer live computation and the existing JSON-snapshot pattern wherever recomputation is cheap; reach for a real table only when it isn't.
- **Revenue modeling granularity**: every SKU, confirmed (not product group/category).

---

## E. Open questions — still genuinely unresolved

1. **WACC / discount rate for DCF (Phase 7)** — the user is researching this separately (real socio-economic factors to weigh) and will supply it. Do not derive or assume a value in the meantime; Phase 7 is blocked until it arrives.
