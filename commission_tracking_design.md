# Sales Reward, Commission & Incentive Tracking — Design Doc

Status: **agreed design, not yet built.** This is the record of a discussion-first design
pass (a first full 4-table implementation was built 2026-09-16, then explicitly reverted
2026-09-17 — the user wanted the whole idea space discussed before any code existed, see
git history around commit `6c2f524`). Source material: `Sales Reward_Incentive Plan.docx`
(provided by the user) plus the chat discussion that resolved every open question in it.

Whoever picks this up: **discuss before building**, same as this doc's own origin — if
scope has drifted since this was written, re-confirm with the user rather than assuming
this doc is still current.

---

## Status at a glance — what's buildable right now vs. genuinely blocked

Two items are real hard blockers, but they block narrower pieces than "the whole doc" —
everything else below is fully specified and can be started without asking the user
anything new.

| Piece | Status | Blocker (if any) |
|---|---|---|
| A.1 Best Performer | ✅ buildable | one minor open point: 3-month averaging method (§A.1) — doesn't block starting, the engine/weights/scope are all specified |
| A.2 Highest Product Sales | ✅ buildable | none |
| A.3 Best Disciplined | ✅ n/a | excluded by design |
| A.4 Best App User | 🚫 **deferred by design** | user will supply the actual list of app data-hit locations when this specific section gets built — not something to chase before then, don't build against the guess in §A.4 |
| B.1/3/4 core (rate/cap/FIFO/gate payout math) | ✅ buildable | none — this is fully specified, including the pooled-FIFO fix |
| B.1/3/4 sales-uptick companion metric | ✅ buildable | resolved 2026-09-17 — the baseline window is a user-adjustable parameter, not fixed (see §B.1/3/4) |
| B.2 Individual Target Achievement | ✅ buildable | one unconfirmed assumption (gate applies here too, §The 100001/100000 gate) — doesn't block starting |
| B.5 New Customer Creation | ⚠️ not scoped | needs real design work (not a blocker exactly — nobody's promised an answer, it just hasn't been designed yet) |
| Product Tracking | ✅ buildable | none |
| The 100001/100000 gate itself | ✅ buildable | none |

**If asked "where do you want to start building," don't re-ask this as an open question —
the table above already answers it.** Good self-contained starting points with zero open
items: Product Tracking, A.2, or B.1/3/4's core payout mechanism (add the uptick metric
once its baseline window is confirmed, as a follow-up, not a prerequisite).

---

## Architecture, agreed up front

- **One results table, not four config tables.** The products/salesmen/sales data already
  exist in the ERP; a "campaign" is a parameter set applied to that existing data, computed
  **once**, and the RESULT is what's worth persisting — not a rebuildable live config system.
  Store: which campaign ran, total paid out, who got it, which products/DOs were involved
  (a compact/delimited or JSON column for the DO references, not a normalized child table).
- Everything below that isn't a one-off payout (the rankings in section A) needs **no
  persistence at all** — compute live off existing data, same "recompute, don't store"
  pattern as every other report in this app.
- The FIFO DO-payment-collected resolver from the reverted first build (`resolve_do_paid_dates`
  — per-customer, full-history, chronological walk, all-or-nothing per DO) is still wanted —
  confirm it's reusable as-is before rewriting it from scratch. See "Pooled FIFO" note below
  for the one real change needed to it.

---

## A. Rankings (no persistence — pure reports)

### A.1 Best Performer
- Scope: **100001 + 100000 combined**, trailing 3 months.
- Reuses `processing/salesman_score.py`'s existing peer-relative scoring engine (same shape:
  positive components build a ceiling, negative components erode it, clipped to [0, 100]),
  but **recalibrated weights**, and **the weights must be adjustable** (a config, not
  hardcoded) — the score recomputes live whenever the weights change, not baked into code.

  Proposed starting weights (confirm/adjust in the UI once built, not fixed by this doc):
  ```
  +45  Sales Target Achievement   (net_sales / target, clipped 0-100%)
  +45  Collection                 (collected / net_sales — same definition
                                    Salesman Score already uses today)
  +5   Unique Products sold       (peer-relative)
  +5   Unique Customers sold to   (peer-relative)
  -20  Returns %                  (peer-relative)   ["Lowest Return"]
  -20  Balance / Dues             (peer-relative)   ["Lowest Outstanding Dues"]
  ```
  Today's actual Salesman Score (for reference/diff, NOT what to ship): `+45 target, +45
  collection, +5 products, +5 customers, -6 returns, -12 balance(2mo), -2 balance(this mo)`.
  The doc's 3 named criteria (Target Achievement / Lowest Return / Lowest Outstanding Dues)
  are all real components above; "Collection" and the two peer-relative bonus components
  are carried over from the existing engine, not newly asked for.

- Payout: **a fixed BDT amount** (e.g. 2,000) to the winner — NOT a %, NOT proportional.
- "Last 3 months" — averaging each month's own score, vs. computing the components off
  pooled 3-month totals directly, is not yet decided — confirm with the user before building.

### A.2 Highest Product Sales
- Current month, ranked by **distinct product count** (breadth, not volume) per salesman.
- No open questions.

### A.3 Best Disciplined Salesperson
- **Excluded** — not trackable from any DB this app touches (selfie/location/dress
  code/response-time compliance has no data source). User's own call, from the source doc.

### A.4 Best App User
- Tracks per-salesman, per-month usage of the mobile app's various functions.
- **STILL OPEN — waiting on the user for the exact list of data-hit locations to count.**
  A reasonable starting guess (NOT confirmed) based on what this app already reads as
  "app-staged, not app-owned" data: `glpmt` entries (App Collections), `opcrn` entries
  (Returns Registry), `feedback` entries, `opdor.xdatedel`/`xdatepay` promise-date entries.
  Do not build against this guess without the user's actual list.

---

## B. Commission Campaigns (the payout side — this is where the one results table lives)

### B.1 + B.3 + B.4 — Product-Specific / Stock Clearance / Slow-Moving (one mechanism)
These three items in the source doc are mechanically identical per the user's own note on
B.4 ("same as product incentives, nothing different") — one feature, not three:

- Pick product(s), set a **rate per unit sold**, an optional **cap per salesman per
  product**.
- Commission earned only if the customer's own DO is **fully collected** by a deadline
  (all-or-nothing per DO — confirmed, not prorated).
- Deadline can differ by **salesman payout group** (e.g. a Dhaka group vs. a District
  group), each with its own collection-deadline date. Salesman-to-group assignment is
  manual — no region/district field exists on `prmst` to derive it automatically.
- **Scope: 100001 + 100000 combined.**
- **Pooled FIFO — a real correction made during this discussion.** `cusid` is CONFIRMED
  shared between 100001 and 100000 (verified against real Postgres: 5,038 of 5,046
  100000-customers carry the exact same `cusid` AND `cusname` in 100001, only 9 minor
  name-typo mismatches) — this is genuinely one shared customer namespace across the two
  ZIDs, not independent per-ZID numbering that happens to collide. **The FIFO resolver must
  pool 100001+100000 sales/collection rows together per customer BEFORE walking the
  chronological FIFO queue** — a real customer's payment history spans both entities as one
  continuous ledger. (The reverted first build's `resolve_do_paid_dates` ran per-ZID; this
  is the one real change it needs, not a rewrite.)
- The "≥500 pcs" stock-clearance threshold from the source doc (B.3) is descriptive context
  for how the user tends to pick products by hand, not an automatic filter to build.
- Whether the 100001/100000 gate (section below) applies here: **yes, confirmed.**
- **Sales uptick check — from the user's very first framing of this feature.** Alongside
  the payout math, also check whether the campaign actually produced a sales lift — current
  campaign-window sales for the picked product(s)/salesman(s) vs. their **previous
  average**. This is an analytical companion metric answering "did running this campaign
  work," separate from "how much do we owe" — surface it alongside the payout report, not
  as a gate on the payout itself unless the user says otherwise.
  - **Baseline window is a user-adjustable parameter, not fixed** — confirmed 2026-09-17.
    Give it a toggle/selectbox (same pattern as other adjustable lookback windows elsewhere
    in this app, e.g. Call Coverage Matrix's "Last 3 months," WhatsFly Bulk Messaging's
    "Time window" slider), with **3 months and 6 months** as the example values given —
    don't hardcode a single window.

### B.2 — Individual Target Achievement
- Named list (fixed, hand-maintained — from the source doc: Sazzad, Sumon Sheikh, Mobarak,
  Sobahan, Jamal, Kuddus, Sekandar–without Special Sales).
- Each person on the list who hits **their own individual target** (already in
  `data/targets.json`) gets a **fixed BDT amount** (e.g. 2,000) — a threshold check per
  person, NOT a ranking, NOT proportional.
- **No overlap/conflict logic needed with A.1.** Confirmed: these are run as alternating
  monthly choices by the user (e.g. Best Performer this month, this individual-target
  commission next month) — "1 campaign per [month]" — the system does not need to arbitrate
  between them; it's an operational scheduling choice, not a data problem.
- Whether the 100001/100000 gate applies here: **assumed yes** (also target-based), but not
  explicitly confirmed — flag with the user before building; may need to be exempt.

### B.5 — New Customer Creation
- Source doc claims this is "already done in Overall Sales Analysis" — **checked, it is
  NOT the right shape.** `processing/overall_sales.py::compute_customer_flow` exists but is
  **area-scoped**, not salesman-scoped, and "new" there means *new to that area this month*
  (a 10-year customer who just started buying in a different area counts as "new"), not
  "first sale ever to the business, attributed to the salesman who landed them."
- **Needs genuinely new logic** — first-ever-sale detection per customer, attributed to the
  salesman on that first sale, not a reuse of the existing customer-flow cohort logic.
- Not scoped/built yet — lowest-detail item in this doc, pick up last.

### The 100001 / 100000 gate
- **Confirmed**: if 100001 or 100000 individually fails to hit its own target, the
  commission for that gated campaign type isn't paid out at all (a company-wide pass/fail
  gate layered on top of the per-salesman/per-product math above).
- **Confirmed source of "did the ZID hit its target"**: the **sum of existing individual
  salesman targets** already in `data/targets.json` for that ZID — explicitly NOT a new
  standalone company-level target value. No new config table/field needed for this.
- Applies to B.1/B.3/B.4 for certain; probably B.2 too (not explicitly confirmed — see
  above).

---

## Product Tracking (separate feature, no dependency on anything above)

- A JSON watchlist, admin-edited, **1–20 item codes** (more "wouldn't make sense" per the
  user) — same convention as `data/targets.json`/`data/public_holidays.json`/
  `data/warehouse_filters.json` already use in this app (small human-curated config file,
  not a DB table).
- A view showing how those specific watched products performed **within the current
  month** whenever the view is opened.
- **Tracks both sales AND returns** for each watched product (confirmed — not sales alone).
- No open questions — ready to scope/build independently of the reward/commission system
  above, whenever picked up.

---

## Open items before/while building

See the status table near the top for which of these actually block starting vs. which are
just unresolved details on an otherwise-buildable piece.

1. **A.4** — deferred by the user's own choice, not a blocker to chase — they'll supply the
   actual list of app data-hit locations when this specific section gets built. Do not build
   against the guess in §A.4 in the meantime, and don't ask for this again before then.
2. **A.1** — "trailing 3 months" averaging method (per-month average vs. pooled totals) not
   yet decided. Doesn't block starting A.1.
3. **A.1** — the -20/-20 Returns/Dues weights (and the 45/45/5/5 carried over from today's
   Salesman Score) are a proposed starting point, not locked — the weights need to be
   adjustable in the UI regardless, so this matters less at build time than it would if the
   weights were going to be hardcoded.
4. **B.2 / gate interaction** — assumed the 100001/100000 gate also applies to B.2, not
   explicitly confirmed. Doesn't block starting B.2.
5. **B.5** — no design work done yet beyond "existing logic doesn't fit, needs something
   new" — scope this properly with the user when it's picked up, don't guess at a shape.
