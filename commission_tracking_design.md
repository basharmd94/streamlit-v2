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
| B.1/3/4 core (rate/cap/FIFO/gate payout math) | ✅ **built 2026-09-19** | none — verified against real Postgres (see §B.1/3/4) |
| B.1/3/4 sales-uptick companion metric | ✅ **built 2026-09-19** | baseline window is a user-adjustable parameter (3mo/6mo), as resolved 2026-09-17 |
| B.2 Individual Target Achievement | ✅ buildable | one unconfirmed assumption (gate applies here too, §The 100001/100000 gate) — doesn't block starting |
| B.5 New Customer Creation | ⚠️ not scoped | needs real design work (not a blocker exactly — nobody's promised an answer, it just hasn't been designed yet) |
| Product Tracking | ✅ **built 2026-09-19** (redesigned version) | none — see §Product Tracking for what shipped |
| The 100001/100000 gate itself | ✅ buildable | none |

**If asked "where do you want to start building," don't re-ask this as an open question —
the table above already answers it.** Good self-contained starting points with zero open
items: Product Tracking, A.2, or B.1/3/4's core payout mechanism (add the uptick metric
once its baseline window is confirmed, as a follow-up, not a prerequisite).

---

## Architecture, agreed up front

- **One table, not four config tables — and revised 2026-09-19 to store the campaign
  DEFINITION, not a computed result.** Originally framed as "compute once, persist the
  result" (a snapshot of what got paid) — **corrected by the user while starting B.1/3/4**:
  the table instead holds campaign *parameters* (name, dates, products, rate/cap, payout
  groups) and the payout is **calculated live whenever that campaign is queried/opened**,
  same "recompute, don't store" pattern as section A's rankings — B section is no longer an
  exception to that pattern. This is safe to recompute repeatedly because the numbers
  naturally stabilize once a group's collection deadline has passed (nothing dated after
  the deadline can still count), so "live" doesn't mean "keeps changing forever."
- Everything in section A (rankings) also needs **no persistence at all** — compute live
  off existing data, same pattern.
- **The FIFO DO-payment-collected resolver mentioned as "from the reverted first build"
  does NOT actually exist anywhere** — checked git history 2026-09-19, that first build was
  never committed (built and reverted entirely in an earlier, uncommitted session). Written
  fresh instead, per-customer/full-history/chronological/all-or-nothing-per-DO, pooled
  across 100001+100000 (see "Pooled FIFO" note below).

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

## B. Commission Campaigns (the payout side — this is where the one campaign-definition table lives)

### B.1 + B.3 + B.4 — Product-Specific / Stock Clearance / Slow-Moving (one mechanism)
These three items in the source doc are mechanically identical per the user's own note on
B.4 ("same as product incentives, nothing different") — one feature, not three:

- Pick product(s), set a **rate per unit sold**, an optional **cap**.
- **Rate and cap both vary PER PRODUCT, not one flat value for the whole campaign —
  confirmed 2026-09-20** (corrected from an earlier single-campaign-wide framing). **Rate
  is the per-unit incentive/discount BDT amount being offered on that product (e.g. 5 or
  3), NOT the product's sales price** — this was explicitly ambiguous and the user
  resolved it directly: "the commissions for these I will set the discount rate so BDT 5
  or BDT 3 etc and per product." Cap is "per recipient per product" (see recipient_type
  below), also settable per product, optional.
- **The commission can be paid to either the SALESMAN or the CUSTOMER on the DO —
  confirmed 2026-09-19/20.** "The underlying payment, date structure, after payment
  structure will remain the same" regardless of which — i.e. DO eligibility (fully
  collected, by which deadline) is identical either way and is always keyed off the DO's
  own salesman's payout group (the deadline is a logistics/territory concept, tied to
  whoever handled the sale, not to whoever ultimately gets paid). Only the final
  aggregation/attribution step differs: group the eligible line items by salesman or by
  customer.
- Commission earned only if the customer's own DO is **fully collected** by a deadline
  (all-or-nothing per DO — confirmed, not prorated).
- Deadline can differ by **salesman payout group**, each with its own collection-deadline
  date. **Group membership is now derived LIVE, not a manual roster — confirmed
  2026-09-20 (second correction).** `prmst.xdisease` (despite its medical-sounding name)
  holds the salesman's current working area(s), comma-separated free text (e.g.
  `"Ibrahimpur, Askona"`). `cacus.xstate` (despite the "state" name) already carries a
  sales-zone/channel classification per customer area (`xcity`) — confirmed real values:
  `Dhaka retail`, `District`, `Dhaka General`, `Nawab pur`, `Kawranbazar`, `Alubazar`,
  `Imamgonj`, `Rahima enterprise`, `Sylhet Retail` — and this exact column is already used
  elsewhere in the app for a Retail/District split
  (`processing/ar_analysis.py`'s "Market" column). **Group names are the real `xstate`
  values themselves, not collapsed to a plain Dhaka/District binary** — the one exception,
  confirmed by the user: `"Sylhet Retail"` folds into `"District"` (it says "retail" but
  Sylhet isn't Dhaka, and the deadline is about physical distance).
  `processing/commission_campaigns.py::build_area_group_map` builds `{xcity -> group}` from
  pooled 100001+100000 `cacus` (majority `xstate` per area — a handful of stray mistagged
  rows per area don't change the winner, e.g. Chittagong is `District` in 371/373 rows);
  `derive_spid_group_map` matches each salesman's `xdisease` area list against that map
  (majority vote across the salesman's own listed areas, ties broken by whichever tied
  group's area is listed first). **This replaced the original plan of a hand-maintained
  `data/commission_payout_groups.json` roster** (see "What shipped" below for that
  mid-flight history) — there's nothing to keep in sync by hand anymore; fixing a
  salesman's group means fixing their area on the ERP side (`prmst`), not in this app.
  **As of 2026-09-20 `prmst.xdisease` is mid-rollout on the live server** — the local
  Postgres mirror still holds old placeholder junk (`"Ok"`, `"OK."`, ...) in most populated
  rows, not real area data yet, so `derive_spid_group_map` correctly returns an empty map
  locally (verified) — don't expect real group resolution until that rollout finishes on
  the live server. `cacus.xstate` itself has no such rollout gap — it's real and usable
  locally today.
- **Scope: 100001 + 100000 combined.**
- **Pooled FIFO — a real correction made during this discussion.** `cusid` is CONFIRMED
  shared between 100001 and 100000 (verified against real Postgres: 5,038 of 5,046
  100000-customers carry the exact same `cusid` AND `cusname` in 100001, only 9 minor
  name-typo mismatches) — this is genuinely one shared customer namespace across the two
  ZIDs, not independent per-ZID numbering that happens to collide. **The FIFO resolver must
  pool 100001+100000 sales/collection rows together per customer BEFORE walking the
  chronological FIFO queue** — a real customer's payment history spans both entities as one
  continuous ledger. (No prior FIFO resolver code actually exists to reuse — see
  Architecture note above — so this is a fresh implementation, pooled from the start.)
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

#### Table schema — `commission_campaigns` (revised 2026-09-20)

One DB table, holds campaign *definitions* only — no salesman/customer/DO/payout numbers
stored, those are all computed live from this row + existing sales/collection data
whenever the campaign is opened. **Salesman→payout-group membership is deliberately NOT in
this table** — confirmed 2026-09-20: it's derived LIVE from `prmst.xdisease` +
`cacus.xstate` (see above), not stored anywhere. The DB table only stores which deadline
THIS campaign gives each group; the group list itself (currently 8: `Dhaka retail`,
`District`, `Dhaka General`, `Nawab pur`, `Kawranbazar`, `Alubazar`, `Imamgonj`,
`Rahima enterprise`) comes from whatever distinct `xstate` values exist in `cacus` at
create-campaign time, not a fixed enum.

```sql
CREATE TABLE commission_campaigns (
    id                SERIAL PRIMARY KEY,
    campaign_name     VARCHAR(200) NOT NULL,
    campaign_type     VARCHAR(50),        -- free-text label: 'Product-Specific' /
                                           -- 'Stock Clearance' / 'Slow-Moving' -- same
                                           -- mechanism either way, just lets the user
                                           -- categorize campaigns in a list view
    recipient_type    VARCHAR(20) NOT NULL DEFAULT 'salesman',  -- 'salesman' or 'customer'
    product_rates     JSONB NOT NULL,     -- {"ITEMCODE1": {"rate": 5.0, "cap": 2000.0},
                                           --  "ITEMCODE2": {"rate": 3.0, "cap": null}}
                                           -- rate = per-unit INCENTIVE/discount amount
                                           -- (e.g. 5 or 3 BDT), NOT the product's sales
                                           -- price -- both rate and cap vary per product
    window_start      DATE NOT NULL,      -- sales window: which DOs are even eligible
    window_end        DATE NOT NULL,
    payout_groups     JSONB NOT NULL,     -- {"Dhaka": "2026-10-15", "District": "2026-10-31"}
                                           -- group name -> THIS campaign's own collection
                                           -- deadline for that group. Salesman roster per
                                           -- group comes from the separate JSON file above.
                                           -- Always salesman-keyed regardless of
                                           -- recipient_type -- see note above.
    uptick_baseline_months INTEGER DEFAULT 3,  -- companion-metric baseline window
    created_by        VARCHAR(50),
    created_at        TIMESTAMPTZ DEFAULT now(),
    notes             TEXT
);
```

(This replaced an earlier `product_codes`/`rate_per_unit`/`cap_per_salesman_per_product`
shape from 2026-09-19 — revised the next day, before the table had any real rows, so it
was a straight drop+recreate, not a migration. The schema itself didn't need a second
revision for the payout-group derivation change — `payout_groups` was always just
`{group_name: deadline}`, independent of where the group *list* comes from.)

#### What shipped (2026-09-19, revised 2026-09-20)

`processing/commission_campaigns.py` (FIFO resolver + payout engine + gate + uptick +
campaign CRUD) and `views/commission_campaigns_view.py` (roster editor, create-campaign
form, campaign list/detail) — wired into the Commissions page's dropdown. The
`commission_campaigns` table was created live via
`db/sql_scripts/create_commission_campaigns_table.sql`.

- **FIFO resolver** (`resolve_do_paid_dates`): pooled per-customer chronological walk —
  a DO is a FIFO queue entry, a collection pays off the oldest open DO(s) first, in full,
  before moving on; leftover collection becomes a "prepaid credit" pool applied to that
  customer's next DO(s) as they appear (handles an advance payment that precedes the DO
  it's really for). **Verified two ways**: hand-computed against a real customer's full
  6-DO/5-collection history (including a collection that split across two DOs) — exact
  match; plus 3 synthetic edge cases (prepaid credit applied to a future DO, a large
  overpay with no further DOs, a DO that never gets paid) — all pass.
  `build_do_totals` carries both salesman AND customer id/name on every DO row (added
  2026-09-20) so either `recipient_type` can be served from the same resolved ledger.
- **Payout**: eligible DOs (in the campaign's sales window, salesman in a roster group,
  paid by that group's deadline) → sum qty of the picked product(s) on that DO → × that
  product's own rate → capped per recipient (salesman or customer) per product → summed
  per recipient. Verified against real data down to the individual line (22 units × ৳5 =
  ৳110, cross-checked against a standalone computation) and cap-clipping verified with a
  deliberately low cap. **Verified the recipient-grouping choice is capping-neutral the
  way it should be**: summed raw (pre-cap) payout came out identical (312) whether grouped
  by salesman or by customer against the same real data — same underlying line items
  either way, only where the cap binds differs, correctly, by recipient granularity.
- **Gate**: `zid_target_sum` (sum of `data/targets.json` entries for the ZID across the
  window's calendar months) vs. `zid_actual_sales` (`final_sales` summed in-window) —
  fails the whole campaign's payout if either ZID misses.
- **Uptick**: campaign-window qty/revenue for the picked product(s) vs. a trailing
  `baseline_months` average ending the day before the window starts; `None` (not a
  crash or a bogus number) when the baseline period has zero sales — hit for real on a
  genuine data gap in testing and confirmed correct, not a bug.
- **Performance**: the resolver is O(rows) via pre-grouping by customer, not the O(rows ×
  customers) version first written (which never finished on ~10k customers / 1M+ events
  combined — see git history if resurrecting an older approach). Runs in ~15-20s over
  full 100001+100000 history; `views/commission_campaigns_view.py` caches it
  (`st.cache_data(ttl=3600)`), shared across every campaign view rather than recomputed
  per campaign.
- **Two real bugs fixed**: (1) `core/db.py::get_data` never commits (it's the SELECT-only
  path) — using it for `INSERT ... RETURNING id` silently lost the insert. Added
  `core/db.py::execute_write_returning` (commits AND fetches the RETURNING row) for this
  and any future write-that-needs-its-id case. (2) A missing per-product cap rendered as
  the literal string `"None"` instead of `.style.format(na_rep=...)`'s intended `"—"` —
  `st.dataframe` doesn't reliably respect Styler `na_rep` here (same class of issue as
  CLAUDE.md's documented TOTAL-row/Styler pitfall) — fixed by pre-formatting the Cap
  column to a display string before it reaches `.style.format()`.
- **Not yet built**: B.2 and B.5 still don't exist (B.2 is fully specified and a natural
  next pick; B.5 needs design work first, see below).

#### Payout-group derivation replaced the manual roster (2026-09-20, same-day follow-up)

The roster was originally shipped as a hand-maintained `data/commission_payout_groups.json`
(admin adds/removes salesmen per group in-app, `views/commission_campaigns_view.py`'s
`_render_roster_editor`) — described above and still visible in git history. Same day, the
user asked to check whether `cacus` already distinguishes a Dhaka/District-style zone per
customer, so group membership could be derived instead of hand-maintained. It does
(`cacus.xstate`, see above) — this was implemented immediately and the manual roster
mechanism (JSON file, CRUD helpers, editor UI) was **removed outright**, not kept as a
fallback: `build_area_group_map`/`derive_spid_group_map` replaced it end to end, and
`views/commission_campaigns_view.py::_render_derived_groups` replaced the editor with a
read-only view (group list + which salesmen resolved, with an explanatory message when
none do). `compute_campaign_payout`'s signature changed from taking a
`payout_groups_roster: dict` (`{group: [spid,...]}`) to a pre-resolved
`spid_group_map: dict` (`{spid: group}`) directly — the DO eligibility/exclusion logic
itself (`spid_group.get(r.spid)`, deadline check, `"salesman not in any payout group"`
reason) is completely unchanged, only where that dict comes from changed.

**Verified**: `build_area_group_map()` against real local Postgres returns the 8 expected
groups (Sylhet Retail correctly folds into District — confirmed via `habiganj -> District`,
Habiganj being the only tested area in that bucket); `derive_spid_group_map`'s majority-vote
and tie-break logic verified against 5 synthetic salesmen (clean majority, an exact tie
broken by first-listed area, a blank area, an unresolvable area name — all 5 behaved as
intended); live end-to-end in the browser — the new "🗺️ Payout Groups — derived from PRMST
area" expander renders the 8 real groups and the correct "no salesman resolves yet"
message, the create-campaign form's per-group deadline inputs source from the same derived
list, and a real campaign was created and its payout computed with an empty
`spid_group_map` (0 salesmen currently resolve, since `xdisease` is still mid-rollout) with
no crash — gate/payout/uptick all rendered correctly at zero. The DO-exclusion-reasons path
itself wasn't re-exercised with a real historical window in this pass (unchanged code,
already verified in the original build above) — flagged only for completeness, not treated
as a gap.

#### For future note (2026-09-19/20) — recipient type across the rest of A/B

Not yet relevant to anything built, but worth knowing before scoping A.1/A.2/A.4/B.2/B.5:
the commission for a given section can go to a **salesman**, a **customer**, or either,
depending on the section — confirmed by the user while starting B.1/3/4, as general
guidance for the whole feature, not specific to this one campaign type:

| Section | Can be paid to |
|---|---|
| A.1 Best Performer | Salesman **or** Customer |
| A.2 Highest Product Sales | Salesman **or** Customer |
| A.4 Best App User | Salesman **only** |
| B.1/3/4 (this section) | Salesman **or** Customer — built |
| B.2 Individual Target Achievement | Salesman **or** Customer |
| B.5 New Customer Creation | Salesman **only** (it's about who landed the customer) |

Don't assume every future section needs a `recipient_type` toggle just because B.1/3/4
has one — confirm scope with the user when each is actually picked up, same as everything
else in this doc.

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

Status: **built 2026-09-19** (`views/commissions.py` / `processing/commissions.py`, branch
`claude/commission-tracking-md-review-84bee6`), the redesigned Before/After version below —
verified end-to-end against real Postgres (add/remove/save round-tripped correctly, the
cutoff-date boundary was checked against real sales rows, and the 6-month clamp was
exercised live in the UI). An earlier plain-MTD version existed briefly (one shared
"current month" window for the whole list) but was fully replaced by this before it was
ever committed — no migration/cleanup needed.

- A JSON watchlist, admin-edited, **1–20 item codes** (more "wouldn't make sense" per the
  user) — same convention as `data/targets.json`/`data/public_holidays.json`/
  `data/warehouse_filters.json` already use in this app (small human-curated config file,
  not a DB table).
- **Before/After comparison, per product:**
  - The user picks a **cutoff date** (≤ today).
  - **After window** = `[cutoff, today]` (inclusive of both ends).
  - **Before window** = `[back_to, cutoff)` — the cutoff date itself belongs to After only,
    so the two windows never overlap or double-count it. `back_to` can be **at most 6
    months before the cutoff** (`processing/commissions.py::min_back_to`).
  - **Each watched product has its own cutoff + back_to** — not one shared pair for the
    whole watchlist. Rendered as **one separate table per product** (own Before row, own
    After row, plus a Change row = After − Before), not a combined multi-product grid.
- **Persistence**: per product, only the item code + its own `cutoff`/`back_to` dates are
  saved — nothing computed. Shape: `data/commission_product_watchlist.json` →
  `{zid: {itemcode: {"cutoff": "YYYY-MM-DD", "back_to": "YYYY-MM-DD"}}}`. All metrics are
  still recomputed live off those saved dates on every view (`build_product_comparisons`),
  same "recompute, don't store" pattern as the rest of this app.
- **Tracks both sales AND returns**, per window — qty sold, sales revenue, qty returned,
  net qty, sourced from `sales_daily_item`/`returns_daily_item` (`mv_sales_daily_item` /
  `mv_returns_daily_item`), joined to `final_items_view` for name/group/current stock
  (current stock shown as point-in-time context above each product's table, not per-window).
- **UI**: add-product row (product picker + Cutoff + Back To date inputs + Add) followed by
  one edit block per tracked item (its own Cutoff/Back To date inputs + Save + Remove) in
  the "⚙️ Manage Watchlist" expander (admin-only, same as before); the date inputs
  themselves enforce `back_to < cutoff` and the 6-month cap via `min_value`/`max_value`, so
  an invalid combination can't really be entered, not just rejected after the fact.
- **Real bugs hit and fixed while building this** (worth knowing before touching this code
  again): (1) `st.session_state[key] = ...` cannot be assigned in the same script run where
  that widget was already instantiated — resetting the "Add a product" picker after a
  successful add has to happen via a pending-reset flag checked at the *top* of the next
  run, not immediately after the add. (2) A widget's persisted `session_state` value can
  fall outside newly-computed `min_value`/`max_value` bounds when a dependent widget (the
  cutoff date) changes on the same rerun — pre-clamp the stored value before instantiating
  the dependent `Back To` widget, and skip passing `value=` on the run where a clamp just
  happened (otherwise Streamlit raises, or warns about a value set two ways at once).

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

Product Tracking's redesign open items are resolved — see §Product Tracking, now marked built.
