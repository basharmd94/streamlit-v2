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
| A.1 Best Performer | ✅ **built 2026-09-20** | none — see §A.1 for what shipped |
| A.1b Best Performer (Customers) | ✅ **built 2026-09-20** | none — see §A.1b for what shipped. **Not in the original design scope** — added same day, a customer-side sibling to A.1 requested directly |
| A.2 Highest Product Sales | ✅ **built 2026-09-20** | none — see §A.2 for what shipped |
| A.3 Best Disciplined | ✅ n/a | excluded by design |
| A.4 App Usage Commission | ✅ **built 2026-09-21** | none — see §A.4 for what shipped (renamed from "Best App User" — threshold bonus, not a ranking) |
| B.1/3/4 core (rate/cap/FIFO/gate payout math) | ✅ **built 2026-09-19** | none — verified against real Postgres (see §B.1/3/4) |
| B.1/3/4 sales-uptick companion metric | ✅ **built 2026-09-19** | baseline window is a user-adjustable parameter (3mo/6mo), as resolved 2026-09-17 |
| B.2 Individual Target Achievement | ✅ buildable | one unconfirmed assumption (gate applies here too, §The 100001/100000 gate) — doesn't block starting |
| B.5 New Customer Creation | ⚠️ not scoped | needs real design work (not a blocker exactly — nobody's promised an answer, it just hasn't been designed yet) |
| Product Tracking | ✅ **built 2026-09-19** (redesigned version) | none — see §Product Tracking for what shipped |
| The 100001/100000 gate itself | ✅ buildable | none |

**If asked "where do you want to start building," don't re-ask this as an open question —
the table above already answers it.** Product Tracking, A.1, A.1b, A.2, A.4, and B.1/3/4
are all built now (2026-09-19/21) — B.2 is the best remaining self-contained starting point
with zero open items.

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

Status: **built 2026-09-20** (`processing/salesman_score.py::build_pooled_monthly_scores` +
`processing/commission_campaigns.py::compute_best_performer_ranking` +
`views/commission_best_performer_view.py`) — verified against real Postgres.

- Scope: **100001 + 100000 combined**, same pooling every other commission section uses.
- **Confirmed 2026-09-20: calls the EXISTING Salesman Score engine exactly as-is, not the
  "proposed" recalibrated weights above** — explicit instruction: "the weights of sales and
  collections and returns and the negatives will be exactly the same as the engine that is
  built." `build_pooled_monthly_scores` is a new, ZID-agnostic extraction of
  `views/salesman_score.py`'s own row-building pipeline (the shipped, view-embedded version
  is untouched, avoiding regression risk) that calls `compute_salesman_scores` internally —
  so the actual shipped weights are `+45 target, +45 collection, +5 products, +5 customers,
  -6 returns, -12 balance(2mo), -2 balance(this month)` (today's real engine, NOT the
  `-20/-20` proposal above, which was never built). Not admin-adjustable — same fixed
  formula as the Salesman Score tab.
- **Averaging period, confirmed 2026-09-20, resolving the "not yet decided" open point
  above**: admin picks **1-3 months to average**. A salesman's `avg_score` is the mean of
  their score across only the months they actually have a row in (no phantom 0 for a
  month/year with zero activity). This settles the "per-month average vs. pooled totals"
  question in favor of per-month averaging.
- **Reporting month, pinned at setup — corrected 2026-09-20 across two rounds of user
  feedback, same day, right after first shipping this.** Round 1 (a real bug): the window
  was anchored to wall-clock "today" and recomputed live on every view — so a campaign set
  up in September would silently show September+October the moment October began, instead
  of staying on September's own completed results. The user's own words, catching it:
  "once the month is over, I will need the performance of the last month... do you
  understand what I mean?" Round 2 (scope correction, same day): the fix's first draft let
  the reporting month be any of the past 12 months too ("retroactive setup") — the user
  corrected this immediately: **"I can only set this current month and future months... I
  can't set it for a month that already passed."** So: the admin picks an explicit
  **Reporting month** at setup — current month, or any FUTURE month, never a past one (one
  narrow exception: editing an already-existing campaign whose own reporting month has
  since closed keeps that one real month available, so Save can't silently reassign it —
  no OTHER past month is ever offered). `window_start`/`window_end` become the campaign's
  real, load-bearing evaluation bounds (no longer "informational only"), and every later
  view re-derives the N-month list from the STORED reporting month, never from `today`.
  While the reporting month is still genuinely current (or hasn't arrived yet), results
  track live/show zero as appropriate; once it closes, they freeze at that month's final
  numbers forever, however long the campaign is left open before being checked again.
- **Payout mechanism — same per-rank-payout pattern as A.2, not a single fixed amount to
  one winner** (the "fixed BDT amount to the winner" line above was superseded once A.2's
  own ranked-payout mechanism was confirmed and reused here): admin picks **2-10 winners**
  (a wider range than A.2's 2-5) and enters one payout amount per rank position, ranked by
  average score descending.
- **Salesman-only, no `recipient_type` toggle** — a deliberate deviation from the "for
  future note" table further down (§For future note): the score's own inputs (target
  achievement, collection %, AR balance) don't have a customer-equivalent meaning.
- **Reuses the SAME `commission_campaigns` table as B.1/3/4 and A.2 — a 3rd
  `campaign_type` value, `"Best Performer"`** (same "no separate table per campaign type"
  pattern as A.2). `product_rates` (JSONB) holds `{"num_months": M, "num_winners": N,
  "payouts_by_rank": [amt1, ...]}`. Verified A.1's `campaign_type` doesn't leak into A.2's
  or B.1/3/4's existing picker filters and vice versa, same check done for A.2.
- **The 100001/100000 target gate applies here too** (explicit ask, "make sure one hundred
  thousand one and one hundred thousand is applied here as well. The gate") —
  `compute_best_performer_ranking` calls the same `check_gate` B.1/3/4 uses, over the full
  N-month evaluation span; either ZID missing target zeroes `total_payout` but the
  computed ranking still shows, same convention as every other gated section.
- **Same setup/results split as the rest of the feature**: admin Commissions page shows
  only Create/Edit/Delete, Target Management's "💰 Commission Results" shows only the gate
  banner + metrics + one ranking table (Rank/Salesman Code/Salesman/Avg Score/Months
  Scored/Payout, `—` for non-winners).
- **Known pre-existing data quirk, not introduced by this feature**: at least one real
  `spid` (`SA--000290`) maps to two different salesman names in the raw sales data (a
  reused ERP employee code) — the existing, shipped Salesman Score tab already builds one
  row per `(spid, spname)` pair and would show this as two rows too. Here it surfaces as a
  `months_scored` count that can exceed `num_months` for that one spid. Deliberately not
  deduped, since doing so would diverge from "exactly the same engine."
- **Verified** against real Postgres: a standalone script confirmed `build_pooled_monthly_scores`
  + `compute_best_performer_ranking` reproduce the correct averaged ranking for a real
  2-month window (Total Payout ৳9,500 exactly matching 3 configured rank amounts, gate
  passed against real ZID sales totals); the campaign_type filter isolation re-verified
  with a live create/list/delete round trip; live in the browser as a real admin — created
  a real 2-month/3-winner campaign, confirmed the setup page shows no results, confirmed
  Target Management's Commission Results computed the identical ranking/payout/gate the
  standalone script found.
- **Reporting-month pinning fix separately verified**: a standalone script confirmed the
  window/month derivation for a current-month anchor, a past-month anchor, and a January
  anchor crossing a year boundary; a direct regression test built the same past-anchored
  campaign twice (simulating opening it "now" vs. "again later") and got a byte-identical
  ranking both times. Live in the browser: created a real campaign anchored to a **closed**
  month (August, with today in September) — setup correctly showed "...ending August 2026
  (closed — final numbers)", and Target Management computed a ranking exactly matching the
  standalone script's own past-month-anchor result (same top 3, same scores, ৳7,500 total).

### A.1b Best Performer (Customers)

**Not in the original design scope — added 2026-09-20, same day as A.1**, requested
directly as a customer-side sibling: "same logic, everything [as the salesman version]...
but this would follow the customer scoring that is already in marketing analysis." Status:
**built 2026-09-20** (`processing/commission_campaigns.py::compute_customer_best_performer_ranking`
+ `views/commission_customer_best_performer_view.py`) — verified against real Postgres.

- Ranks CUSTOMERS by their **existing Customer Score** —
  `processing/marketing.py::build_customer_marketing_table`'s own `composite_score`, the
  EXACT same engine Marketing Analysis's own 📊 Customer Scoring already uses, called
  unmodified (weights: 25% total sales, 20% monthly activity rate, 15% YoY sales growth,
  15% avg days to collection (lower better), 10% total collection, 10% avg order interval
  (lower better), 5% YoY collection growth — all peer-relative, min-max scaled within
  whichever customer population is pulled). **2-100 winners** (explicit ask — a much wider
  range than A.1's 2-10 or A.2's 2-5).
- **Pays out by RANK-BAND TIER, not a distinct amount per individual rank — corrected
  2026-09-20, same day, right after first shipping this with per-rank amounts.** With up
  to 100 winners, entering a separate BDT figure for every single rank was "hectic"
  (explicit ask). The admin now defines up to 10 rank BANDS instead (`start_rank..end_rank`
  inclusive — a band can be a single rank, e.g. `1-1`, for an individually-differentiated
  top prize, or a wide range, e.g. `11-50`, sharing one reward). **Each tier can also carry
  a physical GIFT, not just cash** — explicit ask: "it doesn't necessarily have to be a
  cash prize... a mug or a pen or a cap." A tier can have a BDT amount, a gift, or both.
  Mirrors the already-shipped **Quantity-Tier Revenue Simulation** feature elsewhere in
  this app (Sales Analysis → Order Analytics → Product Orders) — same "admin-defined tier
  slots, leave a slot at 0 to disable it" pattern, reused rather than inventing a new one.
  `product_rates["tiers"]` = `[{"start_rank", "end_rank", "amount", "gift"}, ...]`.
  Overlapping tiers and tiers with neither an amount nor a gift are rejected at save time
  (`views/commission_customer_best_performer_view.py::_validate_tiers`). The results table
  gets a new **Gift** column alongside Payout (BDT); `total_payout` sums only the BDT side
  (gifts aren't priced).
- **ZID pooling — corrected 2026-09-20, same day, right after first shipping this as
  single-ZID-only.** Original assumption: a customer code is only unique WITHIN one ZID
  (unlike a salesman, who genuinely works both), so never pool. **Wrong for 100001/100000
  specifically** — the user ran a real audit: out of ~10,000 customers, 100001 (HMBR) and
  100000 (GI Corporation) share the SAME `cacus` customer code 99.9% of the time (~10
  mismatches, none of them current/active customers). Business reason, the user's own
  words: "when a salesman goes to a customer, the customer doesn't understand the
  difference between 100,000 and 100,001 — they just buy products from us directly." So
  100001+100000 ARE pooled together for scoring, same as A.1 (salesmen) and B.1/3/4 already
  pool them, for the same underlying reason (one shared field sales team/customer
  relationship). **100005 (Zepto) stays separate** — a genuinely independent consumer
  brand, no shared codes, explicitly confirmed to stay its own group.
  `views/commission_customer_best_performer_view.py::_ZID_GROUPS` is the single source of
  truth for this grouping. The campaign's own ZID GROUP (a list, not a single zid) is
  chosen at setup (whichever business is active then) and PINNED in `product_rates` — same
  "pin at setup, don't let it drift" discipline as A.1's reporting month. Verified via a
  standalone script: 2,675 real customers genuinely appear in both 100001 and 100000
  individually, and a pooled score's `total_sales` for a sample customer exactly equals
  the sum of both businesses' individual totals (760,432 = 588,240 + 172,192).
- **A reporting YEAR, not a reporting month — no month-averaging, a second deliberate
  departure from A.1, confirmed with the user before building** (via AskUserQuestion —
  architecturally different enough from A.1 that guessing wrong would have meant a redo).
  `build_customer_marketing_table` only filters/aggregates by calendar YEAR (confirmed by
  reading `_sales_metrics`/`_collection_metrics`: `monthly_activity_rate` divides by
  `len(years) * 12`, assuming full years; YoY growth needs full-year comparisons) — there's
  no clean way to cap it to a partial month without changing its own internal logic, which
  "follow the customer scoring that is already in marketing analysis" explicitly meant NOT
  to do (reuse as-is, don't reimplement). So the admin picks a **Reporting year** instead —
  current year or later only, same no-retroactive-setup rule as A.1's reporting month
  (confirmed with the user for A.1: "I can't set it for a [period] that already passed"),
  applied here at the year level. One narrow exception in the edit form, same pattern as
  A.1: an already-existing campaign whose own reporting year has since closed keeps that
  ONE real year available/selected (so Save can't silently reassign it), without offering
  any other past year as a fresh choice.
- **No 100001/100000 target gate** — that check is specific to the shared sales team
  between those two ZIDs; it doesn't generalize to a single-ZID mechanism that can also be
  used for 100005 (no shared team at all). `total_payout` is always the raw computed
  figure, never zeroed.
- **Same setup/results split as the rest of the feature**: admin Commissions page shows
  only Create/Edit/Delete, Target Management's "💰 Commission Results" shows only 2 metrics
  + one ranking table (Rank/Customer Code/Customer/Score/Payout, `—` for non-winners). The
  campaign picker lists every Customer Best Performer campaign regardless of which
  business group is currently active in the sidebar (each campaign carries its own pinned
  zid group) — only CREATING a new one is scoped to the currently-active business group.
- **Verified** against real Postgres: a standalone script scored real 100001 customers
  (3,610 scored, top-ranked `Rahima Enterprise` at 76.6 single-ZID / 79.2 pooled) and
  confirmed a 5-winner ranking's total payout exactly matched the 5 configured rank
  amounts; a second script confirmed campaign_type isolation (invisible to A.1/A.2/B.1/3/4's
  pickers) via a live create/list/delete round trip, and exercised the 100-winner edge
  case (100 winners × ৳100 = exactly ৳10,000 total, matching
  `min(100, scored_customers) × 100`); the past-year edit exception confirmed via script;
  a separate script confirmed real 100001+100000 pooling (2,675 overlapping customers,
  pooled totals exactly equal to the sum of both businesses' individual totals) and that
  Zepto (100005) stays unpooled. Live in the browser as a real admin — created a real
  3-winner campaign while GI Corporation (100000) was active, confirmed it correctly
  resolved to "HMBR + GI Corporation (pooled)", confirmed the setup page shows no results,
  and confirmed Target Management's Commission Results computed the identical pooled
  ranking (real customers, real scores) with the correct ৳6,000 total payout.
- **Tier-payout redesign separately verified**: `_validate_tiers` confirmed to accept a
  valid 6-tier config, reject overlapping tiers, reject a tier with neither an amount nor
  a gift, and reject an empty tier list; the engine test (a 6-tier config spanning ranks
  1-100 against real pooled 100001+100000 scores) confirmed the exact expected total
  payout (৳41,000) and the exact expected payout/gift at every tier boundary (rank 1 =
  ৳10,000 alone, ranks 3-5 = ৳3,000 + "Mug" each, rank 11 = "Cap" only with ৳0 payout, rank
  101 beyond num_winners = nothing). Live in the browser: created a real 2-tier campaign
  (Rank 1 = ৳5,000 alone; Ranks 2-5 = "Mug" only) — the setup summary correctly rendered
  "Rank 1: ৳5,000 · Rank 2-5: Mug", and Target Management's results table showed the
  correct split across a new Gift column, with Total Payout correctly counting only the
  ৳5,000 cash tier (the gift-only tier contributes nothing to the BDT total, by design).

### A.2 Highest Product Sales

Status: **built 2026-09-20** (`processing/commission_campaigns.py::compute_highest_product_sales_ranking`
+ `views/commission_rankings_view.py`) — verified against real Postgres.

- Ranked by **distinct product count** (breadth, not volume) per recipient (salesman or
  customer), within an admin-set window (defaults to the current calendar month, but not
  hardcoded — an admin can run it for any window, same as B.1/3/4).
- **Payout mechanism confirmed 2026-09-20**, resolving the one thing the original doc left
  unstated: "Yes, both of these are ranked... the number of people who get the prize [is
  variable]... up till five... this could be two till five... the prizes will be given out
  on a ranked basis... I can enter the amount that I want to give out... for the first,
  second, third, based on the number of people I want to give it out to." So: the admin
  picks **how many winners (2-5)**, then enters **one payout amount per rank position**
  (1st, 2nd, ... up to that many) — the app ranks recipients by distinct product count and
  assigns each rank's own amount to whoever lands there. Not a % or proportional payout,
  and not a single flat amount to everyone in the top N — each rank has its own figure.
- **Reuses the SAME `commission_campaigns` table as B.1/3/4 — explicit ask: "I don't want
  to create a different table for every campaign... designate data to the right columns if
  it's possible. If not, just use the JSONB column that's there."** `campaign_name`,
  `recipient_type`, `window_start`/`window_end` are reused with their existing meaning;
  `product_rates` (JSONB) is repurposed to hold `{"num_winners": N, "payouts_by_rank":
  [amt1, amt2, ...]}` instead of per-product rates; `cap`/`payout_groups`/
  `uptick_baseline_months` are unused for these rows. `campaign_type` (documented
  elsewhere as "just a UI category" for B.1/3/4's own 3 values) now also doubles as the
  mechanism discriminator — `"Highest Product Sales"` for these rows — so each section's
  own campaign picker filters to its own `campaign_type` value(s) and never shows the
  other's rows. **A real bug caught during this same build**: B.1/3/4's own picker
  (`commission_campaigns_view.py::render`) wasn't filtering by `campaign_type` at all
  before this — a ranking campaign would have shown up in its dropdown and likely crashed
  its Edit form (which assumes `product_rates` is `{itemcode: rate}`, not
  `{num_winners, payouts_by_rank}`). Fixed in the same pass.
- **Same setup/results split as the rest of the feature**: admin Commissions page shows
  only Create/Edit/Delete, Target Management's "💰 Commission Results" shows only the
  ranking table — one table, every recipient with sales activity in the window (not just
  the winners), columns Rank / Code / Name / Distinct Products / Total Qty / Payout (BDT,
  `—` for non-winners) — same "show everyone with an outcome column" pattern as B.1/3/4's
  `line_items` table.
- Ties (identical distinct-product-count at a cutoff rank) are broken by total quantity
  sold, then by recipient code — not an explicit rule from the user, flagged as an
  assumption in the code, not confirmed.
- **Verified** against real Postgres: a synthetic tie-break test (two salesmen tied on
  distinct-product-count, correctly ranked by quantity as the tiebreak); the
  `commission_campaigns`-sharing fix (created one B.1/3/4 row and one A.2 row in the same
  table, confirmed each section's picker only ever sees its own); live in the browser
  against real sales data (2024, a wide historical window) — Total Payout ৳9,000 (5,000 +
  3,000 + 1,000) exactly matched the 3 configured rank amounts, and the full ranking table
  correctly showed real salesmen (Md. Suruj Mia #1 at 211 distinct products, etc.) with the
  correct rank order, only the top 3 carrying a payout, everyone else showing `—`.

### A.3 Best Disciplined Salesperson
- **Excluded** — not trackable from any DB this app touches (selfie/location/dress
  code/response-time compliance has no data source). User's own call, from the source doc.

### A.4 App Usage Commission

Status: **built 2026-09-21** (`processing/commission_campaigns.py::compute_app_usage_scores` +
`compute_app_usage_bonus` + `views/commission_app_usage_view.py`) — verified against real Postgres.
Renamed from "Best App User" to "App Usage Commission" to match the shipped mechanism (a threshold
bonus, not a ranking/"best" contest).

Resolves the "STILL OPEN" status above: the user supplied the real mobile ERP API's own DB impact
map (`mobile_order_api_data_map.md`/`.json`, traced from source and verified column-for-column
against live Postgres) and, after a discussion grounded in real data, specified exactly what to
score. **Not the earlier guessed list** (`glpmt`/`opcrn`/`feedback`/`opdor` promise entries) — the
real design uses `opmob` (orders + GPS), `opcrn` (returns), `opdor.xdatepay` (promised payment), and
`glpmt` (collections); `feedback` was not part of the final ask.

- **5 weighted components, a composite 0-100 score per salesman** (confirmed weights: 4% + 24%×4 =
  100%, the user's own numbers):
  - **4% Orders** — order count this month (`opmob`, grouped by `invoiceno`+`invoicesl`, NOT
    `xordernum` which is NULL on fresh mobile orders per the data map), peer-relative, higher better.
  - **24% Location** — 50% GPS fill rate (`opmob.xlat`/`xlong` present) + 50% GPS **distinctness**
    rate (distinct coordinates ÷ GPS-tagged orders) — the distinctness half is what actually catches
    "always logging the same fake spot," confirmed as a real, detectable pattern against live data
    before building (one real salesman: 133 distinct coordinates out of 3,632 GPS-tagged orders,
    ~3.7%, vs. a healthy peer at ~31%).
  - **24% Return hygiene** — % of the salesman's own returns (`opcrn.xemp`) NOT still stuck
    `"1-Open"` more than a 14-day grace period, measured against real *today* (not the reporting
    month's own end, so a return opened near month-end still gets a fair grace period). Confirmed
    against live data this is a weak/near-universal signal on its own (~99.9% of ALL returns
    eventually reach `"3-Issued"` regardless of salesman) — kept anyway per the user's own
    weighting, since it still penalizes genuinely-stuck outliers.
  - **24% Promised payment entry** — % of the salesman's delivery orders (`opdor.xsp`) this month
    with `xdatepay` filled in. Confirmed near-zero adoption in real data (334 of 594,723 `opdor` rows
    ever have it set; ~0% for the top-volume 2026 salesmen) — the user confirmed this reflects real
    (not a local-mirror rollout gap) low usage, so this component rewards genuine early adopters
    from a near-0 baseline rather than penalizing everyone equally.
  - **24% Collections** — count of `glpmt` entries (`xemp`) this month, peer-relative. Same
    near-zero-adoption confirmation (9 total `glpmt` rows locally, ever).
  - A salesman with zero *eligible* returns/delivery-orders that specific month (nothing to
    evaluate for just that one component) gets a **neutral 50** on that component, not punished
    with a 0 or rewarded with a 100.
- **Payout is a THRESHOLD BONUS, not a ranking — a genuinely different mechanism from A.1/A.1b/A.2's
  rank-based payouts, closer to B.2's own (not yet built) "clear your own bar" shape.** Explicit ask:
  "create a score from 1 to 100, whoever scores more than 90 gets a fixed commission." Admin sets a
  `threshold` (defaults to 90) and one flat `bonus_amount` (BDT) — every salesman scoring ABOVE the
  threshold gets that same amount; no gate, `total_payout = qualified_count × bonus_amount`.
- **Salesman population = every spid appearing in `opmob` that reporting month** (placed at least
  one real order via the app) — a salesman with zero app orders that month isn't scored at all,
  matching the feature's own premise ("an incentive to all who actively used the app").
- **Reporting month, current-or-future only, pinned at setup — reuses A.1's exact rule and
  reasoning**, not re-litigated: single month only (no multi-month averaging — this is a monthly
  compliance bonus, not a ranking that benefits from smoothing). Live while the reporting month is
  ongoing, frozen once it closes.
- **Pooled 100001+100000, NOT per-ZID-group like A.1b** — this is a salesman behavior metric (like
  A.1), not a per-business customer metric. Confirmed against real `opmob` data before choosing
  scope: 346,606 orders/67 salesmen on 100001, 31,994/38 on 100000, vs. only 63 orders/14 users on
  100005 (Zepto) — negligible, excluded, same as A.1's own scope.
- Three new raw queries (`core/queries.py::get_app_usage_orders`/`get_app_usage_returns`/
  `get_app_usage_delivery_orders`, registered in `core/analytics.py`) — none of the existing
  `opmob`/`opcrn`/`opdor` Analytics entries exposed what this needed (no `xlat`/`xlong`, unsafe
  `xordernum` grouping, or an open-only `xstatuscrn` filter that excludes the very rows needed to
  compute a stuck-rate).
- **Verified** against real Postgres: engine tested against real August 2026 data (31 scored
  salesmen, real score spread 33.5–72.5, zero qualifying at a 90 threshold — a realistic outcome
  given real adoption levels, not a bug); campaign_type isolation confirmed via a live create/list/
  delete round trip (invisible to A.1/A.1b/A.2/B.1/3/4's own pickers). Live in the browser as a real
  admin — created a real September 2026 (current month) campaign at threshold 20, confirmed the
  setup page shows no results, and confirmed Target Management's Commission Results computed 26
  real scored salesmen with the correct ৳39,000 total payout (26 × ৳1,500), full per-component
  breakdown rendering correctly (GPS Fill %, GPS Distinct %, Return Stuck %, Promised Pay %,
  Collections Logged, Qualified, Payout).
- **Noted for later, not built**: a "Number of Unique Customers" ranking, the same shape as A.2 but
  ranking recipients by distinct CUSTOMER count instead of distinct product count.

---

## B. Commission Campaigns (the payout side — this is where the one campaign-definition table lives)

### B.1 + B.3 + B.4 — Product-Specific / Stock Clearance / Slow-Moving (one mechanism)
These three items in the source doc are mechanically identical per the user's own note on
B.4 ("same as product incentives, nothing different") — one feature, not three:

- Pick product(s), set a **rate per unit sold**, an optional **cap**.
- **Rate varies PER PRODUCT; cap is ONE value for the whole campaign — confirmed
  2026-09-20, in two steps.** First correction: rate is the per-unit incentive/discount
  BDT amount being offered on that product (e.g. 5 or 3), NOT the product's sales price —
  this was explicitly ambiguous and the user resolved it directly: "the commissions for
  these I will set the discount rate so BDT 5 or BDT 3 etc and per product," and at the
  same time both rate AND cap were made to vary per product (corrected from an earlier
  single-campaign-wide framing). **Second, same-day correction**: the per-product cap was
  a misunderstanding on top of the first correction — the user clarified directly: "the
  cap is one cap that applies to all salesman. Not per product. I think I did not explain
  this properly before." The cap is ONE ceiling for the whole campaign, capping a single
  recipient's (salesman or customer, per recipient_type below) TOTAL payout summed across
  every product in the campaign — not a separate ceiling per product line.
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
    product_rates     JSONB NOT NULL,     -- {"ITEMCODE1": 5.0, "ITEMCODE2": 3.0}
                                           -- rate = per-unit INCENTIVE/discount amount
                                           -- (e.g. 5 or 3 BDT), NOT the product's sales
                                           -- price -- varies per product, cap does not
                                           -- (see `cap` column below)
    cap               NUMERIC(18,2),      -- ONE ceiling for the whole campaign -- caps a
                                           -- single recipient's TOTAL payout, summed
                                           -- across every product. NULL = no cap.
    window_start      DATE NOT NULL,      -- sales window: which DOs are even eligible
    window_end        DATE NOT NULL,
    payout_groups     JSONB NOT NULL,     -- {"Dhaka retail": "2026-10-15", "District": "2026-10-31"}
                                           -- group name -> THIS campaign's own collection
                                           -- deadline for that group. Salesman group
                                           -- membership is derived live, not stored here
                                           -- (see above). Always salesman-keyed regardless
                                           -- of recipient_type -- see note above.
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
- **Payout**: eligible DOs (in the campaign's sales window, salesman in a payout group,
  paid by that group's deadline) → sum qty of the picked product(s) on that DO → × that
  product's own rate → summed per recipient ACROSS every product → then the single
  campaign-wide cap (if set) is applied ONCE to that recipient's total (see the
  same-day cap redesign note below — this bullet already reflects the corrected,
  shipped behavior, not the original per-product-cap version). Verified against real
  data down to the individual line (22 units × ৳5 = ৳110, cross-checked against a
  standalone computation). **Verified the recipient-grouping choice is capping-neutral
  the way it should be**: summed raw (pre-cap) payout came out identical (312) whether
  grouped by salesman or by customer against the same real data — same underlying line
  items either way, only where the cap binds differs, correctly, by recipient
  granularity.
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

#### Cap redesign, Product Tracking prorated row, and a Target Management results view (2026-09-20)

Three follow-up asks in one message, after the user tried the shipped feature on the live
server:

1. **Cap corrected from per-product to campaign-wide.** The user's own words: "In
   product/stock clearance commissions, the cap is one cap that applies to all salesman.
   Not per product. I think I did not explain this properly before." `product_rates`
   collapsed from `{itemcode: {"rate", "cap"}}` to plain `{itemcode: rate}`, and a new
   top-level `cap` column on `commission_campaigns` holds ONE ceiling for the whole
   campaign. `compute_campaign_payout` now sums `quantity x rate` per product per
   recipient with no capping at that level, groups to `recipient_total` (raw, summed
   across every product), and applies the cap once there
   (`raw_total_payout.clip(upper=cap)`). The table had zero real rows on the live server
   (confirmed with the user before touching the schema) so this was a straight
   drop+recreate of `commission_campaigns`, not a migration. UI: the create-campaign form
   now has ONE "Cap (BDT, 0 = no cap)" input below the per-product rate rows (not one per
   product); the campaign detail view shows a "Raw Total (BDT)" column alongside "Payout
   (BDT)" in the per-recipient table only when a cap is actually set (otherwise the two
   would always be identical, which is just noise), and the per-product breakdown table
   dropped its "Cap"/"Capped Payout" columns entirely — a line there is just
   quantity × rate, uncapped, with a caption pointing at the campaign-wide cap shown
   above. **Verified**: a synthetic case (one salesman, two products, 150+90=240 raw)
   correctly capped to 100 at the recipient-total level, not per product; live in the
   browser, a real 2-product campaign with rate 5/3 and cap 100 rendered the single Cap
   input and the correct capped-total caption end to end.

2. **Product Tracking gained a prorated "Before Avg" row.** The user's own example: cutoff
   2026-09-13, back_to 2026-08-01 (Before spans ~6 weeks), today 2026-09-20 (After has
   only run ~1 week so far) — Before totaled 453 units, After only 61 so far. Comparing
   453 to 61 directly is misleading since the two windows are wildly different lengths (43
   days vs. 8); the user wants a fair baseline: "453/6 thats 75.5. Means this product lost
   sales over the last 7 days." `processing/commissions.py::build_product_comparisons` now
   computes `before_days = (cutoff - back_to).days`, `after_days = (as_of - cutoff).days +
   1`, and a new `before_avg_prorated` dict = `before_total × (after_days / before_days)`
   for every metric (qty sold, revenue, qty returned, net qty) — the exact-day-count
   version of the user's own hand-rounded-to-weeks approximation (453 × 8/43 = 84.28 here,
   vs. the user's manually-rounded 75.5; same direction, more precise). Rendered as a new
   row between Before and After (`_render_comparison_table` in `views/commissions.py`),
   and the **Change row's formula changed from `After − Before` to `After − Prorated
   Avg`** — comparing After against the raw (much longer) Before total would always make
   it look artificially small, the prorated row is the fair comparison. Both captions and
   the CSV export were updated to include the new row. **Verified**: a script reproducing
   the user's own numbers (453 before, cutoff/back_to as given) confirms before_days=43,
   after_days=8, prorated=84.28; live in the browser, a real watchlist product rendered
   all 4 rows (Before 25, Before Avg (prorated to 1d) 1, After 0, Change -1) correctly.

3. **A read-only "Commission Results" view in Target Management, for managers.** The
   user's own framing: "the admin will setup the commissions, But i would like a radio
   view within target management which just shows the result of whatever commission
   campaign my managers would like to see... no setup just result based on campaign
   choice," explicitly scoped to cover every section "from product tracker to new
   customer creation," present and future. Rather than duplicate the Commissions page's
   section list, `views/commissions.py::display_commissions_page` was refactored: the
   `_SECTIONS` dict became `_sections(read_only, key_suffix)` (a function, so every
   section renderer can be parameterized) behind a new shared
   `render_section_picker(zid, read_only, key_suffix)`. Every section renderer that has
   setup controls (`_render_product_tracking`'s watchlist editor,
   `commission_campaigns_view.render`'s roster/create/delete) now takes `read_only: bool`
   and hides those controls when `True` — **regardless of the viewer's own role**, not
   just because a non-admin already couldn't see them; this guarantees the Target
   Management mount point is never a setup surface even for an admin who opens it from
   there. `views/target_management.py` gained a new `"💰 Commission Results"` radio option
   calling `commissions.render_section_picker(zid, read_only=True, key_suffix="_tm")` —
   no new page_permissions grant needed, since Target Management access already covers
   every role that should see this. Also added while touching the shared detail
   renderer: a 🟢 Ongoing / 🔴 Closed status badge on each campaign (`window_end >= today`)
   — the user asked for "whether its on going or not" alongside the result, and this
   applies wherever `_render_campaign_detail` is used, not just the new view. **Verified**:
   logged in as a real non-admin role (`sales`, which already has Target Management
   access) and confirmed the new radio option shows both Product Tracking and the Stock
   Clearance campaign with zero setup controls (no Manage Watchlist expander, no Payout
   Groups/Create New Campaign expanders, no Delete button) while still showing full
   detail, status, and current result — same campaign, same data, as the admin-only
   Commissions page.

#### Edit Campaign + active-employee-only payout groups (2026-09-20, same-day follow-up)

The user (as admin) hit a real problem after using the merged feature: "I made a mistake
with the dates while setting up the campaign" — the only fix available was delete +
recreate, losing every other field entered. Explicit ask: "can you make an option of
editing and changing or deleting the campaign whichever you think is better." Since a
campaign is a pure definition with nothing pre-computed depending on it (payout is always
computed live from the row + current data), editing in place is strictly safer than
delete+recreate and was built as the primary fix, alongside keeping the existing Delete.

- **`processing/commission_campaigns.py::update_campaign`** — same field list as
  `create_campaign`, `UPDATE ... WHERE id = %s`, leaves `id`/`created_by`/`created_at`
  untouched.
- **`views/commission_campaigns_view.py`** — the Create form's body was factored out into
  a shared `_render_campaign_form(available_groups, items_df, existing=None)`: `existing`
  is `None` for Create (unchanged behavior) or a campaign dict for Edit (every field
  pre-filled — product multiselect defaults to the campaign's own products, rate/cap/dates
  default to its own values, per-group deadlines default to the campaign's own saved
  deadline where one exists, otherwise today). A new `_render_edit_campaign(campaign)`
  wraps it in a "✏️ Edit Campaign" expander next to the existing Delete button.
- **Real latent bug fixed in the process, not just for Edit**: the Sales window end
  `date_input`'s `min_value=window_start` could raise if a stored value fell below a newly
  moved `window_start` on a rerun — the exact same class of issue as the Product Tracking
  Cutoff/Back-To widget-clamp bug already documented above (CLAUDE.md's "two real
  Streamlit widget-state bugs" note). Fixed with the identical pattern (pre-clamp
  session_state before the dependent widget is instantiated,
  `_clamp_min_session_date`) — applies to both Create and Edit now, since they share the
  same form body.
- **Verified**: `update_campaign` against real local Postgres — created a campaign, updated
  every field (name, products+rate, cap, window dates, a payout-group deadline, baseline
  months, notes) in one call, confirmed every field round-tripped correctly and `id`
  stayed the same. Live in the browser: opened Edit on a real campaign and confirmed EVERY
  field pre-filled exactly right, including the one payout group that actually had a saved
  deadline (`Dhaka retail: 2026/02/01`) vs. every other group correctly defaulting to
  today (never touched before); changed the campaign name via the form and saved,
  confirmed the campaign list label and detail view picked up the new name immediately
  with the same id and unchanged dates. (Date-picker fields specifically couldn't be
  driven through the browser-automation tool used for this verification — same tool
  limitation hit earlier with Product Tracking's date inputs — so the actual date-field
  round-trip was verified via the direct script test instead, which is the more precise
  check anyway.)

**Second, unrelated fix in the same message**: "make sure the active emp status employees
are only taken into account." `derive_spid_group_map` (the live payout-group derivation
from `prmst.xdisease`, see above) was pulling every `prmst` row that passed the
`valid_spids` filter (real sales history) regardless of current employment status — a
since-resigned/terminated employee who made real sales while still employed would still
resolve to a payout group. Now also filters `prmst.xstatusemp == 'A-Active'`
(case-insensitive), checked in ADDITION to `valid_spids`, not instead of it —
`valid_spids` answers "is this really a salesman," this answers "are they still employed
today." **Verified** with 5 synthetic prmst rows (`A-Active`, `R-Resigned`, `T-Terminated`,
`H-Hold`, and a lowercase `a-active` to confirm case-insensitivity) — only the two active
ones resolved.

#### Reporting consolidated to 1-2 tables, setup and results split across two pages (2026-09-20)

A voice-transcribed follow-up message, several asks at once, all pointing at the same
underlying principle the user stated directly: "the admin will setup the commissions...
target management you can see the results... maximum one or two tables... consolidated in
a very direct kind of way."

**1. Setup and results are now two separate surfaces, never both on one page.** "My admin
commissions page where I edit, that place, whatever, let's just leave like the editing and
the stuff, no need to show anything else, honestly." The admin Commissions page now shows
ONLY setup controls (watchlist editor for Product Tracking; roster view, create form, edit
form, delete button for B.1/3/4) — it never computes a payout at all anymore, so opening a
campaign to fix a typo no longer pays the ~15-20s FIFO cost. Target Management's "💰
Commission Results" mode shows ONLY the results below — never setup. Implemented via a
`read_only: bool` threaded through every section renderer, gating setup controls
regardless of the viewer's own role (see CLAUDE.md's "Architecture: setup vs. results" for
the mechanics).

**2. B.1/3/4's own results collapsed to 2 tables.** Previously: a Per-Recipient Payout
table, a separate Per-Recipient-Per-Product breakdown expander, a separate Excluded-DOs
expander (two tables inside it), and bare Sales Uptick metric cards — five-ish separate
pieces. The user's own words: "I want to see one table... the salesman, the DO number, the
product code, the quantity sold... the fixed amount of commissions... and finally the
total amount paid... And I want to see what salesmen were removed and what not... in a
secondary table, how about we see the before after, like you do in product tracking."
- **Table 1** — `processing/commission_campaigns.py::compute_campaign_payout`'s new
  `line_items`: one row per (recipient, DO, product) for EVERY in-window DO carrying a
  picked product, eligible or excluded alike, with a `Status` column (Eligible, or the
  specific exclusion reason) right there in the same table — this is how "what salesmen
  were removed and what not" is answered, not a separate table. Columns match the user's
  own list exactly: Salesman (Code + Name), DO Number, Product Code, Qty, Rate, Payout
  (0 for excluded rows — never a phantom uncapped number), Status.
- **Table 2** — a new per-product Before/After table
  (`compute_campaign_product_before_after`), Before = the campaign's existing baseline
  window, After = the campaign's own sales window (capped to today if still ongoing). Uses
  the exact same rendering component as Product Tracking's own redesign (see below) — one
  table, one row per picked product, a Metric dropdown above it.
- The gate banner and 3 top-line metrics (Total Payout / Eligible DOs / Excluded DOs) are
  kept — not tables, just a compact status header above the two tables.

**3. Product Tracking collapsed the same way** — "the product tracking honestly can be the
same way. Instead of looking at five tables, I would like it to be in one table... the
before after can only be in the column header." One table, rows = every tracked product,
columns = Before / Before Avg (Prorated) / After / Change, for whichever metric is
currently picked.

**4. A Metric dropdown replaces always-showing every number at once** — "the total revenue
is not very necessary. You can have a filter... to see if you want to see returns, if you
want to see net revenue, or you want to see net units sold." Three choices only: Net Units
Sold, Returns, Net Revenue (gross Qty Sold / gross Sales Revenue dropped from the picker,
still computed internally since Net Revenue needs gross revenue as an input). **Net
Revenue is a genuinely new metric** — revenue net of the return's own BDT value, not just
net of return quantity — which required switching both Product Tracking's and the
campaign's own returns data source from the lightweight `returns_daily_item` MV
(quantity-only) to the full `"return"` Analytics table (`get_return_data`, has
`treturnamt`).

**Shared engine, not two parallel implementations**: `processing/commissions.py` gained
`_window_metrics` (the 5-metric bundle: qty sold, sales revenue, qty returned, net qty, net
revenue, for one window) and `_prorate_before` (the day-count proration), both used by
BOTH Product Tracking's `build_product_comparisons` and B.1/3/4's new
`compute_campaign_product_before_after` — one before/after "engine," two different
window-boundary sources. The actual table rendering is one shared component,
`views/commission_shared.py::render_before_after_table`, used by both
`views/commissions.py` (Product Tracking) and `views/commission_campaigns_view.py`
(the campaign's Table 2) — new file, specifically to avoid a circular import between those
two view modules (commissions.py already imports commission_campaigns_view, so the shared
piece couldn't live in either of them without creating a cycle).

**Verified** end-to-end against real Postgres, deliberately including a real
high-sales-volume item (not just a sparse test item) to exercise a real mix of
eligible/excluded rows: item `1281` (61k+ historical DOs) over a real 2024 window produced
1,399 rows in `line_items`, all correctly showing real salesman names, DO numbers,
quantities, and the exclusion reason `"Excluded: salesman not in any payout group"`
(expected — `derive_spid_group_map` still resolves nothing locally, per `xdisease`'s
ongoing live-server rollout, not a bug); the same campaign's Table 2 showed correct real
Before/Prorated/After figures. Separately confirmed live: the admin page's campaign detail
view loads without ever showing the "Computing payout..." spinner (the FIFO path is
provably skipped); the Target Management results view shows the full report with zero
setup controls anywhere; Product Tracking's consolidated table correctly switches between
all 3 metrics including real ৳ figures for Net Revenue.

#### For future note (2026-09-19/20) — recipient type across the rest of A/B

Not yet relevant to everything below, but worth knowing before scoping B.2/B.5: the
commission for a given section can go to a **salesman**, a **customer**, or either,
depending on the section — confirmed by the user while starting B.1/3/4, as general
guidance for the whole feature, not specific to this one campaign type:

| Section | Can be paid to |
|---|---|
| A.1 Best Performer | **Salesman only — built 2026-09-20**, a deliberate deviation from this table's original "Salesman or Customer" guess: the score engine's own inputs (target achievement, collection %, AR balance) don't have a customer-equivalent meaning |
| A.2 Highest Product Sales | Salesman **or** Customer — built |
| A.4 App Usage Commission | **Salesman only — built 2026-09-21**, confirmed by construction: every component (orders, GPS, returns, promised payment, collections) is inherently a salesman behavior, no customer-equivalent meaning |
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
    whole watchlist.
- **Reporting — one consolidated table across every tracked product, not one table per
  product (redesigned 2026-09-20).** Rows = products; columns = Before / Before Avg
  (Prorated) / After / Change, for ONE metric picked from a dropdown above the table (Net
  Units Sold / Returns / Net Revenue — gross Qty Sold/Sales Revenue are computed internally
  but not user-selectable). Rendered by the shared
  `views/commission_shared.py::render_before_after_table` component — also used by
  B.1/3/4's own campaign-product Before/After table, see §B.1/3/4's own redesign note. Net
  Revenue required switching the returns data source from the lightweight
  `returns_daily_item` MV (qty only) to the full `"return"` Analytics table (has the
  return's own BDT value, `treturnamt`).
- **Setup (watchlist editor) and results (the table above) are two separate surfaces** —
  the admin Commissions page shows only the editor, Target Management's "💰 Commission
  Results" mode shows only the table, never both on one page. See §B.1/3/4's redesign note
  for the shared `read_only` mechanism.
- **Persistence**: per product, only the item code + its own `cutoff`/`back_to` dates are
  saved — nothing computed. Shape: `data/commission_product_watchlist.json` →
  `{zid: {itemcode: {"cutoff": "YYYY-MM-DD", "back_to": "YYYY-MM-DD"}}}`. All metrics are
  still recomputed live off those saved dates on every view (`build_product_comparisons`),
  same "recompute, don't store" pattern as the rest of this app.
- **Tracks both sales AND returns**, per window — qty sold, sales revenue, qty returned,
  net qty, net revenue, sourced from `sales_daily_item` + the full `"return"` table, joined
  to `final_items_view` for name/group/current stock.
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

1. **B.2 / gate interaction** — assumed the 100001/100000 gate also applies to B.2, not
   explicitly confirmed. Doesn't block starting B.2.
2. **B.5** — no design work done yet beyond "existing logic doesn't fit, needs something
   new" — scope this properly with the user when it's picked up, don't guess at a shape.

A.4's own open item is resolved — see §A.4, now marked built.

Product Tracking's, A.1's, and A.2's own open items are all resolved — see their own
sections, now marked built.
