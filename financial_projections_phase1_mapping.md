
---

## Step 1b — May/June 2026 cross-check, GTC Admin + Motiur vs local GL (ZID 100001)

Done 2026-10-20 against the local `da` Postgres (`glheader` + `gldetail` + `glmst` + `glsub`),
period 5 and 6 of 2026. Sheet columns: May-26 = col AD, Jun-26 = col AE.

**How "a payment" was defined in the GL** (confirmed by reading every BPAY voucher line by line):
payment vouchers are `CPAY` (cash) and `BPAY` (bank). The *paid-for* side is every non-funding
debit line; the *funding* side is a credit to cash/bank (`01010001`, `01010002`, `01020001`)
**or straight to the Dhaka Bank CC overdraft `10010003`** — several BPAYs (e.g. `BPAY003180`,
`BPAY003182`) have no "Cash at Bank" line at all, they draw the CC directly. Funding total ==
paid-for total in both months (15,147,083.96 May / 24,996,421.36 Jun), so nothing leaks.
`BTJV`/`CWJV`/`BDJV` (bank-transfer / cash-withdrawal / bank-deposit journals) are internal
movements and are excluded; `BTJV` is also where CC interest is charged (Jun: `06300001`
1,198,398.08, non-cash).

**Whole-sheet tie-out:** sheet rows 24 + 28 (international + local) = 14,847,705 May /
24,795,483 Jun vs GL 15,147,084 / 24,996,421 → within 2.0% / 0.8%.

`Advance For International Purchase` (`01050001`) is the import pipeline and has a **sub-ledger
master in `glsub`** — this is what maps the sheet's import rows cleanly:
`0105000101/102` = TT (Kai Hui / Kungfu) · `…104` = L/C charge & commission · `…105/106` = L/C
margin (Kungfu / Kai Hui) · `…107` = Pay Order for Customs Tax & VAT · `…103` insurance ·
`…108` worker entertainment · `…109` carrying & labour · `…110` C&F · `…111` BSTI ·
`…112` custom demand notes · `…113` misc · `…114` freight.

| Row | Sheet label | GL account(s) | Sheet May | GL May | Sheet Jun | GL Jun | Verdict |
|---|---|---|---|---|---|---|---|
| 18 | LC Open | `01050001` sub 104+105+106 | 1,010,570.58 | 1,181,691.96 | 0 | 1,187,004.95 | ✅ same account; amounts near May, sheet Jun is plan |
| 20 | HMBR Payorder (Tax & VAT) | `01050001` sub 107 (+ GI's share via `09060001`) | 0 | 0 | 4,037,063.41 | 3,715,073.41 | ✅ **exact**: 3,715,073.41 + 321,990 (GI's share, same voucher `BPAY003180`) = 4,037,063.41; 321,990 is also GI Test's "GI International Payment" |
| 21 | TT | `01050001` sub 101+102 | 3,496,083.42 | 4,358,800.00 | 5,333,051.64 | 3,105,600.00 | ✅ account right, amounts differ (sheet is plan) |
| 23 | Others (Transport + Hasbolt) | `01050001` other subs | 800,000 | 48,800 | 800,000 | 149,350 | ⚠️ round-number plan, GL far smaller |
| 29 | Tota supplier Local | `09030001` Accounts Payable (Local) | 200,000 | 515,045 | 200,000 | 251,342 | ⚠️ right account, sheet understates |
| 33 | Staff Salary | `09010002` Accrued Salary | 1,541,051 | 1,461,410 | 2,341,051 | 1,474,447 | ⚠️ May −79,641 gap; Jun sheet = May + exactly 800,000 (plan) |
| 34 | Malek Jobeda Loan Payment | `09040007` Malek Jobeda Foundation | 1,000,000 | 1,000,000 | 1,000,000 | 1,000,000 | ✅ **exact both months** |
| 35 | Interest & Loan Paid to Md Sir | `10020001` Loan From MD Sir, `10020016` Loan From Director | 200,000 | 0 | 200,000 | 600,000 | ⚠️ pattern is ~200k/month but split across two accounts and lumpy |
| 40 | Income Tax & Misc | `06290002` Income Tax, `06290001` VAT | 50,000 | 143,046 | 50,000 | 1,218,575 | ⚠️ Jun has a 1,188,575 income-tax payment the sheet didn't plan |
| 42 | Operating Expenses | all `06xxxxxx`+`07xxxxxx` paid via CPAY/BPAY (ex. bonus/tax/interest) | 2,050,000 | 1,999,823 | 2,050,000 | 1,589,222 | ✅ May within 2.4%; (06xx 822k + 07xx 1,177k) |
| 43 | C & F bill | `09050024` | 0 | 150,000 | 0 | 250,000 | ⚠️ sheet shows 0, GL has C&F payments both months |
| 46 | Misc. Expenses (Unforseen) | `13020007` BS-reconciliation suspense | 250,000 | 250,000 | 250,000 | 3,500,000 | ✅ May **exact**; Jun GL is the 3.5M below |
| 47 | Loan to Zepto | `02050011` Loan to Zepto Chemicals | 0 | 64,980 | 0 | 65,380 | ✅ these are Zepto salary payments; the same figures appear in Zepto-M "Additional collection" for those months |
| 48 | Loan to GI corp | `09060001` "Bank balance Received in GI" (CPAY 1.9M/2.8M + small BPAY) | 0 | 2,030,895 | 0 | 3,214,962 | ✅ the real inter-company outflow, sheet row is blank for both months |

### Needs your call (confusions)

1. **UCB Loan Payment (row 31: 3.5M May / 3.0M Jun) has no GL match.** No UCB loan account
   (`10010005/6/8`) moves in May or June. The only 3,500,000 is `BPAY003182` on 2026-06-15,
   booked to `13020007` "Reconciliation of the balance sheet…" (a suspense account), funded from
   the DBL CC. Is that the UCB payment, mis-coded and a month late?
2. **Dhaka Bank / DBL LTR 3,200,000** paid 2026-06-25 (`BPAY003184`, `10010001`) — sheet row 30
   is blank, and I tentatively put it under "BL withdrow" (row 19, sheet Jun 4,784,316.53). Which
   is it?
3. **Sun Flower (row 41, 450k) and Fix iT (row 44, 300k) both have zero GL movement** in both
   months. Sun Flower exists as `09040001` and `10020011`; Fix iT as `10020002`/`10020003` —
   if these were actually paid, they were booked somewhere I haven't mapped.
4. **Staff Salary May gap of 79,641** — accrued salary 1,461,410 + bonus 691,617 + advance /
   PF / director rem / tax-fund 419,594 are all separate GL lines; none combination I tried
   gives 1,541,051 exactly.
5. **Unmapped to any sheet row:** `09030003` Accounts Payable (Internal) 369,411 / 163,091
   (salary of GI/Zepto-side people paid via HMBR); `01050007` AT(+)VAT Rebate Claim ~175k/month;
   `01030002` Recognized Agent 277,851 (May). Plus the sheet has no row for **bonus** (May 691,617).
6. **Rows 30, 32, 36, 37, 38, 39, 45** (Dhaka loan, Rahima, Bijoy sir, DBL/UCB interest,
   CC repayment, Investment) are blank in the sheet *and* have no cash movement in the GL for
   these months — consistent, but unproven for months where they would be non-zero.
7. **Loan to Bijoy sir** → `02050004` "Loan to Directors" is a guess, no movement to confirm it.

---

## Step 1c — Jan/Feb 2026 cross-check (same method as 1b) + corrections to 1b

Sheet columns Z (Jan-26) and AA (Feb-26); GL periods 1 and 2; same CPAY+BPAY definition
(funding == paid-for in both months: 18,443,668.42 / 15,066,822.36). **Jan/Feb sheet values look
like actuals** (non-round, most rows tie to the paisa) — unlike May/Jun, which is largely plan.
Sheet formula also differs: Jan/Feb cash balance is `=Z16+Z17+Z26-Z24-Z25-Z23` with
`Z24 = SUM(Z18:Z22)` (row 23 subtracted separately); May/Jun folds row 23 into row 24.

| Row | Sheet label | GL account(s) | Sheet Jan | GL Jan | Sheet Feb | GL Feb | Verdict |
|---|---|---|---|---|---|---|---|
| 18 | LC Open | `01050001` sub **105/106** (L/C margin only) | 2,208,867.93 | 2,208,867.93 | — | 3,412,000 (see 19) | ✅ Jan exact |
| 19 | BL withdrow | not identified | — | — | 4,517,817.93 | 3,412,000 margin (+1,140,000 `02050017` same voucher `BPAY003153` = 4,552,000) | ⚠️ close to 4,552,000 (34,182 off), unproven |
| 20 | HMBR Payorder | `01050001` sub 107 | 3,685,105.11 | 3,685,105.11 | 4,372,895.61 | 3,547,126.31 (+697,681 GI pay order = 4,244,807.31) | ✅ Jan exact; Feb 128,088.30 unexplained |
| 21 | TT | sub 101/102 | 4,197,600 | 4,197,600 | — | 0 | ✅ exact |
| 22 | Others International Payment | `01050001` subs 103+**104**+108+109+113 | 192,046.38 | 192,046.38 | 124,668.05 | 124,668.05 | ✅ **exact both** (L/C charge `…104` belongs here, not in LC Open) |
| 23 | Others (Transport + Hasbolt) | `09030001` **supplier SUP-000046** | 287,550 | 287,550 | 291,870 | 291,870 | ✅ **exact both** (GRN-009791 / GRN-009808) |
| 29 | Tota supplier Local | `09030001` ex SUP-000046, + `09030003` Gulshan Packaging purchases (ex salary pieces) | 327,500 | 333,000 | 183,575 | 186,305 (183,575 if one 2,730 line excluded) | ⚠️ near; Feb exact minus one line |
| 31 | UCB Loan Payment | `10010006` UCB CC via **BDJV** (bank-deposit journal) | 45,000 | 45,000 (`BDJV001240`) | — | — | ✅ exact — so UCB payments live in `10010006`, not `10010005/8` |
| 33 | Staff Salary | `09010002` Accrued Salary **+ salary-tagged `09030003` "Pay to Gulshan Packaging"** | 1,552,852 | 597,010+871,551+30,000+54,291 | 1,526,496 | 529,223+907,925+28,880+60,468 | ✅ **exact both** |
| 34 | Malek Jobeda | `09040007` | 1,000,000 | 1,000,000 | 1,000,000 | 1,000,000 | ✅ exact |
| 35 | Interest & Loan to Md Sir | `10020001` | 100,000 | 100,000 (`CPAY014840`) | — | 0 | ✅ exact |
| 36 | Loan to Bijoy sir | `02050004` Loan to Directors | — | 0 | 150,000 | 150,000 (`CPAY014896`, "Loan to Director Sir for (Shamee Madam)") | ✅ exact — confirms the guess in 1b |
| 39 | Repayment to DBL-CC A/C | `10010003` Dr via BTJV | 133,892.89 | 5,141,000 total Dr | 1,519,000 | 5,764,000 total Dr | ❌ no clean tie (Feb 1,519,000 can be assembled from BTJV lines but so can many things) |
| 40 | Income Tax & Misc | `06290001` VAT | — | 16,000 | 86,384 | 23,000 | ⚠️ unexplained |
| 42 | Operating Expenses | `06xx`+`07xx` via CPAY | 2,166,034.91 | 1,787,346 | 1,845,742.18 | 1,641,865 | ⚠️ sheet higher by 378,689 / 203,877; cents suggest non-cash/accrual items |
| 43 | C & F bill | `09050024` | 300,000 | 300,000 | 324,000 | 324,000 | ✅ exact |
| 46 | Misc. Expenses (Unforseen) | various | 290,000 | `13020007` 100,000 (Jan) | 173,000 | `06030005` 173,000 (Advocate Sharif renewal) | ✅ Feb exact; ⚠️ Jan not assembled |
| 47 | Loan to Zepto | `02050011` | 67,307 | 67,307 | 45,000 | 45,000 | ✅ exact |
| 48 | Loan to GI corp (Int. Purchase) | **`02050017` "Loan to GI For Property Share"** | — | 0 | 1,140,000 | 1,140,000 (`BPAY003153`) | ✅ exact — settles the open question from Step 1 |

### Corrections to Step 1b (May/Jun)
- **Row 48 = `02050017`, not `09060001`.** `09060001` "Bank balance Received in GI" is HMBR paying
  GI's pay orders, salaries and bank-balance top-ups; GI Test's "GI International Payment"
  Feb-26 1,837,681 = 697,681 (`09060001` pay order) + 1,140,000 (`02050017`), Jan-26 295,302 =
  the `09060001` pay order. In May/Jun `02050017` only has credits (GI repaying 1M
  installments), consistent with the sheet's blank row.
- **Row 23 = supplier SUP-000046 payments** (May 370,080 / Jun 236,280 vs sheet plan 800,000),
  not the other `01050001` sub-ledgers; those are row 22.
- **Row 18 = L/C margin (sub 105/106) only**; L/C charge (104) is row 22.
- **Row 31 UCB** is `10010006` via BDJV journals, so the May/Jun "no UCB movement" finding was
  looking at the wrong accounts.
- **Row 33 salary** = `09010002` + the `09030003` pieces tagged as salary; May still leaves a
  50,161 gap after adding BPAY003172's 29,480, Jun sheet is plan.
- Therefore the 2.0% / 0.8% May/Jun tie-out earlier also benefits from `09060001` being counted
  as GI-bound pass-through outflow, not a sheet row.

### Still unresolved (need you)
1. Row 19 BL withdrow (Feb 4,517,817.93; Jun 4,784,316.53) — no GL account found.
2. Row 39 Repayment to DBL-CC (133,892.89 / 1,519,000).
3. Row 40 Income Tax & Misc Feb 86,384; row 46 Misc Jan 290,000; Feb Payorder gap 128,088.30.
4. Row 42: what extra 378,689 (Jan) / 203,877 (Feb) the sheet includes beyond CPAY 06xx/07xx.
5. Row 29 Jan 5,500 gap.

---

## Step 1d — Totals-first reconciliation, 2025 (ignoring the sheet's line items)

Approach changed per request: ignore the sheet's payment line items and match **totals** —
sheet "HMBR Total Payment" vs total cash out in the GL for ZID 100001, then list the GL
accounts the cash actually went to. Full 77-account Oct–Dec list:
`docs/phase1_cash_out_by_account_oct_dec_2025.csv`.

**GL definition of cash out:** vouchers `CPAY` + `BPAY`; the money-out side is the credit to
`01010001` General Cash, `01020001` Cash at Bank, and the two CC overdraft accounts
`10010003` (Dhaka Bank) / `10010006` (UCB) — payments are sometimes funded straight from the
CC with no bank line. `BTJV`/`CWJV`/`BDJV`/`STJV` are internal transfers and excluded.
Paid-for side (debits) == cash-out side to the rupee in every month.

**Sheet side — the sheet's "HMBR Total Payment" (row 25) is NOT the whole outflow.** The cash
balance formula subtracts more than row 25 (Jan–Sep 2025: Payorder row 20 + International row
24 + row 25; Dec: row 24 + 25 + 23; Oct/Nov are hardcoded). The only fair sheet total is the
implied outflow = Collection + Additional Collection + Balance Forward − Closing cash.

| 2025 | Sheet row 25 | Sheet implied total outflow | GL cash out | GL − sheet | % |
|---|---|---|---|---|---|
| Jan | 9,154,723 | 10,951,962 | 13,011,277 | +2,059,315 | +18.8 |
| Feb | 6,243,986 | 14,670,408 | 18,687,629 | +4,017,221 | +27.4 |
| Mar | 6,103,360 | 7,392,995 | 9,750,815 | +2,357,819 | +31.9 |
| Apr | 4,947,540 | 9,029,704 | 14,604,417 | +5,574,713 | +61.7 |
| May | 7,667,631 | 16,905,472 | 20,815,911 | +3,910,439 | +23.1 |
| Jun | 9,228,437 | 16,821,210 | 17,750,062 | +928,852 | +5.5 |
| Jul | 11,993,375 | 17,307,747 | 17,118,289 | −189,458 | −1.1 |
| Aug | 12,894,389 | 21,892,010 | 18,176,678 | −3,715,332 | −17.0 |
| Sep | 11,847,367 | 17,326,238 | 12,637,246 | −4,688,992 | −27.1 |
| Oct | 11,141,488 | 25,342,795 | 26,359,093 | +1,016,298 | +4.0 |
| Nov | 8,949,294 | 18,733,250 | 20,622,438 | +1,889,188 | +10.1 |
| Dec | 8,471,421 | 20,713,934 | 20,781,997 | +68,063 | +0.3 |

Oct–Dec: sheet 64,789,979 vs GL 67,763,528 (+4.6%). The GL is usually higher in H1 (sheet
likely incomplete / projected for those months) and lower in Aug–Sep.

### Oct–Dec 2025: where the GL cash went (77 accounts; top by Q4 total)

| Account | Name | Oct | Nov | Dec | Q4 | % |
|---|---|---|---|---|---|---|
| `01050001` | Advance For International Purchase | 14,201,307 | 9,783,956 | 11,420,394 | 35,405,657 | 52% |
| `09060001` | Bank balance Received in GI (paid on GI's behalf) | 3,127,016 | 4,013,417 | 2,689,906 | 9,830,339 | 15% |
| `09010002` | Accrued Salary | 2,879,933 | 1,458,004 | 1,456,266 | 5,794,203 | 9% |
| `09040007` | Malek Jobeda Foundation | 1,000,000 | 1,000,000 | 1,000,000 | 3,000,000 | 4% |
| `10020001` | Loan From MD Sir | 2,100,000 | 0 | 0 | 2,100,000 | 3% |
| `09030001` | Accounts Payable (Local) | 130,477 | 95,541 | 960,920 | 1,186,938 | 2% |
| `01010006` | Cash and bank deposits to the Director | 0 | 1,000,000 | 0 | 1,000,000 | 1% |
| `06260001` | Electricity Bill | 353,919 | 351,416 | 215,055 | 920,390 | 1% |
| `09030003` | Accounts Payable (Internal) | 380,268 | 164,297 | 295,145 | 839,710 | 1% |
| `07120002` | Sales Conveyance (TA/DA) | 259,700 | 250,685 | 250,187 | 760,572 | 1% |
| `09050024` | C & F bill / Charge | 240,000 | 275,982 | 200,000 | 715,982 | 1% |
| `07030001` | Expenses for Trading Vehicle | 177,189 | 190,501 | 200,585 | 568,275 | 1% |
| `06210001` | Entertainment | 193,545 | 190,126 | 174,915 | 558,586 | 1% |
| `09010006` | Accrued for Director Remuneration | 98,345 | 193,483 | 186,440 | 478,268 | 1% |
| `11040001` | Security Deposit Fund | 150,000 | 150,000 | 150,000 | 450,000 | 1% |

(remaining 62 accounts together ≈ 6%, all in the CSV)

**`01050001` ties exactly to the sheet's international section.** Its sub-ledgers (`glsub`):
`0105000107` Pay Order for Customs Tax & VAT = Oct 2,341,840 / Nov 2,159,152 / Dec 1,353,328
= the sheet's Payorder row to the rupee; and sheet International (row 24) = `01050001` minus
that Payorder in Oct (11,859,467) and Nov (7,624,804), and = the whole `01050001` in Dec
(11,420,393 vs 11,420,394). Other subs: `…102` TT 8.51M/5.01M/3.05M, `…105` L/C margin
3.25M/2.50M/6.91M, `…104` L/C charges, `…103` insurance, `…108/109` worker/carrying.

**Not explained:** the sheet's local total (row 25) is below the GL's non-`01050001` cash by
Oct 1,016,298 / Nov 1,889,188 / Dec 68,062 (Dec after adding row 23's 822,120). A subset search
over account totals finds many coincidental combinations, so no single set is claimed.

---

## Step 1e — Balance-sheet cash check (Financial Statements logic) + roll-forward, 2025

**How the Financial Statements module gets monthly BS cash** (`core/queries.py::get_gl_details`,
`processing/financial.py::process_data_month` / `cash_open_close`): sums `gldetail.xprime` for
BS accounts by `glheader.xper`, cumulative within the year, period 0 = the `OB--` opening-balance
voucher. **No voucher-type filter and no posting-status filter** — every GL line counts. Its
"Cash & CE" row is *not* just two accounts: `data/hierarchy.json` puts 11 accounts under
`0101` + `0102` (`01010001/2/3/4/5/6/7/8`, `01020001/2/3`).

**Voucher types referenced in code:** payments `CPAY`, `BPAY`, `STJV` (`get_payment_data`,
hard-coded to 100001); AR-ledger queries add `PAY`, `CHQ`, `BTJV` (alongside `RCT/BRCT/CRCT`
for collections). In 2025 `STJV` has **zero** net effect on `01010001`+`01020001`.

### Closing cash, 2025, ZID 100001 (replicated with the real `get_gl_details`)

| | GL 01010001+01020001 | GL full Cash & CE (11 accts) | Sheet "HMBR Cash Balance" | GL(2) − sheet |
|---|---|---|---|---|
| Jan | 3,500,419 | 3,515,965 | 1,229,111 | +2,271,308 |
| Feb | 955,478 | 971,024 | 754,676 | +200,802 |
| Mar | 2,622,009 | 2,637,555 | 2,054,579 | +567,430 |
| Apr | 2,952,016 | 2,967,562 | 1,936,687 | +1,015,329 |
| May | 3,656,369 | 3,671,915 | 1,508,167 | +2,148,202 |
| Jun | 1,670,901 | 1,686,447 | 443,029 | +1,227,872 |
| Jul | 3,794,081 | 3,809,627 | 558,944 | +3,235,138 |
| Aug | 2,112,570 | 2,128,116 | 684,105 | +1,428,465 |
| Sep | 1,743,501 | 1,759,047 | 236,497 | +1,507,004 |
| Oct | 5,862,468 | 5,878,014 | 4,876,249 | +986,219 |
| Nov | 3,549,429 | 3,564,976 | 3,417,508 | +131,921 |
| Dec | 5,372,460 | 5,388,007 | 5,236,394 | +136,067 |

Opening: GL period-0 cash = 2,219,995 (full CE 3,078,541) vs sheet Jan Balance Forward
1,764,951. The sheet's closing cash **only comes close in Nov–Dec** (hardcoded Oct/Nov, formula
Dec). Sheet Dec closing 5,236,393.85 = its Jan-26 Balance Forward, GL Dec = 5,372,460.40.

### Roll-forward: why the closings differ (2025 full year, 01010001+01020001)

| Flow | GL | Sheet | GL − sheet |
|---|---|---|---|
| Receipts (RCT-/BRCT/CRCT) | 230,174,298 | 200,559,168 (Collection + Additional) | +29,615,130 |
| Outflows (CPAY+BPAY from cash, BTJV net, other) | 227,021,833 | 197,087,725 (implied) | +29,934,108 |
| Net change in cash | +3,152,465 | +3,471,443 | −318,978 |

Both sides are ~30M higher in the GL and nearly cancel — the sheet is a *smaller* model of both
inflow and outflow, not a mis-stated one-sided figure.
- **BTJV moves 35.1M net out of cash in 2025**: debits to `10010003` Dhaka Bank CC 29.3M,
  `10010006` UCB CC 5.7M, bank charges 0.1M — i.e. cash used to repay the CC overdrafts
  (sheet rows 39 "Repayment to DBL-CC A/C" / 31 "UCB Loan Payment"). Step 1d's CPAY+BPAY-only
  cash-out totals left this out.
- **Receipts split** (credited accounts): customer AR `01030001` 100.4M vs sheet "HMBR
  Collection" 115.8M (that row is a formula, Sales × 1.01); non-AR receipts 129.7M vs sheet
  "Additional Collection" 84.8M. Non-AR includes `02050001` Loan to Hospital (Development)
  **88.8M** (real repayments: balance fell 150.9M → 62.2M), `09060001` GI bank balance 40.3M,
  `01030002` Recognized Agent 12.3M, `02050017` GI property-share installments 10.6M,
  `01050008` Accrued Against Sales Distribution 8.4M, `02050011` Zepto 2.3M, loans from Md
  Sir/Director/Fix-iT ~4.8M.

### Open questions for the user
(answered by the user; see Step 1f — it supersedes this section)


---

## Step 1f — Re-run after the user's answers (supersedes parts of 1d/1e)

User clarifications: (1) the sheet row to match is **row 27 "HMBR Cash Balance"** against GL
`01010001`+`01020001`; (2) hospital loan repayments are a one-off capital inflow (asset sale),
not operations; (3) CC overdraft repayments **are** in the sheet and must be counted; (4) GL
internal transactions between 100000 (GI), 100001 (HMBR) and 100005 (Zepto) exist but the
treatment was unknown.

### Main finding — the sheet's cash balance excludes GI's custodial money (`09060001`)
`09060001` "Bank balance Received in GI" is a liability on HMBR's books: GI's cash held in HMBR's
bank/cash accounts. **GL cash + balance of `09060001` (a negative number) ≈ sheet row 27**:

| Month | GL cash (2 accts) | `09060001` bal | GL excl. GI | Sheet | Gap before | Gap after |
|---|---|---|---|---|---|---|
| Dec-24 | 2,219,995 | −397,887 | 1,822,108 | 1,764,951 | 455,045 | 57,158 |
| Feb-25 | 955,478 | −185,117 | 770,361 | 754,676 | 200,802 | 15,685 |
| Mar-25 | 2,622,009 | −543,157 | 2,078,852 | 2,054,579 | 567,430 | 24,273 |
| Apr-25 | 2,952,016 | −1,011,527 | 1,940,489 | 1,936,687 | 1,015,329 | 3,802 |
| May-25 | 3,656,369 | −2,064,336 | 1,592,033 | 1,508,167 | 2,148,202 | 83,866 |
| Jul-25 | 3,794,081 | −3,152,291 | 641,790 | 558,944 | 3,235,138 | 82,847 |
| Aug-25 | 2,112,570 | −1,386,776 | 725,794 | 684,105 | 1,428,465 | 41,689 |
| Sep-25 | 1,743,501 | −1,451,815 | 291,686 | 236,497 | 1,507,004 | 55,189 |
| Oct-25 | 5,862,468 | −917,531 | 4,944,937 | 4,876,249 | 986,219 | 68,688 |
| Nov-25 | 3,549,429 | −49,733 | 3,499,696 | 3,417,508 | 131,921 | 82,188 |
| Dec-25 | 5,372,460 | −40,725 | 5,331,735 | 5,236,394 | 136,067 | 95,342 |

Std-dev of the gap fell from 897k to 519k, and to <100k in every month except the outliers
below. The **opening-balance gap (455k at Dec-24) is almost all GI custody** (397,887) — the 5
accounts in the sheet's first mistake were never a data error.

### Remaining residual, explained
- **Jan-25 +1.90M / Feb-25 −1.94M, Jun-25 −1.00M / Jul-25 +1.00M:** pure month-cut-off timing
  on CC-drawdown bank-transfer vouchers (e.g. `BTJV001417` 2025-01-30 +1,900,000 from `10010003`;
  `BTJV001489/88` 2025-06-22/18). They reverse the next month.
- **Sep–Dec 2025 drift of exactly +13,500/month:** a monthly rent receipt (`BRCT`, credited
  `08030002` Floor Rent, sub `0803000201`, bank sub `0102000145`) the sheet does not count.
  The few-tens-of-thousands background residual is likely other small rent/other income.
- **Sheet "Collection" (row 16) is built from the same receipt vouchers as `mv_collection_vouchers`**
  (same cents, e.g. Jan-25 GL 8,906,272.88 vs sheet 8,782,752.88) but differs by thousands to
  ~0.9M per month; no exact subset of GL receipt accounts reproduces rows 16+17.
- **Row 17 Oct-25 formula components** `117331+2000000+53834+2500000+2500000+8500000+2398696+
  30000+1000000+740053`: 2.5M+2.5M+8.5M+2,398,696 = 15,898,696 = **hospital repayments**;
  2,000,000 = Md Sir loan (`10020001`); 1,000,000 = GI property-share installment (`02050017`);
  53,834 = Zepto (`02050011`); 740,053 = `01050008`. GI's `09060001` receipts (2.59M) are *not* in it.

### Hospital repayments — sheet is NOT consistent with "left out"
Comparing sheet rows 16+17 with GL cash receipts (ex-GI): Jul, Oct, Nov, Dec the sheet is far
closer to *including* hospital (Oct: −1.8M with, −17.7M without; Oct's own formula lists it);
only Apr-25 looks like it excluded it. Sum |gap| over 12 months: 31.9M with vs 66.7M without.

### 2026 (same adjustment)
Jan-26 ties (108k) and May-26 (149k); Feb–Apr and Jun–Sep do not (gap 2.0M–12.0M) — consistent
with the sheet's 2026 rows being partly plan (formulas from May, typed forecasts earlier).

---

## Step 1g — GI check, scope of collections, and the Dec-2025 one-month model

Workbook: `docs/phase1_cash_model_dec_2025_HMBR.xlsx` (tabs: model vs sheet, cash lines by
account, GI sheet check, notes). The model's closing cash is **GL `01010001` + `01020001`
exactly** (opening 3,549,429.33 + every GL movement = 5,372,460.40, difference 0.00).

**GI sheet check (ZID 100000).** The "GI Test" tab's cash balance = GI `01010001` + `01020001`
+ `01020003` "GI Bank balance In (GTC)" to the rupee in every month May-24..Dec-25 (Sep-24 off
by 15,792); its monthly change equals the GL change exactly in all of 2025. `01020003` mirrors
HMBR `09060001` (same amount, opposite sign). `GI` and `GI Test` tabs are identical for 2025.
So each sheet excludes the *other* business's money held in its own bank accounts; summed they
give group cash with no double count. GI's "GI Collection" = customer receipts `01030001` +
Recognized Agent `01030003` exactly (e.g. Jan-25 6,586,708 + 182,160 = 6,768,868); its
"Additional Collection" = financing inflows (e.g. Loan from Fix iT 300,000 / 2,361,297).
GI customers pay into HMBR bank accounts: Dec-25 4,250,193 of GI receipts were banked by HMBR.

**What the HMBR sheet's rows really are (Dec-25, exact unless noted):** Collection = `01030001`
+ `01030002` + `11040001` receipts; International rows = `01050001` sub-ledgers (LC Open = 20%
margin, BL withdrawal = 80% margin, Payorder `…107`, TT `…101/102`, Others `…103/104/108/109`)
recorded when paid, even if funded by the CC overdraft; Sales Value = gross trade sales less
SRJV and SRT- returns; Local Purchase = row 23 = SUP-000046 payments. Additional Collection
decodes to every other real inflow except GI pass-through and internal transfers (gap 3,795).
Not tied: purchase block rows 3-7 (typed estimates), the inventory ledger (modelled, not
stock-based), supplier/salary/misc rows by ±5k–60k, and the CC-repayment rows (sheet 2.6M vs
GL 7.79M).

---

## Step 1h — Streamlit's own monthly cash flow (Dec-25) vs the model vs the sheet

Ran the app's exact monthly pipeline (`process_data_month` → `sort_pl_level0` →
`append_net_profit_to_bs_level0` → `cash_open_close` → `make_cashflow_statement_level0`) for ZID
100001, year 2026, months 1–9 (yields 2025-01 … 2026-09). Note: business `100001` carries project
`GULSHAN TRADING`; no cash voucher in 2025–26 sits under a different project, so the project filter
changes nothing here.

**Streamlit "Cash & CE" is 11 accounts** (`0101` ×8 + `0102` ×3), not two:

| Dec-25 | Opening (Nov-30) | Closing (Dec-31) |
|---|---|---|
| Streamlit Opening/Closing Cash & CE | 3,564,975.50 | 5,388,006.60 |
| − `01010004` Staff Welfare | −9,967.00 | −9,967.00 |
| − `01010003` PBL-RFCD-51302053 | −5,579.20 | −5,579.20 |
| = General Cash + Cash at Bank (model) | 3,549,429.30 | 5,372,460.40 |
| − GI custody (`09060001` balance) | −49,733.00 | −40,725.00 |
| − remaining unexplained | −82,188.13 | −95,341.55 |
| = Sheet "HMBR Cash Balance" | 3,417,508.17 | 5,236,393.85 |

Gap model − sheet: 131,921.16 (opening) / 136,066.55 (closing).

**Where the unexplained part comes from** (residual after GI custody, by month, 2025):
Apr 3,802 → May 83,866 → Jul 82,847 → Aug 41,689 → Sep 55,189 → Oct 68,688 → Nov 82,188 → Dec 95,342.
Sep, Oct, Nov each add **exactly 13,500** (13,500 / 13,499 / 13,500); Dec adds 13,153. The only
recurring 13,500 in the GL is the monthly rent receipt `BRCT` credited `08030002` Floor Rent, sub
`0803000201` (bank sub `0102000145`), which starts Sep-25. Confirmed in the sheet's own Oct-25
Additional Collection formula, which lists a 30,000 rent item (`…+30000+…`) but no 13,500 — the other
rent tenant (43,500 = 13,500 + 30,000). So 40,499 of the 82,188 at Nov-30 = three months of the
omitted 13,500 tenant; the other 41,689 was already there at Aug-31 and is not explained
(the residual jumps +80,064 in May and −41,158 in Aug, so it is re-based by typed rows).

Decomposition of the opening gap: 131,921 = GI custody 49,733 + rent omitted Sep–Nov 40,499 +
older residual 41,689. Closing gap: 136,067 = 40,725 + 53,652 + 41,689.

**App finding (not changed):** `cash_open_close` sums all 11 `0101`/`0102` accounts, which include
`01010005` "Advance for Staff Salary" — 1,434,981 at Sep-26 — and `01010008` "Advance for
Manufacturing Project". That is why the app's Sep-26 closing cash (1,768,065.60) is 1,434,981 above
General Cash + Cash at Bank (333,084.60 incl. the two small accounts). Dec-25 is unaffected (those
accounts are 0). The app's own "Cash-flow Check" row also shows ±3,632 in May-25 / Jan-26.

---

## Step 1i — Same analysis for GI (100000) and Zepto (100005), Dec-2025

Workbook: `docs/phase1_cash_model_dec_2025_GI_Zepto.xlsx` (GI model, Zepto model, 2025 sheet-vs-GL
by month, notes). App pipeline run as in 1h (`process_data_month` → `cash_open_close`).

**GI** — Streamlit Cash & CE Dec-25: 226,773.00 → 260,960.00 = the sheet's "GI Cash Balance"
(226,772.94 → 260,959.94) to 6 paise. Cash = General Cash 220,235 + `01020003` "GI Bank balance In
(GTC)" 40,725 (mirror of HMBR `09060001`). **Every cash row decodes exactly**: Collection = customers
`01030001` 7,692,016 + Recognized Agent `01030003` 734,300 = 8,426,316; International = `01050001`
535,746 (subs 0105000103 400,000 + 0105000104 135,746); supplier = `09030001` 5,220,429; Loan
Repayment to GTC = `10020003` 1,000,000; salary = `09010002` 571,272; Accrued S&D 7% = `09010003`
569,295; Mold & Machinaries = `03050001` 205,000; **Operating Expenses 290,387 = the sum of all
remaining outflow lines**; Total Payment 7,856,383; Sales (Old) = gross trade sales less SRJV/SRT-
returns = 9,356,273.29 (sheet 9,356,274). Only the sheet's shipment block (rows 24-26: 85,000 /
575,000 / 170,000) is not in the GL. GI sheet tracks the 3-account GL to the rupee in 11 of 12 months
of 2025 (May +0.50).

**Zepto** — Streamlit Cash & CE Dec-25: 982,616.90 → 808,059.10 (General Cash 679,659 + Cash at
Bank 128,400.10) vs sheet "Balance" 982,617.59 → 808,059.70: gap **0.64**, carried since Nov (Apr-25
had a −73.23 blip, otherwise within ±2). **Every cash row decodes exactly**: Collection =
`01030001` 2,809,050 + Recognized Agent `01030002` 6,515 = 2,815,565; Additional collection 74,500 =
`10020002` Loan From HMBR inflow (HMBR pays Zepto salaries; = HMBR `02050011`); Supplier Payment
1,741,329 = AP Local `09030001` 1,288,475 + AP Internal `09030002` 452,854; Staff Salary `09010002`
302,364; Loan Repayment to HMBR 600,000 + (Salary) 74,500 = `10020002` outflows 674,500 (= HMBR
`02050011` receipts); Director remuneration `06150001` 40,000; **Operating Expenses 306,429.89 = all
remaining outflows** (incl. bank charge 819.89, 07xx/06xx); Total Payment 3,064,622.89. The
salary advance is shown in the sheet as both an inflow and an outflow, netting to zero. Zepto's
sales block is **not** exact: MRP Discount ties (1,227,119.79 vs 1,227,119) but Total Sales With MRP
(sheet 4,791,403 vs 5,042,717.60), Additional Discount (576,777 vs 579,732.12) and MO Value (sheet
1,796,273.11, a +30.4k/month projection, vs 1,452,483.35 manufacturing receipts) do not.

**Internal mirrors found:** Zepto `10020002` Loan From HMBR ↔ HMBR `02050011`; GI `10020003` Loan from
GTC ↔ HMBR `02050017`; GI `09010003` Accrued S&D ↔ HMBR `01050008`; GI `01020003` ↔ HMBR
`09060001`. Unlike HMBR, GI and Zepto sheets are fully cash-basis with no CC-funded payments, which is
why they decode completely.
