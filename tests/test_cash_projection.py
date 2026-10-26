"""Unit tests for processing/cash_projection.py (Cash Model -> Projection)."""
import numpy as np
import pandas as pd

from processing import cash_model as cm
from processing import cash_projection as cp


def test_next_months_rolls_over_the_year():
    assert cp.next_months((2026, 11), 3) == [(2026, 12), (2027, 1), (2027, 2)]
    assert cp.parse_month("2027-02") == (2027, 2) and cp.parse_month("nope") is None


def _orders(**over):
    row = {"po": "IP--9", "shipment": "MDKF009/26,2026-10-04", "po_date": "2026-10-04", "status": "1-Open",
           "total": 1000.0, "currency": "BDT", "lines": 3, "lines_no_rate": 0, "lines_not_in_catalog": 0,
           "lines_value": 1000.0}
    row.update(over)
    return pd.DataFrame([row])


def test_schedule_past_date_moves_to_first_month_and_override_wins():
    first = (2026, 10)
    s = cp.build_shipment_schedule(_orders(po_date="2026-08-20"), {}, first)
    assert s.iloc[0]["month"] == (2026, 10) and s.iloc[0]["value"] == 1000.0 and s.iloc[0]["warning"] == ""
    plan = {"overrides": {"IP--9": {"month": "2027-02", "value": 1500, "note": "Chinese New Year"}},
            "planned": [{"ref": "New", "value": 700, "month": "2026-12"}]}
    s = cp.build_shipment_schedule(_orders(), plan, first)
    assert s["month"].tolist() == [(2026, 12), (2027, 2)]
    assert s[s["source"] == "Open IP"].iloc[0]["value"] == 1500.0


def test_schedule_flags_open_purchases_without_cost_data():
    s = cp.build_shipment_schedule(_orders(lines_no_rate=2, lines_not_in_catalog=1, lines_value=400.0), {}, (2026, 10))
    w = s.iloc[0]["warning"]
    assert "without a cost" in w and "not in the catalogue" in w and "do not add up" in w


def test_import_schedule_follows_lc_ltr_tt_mechanics():
    months = cp.next_months((2026, 9), 6)                       # Oct-26 .. Mar-27
    sched = pd.DataFrame([{"ref": "S", "source": "Planned", "value": 10_000_000.0, "month": (2026, 12), "note": "", "warning": ""}])
    imp = dict(cp.DEFAULT_SETTINGS["imports"])                  # lc 55%, margin 20%, lead 1, ltr 3 months, tt after ltr 1
    drv = {"pay_order_pct": 0.15, "charges_pct": 0.04, "rebate_pct": 0.15}
    a = cp.import_schedule(sched, months, imp, drv, ltr_open=0.0)
    i = months.index((2026, 12))
    assert a["vendor"][i] == 10_000_000
    assert np.isclose(a["margin"][i - 1], 0.20 * 0.55 * 10_000_000)             # 1.1M, one month before arrival
    assert np.isclose(a["ltr_draw"][i - 1], 0.80 * 0.55 * 10_000_000)           # 4.4M when documents arrive, no cash
    assert np.isclose(a["tt"][i], 0.45 * 10_000_000)                            # the TT part of V, month after the LTR
    assert np.isclose(a["customs"][i], 0.30 * 10_000_000)                       # duty + reclaimable VAT/AT in cash
    assert np.isclose(a["landed"][i], 10_000_000 * 1.19)                        # rebate part is not cost
    assert np.isclose(a["ltr_repay"][i - 1 + 3], 4_400_000)                     # repaid within the 3-month limit


def test_existing_ltr_is_run_off_within_the_limit():
    months = cp.next_months((2026, 9), 6)
    imp = dict(cp.DEFAULT_SETTINGS["imports"])
    a = cp.import_schedule(pd.DataFrame(), months, imp, {"pay_order_pct": 0, "charges_pct": 0, "rebate_pct": 0}, ltr_open=3_000_000)
    assert a["ltr_repay"][:3].tolist() == [1_000_000] * 3 and a["ltr_repay"][3:].sum() == 0


def _final(cols):
    row = lambda line, kind, v: {"Line": line, "kind": kind, **{c: v for c in cols}}
    p = pd.DataFrame([row("   Vendor value of shipments received", "line", 0.0), row("Landed cost of shipments received", "subtotal", 0.0),
                      row("   Trading purchases received (items that are sold)", "line", 100.0)])
    i = pd.DataFrame([row("Opening inventory (prior month closing)", "total", 1000.0), row("   Deliveries (DO--)", "line", -240.0),
                      row("   Customer returns (SR--)", "line", 20.0), row("   Goods received & other receipts", "line", 150.0),
                      row("Closing inventory", "total", 1000.0), row("   Gross sales (trade value)", "line", 250.0),
                      row("   − Discount", "line", -5.0), row("   − Customer returns (AR credits, SRJV/SRT-)", "line", -20.0),
                      row("Net sales (after returns)", "total", 225.0)])
    c = pd.DataFrame([row("Opening cash & CE (app)", "total", 0.0), row("COLLECTIONS (sales operations)", "header", np.nan),
                      row("   Customer receipts", "line", 200.0), row("Total collections", "subtotal", 200.0),
                      row("ADDITIONAL INFLOWS (not from sales operations)", "header", np.nan),
                      row("Total additional inflows", "subtotal", 0.0), row("TOTAL CASH IN", "total", 200.0),
                      row("PAYMENTS (cash out)", "header", np.nan), row("   Operating expenses", "line", -60.0),
                      row("TOTAL CASH OUT", "subtotal", -60.0), row("NET CHANGE IN CASH", "total", 140.0),
                      row("Closing cash — model (opening + net change)", "total", 140.0)])
    return cm.build_final_model(p, i, c, cols)


def test_projection_chains_cash_and_inventory_and_honours_settings():
    cols = ["Jul-26", "Aug-26", "Sep-26"]
    final = _final(cols)
    settings = cp.load_settings("none", path="/nonexistent.json")
    settings["horizon_months"] = 3
    settings["known_items"] = [{"label": "Hospital", "direction": "in", "amount": 500, "month": "2026-11", "line": None}]
    settings["lines"] = {"   Operating expenses": {"mode": "avg3", "adjust_pct": -10.0, "fixed": 0}}
    sched = pd.DataFrame(columns=["ref", "source", "value", "month", "note", "warning"])
    t, pcols, info = cp.project(final, cols, (2026, 9), 1000.0, sched, settings, ltr_open=0.0, shipments_hist=pd.DataFrame())
    assert pcols == ["Oct-26 (P)", "Nov-26 (P)", "Dec-26 (P)"]
    g = lambda line, col: float(t[t["Line"].str.rstrip() == line.rstrip()][col].iloc[0])
    assert g("Opening cash & CE (app)", pcols[0]) == 1000.0                      # anchored on the app's closing cash
    assert np.isclose(g("   Operating expenses", pcols[0]), -54.0)               # −60 average, trimmed 10%
    assert g("   Other inflows", pcols[1]) == 500.0                              # known item lands in its month only
    assert g("   Other inflows", pcols[0]) == 0.0
    for c in pcols:
        assert np.isclose(g("Closing cash — model (opening + net change)", c),
                          g("Opening cash & CE (app)", c) + g("NET CHANGE IN CASH", c))
        assert np.isclose(g("NET CHANGE IN CASH", c), g("TOTAL CASH IN", c) + g("TOTAL CASH OUT", c))
    assert np.isclose(g("Opening cash & CE (app)", pcols[1]), g("Closing cash — model (opening + net change)", pcols[0]))
    # inventory: opening of month 2 is the closing of month 1; net sales = % sold x opening
    assert np.isclose(g("Opening inventory (prior month closing)", pcols[1]), g("Closing inventory", pcols[0]))
    assert np.isclose(g("Net sales (after returns)", pcols[0]), info["drivers"]["pct_sold"] * 1000.0)
    # actual columns are untouched
    assert float(t[t["Line"] == "Closing inventory"]["Sep-26"].iloc[0]) == 1000.0


def test_cc_funded_margin_moves_no_cash_but_cash_funded_does():
    cols = ["Jul-26", "Aug-26", "Sep-26"]
    final = _final(cols)
    sched = pd.DataFrame([{"ref": "S", "source": "Planned", "value": 1000.0, "month": (2026, 11), "note": "", "warning": ""}])
    out = {}
    for funding in ("cc", "cash"):
        s = cp.load_settings("none", path="/nonexistent.json")
        s["horizon_months"] = 3
        s["imports"]["margin_funding"] = funding
        t, pcols, _ = cp.project(final, cols, (2026, 9), 1000.0, sched, s, 0.0, pd.DataFrame())
        out[funding] = float(t[t["Line"].str.strip() == "International: L/C margin & charges"][pcols[0]].iloc[0])
    assert out["cc"] == 0.0 and np.isclose(out["cash"], -0.20 * 0.55 * 1000.0)


def test_settings_roundtrip_and_defaults(tmp_path):
    path = str(tmp_path / "s.json")
    s = cp.load_settings("100001", path=path)
    assert s["horizon_months"] == 12 and s["imports"]["lc_pct"] == 0.55
    s["imports"]["lc_pct"] = 0.6
    cp.save_settings("100001", s, path=path)
    assert cp.load_settings("100001", path=path)["imports"]["lc_pct"] == 0.6
    assert cp.load_settings("100000", path=path)["imports"]["lc_pct"] == 0.55      # other business untouched
    cp.save_plan("100001", {"overrides": {"IP--1": {"month": "2026-12"}}, "planned": [], "holidays": []}, "u", path=str(tmp_path / "p.json"))
    assert cp.load_plan("100001", path=str(tmp_path / "p.json"))["overrides"]["IP--1"]["month"] == "2026-12"


# ── debt engine ──────────────────────────────────────────────────────────────
def _debts(**over):
    d = cp.default_settings_for("100001")["debts"]
    d["deposits"] = []
    for k, v in over.items():
        if k in d["items"]:
            d["items"][k].update(v)
        else:
            d[k] = v
    return d


def _run(debts, base_net, opening=0.0, margin=None, customs=None, floor=0.0, funding=("cc", "cc"), bal=None):
    T = len(base_net)
    months = cp.next_months((2026, 9), T)
    z = np.zeros(T)
    balances = {"malek": 27_120_000.0, "sunflower": 3_000_000.0, "mdloan": 600_000.0, "mdod": 38_000_000.0,
                "ucb": 25_750_000.0, "dhaka": 12_190_000.0, "director": 330_000.0, "fixit": 2_750_000.0}
    balances.update(bal or {})
    return cp.run_debts(months, np.array(base_net, float), z if margin is None else np.array(margin, float),
                        z if customs is None else np.array(customs, float), opening, balances, debts, floor, *funding)


def test_malek_payment_splits_interest_and_principal_when_cash_allows():
    r = _run(_debts(), [5_000_000.0] * 2)
    assert r["pay"]["malek"][0] == 1_000_000.0
    assert np.isclose(r["malek_interest"][0], 27_120_000 * 0.12 / 12)             # ~271k interest, rest principal
    assert np.isclose(r["princ"]["malek"][0], 1_000_000 - 271_200)
    assert r["bal"]["malek"][0] < 27_120_000


def test_priority_order_decides_who_is_paid_when_cash_is_short_and_arrears_carry():
    # surplus 1.2M: Malek (1.0M) first, then Sunflower gets the remaining 0.2M of its 0.5M
    r = _run(_debts(), [1_200_000.0, 0.0])
    assert r["pay"]["malek"][0] == 1_000_000.0 and np.isclose(r["pay"]["sunflower"][0], 200_000.0)
    # month 2 has no surplus: nothing paid, the shortfall stays owed (arrears) and is due next time
    assert r["pay"]["malek"][1] == 0.0
    r2 = _run(_debts(), [1_200_000.0, 0.0, 5_000_000.0])
    assert r2["pay"]["sunflower"][2] > 500_000.0            # 0.5M due + 0.3M arrears from month 1


def test_shortfall_draws_dhaka_cc_only_up_to_the_target_headroom():
    r = _run(_debts(), [-2_000_000.0], opening=0.0, bal={"dhaka": 12_190_000.0})   # target 12.5M -> headroom 310k
    assert np.isclose(r["draw"][0], 310_000.0) and np.isclose(r["shortfall"][0], 1_690_000.0)
    r = _run(_debts(dhaka={"use_limit": True}), [-2_000_000.0], bal={"dhaka": 12_190_000.0})
    assert np.isclose(r["draw"][0], 2_000_000.0) and r["shortfall"][0] == 0.0         # sanctioned limit allows it


def test_margin_and_customs_use_cc_headroom_then_cash():
    r = _run(_debts(), [0.0], margin=[1_000_000.0], customs=[2_000_000.0], bal={"dhaka": 11_900_000.0})   # headroom 600k
    assert np.isclose(r["cc_margin"][0], 600_000.0) and r["cc_customs"][0] == 0.0
    assert np.isclose(r["cash_margin"][0], 400_000.0) and np.isclose(r["cash_customs"][0], 2_000_000.0)
    r = _run(_debts(), [0.0], margin=[1_000_000.0], customs=[0.0], funding=("cash", "cash"))
    assert r["cc_margin"][0] == 0.0 and r["cash_margin"][0] == 1_000_000.0


def test_deposits_go_only_where_entered_and_an_excess_is_flagged_not_spilled():
    d = _debts()
    d["deposits"] = [{"label": "Hospital", "amount": 30_000_000.0, "month": "2026-10", "dest": "ucb"}]
    r = _run(d, [0.0], bal={"ucb": 25_750_000.0, "dhaka": 12_190_000.0})
    assert np.isclose(r["dep_noncash"][0], 25_750_000.0) and np.isclose(r["bal"]["ucb"][0], 0.0)
    assert np.isclose(r["dep_unapplied"][0], 4_250_000.0)                        # NOT moved to Dhaka automatically
    assert np.isclose(r["bal"]["dhaka"][0], 12_190_000.0) and r["dep_cash"][0] == 0.0
    d["deposits"] = [{"label": "a", "amount": 25_750_000.0, "month": "2026-10", "dest": "ucb"},
                     {"label": "b", "amount": 3_000_000.0, "month": "2026-10", "dest": "dhaka"},
                     {"label": "c", "amount": 1_250_000.0, "month": "2026-10", "dest": "cash"}]
    r = _run(d, [0.0], bal={"ucb": 25_750_000.0, "dhaka": 12_190_000.0})
    assert np.isclose(r["bal"]["dhaka"][0], 9_190_000.0) and np.isclose(r["dep_cash"][0], 1_250_000.0)
    assert r["dep_unapplied"][0] == 0.0


def test_ucb_is_manual_by_default_nothing_pays_it_automatically():
    d = _debts()
    assert d["items"]["ucb"]["mode"] == "off"
    r = _run(d, [50_000_000.0], bal={"ucb": 25_750_000.0, "dhaka": 0.0, "director": 0.0})
    assert r["pay"]["ucb"][0] == 0.0
    d["items"]["ucb"].update({"mode": "lump", "amount": 2_000_000.0, "due_month": "2026-10"})   # a manual entry pays it
    assert _run(d, [50_000_000.0], bal={"ucb": 25_750_000.0})["pay"]["ucb"][0] == 2_000_000.0


def test_payouts_come_first_then_the_cc_tops_cash_back_up_to_the_floor():
    # cash 1.5M, nothing else: Malek's 1.0M is paid even though it leaves 0.5M < floor 1M, then the CC tops up
    r = _run(_debts(), [1_500_000.0], floor=1_000_000.0, bal={"dhaka": 5_000_000.0})
    assert r["pay"]["malek"][0] == 1_000_000.0
    assert np.isclose(r["draw"][0], 500_000.0 - 0.0 + 0.0) or np.isclose(r["draw"][0], 1_000_000.0 - (1_500_000.0 - 1_000_000.0 - 500_000.0))
    assert np.isclose(r["cash"][0], 1_000_000.0)                                  # held at the floor after payments
    # sweeps (CC paydown) only ever use cash ABOVE the floor
    r = _run(_debts(dhaka={"mode": "sweep"}), [3_000_000.0], floor=1_000_000.0, bal={"dhaka": 9_000_000.0, "malek": 0.0, "sunflower": 0.0,
                                                                              "mdloan": 0.0, "mdod": 0.0, "director": 0.0})
    assert np.isclose(r["pay"]["dhaka"][0], 2_000_000.0) and np.isclose(r["cash"][0], 1_000_000.0)


def test_ltr_interest_accrues_monthly_and_is_paid_in_cash_at_quarter_end():
    cols = ["Jul-26", "Aug-26", "Sep-26"]
    final = _final(cols)
    sched = pd.DataFrame(columns=["ref", "source", "value", "month", "note", "warning"])

    def interest_row(rate):
        s = cp.load_settings("100001", path="/nonexistent.json")
        s["horizon_months"] = 3
        s["debts"]["deposits"] = []
        s["cash_floor"] = 0.0
        s["imports"]["ltr_rate"] = rate
        t, pcols, _ = cp.project(final, cols, (2026, 9), 100_000_000.0, sched, s, ltr_open=3_000_000.0,
                                 shipments_hist=pd.DataFrame(), app_debts={"09040007": -27_000_000.0})
        return [float(t[t["Line"].str.strip() == "Interest & bank charges"][c].iloc[0]) for c in pcols]

    with_ltr, without = interest_row(0.1455), interest_row(0.0)
    diff = [a - b for a, b in zip(with_ltr, without)]
    assert diff[0] == 0.0 and diff[1] == 0.0                                    # Oct, Nov: accruing, nothing paid
    # LTR 3.0M run off over 3 months: outstanding 3M, 2M, 1M at the start of Oct, Nov, Dec
    assert np.isclose(diff[2], -(3.0 + 2.0 + 1.0) * 1_000_000 * 0.1455 / 12)    # paid in Dec (quarter-end)


def test_cc_interest_is_added_to_the_balance_only_at_quarter_end():
    r = _run(_debts(), [0.0] * 4, bal={"ucb": 10_000_000.0, "dhaka": 0.0})      # Oct, Nov, Dec, Jan
    assert r["int_cc"][0] == 0.0 and r["int_cc"][1] == 0.0
    assert np.isclose(r["int_cc"][2], 10_000_000 * 0.1455 / 12 * 3)             # charged in Dec
    assert r["bal"]["ucb"][2] > 10_000_000 and r["bal"]["ucb"][1] == 10_000_000


def test_md_loan_is_paid_in_its_due_month_and_projection_cash_chain_holds_with_debts():
    cols = ["Jul-26", "Aug-26", "Sep-26"]
    final = _final(cols)
    s = cp.load_settings("100001", path="/nonexistent.json")
    s["horizon_months"] = 6
    s["debts"]["deposits"] = []
    sched = pd.DataFrame(columns=["ref", "source", "value", "month", "note", "warning"])
    app_debts = {"10010003": -12_190_000.0, "10010006": -25_750_000.0, "10010007": -38_487_187.0, "09040007": -27_120_000.0,
                 "09040001": -3_000_000.0, "10020001": -600_000.0, "10020002": -2_750_000.0, "10020016": -330_000.0}
    t, pcols, info = cp.project(final, cols, (2026, 9), 100_000_000.0, sched, s, 0.0, pd.DataFrame(), app_debts=app_debts)
    g = lambda line, col: float(t[t["Line"].str.rstrip() == line.rstrip()][col].iloc[0])
    for c in pcols:
        assert np.isclose(g("Closing cash — model (opening + net change)", c), g("Opening cash & CE (app)", c) + g("NET CHANGE IN CASH", c))
        assert np.isclose(g("NET CHANGE IN CASH", c), g("TOTAL CASH IN", c) + g("TOTAL CASH OUT", c))
    assert np.allclose(info["closing_cash"], info["debts"]["cash"])           # table cash == engine cash, every month
    nov = pcols[1]
    assert g("   Loans repaid (directors, group, others)", nov) <= -600_000.0 + 1e-6
    assert g("   Other-party loans (0904: Malek Jobeda, Water Lily …)", pcols[0]) < 0
