"""Unit tests for processing/cash_model.py (Financial Statements -> Cash Model).

The contract that matters: the cash statement chains opening -> flows -> closing
and exposes the difference against the app's own Cash & CE, so a bad
categorisation or a sign error shows up as a non-zero "Difference" row.
"""
import numpy as np
import pandas as pd

from processing import cash_model as cm

SUBS = {
    "0105000104": "L/C Charge & Commission",
    "0105000105": "L/C Margin 100% Kungfu Int.",
    "0105000107": "Pay Order for Customs Tax & Vat",
    "0105000102": "Payment By T T For Kungfu Int",
    "0105000110": "C & F Charge",
}


def test_window_spans_previous_year_then_current_year():
    months = cm.window_months(2026, 9)
    assert months[0] == (2025, 1) and months[11] == (2025, 12)
    assert months[12] == (2026, 1) and months[-1] == (2026, 9) and len(months) == 21
    assert cm.month_label((2026, 3)) == "Mar-26"


def test_sub_kind_uses_names_not_codes():
    assert cm.sub_kind(SUBS["0105000107"]) == "payorder"
    assert cm.sub_kind(SUBS["0105000105"]) == "lc"          # margin = payment toward vendor value
    assert cm.sub_kind(SUBS["0105000104"]) == "lcchg"       # charge/commission = landed cost
    assert cm.sub_kind(SUBS["0105000102"]) == "tt"
    assert cm.sub_kind(SUBS["0105000110"]) == "other"
    # same code, different entity name -> different kind (GI's 0105000104 is a pay order)
    assert cm.sub_kind("Pay Order for Customs Tax & Vat") == "payorder"


def test_categorize_flow_inflows_and_outflows():
    assert cm.categorize_flow("01030001", None, "in") == ("Collections", "Customer receipts")
    assert cm.categorize_flow("01030003", None, "in")[0] == "Collections"        # recognized agent (GI)
    assert cm.categorize_flow("08030002", None, "in")[1].startswith("🔁 Rent")
    assert cm.categorize_flow("02050001", None, "in")[1].startswith("Hospital loan")
    assert cm.categorize_flow("10010003", None, "in")[1] == "Bank overdraft (CC) drawn"
    assert cm.categorize_flow("10010003", None, "out")[1] == "Bank overdraft (CC) repaid"
    assert cm.categorize_flow("09010002", None, "out")[1] == "Salaries, bonus & staff advances"
    assert cm.categorize_flow("07120002", None, "out")[1] == "Operating expenses"
    assert cm.categorize_flow("01050001", "0105000107", "out", SUBS)[1].startswith("International: pay order")
    assert cm.categorize_flow("01050001", "0105000105", "out", SUBS)[1].startswith("International: L/C")
    assert cm.categorize_flow("09060001", None, "out")[1].startswith("GI cash passing")


def _flows(rows):
    return pd.DataFrame(rows, columns=["year", "month", "ac_code", "xsub", "vtype", "inflow", "outflow", "lines"])


def test_cash_statement_chains_and_ties_to_app():
    months = [(2025, 1), (2025, 2)]
    flows = _flows([
        (2025, 1, "01030001", None, "RCT-", 1000.0, 0.0, 5),
        (2025, 1, "09030001", None, "CPAY", 0.0, -400.0, 3),
        (2025, 2, "01030001", None, "RCT-", 500.0, 0.0, 2),
        (2025, 2, "10010003", None, "BTJV", 300.0, -100.0, 2),
    ])
    app_closing = {(2025, 1): 1600.0, (2025, 2): 2300.0}
    cash = cm.build_cash_statement(flows, pd.DataFrame(), app_closing, 1000.0, months, SUBS)
    get = lambda line, col: float(cash[cash["Line"] == line][col].iloc[0])
    assert get("Opening cash & CE (app)", "Jan-25") == 1000.0
    assert get("Opening cash & CE (app)", "Feb-25") == 1600.0             # previous column's APP closing
    assert get("NET CHANGE IN CASH", "Jan-25") == 600.0
    assert get("Closing cash — model (opening + net change)", "Jan-25") == 1600.0
    assert get("Difference (model − app)", "Jan-25") == 0.0
    assert get("Difference (model − app)", "Feb-25") == 0.0               # 1600 + 500 + 300 - 100 = 2300
    assert get("TOTAL CASH OUT", "Feb-25") == -100.0                      # CC repayment is a cash outflow


def test_cash_statement_difference_row_flags_a_missing_flow():
    months = [(2025, 1)]
    flows = _flows([(2025, 1, "01030001", None, "RCT-", 1000.0, 0.0, 1)])
    cash = cm.build_cash_statement(flows, pd.DataFrame(), {(2025, 1): 1500.0}, 0.0, months, SUBS)
    diff = float(cash[cash["Line"] == "Difference (model − app)"]["Jan-25"].iloc[0])
    assert diff == -500.0


def test_cc_funded_payments_are_memo_only_not_cash():
    months = [(2025, 1)]
    flows = _flows([(2025, 1, "01030001", None, "RCT-", 100.0, 0.0, 1)])
    cc = pd.DataFrame([{"year": 2025, "month": 1, "ac_code": "01050001", "xsub": "0105000107", "amount": 700.0}])
    cash = cm.build_cash_statement(flows, cc, {(2025, 1): 100.0}, 0.0, months, SUBS)
    assert float(cash[cash["Line"] == "NET CHANGE IN CASH"]["Jan-25"].iloc[0]) == 100.0   # CC-funded did not move cash
    memo = cash[cash["Line"] == "Total paid from CC overdraft"]["Jan-25"].iloc[0]
    assert memo == -700.0


def test_parse_kf_matches_shipment_and_remark_formats():
    assert cm.parse_kf("MDKF008/25,2025-12-17") == (8, 25)
    assert cm.parse_kf("Purchase bill Payment Against L/C KF 008/2025") == (8, 25)
    assert cm.parse_kf("Transfer to Vat Rebate Against L/C KF007/2025") == (7, 25)
    assert cm.parse_kf("no shipment here") is None


def test_shipments_split_capitalised_pay_order_from_rebate_transfer():
    ships = pd.DataFrame([{"po": "IP--1", "shipment": "MDKF008/25,2025-12-17", "grn": "IGRN1",
                           "grn_date": "2025-12-17", "vendor_value": 1000.0, "currency": "BDT"}])
    settle = pd.DataFrame([
        {"year": 2025, "month": 12, "voucher": "J1", "xsub": "0105000107", "amount": 100.0, "remark": "Payment Against L/C KF 008/2025"},
        {"year": 2026, "month": 1, "voucher": "J2", "xsub": "0105000107", "amount": 40.0, "remark": "Transfer to Vat Rebate Against L/C KF008/2025"},
        {"year": 2025, "month": 12, "voucher": "J1", "xsub": "0105000110", "amount": 30.0, "remark": "Payment Against L/C KF 008/2025"},
        {"year": 2025, "month": 12, "voucher": "J1", "xsub": "0105000105", "amount": 900.0, "remark": "Payment Against L/C KF 008/2025"},
    ])
    out = cm.build_shipments(ships, settle, SUBS).iloc[0]
    assert out["Pay order (capitalised)"] == 100.0
    assert out["VAT/AT rebate transfer"] == 40.0
    assert out["Other import costs"] == 30.0
    assert out["Landed cost"] == 1000.0 + 100.0 + 30.0          # margin is financing, rebate is reclaimable


def test_inventory_ledger_closes_exactly_and_prices_use_trailing_sales():
    months = [(2025, 1), (2025, 2)]
    stock = pd.DataFrame([
        {"itemcode": "A", "doctype": "GRN-", "ym": 0, "qty": 10.0},          # opening before window
        {"itemcode": "A", "doctype": "DO--", "ym": 202501, "qty": -4.0},
        {"itemcode": "A", "doctype": "GRN-", "ym": 202502, "qty": 6.0},
    ])
    sales = pd.DataFrame([{"itemcode": "A", "ym": 202501, "units": 4.0, "gross": 400.0, "discount": 0.0}])
    returns = pd.DataFrame([{"year": 2025, "month": 1, "amount": 50.0}])
    inv, _ = cm.build_inventory_sales(stock, sales, returns, pd.DataFrame(), months)
    row = lambda name, col: float(inv[inv["Line"].str.strip() == name][col].iloc[0])
    assert row("Closing inventory", "Jan-25") == 6 * 100.0               # price = 400/4 = 100
    assert row("Closing inventory", "Feb-25") == 12 * 100.0
    chk = inv[inv["Line"].str.contains("check")].iloc[0]
    assert abs(float(chk["Jan-25"])) < 1e-6 and abs(float(chk["Feb-25"])) < 1e-6
    assert row("Net sales (after returns)", "Jan-25") == 400.0 - 50.0
    assert np.isclose(row("% of opening inventory sold", "Jan-25"), 350.0 / 1000.0)
