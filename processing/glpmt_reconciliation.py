# processing/glpmt_reconciliation.py
"""
Per-customer reconciliation for the "App Collections" panel
(views/glpmt_shared.py) -- redesigned per explicit ask, once it became
clear the real point of this table was never "list every payment promise
a salesman logged," but to verify the WHOLE story completed: a delivery
(DO) happened, the salesman logged the customer's promised payment date
on their behalf, and (hopefully) a real collection (RCT) came in
afterward. This module builds one row PER CUSTOMER (not per glpmt entry)
-- their latest DO, latest RCT, latest glpmt entry, and their current AR
balance -- side by side, so staff can see at a glance whether the chain
actually completed.
"""

import numpy as np
import pandas as pd

from processing.salesman_due import prep_ar_ledger, build_latest_sale_collection_report


def _latest_do_per_customer(sales_df: pd.DataFrame) -> pd.DataFrame:
    """One row per customer: their single most recent DO -- date, voucher
    number, and total amount (final_sales, summed across that DO's own
    line items -- one DO can have several)."""
    cols = ["cusid", "do_date", "do_number", "do_amount"]
    if sales_df is None or sales_df.empty:
        return pd.DataFrame(columns=cols)
    d = sales_df.copy()
    d["cusid"] = d["cusid"].astype(str)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date", "voucher"])
    if d.empty:
        return pd.DataFrame(columns=cols)
    per_do = d.groupby(["cusid", "voucher"], as_index=False).agg(
        do_date=("date", "first"), do_amount=("final_sales", "sum"),
    )
    per_do = per_do.sort_values(["cusid", "do_date"], kind="mergesort")
    latest = per_do.groupby("cusid", as_index=False).last()
    return latest.rename(columns={"voucher": "do_number"})[cols]


def _latest_rct_per_customer(collection_df: pd.DataFrame) -> pd.DataFrame:
    """One row per customer: their single most recent RCT (collection
    voucher, mv_collection_vouchers -- one row per voucher already) --
    date, voucher number, and amount."""
    cols = ["cusid", "rct_date", "rct_number", "rct_amount"]
    if collection_df is None or collection_df.empty:
        return pd.DataFrame(columns=cols)
    d = collection_df.copy()
    d["cusid"] = d["cusid"].astype(str)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])
    if d.empty:
        return pd.DataFrame(columns=cols)
    d = d.sort_values(["cusid", "date"], kind="mergesort")
    latest = d.groupby("cusid", as_index=False).last()
    return latest.rename(
        columns={"glvoucher": "rct_number", "date": "rct_date", "value": "rct_amount"},
    )[cols]


def _latest_glpmt_entry_per_customer(glpmt_df: pd.DataFrame) -> pd.DataFrame:
    """One row per customer: their most recently ENTERED glpmt row (by
    entry_time, i.e. ztime -- the latest ACTION the salesman took, not
    just the latest promised payment date on file). Carries that row's
    own paydate/payamt/paytype/bankdetail/paystatus/remarks along with
    it, since those describe that one specific entry."""
    if glpmt_df is None or glpmt_df.empty:
        return pd.DataFrame()
    d = glpmt_df.copy()
    d["cusid"] = d["cusid"].astype(str)
    d["entry_time"] = pd.to_datetime(d["entry_time"], errors="coerce")
    d["paydate"] = pd.to_datetime(d["paydate"], errors="coerce")
    d = d.dropna(subset=["entry_time"]).sort_values(["cusid", "entry_time"], kind="mergesort")
    return d.groupby("cusid", as_index=False).last()


def _after_flag(df: pd.DataFrame, later_col: str, earlier_col: str) -> np.ndarray:
    """'✅' when later_col is genuinely after earlier_col, '⚠️' when it's
    at-or-before (the check failed), '' when either side is missing --
    a missing date means "nothing to check yet", not "failed the check"."""
    both = df[later_col].notna() & df[earlier_col].notna()
    after = both & (df[later_col] > df[earlier_col])
    return np.select([~both, after], ["", "✅"], default="⚠️")


def build_glpmt_reconciliation(
    glpmt_df: pd.DataFrame, sales_df: pd.DataFrame, collection_df: pd.DataFrame, ar_ledger_df: pd.DataFrame,
) -> pd.DataFrame:
    """The redesigned App Collections table -- one row per customer who
    has at least one glpmt entry (glpmt_df is expected to already be
    filtered by the caller, e.g. by salesman/customer/date-of-entry --
    same filters the panel already had), showing the whole delivery ->
    promised-payment -> actual-collection chain side by side.

    Customer Balance reuses build_latest_sale_collection_report's own AR
    balance calc UNCHANGED, including its existing near-zero filter
    (|balance| < 100 is dropped entirely) -- exactly the "if the balance
    goes close to 0 ... remove that customer from the table" behavior
    asked for, already proven elsewhere in this app (Collection Analysis
    -> Salesman Due) rather than reimplemented here. The merge onto it is
    an INNER join, so a near-zero-balance customer silently drops out of
    this report too, by construction, not a separate filter step.

    Two sanity checks, per explicit ask -- both anchored on the DO date,
    both '' (not a fail) when either date is missing:
    - "Payment After DO": is the salesman's promised payment date after
      the DO it's presumably for?
    - "RCT After DO": is the actual collection date after that DO?
    """
    if glpmt_df is None or glpmt_df.empty:
        return pd.DataFrame()

    latest_glpmt = _latest_glpmt_entry_per_customer(glpmt_df)
    if latest_glpmt.empty:
        return pd.DataFrame()

    # Scope the (potentially full-history, all-customer) sales/collection
    # pulls down to just the customers actually in play here first --
    # cheap, and keeps the per-customer groupbys below from doing
    # wasted work on customers with no glpmt entry at all.
    cust_ids = set(latest_glpmt["cusid"])
    sales_scoped = sales_df[sales_df["cusid"].astype(str).isin(cust_ids)] if sales_df is not None and not sales_df.empty else sales_df
    collection_scoped = collection_df[collection_df["cusid"].astype(str).isin(cust_ids)] if collection_df is not None and not collection_df.empty else collection_df

    latest_do = _latest_do_per_customer(sales_scoped)
    latest_rct = _latest_rct_per_customer(collection_scoped)

    balance_report = build_latest_sale_collection_report(prep_ar_ledger(ar_ledger_df))
    balance = balance_report[["Customer Code", "Current Balance"]].rename(
        columns={"Customer Code": "cusid", "Current Balance": "balance"},
    )
    balance["cusid"] = balance["cusid"].astype(str)

    out = latest_glpmt.merge(latest_do, on="cusid", how="left")
    out = out.merge(latest_rct, on="cusid", how="left")
    out = out.merge(balance, on="cusid", how="inner")

    if out.empty:
        return out

    out["Payment After DO"] = _after_flag(out, "paydate", "do_date")
    out["RCT After DO"] = _after_flag(out, "rct_date", "do_date")

    return out
