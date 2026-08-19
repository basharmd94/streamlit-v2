# processing/marketing_leads.py
# Pure pandas transforms for the Marketing Leads CRM — no st.* calls.
# Covers: CSV/Excel upload normalization, individual-lead summary table
# (Table 1), and the all-call-logs table (Table 2).

from __future__ import annotations

import json
import uuid

import numpy as np
import pandas as pd

# ── Facebook Lead Ads export — known columns map 1:1 to marketing_leads. ────
# Any other column (different lead forms carry different custom questions —
# e.g. a Bengali institution-type question that won't recur on every form)
# is preserved per-row in extra_fields instead of requiring a schema change
# every time a new lead form ships.
_FIXED_COLS = [
    "created_time", "ad_id", "ad_name", "adset_id", "adset_name",
    "campaign_id", "campaign_name", "form_id", "form_name",
    "is_organic", "platform", "full_name", "work_phone_number",
    "company_name", "street_address", "job_title", "inbox_url", "lead_status",
]
_ID_COL = "id"  # -> fb_lead_id (renamed to avoid clashing with our own serial PK)


def _to_bool(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "t"):
        return True
    if s in ("false", "0", "no", "f"):
        return False
    return None


def parse_leads_upload(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize an uploaded Facebook Lead Ads CSV/Excel export.

    Returns one row per lead: fb_lead_id + the fixed columns + extra_fields
    (a JSON string per row, or None). zid / uploaded_by are NOT set here —
    the view adds them (from session state) before insert, since the export
    never carries them.

    Raises ValueError if the file doesn't look like a lead export (no 'id' col).
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if _ID_COL not in df.columns:
        raise ValueError(
            "Upload is missing the 'id' column (the lead-gen platform's own lead id). "
            "Is this a Facebook Lead Ads export?"
        )

    df = df.rename(columns={_ID_COL: "fb_lead_id"})
    df["fb_lead_id"] = df["fb_lead_id"].astype(str).str.strip()
    df = df[df["fb_lead_id"] != ""].copy()
    df = df.drop_duplicates(subset=["fb_lead_id"])  # de-dup within the same upload batch

    known = set(_FIXED_COLS) | {"fb_lead_id"}
    extra_cols = [c for c in df.columns if c not in known]

    if extra_cols:
        def _row_extra(row):
            d = {c: row[c] for c in extra_cols if pd.notna(row[c]) and str(row[c]).strip() != ""}
            return json.dumps(d, ensure_ascii=False) if d else None
        df["extra_fields"] = df.apply(_row_extra, axis=1)
    else:
        df["extra_fields"] = None

    for col in _FIXED_COLS:
        if col not in df.columns:
            df[col] = None

    df["created_time"] = pd.to_datetime(df["created_time"], errors="coerce", utc=True)
    df["is_organic"] = df["is_organic"].apply(_to_bool)

    out_cols = ["fb_lead_id"] + _FIXED_COLS + ["extra_fields"]
    out = df[out_cols].copy()
    out = out.astype(object).where(pd.notna(out), None)  # NaN/NaT -> None for a clean DB insert
    return out.reset_index(drop=True)


def build_leads_upload_template() -> pd.DataFrame:
    """One-row example CSV in exactly the shape parse_leads_upload expects —
    a clean, English-only column set for a CRM manager to fill in by hand and
    re-upload, instead of a real Facebook Lead Ads export whose custom
    per-form questions (e.g. a Bengali institution-type question) land as
    unlabeled extra_fields and cause confusion about what each column means.

    Columns exactly match _ID_COL + _FIXED_COLS, in the same order the
    parser reads them in -- extending this template and _FIXED_COLS out of
    sync would silently start routing a "new" column into extra_fields
    again, so keep them together if either ever changes.
    """
    example = {
        "id": "LEAD-0001",
        "created_time": "2026-01-15 10:30:00",
        "ad_id": "", "ad_name": "", "adset_id": "", "adset_name": "",
        "campaign_id": "", "campaign_name": "", "form_id": "", "form_name": "",
        "is_organic": "", "platform": "Manual",
        "full_name": "Jane Doe",
        "work_phone_number": "01711234567",
        "company_name": "ABC Traders",
        "street_address": "123 Main Road, Dhaka",
        "job_title": "Purchase Manager",
        "inbox_url": "", "lead_status": "",
    }
    out_cols = ["id"] + _FIXED_COLS
    return pd.DataFrame([example])[out_cols]


def build_manual_lead_row(
    full_name: str,
    work_phone_number: str,
    company_name: str = "",
    job_title: str = "",
    street_address: str = "",
    notes: str = "",
) -> pd.DataFrame:
    """One-row DataFrame in the same shape as parse_leads_upload's output, for
    a lead entered by hand (phone call / walk-in) rather than a platform export.

    fb_lead_id is a synthetic id ("manual-<hex>") — it still works as the
    cacus.xurl join key for conversion tracking, exactly like a real Facebook
    lead id; staff just paste this generated id instead.
    """
    fb_lead_id = f"manual-{uuid.uuid4().hex[:12]}"
    notes = (notes or "").strip()
    extra = {"notes": notes} if notes else None

    row = {
        "fb_lead_id": fb_lead_id,
        "created_time": pd.Timestamp.now(tz="UTC"),
        "ad_id": None, "ad_name": None, "adset_id": None, "adset_name": None,
        "campaign_id": None, "campaign_name": None, "form_id": None, "form_name": None,
        "is_organic": None,
        "platform": "manual",
        "full_name": (full_name or "").strip(),
        "work_phone_number": (work_phone_number or "").strip(),
        "company_name": (company_name or "").strip() or None,
        "street_address": (street_address or "").strip() or None,
        "job_title": (job_title or "").strip() or None,
        "inbox_url": None,
        "lead_status": "manual",
        "extra_fields": json.dumps(extra, ensure_ascii=False) if extra else None,
    }
    out_cols = ["fb_lead_id"] + _FIXED_COLS + ["extra_fields"]
    return pd.DataFrame([row])[out_cols]


def build_lead_summary_table(
    leads_df: pd.DataFrame,
    cacus_links_df: pd.DataFrame,
    call_logs_df: pd.DataFrame,
) -> pd.DataFrame:
    """Table 1 — one row per lead: latest call info + converted customer code.

    leads_df:       marketing_leads rows for this ZID (id, fb_lead_id, ...)
    cacus_links_df: cusid, cusname, fb_lead_id — cacus.xurl matches, live
    call_logs_df:   all call log rows for this ZID (any lead)
    """
    if leads_df is None or leads_df.empty:
        return pd.DataFrame()

    out = leads_df.copy()

    # ── Converted customer code, via cacus.xurl == fb_lead_id ────────────────
    if cacus_links_df is not None and not cacus_links_df.empty:
        links = (
            cacus_links_df[["cusid", "cusname", "fb_lead_id"]]
            .dropna(subset=["fb_lead_id"])
            .drop_duplicates("fb_lead_id")
        )
        out = out.merge(links, on="fb_lead_id", how="left")
    else:
        out["cusid"] = None
        out["cusname"] = None

    # ── Latest call per lead ──────────────────────────────────────────────────
    if call_logs_df is not None and not call_logs_df.empty:
        cl = call_logs_df.copy()
        cl["called_at"] = pd.to_datetime(cl["called_at"], errors="coerce")
        latest = (
            cl.sort_values("called_at")
              .groupby("lead_id", as_index=False)
              .last()[["lead_id", "called_at", "outcome", "next_visit_date", "notes"]]
              .rename(columns={
                  "called_at": "last_called",
                  "outcome": "last_outcome",
                  "notes": "last_notes",
              })
        )
        out = out.merge(latest, left_on="id", right_on="lead_id", how="left")
        out = out.drop(columns=["lead_id"], errors="ignore")
    else:
        out["last_called"] = None
        out["last_outcome"] = None
        out["next_visit_date"] = None
        out["last_notes"] = None

    return out.sort_values("created_time", ascending=False, na_position="last").reset_index(drop=True)


def build_lead_call_log_table(call_logs_df: pd.DataFrame) -> pd.DataFrame:
    """Table 2 — every call log row for the ZID, newest first, ready for
    date-called / next-visit filtering at the view level."""
    if call_logs_df is None or call_logs_df.empty:
        return pd.DataFrame()
    df = call_logs_df.copy()
    df["called_at"] = pd.to_datetime(df["called_at"], errors="coerce")
    df["next_visit_date"] = pd.to_datetime(df["next_visit_date"], errors="coerce")
    return df.sort_values("called_at", ascending=False).reset_index(drop=True)
