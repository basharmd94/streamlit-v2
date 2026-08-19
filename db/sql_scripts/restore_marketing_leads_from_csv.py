#!/usr/bin/env python3
"""
Restore marketing_leads (and optionally marketing_lead_call_log) from CSV
files exported by hand -- e.g. via pgAdmin's own Export Data feature --
after the tables have been dropped and recreated with the new area/
lead_cost columns.

Usage:
    python db/sql_scripts/restore_marketing_leads_from_csv.py \
        --leads-csv marketing_leads.csv \
        --calllog-csv marketing_lead_call_log.csv

--calllog-csv is optional -- omit it to restore marketing_leads only.

Connects using config/global_db.ini by default, same as the rest of the
app (run this from within the repo checkout, or pass connection details
explicitly). Pass --host/--dbname/--user/--password/--port to point at a
different server instead -- handy for a dry run against a local/test copy
before doing it for real.

--- Why this needs to be more than "just re-insert the rows" ---
marketing_lead_call_log.lead_id is a foreign key straight to
marketing_leads.id. Letting SERIAL renumber restored rows from 1 would
silently repoint every call log at the wrong lead (or none at all) --
no error, just wrong data. This script explicitly includes `id` in the
INSERT column list for both tables, which inserts the literal value from
the CSV instead of invoking the SERIAL default, then resets both SERIAL
sequences afterward (via pg_get_serial_sequence/setval) so the next
lead/call-log saved through the app doesn't collide with a restored id.

Column-flexible by design: inserts whatever columns are actually present
in each CSV's header row (matched by name against a fixed allowlist, not
a hardcoded position or count), so it works whether the export happens to
include area/lead_cost or not, and regardless of column order -- you
don't need to pre-arrange anything before exporting from pgAdmin.

dtype=str on every CSV read, and encoding="utf-8-sig" -- pandas would
otherwise (a) infer numeric-looking columns like work_phone_number/
fb_lead_id/id as int64 and silently drop leading zeros (every Bangladeshi
phone number would come back corrupted), and (b) choke on the UTF-8 BOM
Windows tools like pgAdmin commonly prepend to CSV exports, which would
otherwise turn the first header into "﻿id" and make the required
`id` column look missing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

_LEADS_COLS = {
    "id", "zid", "fb_lead_id", "created_time", "ad_id", "ad_name", "adset_id",
    "adset_name", "campaign_id", "campaign_name", "form_id", "form_name",
    "is_organic", "platform", "full_name", "work_phone_number", "company_name",
    "street_address", "area", "job_title", "inbox_url", "lead_status",
    "lead_cost", "extra_fields", "lead_stage", "uploaded_by", "uploaded_at",
}
_CALLLOG_COLS = {
    "id", "zid", "lead_id", "called_at", "called_by", "outcome",
    "next_visit_date", "notes",
}


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.astype(object)
    # Blank cells read back as pandas NaN (a float) even under dtype=str --
    # left as-is, psycopg2 renders that as the literal 'NaN'::float, which
    # fails against every non-numeric column (jsonb, text, timestamp, ...).
    # Must become a real Python None so it binds as SQL NULL instead.
    df = df.where(pd.notna(df), None)
    # Some export tools also write NULL out as the literal text "NULL" --
    # catch that too, now that real gaps are already None (so this can't
    # accidentally match one).
    df = df.where(~df.isin(["NULL", "null"]), None)
    return df


def _restore_table(cur, table: str, allowed_cols: set, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    cols = list(df.columns)
    if "id" not in cols:
        raise ValueError(f"{table}: CSV has no 'id' column -- required to preserve original ids.")
    unknown = set(cols) - allowed_cols
    if unknown:
        raise ValueError(f"{table}: CSV has unrecognized column(s): {sorted(unknown)}")

    col_list = ", ".join(cols)
    sql = f"INSERT INTO {table} ({col_list}) VALUES %s"
    rows = [tuple(r) for r in df.itertuples(index=False, name=None)]
    execute_values(cur, sql, rows)
    return len(rows)


def _reset_sequence(cur, table: str) -> None:
    if table not in ("marketing_leads", "marketing_lead_call_log"):
        raise ValueError(f"Refusing to reset sequence for unexpected table: {table}")
    cur.execute(
        f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
        f"COALESCE((SELECT MAX(id) FROM {table}), 1))"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--leads-csv", required=True, help="Path to the marketing_leads export.")
    parser.add_argument("--calllog-csv", help="Path to the marketing_lead_call_log export (optional).")
    parser.add_argument("--host")
    parser.add_argument("--dbname")
    parser.add_argument("--user")
    parser.add_argument("--password")
    parser.add_argument("--port", type=int, default=5432)
    args = parser.parse_args()

    if args.host and args.dbname and args.user:
        db_params = {
            "host": args.host, "dbname": args.dbname, "user": args.user,
            "password": args.password, "port": args.port,
        }
    else:
        from config.settings import get_db_params
        db_params = get_db_params()

    print(f"Connecting to {db_params.get('dbname')}@{db_params.get('host')}...")
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()

    try:
        leads_df = _load_csv(args.leads_csv)
        print(f"Restoring {len(leads_df)} row(s) into marketing_leads...")
        n_leads = _restore_table(cur, "marketing_leads", _LEADS_COLS, leads_df)
        _reset_sequence(cur, "marketing_leads")

        n_calllog = 0
        if args.calllog_csv:
            calllog_df = _load_csv(args.calllog_csv)
            print(f"Restoring {len(calllog_df)} row(s) into marketing_lead_call_log...")
            n_calllog = _restore_table(cur, "marketing_lead_call_log", _CALLLOG_COLS, calllog_df)
            _reset_sequence(cur, "marketing_lead_call_log")

        conn.commit()
        print(f"\nDone. {n_leads} lead(s) and {n_calllog} call log(s) restored.")
        print("Compare these counts against what pgAdmin showed before you dropped the tables.")
    except Exception as e:
        conn.rollback()
        print(f"\nFAILED, rolled back -- nothing was written. {e}")
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
