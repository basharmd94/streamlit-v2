# processing/wf_curated_list.py
"""
Hand-curated customer list for WhatsFly Bulk Messaging -- an alternative
audience-building path alongside the filter builder (processing/
wf_bulk_audience.py): browse the full customer directory for a ZID (no
area/salesman/score/etc. filters at all -- deliberately just the raw
list, per explicit ask) and pick individual customers onto a saved list
by hand.

One curated list per ZID, not multiple named lists -- kept simple per
what was asked; revisit if more than one saved list is ever needed.

Persisted in the SAME whatsapp_webhooks database / streamlit_campaign_writer
role as processing/wf_bulk_campaign.py (see
whatsapp_webhook/add_curated_list_table.sql + add_curated_list_grants.sql)
-- reuses that module's own connection helper rather than duplicating the
config-loading logic, same pattern as processing/wf_template_mapping.py.
"""

import pandas as pd

from processing.wf_bulk_campaign import _get_conn, WfBulkCampaignDBConfigError  # noqa: F401 -- re-exported for callers


def list_curated(zid: str) -> pd.DataFrame:
    """Everything currently on the curated list for this ZID, newest
    addition first."""
    with _get_conn() as conn:
        return pd.read_sql(
            "SELECT * FROM curated_list_contacts WHERE zid = %s ORDER BY added_at DESC",
            conn, params=(zid,),
        )


def add_to_curated(zid: str, customers: list, added_by: str) -> None:
    """`customers`: [{"cusid", "cusname", "cusmobile", "whatsapp", "area"},
    ...] -- straight from cacus_directory rows the user picked out of the
    full list. Upsert per customer (UPDATE-then-INSERT-if-0-rows, same
    convention as every other upsert in this app) so re-adding an
    already-curated customer just refreshes their snapshot'd
    name/mobile/whatsapp/area instead of erroring against the UNIQUE
    (zid, cusid) constraint -- cacus_directory data can drift over time,
    so a re-add is a legitimate way to refresh a stale row, not just a
    no-op."""
    if not customers:
        return
    with _get_conn() as conn, conn.cursor() as cur:
        for c in customers:
            cur.execute(
                """
                UPDATE curated_list_contacts
                SET cusname = %s, cusmobile = %s, whatsapp = %s, area = %s, added_by = %s, added_at = now()
                WHERE zid = %s AND cusid = %s
                """,
                (c.get("cusname"), c.get("cusmobile"), c.get("whatsapp"), c.get("area"), added_by, zid, c["cusid"]),
            )
            if cur.rowcount == 0:
                cur.execute(
                    """
                    INSERT INTO curated_list_contacts (zid, cusid, cusname, cusmobile, whatsapp, area, added_by)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (zid, c["cusid"], c.get("cusname"), c.get("cusmobile"), c.get("whatsapp"), c.get("area"), added_by),
                )
        conn.commit()


def remove_from_curated(zid: str, cusid: str) -> None:
    """A real DELETE, not a flag flip -- same reasoning as
    wf_bulk_campaign.remove_opt_out: there's no "inactive" state worth
    preserving, and re-adding later is just a fresh row."""
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM curated_list_contacts WHERE zid = %s AND cusid = %s", (zid, cusid))
        conn.commit()
