# processing/wf_bulk_campaign.py
"""
Bulk Messaging campaign persistence + send engine (Phase 2 of
Whatsfly_Integration_docs/bulk-send-build-plan.md).

Deliberately generic and template-agnostic: this module knows nothing about
WHICH template is being sent or HOW a recipient's variables get built --
that stays each real campaign's own handler's job (see
views/marketing.py::_WF_BULK_TEMPLATE_VIEWS and Phase 6 of the build plan,
"collaborative, per-template" -- not a generic self-service mapping system).
This module only owns what's genuinely shared across every campaign:
exclusions (opt-out + per-template cooldown), campaign/recipient
persistence, and the send-loop mechanics (pacing, failure classification,
progress reporting).

Runs directly inside Streamlit now, not a separate background worker --
see the build plan's "Architecture, revised" section for why (campaigns
are staying small, max ~300 recipients; a background service is
infrastructure for a scale problem that doesn't exist yet).

DB access: a SEPARATE, narrow-write Postgres role (streamlit_campaign_writer,
config/whatsapp_webhook_campaign_db.ini) into the SAME whatsapp_webhooks
database core/whatsapp_webhook_db.py already reads from -- INSERT/UPDATE/
SELECT on exactly campaigns/campaign_recipients/contact_opt_outs, nothing
else (see whatsapp_webhook/add_streamlit_campaign_role.sql). Plain
psycopg2, one short-lived connection per call -- matches
core/whatsapp_webhook_db.py's own rationale (occasional feature, not a hot
path; a connection pool would be premature).
"""

import time
from contextlib import contextmanager

import pandas as pd
import psycopg2
import psycopg2.extras

from config.settings import get_whatsapp_webhook_campaign_db_params

DEFAULT_COOLDOWN_DAYS = 30
DEFAULT_PACE_SECONDS = 2.0


class WfBulkCampaignDBConfigError(Exception):
    """config/whatsapp_webhook_campaign_db.ini is missing or incomplete."""


def _conn_params() -> dict:
    params = get_whatsapp_webhook_campaign_db_params()
    if not params:
        raise WfBulkCampaignDBConfigError(
            "config/whatsapp_webhook_campaign_db.ini not found (or missing "
            "host/port/dbname/user/password). Create it with:\n\n"
            "[whatsapp_webhook_campaign_db]\n"
            "host = YOUR_SERVER\n"
            "port = 5432\n"
            "dbname = whatsapp_webhooks\n"
            "user = streamlit_campaign_writer\n"
            "password = YOUR_PASSWORD\n"
        )
    return params


@contextmanager
def _get_conn():
    conn = psycopg2.connect(**_conn_params())
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Exclusions -- computed before a campaign's recipient rows are ever
# inserted, each with an explicit count shown to the user (never a silent
# drop, same "never silently drop" pattern as the audience filter builder's
# phone-completeness gate).
#
# 100001/100000 combined scope: confirmed against real data (9,958 of 9,959
# cusids appear in BOTH ZIDs' cacus tables, same cusid on both sides in the
# overwhelming majority of cases) that `cacus` is genuinely a SHARED
# customer master between these two ZIDs specifically -- not two separate
# customer lists that happen to overlap. Opt-out and cooldown are both a
# promise to a real customer/phone number, so a customer opted out (or
# recently sent a template) while looking at 100001 must also be excluded
# from a campaign built under 100000, and vice versa -- otherwise the
# exclusion silently doesn't hold, since it never even checks the sibling
# ZID's rows. Same combined-scope treatment this app already gives 100001+
# 100000 elsewhere (Salesman Due's own combined-scope path, per CLAUDE.md).
# 100005/100009 have no evidence of this overlap (separate customer base,
# no shared sales team) and stay strictly single-ZID.
# ---------------------------------------------------------------------------

_SHARED_CUSTOMER_ZIDS = {"100001", "100000"}


def _combined_zids(zid: str) -> list:
    """[zid] normally; [100001, 100000] when zid is either of those two,
    since they share one real customer base (see module comment above)."""
    zid = str(zid)
    if zid in _SHARED_CUSTOMER_ZIDS:
        return sorted(_SHARED_CUSTOMER_ZIDS)
    return [zid]


def get_opted_out_cusids(zid: str) -> set:
    """cusids opted out for this zid (or, for 100001/100000, for either of
    the two combined ZIDs) -- excluded from every future campaign
    regardless of template, until manually un-opted (Phase 5)."""
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT cusid FROM contact_opt_outs WHERE zid = ANY(%s)", (_combined_zids(zid),))
        return {row[0] for row in cur.fetchall()}


def list_opt_outs(zid: str) -> pd.DataFrame:
    """Every currently opted-out customer for this ZID (combined with its
    sibling for 100001/100000 -- see module comment above, so the
    management page never shows a false "not opted out" for a customer
    who was opted out while viewing the other ZID), newest first -- the
    Opt-Out Management view's (Phase 5) main table."""
    with _get_conn() as conn:
        return pd.read_sql(
            "SELECT * FROM contact_opt_outs WHERE zid = ANY(%s) ORDER BY opted_out_at DESC",
            conn, params=(_combined_zids(zid),),
        )


def set_opt_out(zid: str, cusid: str, opted_out_by: str, reason: str = None) -> None:
    """Opts a customer out of every future bulk campaign for this ZID --
    upsert (UPDATE-then-INSERT-if-0-rows, same convention as every other
    upsert in this app) so re-opting-out an already-opted-out customer
    just refreshes the reason/timestamp instead of erroring against the
    UNIQUE (zid, cusid) constraint.

    The UPDATE half checks the COMBINED zid set (100001/100000 share one
    customer base -- see module comment above) so re-opting-out the same
    customer from the sibling ZID refreshes the existing row instead of
    creating a redundant second one; the INSERT half (only reached when
    truly nothing exists yet in either ZID) always writes under the
    zid actually passed in -- which ZID's row it lands under doesn't
    matter for enforcement, since every exclusion check already looks
    across the combined set too."""
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE contact_opt_outs SET opted_out_at = now(), opted_out_by = %s, reason = %s "
            "WHERE zid = ANY(%s) AND cusid = %s",
            (opted_out_by, reason, _combined_zids(zid), cusid),
        )
        if cur.rowcount == 0:
            cur.execute(
                "INSERT INTO contact_opt_outs (zid, cusid, opted_out_by, reason) VALUES (%s, %s, %s, %s)",
                (zid, cusid, opted_out_by, reason),
            )
        conn.commit()


def remove_opt_out(zid: str, cusid: str) -> None:
    """Opts a customer back in -- a real DELETE, not a flag flip, since
    there's no "inactive opt-out" state worth preserving; if they're
    opted out again later, that's a fresh row with its own reason/
    timestamp, not a revived old one.

    Deletes across the COMBINED zid set (100001/100000 -- see module
    comment above), not just the currently active one, so removing an
    opt-out actually removes it regardless of which ZID the original row
    happened to be stored under."""
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM contact_opt_outs WHERE zid = ANY(%s) AND cusid = %s", (_combined_zids(zid), cusid))
        conn.commit()


def get_cooldown_blocked_cusids(zid: str, template_id: str, cooldown_days, candidate_cusids) -> set:
    """cusids among `candidate_cusids` already sent THIS SAME template_id
    within the last `cooldown_days` days, checked across the COMBINED zid
    set (100001/100000 -- see module comment above; a template sent under
    100001 today counts against the cooldown for a 100000 campaign
    tomorrow, since it's the same real customer/phone number either way)
    -- per-template scope (explicit choice: a different template is never
    blocked by this), verified correct in Phase 1 against realistic
    scratch data. status='sent' only -- a failed attempt never reached
    the customer, so it shouldn't count toward the cooldown."""
    candidates = list(candidate_cusids)
    if not candidates:
        return set()
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT cr.cusid
            FROM campaign_recipients cr
            JOIN campaigns c ON c.id = cr.campaign_id
            WHERE cr.zid = ANY(%s)
              AND c.template_id = %s
              AND cr.status = 'sent'
              AND cr.sent_at >= now() - (%s || ' days')::interval
              AND cr.cusid = ANY(%s)
            """,
            (_combined_zids(zid), template_id, str(int(cooldown_days)), candidates),
        )
        return {row[0] for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# Campaign / recipient persistence
# ---------------------------------------------------------------------------

def create_campaign(*, zid: str, template_name: str, template_id: str,
                     header_image_url, variable_mapping: dict, filters_used: list,
                     cooldown_days: int, total_recipients: int, created_by: str) -> int:
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO campaigns
                (zid, template_name, template_id, header_image_url,
                 variable_mapping, filters_used, cooldown_days,
                 total_recipients, created_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (zid, template_name, template_id, header_image_url,
             psycopg2.extras.Json(variable_mapping or {}), psycopg2.extras.Json(filters_used or []),
             int(cooldown_days), int(total_recipients), created_by),
        )
        campaign_id = cur.fetchone()[0]
        conn.commit()
        return campaign_id


def insert_recipients(campaign_id: int, zid: str, recipients: list) -> None:
    """`recipients`: [{"cusid", "cusname", "phone_number", "variables"}, ...].
    `variables` should already be in the template's own variable_map
    position order -- this layer just stores it, it doesn't reorder
    anything."""
    if not recipients:
        return
    rows = [
        (campaign_id, zid, r["cusid"], r.get("cusname"), r["phone_number"],
         psycopg2.extras.Json(r.get("variables") or {}))
        for r in recipients
    ]
    with _get_conn() as conn, conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO campaign_recipients
                (campaign_id, zid, cusid, cusname, phone_number, variables_sent)
            VALUES %s
            """,
            rows,
        )
        conn.commit()


def mark_campaign_started(campaign_id: int) -> None:
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE campaigns SET status = 'in_progress', started_at = now() WHERE id = %s",
            (campaign_id,),
        )
        conn.commit()


def mark_campaign_completed(campaign_id: int) -> None:
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE campaigns SET status = 'completed', completed_at = now() WHERE id = %s",
            (campaign_id,),
        )
        conn.commit()


def get_pending_recipients(campaign_id: int) -> pd.DataFrame:
    """Only rows still 'pending' for this campaign -- this is what makes
    re-running after a crash naturally safe (Phase 1's crash-safety note):
    an already-attempted row (sent or failed) is never re-selected here,
    no separate resume/checkpoint logic needed."""
    with _get_conn() as conn:
        return pd.read_sql(
            "SELECT id, cusid, cusname, phone_number, variables_sent FROM campaign_recipients "
            "WHERE campaign_id = %s AND status = 'pending' ORDER BY id",
            conn, params=(campaign_id,),
        )


def update_recipient_result(recipient_id: int, *, status: str, wamid=None,
                             failure_type=None, error_detail=None) -> None:
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE campaign_recipients
            SET status = %s, wamid = %s, failure_type = %s, error_detail = %s,
                sent_at = CASE WHEN %s = 'sent' THEN now() ELSE sent_at END
            WHERE id = %s
            """,
            (status, wamid, failure_type, error_detail, status, recipient_id),
        )
        conn.commit()


def update_recipient_whatsfly_reconciliation(recipient_id: int, *, status, error_detail, raw) -> None:
    """Persists a manual WhatsFly status pull (Campaign History's
    "🔄 Reconcile Unconfirmed", views/marketing.py) for a recipient whose
    webhook never reported anything at all (see CLAUDE.md's 2026-09-15
    incident). `status`/`error_detail` are a best-effort parse of
    WhatsFly's get/message-status response -- shape not yet confirmed
    against the live account, see core/whatsfly.py::get_message_status --
    `raw` (the full response dict) is ALWAYS stored regardless, so a
    wrong parse guess doesn't lose the underlying data; it can be
    re-parsed from whatsfly_raw later without another live call."""
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE campaign_recipients
            SET whatsfly_status = %s, whatsfly_checked_at = now(),
                whatsfly_error_detail = %s, whatsfly_raw = %s
            WHERE id = %s
            """,
            (status, error_detail, psycopg2.extras.Json(raw) if raw is not None else None, recipient_id),
        )
        conn.commit()


def get_campaign_recipients(campaign_id: int) -> pd.DataFrame:
    """Full recipient list for a campaign -- feeds the end-of-loop summary
    here, and Phase 4's revisit-a-past-campaign view. SELECT * so the
    whatsfly_status/whatsfly_checked_at/whatsfly_error_detail/whatsfly_raw
    reconciliation columns (see update_recipient_whatsfly_reconciliation)
    flow through automatically, no query change needed when they're added."""
    with _get_conn() as conn:
        return pd.read_sql(
            "SELECT * FROM campaign_recipients WHERE campaign_id = %s ORDER BY id",
            conn, params=(campaign_id,),
        )


def list_campaigns(zid: str = None) -> pd.DataFrame:
    """Every campaign, newest first -- the Campaign History view's (Phase
    4) main table and the source for its aggregate overview. `zid=None`
    returns every ZID's campaigns, including the "TEST" sentinel used by
    the manual test-number-list mode (Phase 3) -- callers that want to
    exclude test runs from a real overview should filter zid != 'TEST'
    themselves, since whether to include them is a display choice, not
    something this layer should assume."""
    with _get_conn() as conn:
        if zid is None:
            return pd.read_sql("SELECT * FROM campaigns ORDER BY created_at DESC", conn)
        return pd.read_sql(
            "SELECT * FROM campaigns WHERE zid = %s ORDER BY created_at DESC", conn, params=(zid,),
        )


def update_campaign_cost(campaign_id: int, actual_cost) -> None:
    """Sets/updates campaigns.actual_cost -- always entered after the
    fact (see add_campaign_cost_column.sql: Meta's own bill is a
    calendar-month total covering every message sent that month, not a
    per-campaign figure Meta ever hands back), so this has no "set once"
    assumption -- re-saving a corrected number later is exactly the
    expected use, not a special case."""
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE campaigns SET actual_cost = %s WHERE id = %s",
            (actual_cost, campaign_id),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Curated List -- a hand-picked alternative to the filter-builder audience,
# for when the target list is better chosen by eye than by any filter (see
# views/marketing.py::_show_wf_curated_list). Deliberately reuses
# campaigns/campaign_recipients directly instead of its own table: a
# curated list genuinely IS a campaign's audience, just one built by hand
# and not sent yet -- see CLAUDE.md / bulk-send-build-plan.md for the
# design discussion this settled from (a dedicated curated_list_contacts
# table was built first, then dropped in favor of this reuse once it was
# pointed out there was no real need for a separate table).
#
# One "draft" campaigns row per ZID, found/created by
# _CURATED_SENTINEL_TEMPLATE_ID (an empty template_id -- a real campaign
# always has a real one, so this can never collide) -- stays at
# status='pending' for as long as it's just being curated (a real
# campaign moves to 'in_progress' the moment run_send_loop starts, so a
# lingering 'pending' row with no template is unambiguous). filters_used
# keeps its normal '[]' default, which already IS the "hand-curated, not
# filter-built" signal -- no extra column needed for that either.
# ---------------------------------------------------------------------------

_CURATED_SENTINEL_TEMPLATE_ID = ""


def find_curated_campaign_id(zid: str) -> int:
    """The existing curated-list campaign id for this ZID, or None if
    nothing has been curated yet."""
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM campaigns WHERE zid = %s AND template_id = %s ORDER BY created_at ASC LIMIT 1",
            (zid, _CURATED_SENTINEL_TEMPLATE_ID),
        )
        row = cur.fetchone()
        return row[0] if row else None


def get_or_create_curated_campaign_id(zid: str, created_by: str) -> int:
    """Reuses create_campaign() as-is -- the blank template_name/
    template_id satisfy those columns' NOT NULL constraint without
    needing a schema change (an empty string is not NULL)."""
    existing = find_curated_campaign_id(zid)
    if existing is not None:
        return existing
    return create_campaign(
        zid=zid, template_name="", template_id=_CURATED_SENTINEL_TEMPLATE_ID,
        header_image_url=None, variable_mapping={}, filters_used=[],
        cooldown_days=0, total_recipients=0, created_by=created_by,
    )


def _sync_campaign_total_recipients(campaign_id: int) -> None:
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE campaigns SET total_recipients = "
            "(SELECT COUNT(*) FROM campaign_recipients WHERE campaign_id = %s) WHERE id = %s",
            (campaign_id, campaign_id),
        )
        conn.commit()


def add_curated_recipients(campaign_id: int, zid: str, customers: list) -> int:
    """`customers`: [{"cusid", "cusname", "phone_number"}, ...] -- already
    resolved to a real WhatsApp-ready phone_number by the caller (see
    processing/common.py::customer_whatsapp_numbers), and already
    filtered by the caller to exclude anyone already on this campaign's
    list -- UNIQUE (campaign_id, cusid) would reject a literal duplicate
    anyway, but pre-filtering keeps the UI from ever offering an
    already-curated customer as if adding them again were a normal
    action. Reuses insert_recipients() as-is. Returns the count added."""
    if not customers:
        return 0
    insert_recipients(campaign_id, zid, customers)
    _sync_campaign_total_recipients(campaign_id)
    return len(customers)


def remove_curated_recipient(campaign_id: int, cusid: str) -> None:
    """A real DELETE -- safe only because this campaign has never
    started (see the module comment above: a real in-progress/completed
    campaign never deletes a recipient row, only a still-'pending'
    curated list does)."""
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM campaign_recipients WHERE campaign_id = %s AND cusid = %s",
            (campaign_id, cusid),
        )
        conn.commit()
    _sync_campaign_total_recipients(campaign_id)


# ---------------------------------------------------------------------------
# Failure classification -- a starting rule, not a settled one. Expected to
# be refined once real WhatsFly failure responses are actually seen (same
# evidence-based approach as the WhatsFly template-contract work
# elsewhere in this app), per the build plan.
# ---------------------------------------------------------------------------

def classify_send_outcome(resp=None, exc: Exception = None) -> tuple:
    """Returns (status, wamid, failure_type, error_detail).
    - A raised exception (network/timeout/connection) -> transient: the
      request never really reached WhatsFly's decision logic.
    - A non-2xx / explicit error envelope FROM WhatsFly -> permanent: a
      rejected request (e.g. bad number format) won't succeed on a blind
      resend.
    - Success -> 'sent', with whatever message id WhatsFly returned
      (best-effort key-guessing, same defensive stance as the rest of
      this account's WhatsFly integration -- the exact response shape for
      a template send isn't confirmed yet)."""
    if exc is not None:
        return "failed", None, "transient", str(exc)
    try:
        body = resp.json()
    except ValueError:
        body = None
    status_val = body.get("status") if isinstance(body, dict) else None
    is_ok = status_val in ("1", 1, True) or (resp.ok and status_val is None)
    if is_ok:
        wamid = None
        if isinstance(body, dict):
            for key in ("wa_message_id", "message_id", "wamid", "id"):
                v = body.get(key)
                if v:
                    wamid = str(v)
                    break
        return "sent", wamid, None, None
    if isinstance(body, dict):
        error_detail = str(body.get("message") or body)
    else:
        error_detail = (getattr(resp, "text", "") or "")[:500]
    return "failed", None, "permanent", error_detail


def run_send_loop(campaign_id: int, send_fn, pace_seconds: float = DEFAULT_PACE_SECONDS, progress_cb=None) -> dict:
    """The actual loop -- generic over `send_fn(phone_number, variables) ->
    requests.Response`, supplied by the caller (built from whichever
    template this campaign is using). Only ever processes rows still
    'pending' for `campaign_id` (see get_pending_recipients), so calling
    this again after a crash is safe by construction. `progress_cb(done,
    total)`, if given, is called after each recipient -- e.g. to drive an
    st.progress() bar. No retry (a failure is written once, then left
    alone); no stop/cancel path (nothing to wire one up to -- it's a plain
    loop, per the build plan's Q3/Q5 decisions).

    Returns {"sent": n, "failed": n, "total": n}."""
    pending = get_pending_recipients(campaign_id)
    total = len(pending)
    counts = {"sent": 0, "failed": 0, "total": total}
    for i, row in enumerate(pending.itertuples(index=False)):
        variables = row.variables_sent if isinstance(row.variables_sent, dict) else {}
        try:
            resp = send_fn(row.phone_number, variables)
            status, wamid, failure_type, error_detail = classify_send_outcome(resp=resp)
        except Exception as e:
            status, wamid, failure_type, error_detail = classify_send_outcome(exc=e)
        update_recipient_result(
            row.id, status=status, wamid=wamid, failure_type=failure_type, error_detail=error_detail,
        )
        counts[status] += 1
        if progress_cb:
            progress_cb(i + 1, total)
        if i < total - 1:
            time.sleep(pace_seconds)
    return counts
