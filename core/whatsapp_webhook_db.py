# core/whatsapp_webhook_db.py
"""
Read-only access to the whatsapp_webhook service's own Postgres database
(whatsapp_webhooks — separate from this app's `da` database, see
whatsapp_webhook/schema.sql and whatsapp_webhook/HANDOFF.md).

This app never writes here — that's the webhook receiver's own job
(whatsapp_webhook/db.py). This module exists solely for Marketing >
WhatsApp Message Log, so a message sent via the Direct WhatsApp panel (or a
real customer reply) can be verified end-to-end without leaving Streamlit.

Credentials: config/whatsapp_webhook_db.ini (gitignored, same convention as
config/direct_whatsapp.ini), via config.settings.get_whatsapp_webhook_db_params.
Use a dedicated, low-privilege, SELECT-only Postgres role here (not
webhook_svc, which the webhook service itself uses to write) — least
privilege, matching the role split already established when
whatsapp_webhook/ was deployed.

Plain psycopg2, one short-lived connection per call — same rationale as
whatsapp_webhook/db.py: this is a dashboard viewed occasionally, not a hot
path, so a connection pool (like core/db.py's ThreadedConnectionPool for the
main `da` database) would be premature here.
"""

from contextlib import contextmanager

import psycopg2
import psycopg2.extras

from config.settings import get_whatsapp_webhook_db_params


class WhatsAppWebhookDBConfigError(Exception):
    """config/whatsapp_webhook_db.ini is missing or incomplete."""


def _conn_params() -> dict:
    params = get_whatsapp_webhook_db_params()
    if not params:
        raise WhatsAppWebhookDBConfigError(
            "config/whatsapp_webhook_db.ini not found (or missing host/port/"
            "dbname/user/password). Create config/whatsapp_webhook_db.ini with:\n\n"
            "[whatsapp_webhook_db]\n"
            "host = YOUR_SERVER\n"
            "port = 5432\n"
            "dbname = whatsapp_webhooks\n"
            "user = streamlit_reader\n"
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


def get_counts() -> dict:
    """{'webhook_events': N, 'messages': N, 'contacts': N} — quick sanity
    numbers for the top of the Message Log view. Returns zeros (rather than
    raising) on any DB error, since this is a secondary readout, not the
    main table — a real config problem still surfaces via get_recent_messages."""
    try:
        with _get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT (SELECT COUNT(*) FROM webhook_events),
                          (SELECT COUNT(*) FROM messages),
                          (SELECT COUNT(*) FROM contacts)"""
            )
            events, messages, contacts = cur.fetchone()
            return {"webhook_events": events, "messages": messages, "contacts": contacts}
    except WhatsAppWebhookDBConfigError:
        raise
    except Exception:
        return {"webhook_events": 0, "messages": 0, "contacts": 0}


def get_recent_messages(limit: int = 100) -> list[dict]:
    """Most recent messages (both directions), newest first, with the
    contact's display name joined in when known."""
    with _get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """SELECT m.wamid, m.direction, m.contact_phone, c.name AS contact_name,
                      m.message_type, m.template_name, m.content, m.current_status,
                      m.message_timestamp, m.created_at
               FROM messages m
               LEFT JOIN contacts c ON c.phone_number = m.contact_phone
               ORDER BY COALESCE(m.message_timestamp, m.created_at) DESC
               LIMIT %s""",
            (limit,),
        )
        return list(cur.fetchall())


def get_status_history(wamid: str) -> list[dict]:
    """Full status timeline for one message (sent/delivered/read/failed),
    oldest first — for drilling into one specific test message."""
    with _get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """SELECT status, error_code, error_title, event_timestamp
               FROM message_status_events
               WHERE wamid = %s
               ORDER BY event_timestamp ASC""",
            (wamid,),
        )
        return list(cur.fetchall())
