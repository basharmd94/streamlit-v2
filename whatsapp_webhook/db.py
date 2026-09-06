# whatsapp_webhook/db.py
"""
Database access for the WhatsApp webhook receiver — a separate PostgreSQL
database, isolated from the main app's `da` database (see
../WhatsApp_Integration_docs/whatsapp-webhook-build.md, "Database" section
and schema.sql in this folder).

Plain psycopg2, one short-lived connection per call via get_conn(). Traffic
here is a handful of webhook POSTs a minute at most during local/sandbox
testing, so a connection pool (like core/db.py's ThreadedConnectionPool in
the main app) would be premature — add one if this ever needs real
production volume.
"""

import json
import os
from contextlib import contextmanager

import psycopg2

# Meta's status progression (sent -> delivered -> read, or failed at any
# point) — used to guard messages.current_status against being clobbered by
# an out-of-order duplicate. Meta's delivery is at-least-once, so a retried
# "sent" callback can legitimately arrive after "read" already landed.
_STATUS_RANK = {"sent": 1, "delivered": 2, "read": 3, "failed": 4}


def _conn_params() -> dict:
    return {
        "host": os.environ.get("WEBHOOK_DB_HOST", "localhost"),
        "port": os.environ.get("WEBHOOK_DB_PORT", "5432"),
        "dbname": os.environ["WEBHOOK_DB_NAME"],
        "user": os.environ["WEBHOOK_DB_USER"],
        "password": os.environ["WEBHOOK_DB_PASSWORD"],
    }


@contextmanager
def get_conn():
    conn = psycopg2.connect(**_conn_params())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def insert_webhook_event(conn, raw_payload: dict, signature_valid: bool) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO webhook_events (signature_valid, raw_payload)
               VALUES (%s, %s) RETURNING id""",
            (signature_valid, json.dumps(raw_payload)),
        )
        return cur.fetchone()[0]


def mark_event_processed(conn, event_id: int, status: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """UPDATE webhook_events SET processed_at = now(), processing_status = %s
               WHERE id = %s""",
            (status, event_id),
        )


def upsert_contact(conn, phone_number: str, wa_id, name) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO contacts (phone_number, wa_id, name)
               VALUES (%s, %s, %s)
               ON CONFLICT (phone_number) DO UPDATE SET
                   wa_id = EXCLUDED.wa_id,
                   name = COALESCE(EXCLUDED.name, contacts.name),
                   updated_at = now()""",
            (phone_number, wa_id, name),
        )


def insert_inbound_message(conn, *, wamid, phone_number_id, contact_phone,
                            message_type, content, message_timestamp) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO messages (wamid, direction, phone_number_id, contact_phone,
                                     message_type, content, current_status, message_timestamp)
               VALUES (%s, 'inbound', %s, %s, %s, %s, 'received', %s)
               ON CONFLICT (wamid) DO NOTHING""",
            (wamid, phone_number_id, contact_phone, message_type,
             json.dumps(content), message_timestamp),
        )


def ensure_outbound_stub(conn, *, wamid, phone_number_id, contact_phone) -> None:
    """A status callback can arrive for a message this service never saw
    sent — today, sends only go through the Streamlit app's
    core/direct_whatsapp.py, which doesn't write into this database (see
    the build doc's own send/receive split — this service only covers the
    receive side so far). Creates a minimal placeholder row on first sight
    of the wamid so message_status_events' FK has something to point at.
    ON CONFLICT DO NOTHING means a real send-side integration later just
    becomes a no-op here, preserving whichever row (real or stub) already
    exists rather than overwriting it."""
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO messages (wamid, direction, phone_number_id, contact_phone, message_type)
               VALUES (%s, 'outbound', %s, %s, 'unknown')
               ON CONFLICT (wamid) DO NOTHING""",
            (wamid, phone_number_id, contact_phone),
        )


def insert_status_event(conn, *, wamid, status, error_code, error_title,
                         event_timestamp, webhook_event_id) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO message_status_events
                   (wamid, status, error_code, error_title, event_timestamp, webhook_event_id)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (wamid, status) DO NOTHING""",
            (wamid, status, error_code, error_title, event_timestamp, webhook_event_id),
        )
        cur.execute("SELECT current_status FROM messages WHERE wamid = %s", (wamid,))
        row = cur.fetchone()
        current = row[0] if row else None
        if _STATUS_RANK.get(status, 0) >= _STATUS_RANK.get(current, 0):
            cur.execute(
                "UPDATE messages SET current_status = %s WHERE wamid = %s",
                (status, wamid),
            )


def upsert_template(conn, *, name, language, category, status) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO templates (name, language, category, status, last_checked_at)
               VALUES (%s, %s, %s, %s, now())
               ON CONFLICT (name, language) DO UPDATE SET
                   category = COALESCE(EXCLUDED.category, templates.category),
                   status = EXCLUDED.status,
                   last_checked_at = now()""",
            (name, language, category, status),
        )


def insert_account_alert(conn, *, event_type, payload) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO account_alerts (event_type, payload) VALUES (%s, %s)",
            (event_type, json.dumps(payload)),
        )
