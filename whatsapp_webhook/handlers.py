# whatsapp_webhook/handlers.py
"""
Per-field routing for a verified webhook payload — see "Webhook fields
reference" in ../WhatsApp_Integration_docs/whatsapp-webhook-build.md.

Runs as a FastAPI BackgroundTask (see main.py) so the webhook POST can ack
Meta with 200 immediately per the doc's "respond fast, process later" rule.
FastAPI's built-in BackgroundTasks is enough at this build phase's actual
scale (sandbox/local testing, a handful of events at a time) — a real task
queue (Celery/RQ) is production-scale machinery, deferred along with actual
deployment per the doc's own phasing.
"""

import logging
from datetime import datetime, timezone

import db as wh_db

logger = logging.getLogger("whatsapp_webhook")

_TEMPLATE_FIELDS = {
    "message_template_status_update",
    "message_template_quality_update",
    "message_template_components_update",
    "template_category_update",
}


def _ts(unix_ts):
    if unix_ts is None:
        return None
    try:
        return datetime.fromtimestamp(int(unix_ts), tz=timezone.utc)
    except (TypeError, ValueError):
        return None


def process_event(event_id: int, payload: dict) -> None:
    try:
        with wh_db.get_conn() as conn:
            for entry in payload.get("entry", []) or []:
                for change in entry.get("changes", []) or []:
                    _dispatch(conn, event_id, change.get("field", ""), change.get("value") or {})
        _mark(event_id, "processed")
    except Exception:
        logger.exception("Failed processing webhook_event id=%s", event_id)
        _mark(event_id, "failed")


def _mark(event_id: int, status: str) -> None:
    try:
        with wh_db.get_conn() as conn:
            wh_db.mark_event_processed(conn, event_id, status)
    except Exception:
        logger.exception("Also failed to mark webhook_event id=%s as %s", event_id, status)


def _dispatch(conn, event_id: int, field: str, value: dict) -> None:
    if field == "messages":
        _handle_messages_field(conn, event_id, value)
    elif field in _TEMPLATE_FIELDS:
        _handle_template_field(conn, value)
    else:
        # phone_number_quality_update, phone_number_name_update, account_update,
        # business_capability_update, account_review_update, account_alerts,
        # security, flows, and anything not yet enumerated — no dedicated
        # table shape for these per the doc, so land them in the generic
        # account_alerts audit table rather than dropping them.
        wh_db.insert_account_alert(conn, event_type=field or "unknown", payload=value)


def _handle_messages_field(conn, event_id: int, value: dict) -> None:
    phone_number_id = (value.get("metadata") or {}).get("phone_number_id")

    contacts_by_wa_id = {
        c.get("wa_id"): (c.get("profile") or {}).get("name")
        for c in value.get("contacts", []) or []
        if c.get("wa_id")
    }

    for msg in value.get("messages", []) or []:
        wa_id = msg.get("from")
        if not wa_id:
            continue
        wh_db.upsert_contact(conn, phone_number=wa_id, wa_id=wa_id, name=contacts_by_wa_id.get(wa_id))
        msg_type = msg.get("type", "unknown")
        # The type-specific payload (text/image/location/interactive/...) —
        # whichever key matches msg_type; falls back to the whole message
        # if a type shows up that isn't under a key we recognize.
        content = msg.get(msg_type, msg)
        wh_db.insert_inbound_message(
            conn,
            wamid=msg.get("id"),
            phone_number_id=phone_number_id,
            contact_phone=wa_id,
            message_type=msg_type,
            content=content,
            message_timestamp=_ts(msg.get("timestamp")),
        )

    for status in value.get("statuses", []) or []:
        wa_id = status.get("recipient_id")
        if not wa_id:
            continue
        wh_db.upsert_contact(conn, phone_number=wa_id, wa_id=wa_id, name=None)
        wh_db.ensure_outbound_stub(conn, wamid=status.get("id"), phone_number_id=phone_number_id, contact_phone=wa_id)
        errors = status.get("errors") or []
        first_error = errors[0] if errors else {}
        wh_db.insert_status_event(
            conn,
            wamid=status.get("id"),
            status=status.get("status", "unknown"),
            error_code=str(first_error["code"]) if first_error.get("code") is not None else None,
            error_title=first_error.get("title"),
            event_timestamp=_ts(status.get("timestamp")),
            webhook_event_id=event_id,
        )


def _handle_template_field(conn, value: dict) -> None:
    wh_db.upsert_template(
        conn,
        name=value.get("message_template_name") or value.get("template_name") or "unknown",
        language=value.get("message_template_language") or value.get("language"),
        category=value.get("message_template_category") or value.get("category"),
        status=value.get("event") or value.get("status") or "unknown",
    )
