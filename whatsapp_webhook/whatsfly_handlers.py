# whatsapp_webhook/whatsfly_handlers.py
"""
Per-event-type processing for WhatsFly's own webhook deliveries — see
main.py's WhatsFly section for the route/token/registration setup.

Separate from handlers.py (Meta-direct events): WhatsFly's payload shapes
are completely different — flat, no entry[].changes[] nesting, no shared
"messages" field carrying both directions, no explicit event-type field on
some triggers. Confirmed against real captured payloads for all three
functions here (see git history for the exact examples); Conversation
Status Change has no confirmed shape yet and stays phase-1 raw-capture-only
in main.py — no handler for it here until one arrives.

Each function runs as a FastAPI BackgroundTask, same respond-fast-
process-later pattern as the Meta side. WhatsFly's delivery is not
ordering-guaranteed either — confirmed in practice: a "delivered" status
event for one message arrived at our webhook before that same message's
own Outgoing Message event. process_status/process_outgoing are written
to produce the same correct end state regardless of which one runs first.
"""

import logging
import re
from datetime import datetime, timezone

import db as wh_db

logger = logging.getLogger("whatsapp_webhook")

# WhatsFly prefixes an outgoing message's text with this when it had a
# media header — confirmed: "#ATTACHMENT:image#<template text>" on a real
# template send with an image header. No media URL/id anywhere in the
# payload, so the attachment's actual content is unrecoverable from this
# event alone — this only tells us THAT one was there, and what kind.
_ATTACHMENT_RE = re.compile(r"^#ATTACHMENT:([a-zA-Z0-9_]+)#")


def _mark(event_id: int, status: str) -> None:
    try:
        with wh_db.get_conn() as conn:
            wh_db.mark_event_processed(conn, event_id, status)
    except Exception:
        logger.exception("Failed to mark webhook_event id=%s as %s", event_id, status)


def _parse_status_time(status_time):
    """WhatsFly's status_time is a naive "YYYY-MM-DD HH:MM:SS" string,
    confirmed UTC — cross-checked a real delivered/read pair's status_time
    against that same webhook_events row's own received_at (+06 offset)."""
    if not status_time:
        return None
    try:
        return datetime.strptime(status_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        logger.warning("Unparseable status_time: %r", status_time)
        return None


def _strip_attachment_marker(text):
    if not text:
        return text, None
    m = _ATTACHMENT_RE.match(text)
    if not m:
        return text, None
    return text[m.end():], m.group(1)


def process_incoming(event_id: int, payload: dict) -> None:
    """Confirmed shape: chat_id, first_name, label_names, user_message,
    subscriber_id, wa_message_id, whatsapp_bot_id/name/username. No
    explicit event-type field (unlike Message Status Change), no
    timestamp, and only text messages confirmed so far — an incoming
    image/media message's shape is unconfirmed, so message_type is
    hardcoded 'text' until a real example shows otherwise."""
    try:
        chat_id = payload.get("chat_id")
        wamid = payload.get("wa_message_id")
        if not chat_id or not wamid:
            logger.warning("Incoming Message event id=%s missing chat_id/wa_message_id — skipped", event_id)
            _mark(event_id, "failed")
            return

        # whatsapp_bot_id is WhatsFly's own per-number id, NOT the same
        # value as the Meta phone_number_id used elsewhere in this schema
        # for Direct-WhatsApp-sourced rows — see the code review note from
        # when this distinction was first found. Stored in the same column
        # since it serves the same role (which connected number this
        # belongs to), just a different source's id space.
        phone_number_id = str(payload.get("whatsapp_bot_id") or "")

        with wh_db.get_conn() as conn:
            wh_db.upsert_contact(conn, phone_number=chat_id, wa_id=chat_id, name=payload.get("first_name"))
            wh_db.insert_inbound_message(
                conn,
                wamid=wamid,
                phone_number_id=phone_number_id,
                contact_phone=chat_id,
                message_type="text",
                content={"body": payload.get("user_message")},
                message_timestamp=None,
            )
        _mark(event_id, "processed")
    except Exception:
        logger.exception("Failed processing Incoming Message event id=%s", event_id)
        _mark(event_id, "failed")


def process_outgoing(event_id: int, payload: dict) -> None:
    """Confirmed shape: same fields as Incoming Message, plus agent_name
    (present on outgoing, never seen on an incoming row) and no explicit
    event-type field here either — identified by which route/token it
    arrived on (main.py), not by payload content."""
    try:
        chat_id = payload.get("chat_id")
        wamid = payload.get("wa_message_id")
        if not chat_id or not wamid:
            logger.warning("Outgoing Message event id=%s missing chat_id/wa_message_id — skipped", event_id)
            _mark(event_id, "failed")
            return

        phone_number_id = str(payload.get("whatsapp_bot_id") or "")
        body, attachment_type = _strip_attachment_marker(payload.get("user_message"))
        content = {"body": body}
        if attachment_type:
            content["attachment_type"] = attachment_type
        if payload.get("agent_name"):
            content["agent_name"] = payload["agent_name"]

        with wh_db.get_conn() as conn:
            wh_db.upsert_contact(conn, phone_number=chat_id, wa_id=chat_id, name=payload.get("first_name"))
            wh_db.upsert_outbound_message(
                conn,
                wamid=wamid,
                phone_number_id=phone_number_id,
                contact_phone=chat_id,
                message_type=attachment_type or "text",
                content=content,
            )
        _mark(event_id, "processed")
    except Exception:
        logger.exception("Failed processing Outgoing Message event id=%s", event_id)
        _mark(event_id, "failed")


def process_status(event_id: int, payload: dict) -> None:
    """Confirmed shape: the one WhatsFly trigger with an explicit type
    field ("webhook_type": "message_status_change"). message_status values
    confirmed so far ("delivered", "read") match db._STATUS_RANK's
    vocabulary exactly — no mapping needed, passed straight through."""
    try:
        chat_id = payload.get("chat_id")
        wamid = payload.get("wa_message_id")
        status = payload.get("message_status")
        if not chat_id or not wamid or not status:
            logger.warning("Message Status Change event id=%s missing required fields — skipped", event_id)
            _mark(event_id, "failed")
            return

        phone_number_id = str(payload.get("whatsapp_bot_id") or "")

        with wh_db.get_conn() as conn:
            wh_db.upsert_contact(conn, phone_number=chat_id, wa_id=chat_id, name=payload.get("first_name"))
            # A status event can arrive before its own message's Outgoing
            # Message event (confirmed in practice — see module docstring),
            # so ensure a row exists for the FK before insert_status_event.
            wh_db.ensure_outbound_stub(conn, wamid=wamid, phone_number_id=phone_number_id, contact_phone=chat_id)
            wh_db.insert_status_event(
                conn,
                wamid=wamid,
                status=status,
                error_code=None,
                error_title=payload.get("failed_reason"),
                event_timestamp=_parse_status_time(payload.get("status_time")),
                webhook_event_id=event_id,
            )
        _mark(event_id, "processed")
    except Exception:
        logger.exception("Failed processing Message Status Change event id=%s", event_id)
        _mark(event_id, "failed")
