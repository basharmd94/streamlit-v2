# whatsapp_webhook/main.py
"""
WhatsApp Cloud API webhook receiver — see
../WhatsApp_Integration_docs/whatsapp-webhook-build.md for the full build
reference this was built against, and README.md in this folder for local
run instructions (uvicorn + ngrok + Meta App Dashboard registration).

Local/sandbox phase only, per the doc's own scoping — production deployment
(Windows Server 2016, reverse proxy, TLS, persistent service) is explicitly
deferred, not part of this build.
"""

import hmac
import json
import logging
import os

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response

load_dotenv()

import db as wh_db
import handlers
from security import verify_signature

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whatsapp_webhook")

app = FastAPI(title="WhatsApp Webhook Receiver")

# Fail fast at startup if these aren't set — both are required for every
# request this service handles, per the build doc.
VERIFY_TOKEN = os.environ["WHATSAPP_VERIFY_TOKEN"]
APP_SECRET = os.environ["META_APP_SECRET"].encode()

# Optional — the /webhook/whatsfly/{token} route below 404s until this is
# set, rather than the app failing to start. Not required at import time
# because this route is new/being rolled out independently of the
# Meta-direct one above.
WHATSFLY_WEBHOOK_TOKEN = os.environ.get("WHATSFLY_WEBHOOK_TOKEN")


@app.get("/webhook/whatsapp")
async def verify(request: Request):
    """One-time verification handshake, sent by Meta the moment you click
    Save on the webhook config in the App Dashboard. Echo hub.challenge
    back as plain text if hub.verify_token matches; 403 otherwise."""
    params = request.query_params
    if params.get("hub.mode") == "subscribe" and params.get("hub.verify_token") == VERIFY_TOKEN:
        return Response(content=params.get("hub.challenge", ""), media_type="text/plain")
    raise HTTPException(status_code=403)


@app.post("/webhook/whatsapp")
async def receive(request: Request, background_tasks: BackgroundTasks):
    raw_body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")

    if not verify_signature(raw_body, signature, APP_SECRET):
        # Logged even on a bad signature — webhook_events is the raw audit
        # log for every payload that hits this endpoint, forged or not.
        _log_rejected(raw_body)
        raise HTTPException(status_code=403, detail="Invalid signature")

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        # A genuinely invalid request, not a transient failure on our end —
        # 4xx per the doc's retry-behavior rule, so Meta won't retry this.
        raise HTTPException(status_code=400, detail="Malformed JSON")

    with wh_db.get_conn() as conn:
        event_id = wh_db.insert_webhook_event(conn, raw_payload=payload, signature_valid=True)

    # Ack Meta immediately; the actual DB writes/business logic happen after
    # this response is sent, per the doc's "respond fast, process later" rule.
    background_tasks.add_task(handlers.process_event, event_id, payload)
    return {"status": "received"}


def _log_rejected(raw_body: bytes) -> None:
    try:
        parsed = json.loads(raw_body)
    except json.JSONDecodeError:
        parsed = {"_unparseable_raw": raw_body.decode("utf-8", errors="replace")}
    try:
        with wh_db.get_conn() as conn:
            wh_db.insert_webhook_event(conn, raw_payload=parsed, signature_valid=False)
    except Exception:
        logger.exception("Failed to log a rejected (bad-signature) webhook event")


# ---------------------------------------------------------------------------
# /webhook/whatsfly/{token} — WhatsFly's own webhook delivery (Settings >
# Webhook > "Trigger Webhook for Incoming Message / Outgoing Message /
# Message Status Change / Conversation Status Change"), a separate sender
# from Meta's direct contract above.
#
# WhatsFly's own integration guide documents no signing/secret mechanism at
# all for their webhook calls (no header, no shared secret) — unlike Meta's
# X-Hub-Signature-256. So trust here is a random token embedded in the URL
# path itself instead, constant-time compared against WHATSFLY_WEBHOOK_TOKEN
# — paste the full URL (including the token) into WhatsFly's webhook URL
# field, not a bare /webhook/whatsfly.
#
# Phase 1, matching WhatsFly's own guide's suggested build order ("write a
# minimal route that logs the raw request body and returns 200 OK"): just
# capture the raw payload into the same webhook_events table (reusing the
# existing schema — no migration) and leave processing_status at its
# 'pending' default. WhatsFly's payload shape isn't documented anywhere and
# their other API responses have had genuinely surprising shapes before
# (e.g. the template list's "message" wrapper key) — real parsing (a
# handlers.py-equivalent dispatch) comes once a live delivery shows us the
# actual JSON, not before.
# ---------------------------------------------------------------------------


@app.post("/webhook/whatsfly/{token}")
async def receive_whatsfly(token: str, request: Request):
    if not WHATSFLY_WEBHOOK_TOKEN or not hmac.compare_digest(token, WHATSFLY_WEBHOOK_TOKEN):
        # 404, not 403 — don't confirm to a prober that this path exists at all.
        raise HTTPException(status_code=404)

    raw_body = await request.body()
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        payload = {"_unparseable_raw": raw_body.decode("utf-8", errors="replace")}

    with wh_db.get_conn() as conn:
        # signature_valid here means "the URL token matched", not a Meta-
        # style payload signature — repurposing the existing column rather
        # than adding a new one, since the underlying question (do we trust
        # this event's authenticity) is the same.
        event_id = wh_db.insert_webhook_event(conn, raw_payload=payload, signature_valid=True)

    logger.info("WhatsFly webhook event id=%s captured (phase 1: raw log only, not yet parsed)", event_id)
    return {"status": "received"}
