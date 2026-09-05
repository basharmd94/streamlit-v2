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
