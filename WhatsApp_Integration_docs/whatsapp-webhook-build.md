# WhatsApp Cloud API — Webhook Receiver Build Reference

Build context for a FastAPI service that receives WhatsApp Cloud API webhook events directly from Meta (not through WhatsFly). Covers local development, the webhook contract, and the database schema for storing messages.

---

## Scope & current phase

**Right now:** build and test the webhook receiver locally, tunneled to the internet with ngrok, registered against the Meta developer sandbox app and its free test number.

**Deferred to later, not part of this build:** deploying the FastAPI service to the live Windows Server 2016 box (reverse proxy, TLS certificate, running it as a persistent service, firewall/port-forwarding). That's a separate phase once the local build is proven out — don't design around it yet, just don't paint into a corner that makes it harder later (e.g. keep config in environment variables, not hardcoded paths).

**Database:** a separate PostgreSQL database (its own DB, isolated from the existing production Postgres database) — not a different engine. Reasoning: message/webhook data is well-structured relational data; the one genuinely variable part (different payload shapes per message type) is handled with `JSONB` columns rather than a NoSQL store. This keeps SQL joins/aggregations available for reporting later, without adding a second database technology to operate.

---

## Local development flow

1. Run the FastAPI app locally (e.g. `uvicorn main:app --port 8000`).
2. Run `ngrok http 8000` — this gives a public HTTPS forwarding URL (e.g. `https://abc123.ngrok-free.app`).
3. In the Meta App Dashboard → WhatsApp → Configuration → Webhooks, set:
   - **Callback URL:** `https://abc123.ngrok-free.app/webhook/whatsapp`
   - **Verify Token:** any string you choose, stored as an environment variable, matched in code
4. Click Verify and Save — Meta immediately sends a one-time GET request to confirm the endpoint (see verification handshake below).
5. Subscribe to webhook fields: `messages` (required), plus `message_template_status_update` and `phone_number_quality_update` (low-noise, catch template rejections and number throttling early).
6. Send a test message from the sandbox number, reply from the recipient's phone, confirm the event lands in your local logs/database.

**Note on ngrok's free tier:** the forwarding URL changes every time the tunnel restarts, so the callback URL in Meta's dashboard needs re-registering after each restart during development. A paid ngrok static domain avoids this if it becomes annoying.

---

## Meta's webhook mechanics — what to build against

**1. Verification handshake (GET, one-time per registration)**
Meta sends `GET` with `hub.mode=subscribe`, `hub.verify_token`, and `hub.challenge`. Check the token matches, then echo `hub.challenge` back as plain text with a 200. Mismatch → 403.

**2. Signature verification (every POST, non-negotiable)**
Every event carries an `X-Hub-Signature-256` header — an HMAC-SHA256 of the **raw** request body, signed with your Meta App Secret.
- Verify against the raw bytes, before any JSON parsing — parsing can alter byte representation and break the match
- Use a constant-time comparison (`hmac.compare_digest` in Python) to avoid timing-attack leakage
- Reject with 403 on mismatch — this is the only thing standing between your endpoint and forged events (a fake "template paused" event could halt a live send; a fake "delivered" status could corrupt your records)

**3. Respond fast, process later**
Verify the signature, return `200` immediately, then hand off the actual database writes and business logic to a background task/queue. Don't make Meta wait on your processing.

**4. Idempotency — duplicates will happen**
Meta uses at-least-once delivery. Dedupe using `wamid` (WhatsApp message ID) for messages; for status events, dedupe on the combination of `wamid` + `status`, since one message can get up to three separate status callbacks (sent, delivered, read).

**5. Status codes matter for retry behavior**
Return `5xx` for your own transient failures (DB down, timeout) — Meta retries those. Return `4xx` only for genuinely invalid requests (bad signature) — Meta won't retry a 4xx.

**6. One URL, route internally by `field`**
Every event lands on the same endpoint. Each `changes` entry in the payload has a `field` attribute — route your handler logic off that (`messages`, `message_template_status_update`, `phone_number_quality_update`, etc.).

**7. Log `wamid` and `fbtrace_id` on everything**
Needed for escalating delivery problems to Meta support later — without them, debugging is guesswork.

**8. The send API's response is not a delivery confirmation**
A `200` when you call the send endpoint only means "Meta accepted the request." Actual delivery/read/failure status only ever arrives via this webhook.

---

## Webhook fields reference

**`messages`** — the one you'll handle constantly. Two payload shapes:

*Incoming (has a `messages` array):* `text`, `image`/`video`/`audio`/`document`/`sticker`, `location`, `contacts`, `interactive` (button/list reply), `reaction`, `order`, `referral`, `system`, `unsupported`.

*Outgoing status (has a `statuses` array):* `sent` → `delivered` → `read`, or `failed` (with an `errors` array — this is where error code 131026, "message undeliverable," shows up). Includes `pricing` and `conversation` info.

**Template health:** `message_template_status_update` (APPROVED/REJECTED/PAUSED/DISABLED/LIMIT_EXCEEDED), `message_template_quality_update`, `message_template_components_update`, `template_category_update`.

**Number & account health:** `phone_number_quality_update` (Green/Yellow/Red rating and messaging-tier changes), `phone_number_name_update`, `account_update` (policy violations), `business_capability_update`, `account_review_update`, `account_alerts`, `security`, `flows`.

---

## FastAPI skeleton

```python
from fastapi import FastAPI, Request, Response, HTTPException
import hmac, hashlib, os

app = FastAPI()

VERIFY_TOKEN = os.environ["WHATSAPP_VERIFY_TOKEN"]
APP_SECRET = os.environ["META_APP_SECRET"].encode()

@app.get("/webhook/whatsapp")
async def verify(request: Request):
    params = request.query_params
    if params.get("hub.mode") == "subscribe" and params.get("hub.verify_token") == VERIFY_TOKEN:
        return Response(content=params.get("hub.challenge"), media_type="text/plain")
    raise HTTPException(status_code=403)

def verify_signature(raw_body: bytes, signature_header: str) -> bool:
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(APP_SECRET, raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header)

@app.post("/webhook/whatsapp")
async def receive(request: Request):
    raw_body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not verify_signature(raw_body, signature):
        raise HTTPException(status_code=403, detail="Invalid signature")

    payload = await request.json()
    # 1. Insert raw payload into webhook_events immediately (signature_valid=True)
    # 2. Hand off to background task: route by entry.changes[].field, upsert into
    #    messages / message_status_events / templates / account_alerts as appropriate
    return {"status": "received"}
```

This is a starting skeleton, not the full implementation — Claude Code fills in the database writes, background task queue, and per-field routing logic.

---

## Database schema (PostgreSQL, separate database)

```sql
CREATE TABLE webhook_events (
    id BIGSERIAL PRIMARY KEY,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    signature_valid BOOLEAN NOT NULL,
    raw_payload JSONB NOT NULL,
    processed_at TIMESTAMPTZ,
    processing_status TEXT NOT NULL DEFAULT 'pending' -- pending | processed | failed
);

CREATE TABLE contacts (
    id BIGSERIAL PRIMARY KEY,
    phone_number TEXT UNIQUE NOT NULL,
    wa_id TEXT,
    name TEXT,
    customer_code TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE messages (
    id BIGSERIAL PRIMARY KEY,
    wamid TEXT UNIQUE NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('inbound', 'outbound')),
    phone_number_id TEXT NOT NULL,
    contact_phone TEXT NOT NULL REFERENCES contacts(phone_number),
    message_type TEXT NOT NULL,
    template_name TEXT,
    content JSONB,
    current_status TEXT,
    message_timestamp TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_messages_contact_phone ON messages(contact_phone);
CREATE INDEX idx_messages_current_status ON messages(current_status);

CREATE TABLE message_status_events (
    id BIGSERIAL PRIMARY KEY,
    wamid TEXT NOT NULL REFERENCES messages(wamid),
    status TEXT NOT NULL,
    error_code TEXT,
    error_title TEXT,
    event_timestamp TIMESTAMPTZ NOT NULL,
    webhook_event_id BIGINT REFERENCES webhook_events(id),
    UNIQUE (wamid, status)
);
CREATE INDEX idx_status_events_wamid ON message_status_events(wamid);

CREATE TABLE templates (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    language TEXT,
    status TEXT,
    last_checked_at TIMESTAMPTZ,
    UNIQUE (name, language)
);

CREATE TABLE account_alerts (
    id BIGSERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,
    payload JSONB NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

**Design notes:**
- `messages` and `message_status_events` are split so `messages` always reflects current state (fast to query) while the full sent→delivered→read timeline is preserved separately for delivery-rate analysis.
- `content` is `JSONB` rather than fixed columns, since inbound payload shape varies a lot by message type (text vs. button reply vs. media) — avoids a dozen nullable columns.
- `webhook_events` is the raw audit log and idempotency backbone — every payload lands here first, verified or not, before any business logic runs against it.
- The `UNIQUE (wamid, status)` constraint on `message_status_events` is the idempotency guard for duplicate status callbacks.

---

## Deferred to a later phase

- Production deployment to Windows Server 2016 (reverse proxy + TLS cert, running as a persistent service, firewall/port forwarding)
- WhatsFly-specific integration (separate track, separate document)
