# WhatsApp Webhook Receiver

A standalone FastAPI service that receives WhatsApp Cloud API webhook events
directly from Meta (not through WhatsFly) — the receive-side counterpart to
the Streamlit app's "📨 Direct WhatsApp" send panel (`core/direct_whatsapp.py`).
Build reference: [`../WhatsApp_Integration_docs/whatsapp-webhook-build.md`](../WhatsApp_Integration_docs/whatsapp-webhook-build.md).

**Current phase**: local development only, tunneled with ngrok, registered
against Meta's sandbox app + test number. Production deployment (Windows
Server 2016, reverse proxy, TLS, persistent service) is a later, separate
phase — not built here.

Uses the same Python environment as the main Streamlit app
(`streamlitEnv3.10.13`), but its own Postgres database, isolated from `da`.

---

## One-time setup

1. **Install dependencies** (into the shared `streamlitEnv3.10.13` env):
   ```bash
   pip install -r whatsapp_webhook/requirements.txt
   ```

2. **Create the webhook database** and load the schema:
   ```bash
   createdb whatsapp_webhooks
   psql -h localhost -U postgres -d whatsapp_webhooks -f whatsapp_webhook/schema.sql
   ```
   (Use whatever local Postgres user/database name you prefer — just keep
   it consistent with step 3.)

3. **Configure credentials**:
   ```bash
   cp whatsapp_webhook/.env.example whatsapp_webhook/.env
   ```
   Fill in `.env`:
   - `WHATSAPP_VERIFY_TOKEN` — any string you choose (must match what you
     enter in Meta's dashboard in step 6).
   - `META_APP_SECRET` — Meta App Dashboard → App Settings → Basic → App
     Secret (click "Show").
   - `WEBHOOK_DB_*` — match whatever you created in step 2.

   `.env` is gitignored — never commit it.

---

## Running locally

4. **Start the service**:
   ```bash
   cd whatsapp_webhook
   uvicorn main:app --reload --port 8000
   ```

5. **Tunnel it** (separate terminal):
   ```bash
   ngrok http 8000
   ```
   Note the `https://....ngrok-free.app` URL it prints.

6. **Register the webhook** in Meta App Dashboard → WhatsApp → Configuration
   → Webhooks:
   - **Callback URL**: `https://<your-ngrok-url>/webhook/whatsapp`
   - **Verify Token**: the same string as `WHATSAPP_VERIFY_TOKEN` in `.env`
   - Click **Verify and Save** — Meta sends a one-time GET to confirm.
   - **Subscribe to fields**: `messages` (required), plus
     `message_template_status_update` and `phone_number_quality_update`.

7. **Test**: send a message from the sandbox number, or send one *to* it
   (e.g. via the Streamlit app's Direct WhatsApp panel) and reply from the
   recipient's phone. Confirm the event lands in the database:
   ```bash
   psql -h localhost -U postgres -d whatsapp_webhooks -c "select * from webhook_events order by id desc limit 5;"
   psql -h localhost -U postgres -d whatsapp_webhooks -c "select * from messages order by id desc limit 5;"
   ```

**ngrok's free tier**: the forwarding URL changes every time the tunnel
restarts, so the callback URL in Meta's dashboard needs re-registering after
each restart during development.

---

## Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI app — GET verify handshake, POST receive endpoint |
| `security.py` | HMAC-SHA256 signature verification (raw body, constant-time compare) |
| `handlers.py` | Routes a verified payload by `changes[].field`, runs as a background task |
| `db.py` | Plain psycopg2 access to the separate webhook database |
| `schema.sql` | The 6-table schema (see build doc for design notes) |
| `.env.example` | Credential/config template — copy to `.env` and fill in |

## Known gaps / follow-ups

- **Outbound messages sent via the Streamlit app aren't recorded here at
  send time** — `core/direct_whatsapp.py` calls Meta directly and doesn't
  write into this database. A status callback for such a message still
  gets captured (via `db.ensure_outbound_stub`, which creates a minimal
  placeholder `messages` row on first sight of an unknown `wamid`), but
  `template_name`/`content` stay empty for those rows. Wiring the send
  panel to write a real outbound row (capturing the `wamid` from Meta's
  send response) would fill this in — not done yet, out of scope for this
  build phase.
- **No task queue** — background work runs via FastAPI's built-in
  `BackgroundTasks`, which is in-process and lost if the service restarts
  mid-task. Fine for local/sandbox testing; a real queue (Celery/RQ) would
  be a production-phase concern.
- **No automated tests yet.**
