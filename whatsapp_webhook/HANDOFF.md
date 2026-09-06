# WhatsApp Webhook Receiver — Handoff Summary

Status as of this write-up: **code fully built and verified locally against a mocked DB. Not yet deployed anywhere real.** The user is picking this up in a separate session to deploy directly to their own production Windows Server 2016 box — skipping the local ngrok/ sandbox dev loop the original build doc scoped, since they run their own server and can revert via git if needed.

Read `WhatsApp_Integration_docs/whatsapp-webhook-build.md` first — that's the original build reference this was built against (webhook mechanics, field taxonomy, schema design rationale). This file is the "what happened since then + what's next" summary.

---

## What this service is

The receive-side counterpart to the Streamlit app's **"📨 Direct WhatsApp"** panel (`core/direct_whatsapp.py`, `views/marketing.py`). That panel only *sends* — a webhook is the only way delivery/read/failure status and inbound customer replies ever arrive; a send's own `200` response only means "Meta accepted the request," never a delivery confirmation.

A fully separate FastAPI service — own process, own `.env` credentials, own Postgres database (isolated from the main app's `da` database) — though it shares the `streamlitEnv3.10.13` Python env rather than a dedicated venv (that was an explicit choice made mid-build; easy to split into its own venv later if it ever becomes a maintenance problem).

---

## What's built (all committed, `main` branch)

| File | Responsibility |
|---|---|
| `main.py` | FastAPI app — GET verify handshake (`hub.challenge` echo), POST receiver |
| `security.py` | HMAC-SHA256 signature verification (raw body, constant-time compare) |
| `handlers.py` | Routes a verified payload by `changes[].field`, runs as a `BackgroundTask` |
| `db.py` | Plain psycopg2 access — one connection per call, no pool yet (traffic is low-volume) |
| `schema.sql` | The 6-table schema from the build doc, unmodified |
| `.env.example` | Credential/config template |
| `README.md` | Local dev setup steps (createdb, ngrok, Meta sandbox registration) — still useful reference even though the local-testing phase is being skipped |

Key implementation details, in case they need re-deriving without re-reading the original conversation:

- **Signature check is the entire trust boundary**: raw body bytes, before any JSON parsing (parsing can change byte representation and break the HMAC match), `hmac.compare_digest` for constant-time comparison, 403 on mismatch.
- **Respond fast, process later**: the verified payload is logged to `webhook_events` synchronously (this is the idempotency/audit backbone — every payload lands here first), then routed via FastAPI's built-in `BackgroundTasks` so Meta gets its `200` immediately. A DB/processing failure inside the background task is caught and recorded as `processing_status='failed'` on the `webhook_events` row — never raised, since the response already went out.
- **`messages` field has two shapes in one payload**: an inbound `messages[]` array and an outbound `statuses[]` array. Both are handled in `handlers._handle_messages_field`.
- **Status idempotency**: status events dedupe on `(wamid, status)` (`UNIQUE` constraint, `ON CONFLICT DO NOTHING`) since Meta's at-least-once delivery can send the same status twice. `messages.current_status` only ever moves *forward* through a rank guard (`sent < delivered < read`, `failed` terminal) — so a retried, out-of-order `sent` callback arriving after `read` already landed can't clobber the more recent state.
- **Everything not explicitly modeled lands in `account_alerts`** (`phone_number_quality_update`, `account_update`, `business_capability_update`, etc.) rather than being silently dropped — a generic `event_type` + `payload JSONB` catch-all.
- **Known, documented gap — the outbound-stub workaround**: `core/direct_whatsapp.py` (the send panel) calls Meta directly and does **not** write into this database at send time — the two services aren't wired together yet. So a delivery-status callback can arrive for a `wamid` this service never recorded, and `message_status_events.wamid` has a hard FK to `messages.wamid`. Fixed via `db.ensure_outbound_stub`: on first sight of an unrecognized `wamid`, it inserts a minimal placeholder `messages` row (`direction='outbound'`, `message_type='unknown'`, no `content`/`template_name`) before the status event is recorded, using `ON CONFLICT DO NOTHING` so a future real send-side integration becomes a no-op rather than overwriting real data. **This is still open** — if/when `core/direct_whatsapp.py` is taught to write a real outbound row (capturing the `wamid` Meta returns at send time), these stub rows stop being created and existing ones stay as harmless historical thin rows.
- **No task queue** — `BackgroundTasks` is in-process; a task in flight is lost if the process restarts mid-task. Acceptable for current volume; a real queue (Celery/RQ) is explicitly a later concern, not built.
- **No committed automated tests.** Verification so far was manual, in-session, via `fastapi.testclient.TestClient` against a mocked `db.get_conn` — confirmed the verify handshake (correct/wrong token), signature rejection (including that a DB failure while logging the rejection still returns 403, not 500), the happy path, and malformed JSON. None of that was saved as a `tests/` file — worth formalizing into a real pytest suite before this goes live, so regressions get caught automatically rather than needing another manual pass.

---

## Decision: deploying straight to production, not the local dev loop

The build doc's own phasing was local dev only (uvicorn + ngrok + Meta sandbox), with production deployment explicitly deferred. The user owns their Windows Server 2016 box directly and chose to skip straight to deploying there instead — comfortable reverting via git if something goes wrong. **Nothing has been deployed yet as of this handoff** — no Postgres database created anywhere, no `.env` filled in on any real server, no service installed.

### Open decisions, not yet answered

- **Domain/subdomain** to point at the server (e.g. `webhook.yourdomain.com`) — required, since Meta only accepts a real CA-trusted HTTPS callback URL, not a bare IP or self-signed cert.
- **Which Meta app/WABA** this points at — the same sandbox/test app used so far (lower stakes, no other consequences), or the real production WABA/number. If the latter, check the App Dashboard for Business Verification / Live-mode status first.
- **Reverse proxy choice** — recommended default is **Caddy** (automatic Let's Encrypt HTTPS, minimal config, runs as a native Windows binary) over IIS+ARR+win-acme, unless the server already runs IIS for something else.
- **Postgres location** — assumed to be the same Postgres instance that already hosts `da` (just a new, separate database, per the build doc's own isolation design), not verified against the actual server.

### Deployment steps drafted so far (not yet executed)

Recommended stack: **NSSM** (wraps `uvicorn` — and Caddy — as real Windows services: auto-start, auto-restart on crash) + **Caddy** as reverse proxy.

1. `git pull origin main` on the server (this repo already contains `whatsapp_webhook/` in full).
2. Dedicated venv inside `whatsapp_webhook/`, `pip install -r requirements.txt`.
3. `createdb whatsapp_webhooks` on the server's Postgres, load `schema.sql` into it.
4. `.env` from `.env.example` — `WHATSAPP_VERIFY_TOKEN` (any string), `META_APP_SECRET` (App Dashboard → App Settings → Basic), `WEBHOOK_DB_*`.
5. DNS: point the chosen subdomain at the server's public IP. Firewall: open inbound 443 (+ 80 briefly, for the ACME challenge).
6. Caddy reverse proxy — `Caddyfile` with `reverse_proxy localhost:8000` under the subdomain block; Caddy handles cert issuance/renewal automatically once DNS/ports are live.
7. NSSM: install `uvicorn main:app --host 127.0.0.1 --port 8000` (bound to localhost only — Caddy is the only public-facing process) as one service, Caddy as a second service; redirect stdout/stderr to a log file on both (no console on a running Windows service).
8. Register the webhook in Meta App Dashboard: callback `https://<subdomain>/webhook/whatsapp`, verify token matching `.env`, subscribe to `messages` + `message_template_status_update` + `phone_number_quality_update`.
9. Confirm receipt: `select * from webhook_events order by id desc limit 5;` after Meta's Verify-and-Save handshake and a real test message.

None of steps 1–9 have been run against the real server yet — this is the plan, to be executed and debugged step-by-step in the next session.

---

## Related context (send-side, separate but connected)

- `core/direct_whatsapp.py` + `config/direct_whatsapp.ini` — the send-only panel this webhook complements. Already working end-to-end (confirmed via a real sent-and-received test message), after resolving a `(#131005) Access denied` OAuth error that turned out to be a Meta-side token/permission issue, not a payload-format bug — fixed by regenerating credentials via Meta App Dashboard → WhatsApp → **Getting Started** (full detail in this session's saved memory, not repeated here since it's a send-side, not receive-side, concern).
- The two services are **not wired together** — see the outbound-stub gap above. That integration is a legitimate future task, not started.
