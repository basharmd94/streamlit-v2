# WhatsApp Screencast Prep — Handoff Summary

Status as of this write-up: production WhatsApp send/receive infrastructure is fully built and live-tested end to end. This doc exists to brief a follow-up session on **polishing the Streamlit-side messaging UI so it looks clean and legible in two screen recordings for Meta App Review** — the actual visual/UX design work should happen live in that session, not be dictated here. This file captures the context and constraints so that session doesn't have to re-derive them.

---

## What this is for

Meta's App Review requires a screen recording per requested permission, showing the permission actually being exercised end-to-end. We're requesting two:

- **`whatsapp_business_messaging`** — sending/receiving messages
- **`whatsapp_business_management`** — managing message templates, phone number status

The "end-to-end experience" Meta asks for is **cause → effect**: your system sends → a real WhatsApp user receives it → they reply → your system shows the reply and the delivery/read status. For an API-only integration there's no traditional "user clicking through your app's UI" — the recipient's phone *is* the user experience. Nothing needs to be invented; the existing pipeline already does this. The goal here is just making the **Streamlit side** of that pipeline look clean on camera, not building new functionality.

---

## Business/App context (for accuracy in anything shown on screen)

- Meta App name: **HMBR_WA_Marketing**
- Account holder: **Fixit HMBR Ltd.**, part of the **HMBR Tools & Chemicals Ltd.** group (parent company), which also includes **Zepto Enterprises**
- Privacy Policy / Terms of Use are live at `https://www.hmbr.com.bd/privacy-policy/` and `https://www.hmbr.com.bd/terms-of-use/` (linked in the site's header menu)
- Production webhook is live: `https://webhook.hmbr.com.bd/webhook/whatsapp`, served by Caddy (TLS via `tls-alpn-01`, since port 80 is occupied by an unrelated live IIS site) reverse-proxying to the `whatsapp_webhook` FastAPI service (NSSM service `WhatsAppWebhook`, port 8001, on the same Windows Server 2016 box as this app). Fully tested — real sent/received messages confirmed landing in its `whatsapp_webhooks` Postgres database.

---

## What's already built (current state, before any polish)

All in the main Streamlit app (this repo), under Marketing Analysis's mode radio:

| Mode | File | What it does |
|---|---|---|
| 📨 Direct WhatsApp | `views/marketing.py::_show_direct_whatsapp_messaging` | Sends one message (plain text or approved template) to one hand-entered number, via `core/direct_whatsapp.py`. Built as an exploratory test panel, not polished UI. |
| 📥 WhatsApp Message Log | `views/marketing.py::_show_whatsapp_message_log` | Read-only viewer into the webhook's own `whatsapp_webhooks` database (`core/whatsapp_webhook_db.py`) — recent messages table + a per-message drill-down (raw payload + full `sent → delivered → read` status timeline). Manual "🔄 Refresh" button, no auto-polling. |

Backing pieces (shouldn't need changes, just context):
- `core/direct_whatsapp.py` — `send_text`, `send_template`, `get_templates`, `upload_media`, against Meta's Cloud API directly.
- `core/whatsapp_webhook_db.py` — `get_recent_messages`, `get_status_history`, `get_counts`. Read-only, separate low-privilege DB role (`streamlit_reader`), never writes.
- Credentials: `config/direct_whatsapp.ini` (production `access_token`/`phone_number_id`/`waba_id`) and `config/whatsapp_webhook_db.ini` — both gitignored, already filled in on the server with production values.

---

## What each screencast needs to show

### Messaging screencast (`whatsapp_business_messaging`)
1. Send a template message to a real number, from the Streamlit UI.
2. The message arriving on that real phone (phone screen/camera in frame — outside Streamlit, just staging).
3. A reply typed from that phone.
4. Back in Streamlit: the reply appears, and the original message's status visibly progresses `sent → delivered → read`.

### Management screencast (`whatsapp_business_management`)
1. A template's approval status visible (this could be Meta's own WhatsApp Manager UI, or surfaced in Streamlit via `get_templates()` — open question, see below).
2. Optionally: phone number quality rating / messaging limits.

Record these as **two separate clips**, one per permission — the review form ties each screencast to its own permission slot.

---

## Open design decisions — figure out live, not pre-answered here

- **Live status updates**: right now the Message Log requires manually clicking "🔄 Refresh" to see status progress. For a clean recording, decide: click Refresh at the right moment on camera (simplest, zero new code), or build a lightweight auto-refresh (e.g. `st_autorefresh` on a short interval) so the status visibly updates on its own. Auto-refresh is nicer on camera but is new behavior worth deciding on deliberately, not defaulting into.
- **Send + Log in one flow**: currently two separate radio modes. Worth considering a combined view (send panel with the log directly below it) just for recording clarity, so the reviewer doesn't have to mentally track a mode switch — or keep them separate and make the switch itself part of a clean recorded flow. Either is fine; pick based on what looks best once you're actually looking at it.
- **Template management surface**: does the management screencast use Meta's own WhatsApp Manager UI (zero Streamlit work needed), or should `get_templates()`'s output get a real display in Streamlit first? If the latter, that's new UI, not just polish — scope it explicitly before starting.
- **Visual noise**: Marketing Analysis's page has a lot of surrounding UI (ZID selector, unrelated mode options in the same radio, sidebar filters) that will be in frame and isn't relevant to either screencast. Decide whether to just film the relevant section cropped/zoomed, or whether it's worth temporarily simplifying what's visible.
- **Status display**: current status shows as a plain text column (`sent`/`delivered`/`read`/`failed`). A small polish candidate: colored badges/chips instead of plain text, so a reviewer skimming the recording can follow the status change without reading closely — purely cosmetic, not required.

---

## Constraints to keep in mind while polishing

- Don't touch `core/direct_whatsapp.py` or `core/whatsapp_webhook_db.py`'s actual data logic unless something is genuinely broken — this doc is about presentation, not re-architecting an already-working, already-tested pipeline.
- Whatever ships here is real production UI other people (sales/support staff) may end up using too, not a throwaway demo screen — keep it consistent with the rest of `views/marketing.py`'s existing conventions (one public entry point per mode, no raw pandas manipulation in the view layer, etc. — see this repo's `CLAUDE.md`).
- The account/company names and URLs cited above are what Meta's reviewers will cross-check against — if anything on screen shows a different business name or an unrelated ZID's data, fix that before recording, not after.
