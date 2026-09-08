# core/direct_whatsapp.py
"""
Thin client for Meta's own WhatsApp Cloud API, called directly
(graph.facebook.com) — no WhatsFly in between. See core/whatsfly.py for the
WhatsFly-routed equivalent; the two are kept as separate clients/credentials
since they hit different endpoints against (for now) a separate Meta test
WhatsApp Business Account + test number, not the real WhatsFly-routed number.

Unlike whatsfly.py, this follows Meta's own published Cloud API docs
(https://developers.facebook.com/docs/whatsapp/cloud-api/) directly rather
than reverse-engineering a response shape — so no defensive multi-key
guessing here, the contract is documented.

Credentials come from config/direct_whatsapp.ini (gitignored, same
convention as config/whatsfly.ini) via
config.settings.get_direct_whatsapp_params — never hardcoded, never
committed.
"""

import requests

from config.settings import get_direct_whatsapp_params

_TIMEOUT = 20
_DEFAULT_API_VERSION = "v21.0"


class DirectWhatsAppConfigError(Exception):
    """config/direct_whatsapp.ini is missing or incomplete."""


def get_credentials() -> dict:
    params = get_direct_whatsapp_params()
    if not params:
        raise DirectWhatsAppConfigError(
            "config/direct_whatsapp.ini not found (or missing access_token/"
            "phone_number_id/waba_id). Create config/direct_whatsapp.ini with:\n\n"
            "[direct_whatsapp]\n"
            "access_token = YOUR_TEST_ACCESS_TOKEN\n"
            "phone_number_id = YOUR_TEST_PHONE_NUMBER_ID\n"
            "waba_id = YOUR_WHATSAPP_BUSINESS_ACCOUNT_ID\n"
            "graph_api_version = v21.0\n"
        )
    return params


def _base_url(creds: dict) -> str:
    version = creds.get("graph_api_version") or _DEFAULT_API_VERSION
    return f"https://graph.facebook.com/{version}"


def _headers(creds: dict) -> dict:
    return {"Authorization": f"Bearer {creds['access_token']}"}


def get_templates() -> dict:
    """GET /{waba_id}/message_templates — Meta's documented shape:
    {"data": [{name, components, language, status, category, id}, ...],
    "paging": {...}}."""
    creds = get_credentials()
    resp = requests.get(
        f"{_base_url(creds)}/{creds['waba_id']}/message_templates",
        headers=_headers(creds),
        params={"fields": "name,components,language,status,category", "limit": 100},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def send_text(phone_number: str, message: str) -> requests.Response:
    """POST /{phone_number_id}/messages, type=text — session message only,
    within Meta's 24h customer-service window since the recipient last
    messaged the test number. Returns the raw Response so the caller can
    show status code + body as-is."""
    creds = get_credentials()
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": phone_number,
        "type": "text",
        "text": {"preview_url": False, "body": message},
    }
    return requests.post(
        f"{_base_url(creds)}/{creds['phone_number_id']}/messages",
        headers=_headers(creds),
        json=payload,
        timeout=_TIMEOUT,
    )


def upload_media(file_bytes: bytes, filename: str, mime_type: str) -> dict:
    """POST /{phone_number_id}/media (multipart) — Meta's documented shape,
    returns {"id": "<media_id>"} on success. That id feeds a template's
    header image parameter directly (no separate hosted URL needed, unlike
    WhatsFly's flow)."""
    creds = get_credentials()
    resp = requests.post(
        f"{_base_url(creds)}/{creds['phone_number_id']}/media",
        headers=_headers(creds),
        data={"messaging_product": "whatsapp"},
        files={"file": (filename, file_bytes, mime_type)},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def send_template(phone_number: str, template_name: str, language_code: str, components: list) -> requests.Response:
    """POST /{phone_number_id}/messages, type=template — Meta's documented
    nested `template: {name, language: {code}, components: [...]}` shape
    (the real thing WhatsFly's own "Meta Cloud API style" fallback option
    imitates). Variables are positional here (no per-variable name like
    WhatsFly's `templateVariable-<name>-<n>` — that was a WhatsFly-specific
    convention, not part of Meta's own contract): each `{{n}}` in the body
    is filled by the nth entry of the body component's `parameters` list."""
    creds = get_credentials()
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": phone_number,
        "type": "template",
        "template": {
            "name": template_name,
            "language": {"code": language_code},
            "components": components,
        },
    }
    return requests.post(
        f"{_base_url(creds)}/{creds['phone_number_id']}/messages",
        headers=_headers(creds),
        json=payload,
        timeout=_TIMEOUT,
    )
