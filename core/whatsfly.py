# core/whatsfly.py
"""
Thin client for the WhatsFly API — see Whatsfly_Integration_docs/whatsfly-integration-guide.md
for the full API reference this was built against.

Current build phase (per that guide): single-message test only. This module
covers just what that needs — session text send, template list, and a
generic template send (WhatsFly doesn't publish one fixed "send template"
path; the dashboard's template picker hands you a per-template endpoint, so
`send_template` takes the endpoint as an argument rather than hardcoding one).
No receive-side / webhook handling here — that's a separate always-on
service (FastAPI), not this Streamlit app, per the guide.

Credentials come from config/whatsfly.ini (gitignored, same convention as
every other *.ini in config/) via config.settings.get_whatsfly_params —
never hardcoded, never committed.
"""

import json

import requests

from config.settings import get_whatsfly_params

BASE_URL = "https://app.whatsfly.net/api/v1"
_TIMEOUT = 20


class WhatsFlyConfigError(Exception):
    """config/whatsfly.ini is missing or incomplete."""


def get_credentials() -> dict:
    params = get_whatsfly_params()
    if not params:
        raise WhatsFlyConfigError(
            "config/whatsfly.ini not found (or missing api_token/phone_number_id). "
            "Create config/whatsfly.ini with:\n\n"
            "[whatsfly]\n"
            "api_token = YOUR_TOKEN\n"
            "phone_number_id = YOUR_PHONE_NUMBER_ID\n"
        )
    return params


def get_templates() -> dict:
    """GET/POST /whatsapp/get/template/list — raw JSON response.
    Shape isn't nailed down yet against the live account, so callers should
    treat the result defensively and show it raw for the first real check."""
    creds = get_credentials()
    resp = requests.post(
        f"{BASE_URL}/whatsapp/get/template/list",
        data={"apiToken": creds["api_token"], "phone_number_id": creds["phone_number_id"]},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def send_text(phone_number: str, message: str) -> requests.Response:
    """POST /whatsapp/send — session message only (24h window since the
    recipient last messaged the business number). Returns the raw Response
    so the caller can show status code + body as-is (this is the exploratory
    "what feedback do we get" phase, not a hardened wrapper)."""
    creds = get_credentials()
    return requests.post(
        f"{BASE_URL}/whatsapp/send",
        data={
            "apiToken": creds["api_token"],
            "phone_number_id": creds["phone_number_id"],
            "phone_number": phone_number,
            "message": message,
        },
        timeout=_TIMEOUT,
    )


def upload_media(file_bytes: bytes, filename: str, mime_type: str) -> dict:
    """POST /whatsapp/upload/media (multipart) — field name must be exactly
    `media_file` per the guide. Returns the raw JSON response; shape is
    unconfirmed against the live account (first real use of this endpoint
    here), same defensive stance as get_templates — caller should show it
    raw so a wrong field-name guess when reading the id/url back is visible."""
    creds = get_credentials()
    resp = requests.post(
        f"{BASE_URL}/whatsapp/upload/media",
        data={"apiToken": creds["api_token"], "phone_number_id": creds["phone_number_id"]},
        files={"media_file": (filename, file_bytes, mime_type)},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def send_template(phone_number: str, endpoint: str, extra_params: dict) -> requests.Response:
    """Generic template send. `endpoint` is whatever path (or full URL) the
    WhatsFly dashboard's template picker generated for the chosen template —
    per the build guide there's no single documented path for this, so it's
    a parameter here rather than a fixed route. `extra_params` is merged in
    on top of the standard apiToken/phone_number_id/phone_number trio (e.g.
    template_name, language_code, variables) and can override any of them."""
    creds = get_credentials()
    payload = {
        "apiToken": creds["api_token"],
        "phone_number_id": creds["phone_number_id"],
        "phone_number": phone_number,
    }
    payload.update(extra_params or {})
    # requests can't form-encode a nested dict/list (e.g. a Meta Cloud API
    # style `template: {name, language, components}` object) — it would get
    # silently mangled into a Python repr string instead of JSON. This API's
    # own convention elsewhere (assign-custom-fields' `custom_fields`) passes
    # structured values as a JSON-encoded STRING inside a flat form field, so
    # match that here rather than guessing between requests' data= vs json=.
    payload = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in payload.items()}
    url = endpoint if endpoint.startswith("http") else f"{BASE_URL}{endpoint}"
    return requests.post(url, data=payload, timeout=_TIMEOUT)
