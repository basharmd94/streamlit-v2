# whatsapp_webhook/security.py
"""
HMAC-SHA256 signature verification for Meta's webhook POSTs — see
"Signature verification" in
../WhatsApp_Integration_docs/whatsapp-webhook-build.md.

Verifies against the RAW request body, before any JSON parsing — parsing
can change the byte representation and break the match. Uses a
constant-time comparison to avoid timing-attack leakage. This is the entire
trust boundary for this service: a forged event that got past this check
could fake a "template paused" alert or corrupt delivery records.
"""

import hashlib
import hmac


def verify_signature(raw_body: bytes, signature_header: str, app_secret: bytes) -> bool:
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(app_secret, raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header)
