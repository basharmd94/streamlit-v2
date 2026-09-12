from configparser import ConfigParser
from pathlib import Path

CONFIG_DIR   = Path(__file__).resolve().parent
LOG_INI      = CONFIG_DIR / "logging.ini"
DB_INI       = CONFIG_DIR / "global_db.ini"
WHATSFLY_INI = CONFIG_DIR / "whatsfly.ini"
DIRECT_WHATSAPP_INI = CONFIG_DIR / "direct_whatsapp.ini"
WHATSAPP_WEBHOOK_DB_INI = CONFIG_DIR / "whatsapp_webhook_db.ini"
WHATSAPP_WEBHOOK_CAMPAIGN_DB_INI = CONFIG_DIR / "whatsapp_webhook_campaign_db.ini"

def get_db_params(section: str = "database") -> dict:
    parser = ConfigParser()
    parser.read(DB_INI)
    if not parser.has_section(section):
        raise ValueError(f"Section [{section}] not found in {DB_INI}")
    return dict(parser.items(section))


def get_whatsfly_params(section: str = "whatsfly") -> dict | None:
    """Returns {'api_token': ..., 'phone_number_id': ...} or None if
    config/whatsfly.ini (gitignored, like every other *.ini here) doesn't
    exist yet or is missing the section — never raises, so callers can show
    a friendly setup message instead of crashing the page."""
    if not WHATSFLY_INI.exists():
        return None
    parser = ConfigParser()
    parser.read(WHATSFLY_INI)
    if not parser.has_section(section):
        return None
    params = dict(parser.items(section))
    if not params.get("api_token") or not params.get("phone_number_id"):
        return None
    return params


def get_direct_whatsapp_params(section: str = "direct_whatsapp") -> dict | None:
    """Returns {'access_token', 'phone_number_id', 'waba_id', 'graph_api_version'}
    or None if config/direct_whatsapp.ini (gitignored, like every other *.ini
    here) doesn't exist yet or is missing the section — never raises, so
    callers can show a friendly setup message instead of crashing the page.

    This is Meta's own WhatsApp Cloud API used directly (graph.facebook.com),
    not routed through WhatsFly — meant for a separate Meta test WABA + test
    number, so it's a distinct ini/section from config/whatsfly.ini rather
    than reusing those credentials."""
    if not DIRECT_WHATSAPP_INI.exists():
        return None
    parser = ConfigParser()
    parser.read(DIRECT_WHATSAPP_INI)
    if not parser.has_section(section):
        return None
    params = dict(parser.items(section))
    if not params.get("access_token") or not params.get("phone_number_id") or not params.get("waba_id"):
        return None
    return params


def get_whatsapp_webhook_db_params(section: str = "whatsapp_webhook_db") -> dict | None:
    """Returns {'host', 'port', 'dbname', 'user', 'password'} or None if
    config/whatsapp_webhook_db.ini (gitignored, like every other *.ini here)
    doesn't exist yet or is missing the section — never raises, so callers
    can show a friendly setup message instead of crashing the page.

    This is READ-ONLY access to the separate whatsapp_webhook service's own
    Postgres database (whatsapp_webhooks, isolated from this app's `da` —
    see whatsapp_webhook/schema.sql), for Marketing > WhatsApp Message Log.
    Use a dedicated low-privilege SELECT-only role here (streamlit_reader),
    never the webhook service's own webhook_svc role (which has full
    INSERT/UPDATE on every table there) and never the newer
    streamlit_campaign_writer role either (get_whatsapp_webhook_campaign_db_params
    below) — this function's whole point is staying read-only."""
    if not WHATSAPP_WEBHOOK_DB_INI.exists():
        return None
    parser = ConfigParser()
    parser.read(WHATSAPP_WEBHOOK_DB_INI)
    if not parser.has_section(section):
        return None
    params = dict(parser.items(section))
    params.setdefault("port", "5432")
    if not params.get("host") or not params.get("dbname") or not params.get("user") or not params.get("password"):
        return None
    return params


def get_whatsapp_webhook_campaign_db_params(section: str = "whatsapp_webhook_campaign_db") -> dict | None:
    """Returns {'host', 'port', 'dbname', 'user', 'password'} or None if
    config/whatsapp_webhook_campaign_db.ini (gitignored, like every other
    *.ini here) doesn't exist yet or is missing the section — never raises,
    so callers can show a friendly setup message instead of crashing the
    page.

    A SEPARATE, narrow WRITE role (streamlit_campaign_writer — see
    whatsapp_webhook/add_streamlit_campaign_role.sql) into the SAME
    whatsapp_webhooks database get_whatsapp_webhook_db_params above already
    reads from, but scoped to INSERT/UPDATE/SELECT on only the three Bulk
    Messaging campaign tables (campaigns, campaign_recipients,
    contact_opt_outs) — it cannot touch messages/contacts/webhook_events/
    etc. Kept in its own ini file (not a second section merged into
    whatsapp_webhook_db.ini) so the two roles' credentials stay physically
    separate, matching the one-file-per-credential-set convention already
    used for every other *.ini in this app."""
    if not WHATSAPP_WEBHOOK_CAMPAIGN_DB_INI.exists():
        return None
    parser = ConfigParser()
    parser.read(WHATSAPP_WEBHOOK_CAMPAIGN_DB_INI)
    if not parser.has_section(section):
        return None
    params = dict(parser.items(section))
    params.setdefault("port", "5432")
    if not params.get("host") or not params.get("dbname") or not params.get("user") or not params.get("password"):
        return None
    return params
