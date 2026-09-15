# processing/wf_template_mapping.py
"""
Per-template variable -> customer-attribute (or flat-value) mapping --
Phase 6 of Whatsfly_Integration_docs/bulk-send-build-plan.md, revised.

The original Phase 6 plan was "write a Python handler together, per real
campaign" (see the comment above views/marketing.py::_wf_bulk_default_view).
That assumed campaigns/templates would be defined occasionally, each one
worth a build session. Once templates started getting created ad hoc in
WhatsFly's own dashboard instead, that stopped being the right shape --
this module is a genuinely self-service replacement: map a template's own
variable names (e.g. "CUSCODE", exactly as named when the template was
built in WhatsFly) to a customer attribute or a flat per-campaign value,
ONCE, and every future campaign built around that template can resolve
its variables automatically, no new code required.

Persisted in the SAME whatsapp_webhooks database / streamlit_campaign_writer
role as processing/wf_bulk_campaign.py (see
whatsapp_webhook/add_template_variable_mappings_table.sql +
add_template_mapping_grants.sql) -- reuses that module's own connection
helper rather than duplicating the config-loading logic.
"""

from processing.wf_bulk_campaign import _get_conn, WfBulkCampaignDBConfigError  # noqa: F401 -- re-exported for callers

# {key: label} -- the fixed set of customer/audience attributes a template
# variable can be mapped to. Matches the audience table's own columns in
# views/marketing.py::_show_wf_bulk_messaging exactly, so a mapping made
# here lines up with what a real campaign's audience actually has
# available per customer.
CUSTOMER_ATTRIBUTES = [
    ("cusid", "Customer Code"),
    ("cusname", "Customer Name"),
    ("cusmobile", "Mobile Number"),
    ("whatsapp", "WhatsApp Number"),
    ("area", "Area"),
    ("net_sales", "Net Sales (window)"),
    ("current_balance", "Current Balance"),
    ("current_score", "Current Score"),
]
CUSTOMER_ATTRIBUTE_LABELS = dict(CUSTOMER_ATTRIBUTES)

# {key: label} -- item/product catalog fields a template variable can be
# mapped to (e.g. a template that announces one product's price). Unlike
# CUSTOMER_ATTRIBUTES, these are NOT per-recipient -- one item is picked
# once per campaign (in the Template Mapping preview, and again in the
# actual Bulk Messaging send flow) and the same values go out to every
# recipient, the same "asked once per campaign" shape as flat_value.
# Sourced from views/marketing.py::_wf_item_price_catalog, which reuses the
# existing inventory_overview pull (caitem std price + opspprc's lowest-tier
# wholesale discount, the same GREATEST(std_price - disc, 0) formula used
# throughout this app for "WH Price") -- no new SQL needed.
ITEM_ATTRIBUTES = [
    ("item_code", "Item Code"),
    ("item_name", "Item Name"),
    ("std_price", "Standard Price (List)"),
    ("wh_price", "Wholesale/Discounted Price"),
    ("product_list", "Product List (multiple items — \"Name (Code)\", joined)"),
]
ITEM_ATTRIBUTE_LABELS = dict(ITEM_ATTRIBUTES)

# Which ITEM_ATTRIBUTES keys need a MULTI-item picker (a multiselect,
# resolving to one joined "Name (Code), Name (Code), ..." string across
# every picked item) rather than the single-item picker the other four
# item attributes share. A template's item variable(s) are either all
# single-product or (rarely) all multi-product -- not meaningfully mixed --
# but this is checked per-variable so the picker UI (views/marketing.py::
# _wf_bulk_default_view / _show_wf_template_mapping) can show exactly the
# widget each mapped variable actually needs.
MULTI_ITEM_KEYS = {"product_list"}

# Attributes resolvable straight from cacus_directory alone -- cheap enough
# for a live preview without needing the heavier sales/collection pipeline
# behind Net Sales/Current Balance/Current Score (see
# processing/wf_bulk_audience.py). The other three still show up as
# regular mapping options -- they just can't be preview-resolved here
# without that pipeline, so the preview shows a placeholder for them
# instead of pretending to know the value.
LIGHT_ATTRIBUTES = {"cusid", "cusname", "cusmobile", "whatsapp", "area"}


def get_template_mapping(template_id: str) -> dict:
    """{variable_name: {"source_type", "source_key"}} for one template, or
    {} if nothing has been mapped yet."""
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT variable_name, source_type, source_key FROM template_variable_mappings WHERE template_id = %s",
            (template_id,),
        )
        return {row[0]: {"source_type": row[1], "source_key": row[2]} for row in cur.fetchall()}


def save_template_mapping(template_id: str, template_name: str, mappings: dict, created_by: str) -> None:
    """`mappings`: {variable_name: {"source_type", "source_key"}}. Upserts
    one row per variable -- UPDATE-then-INSERT-if-0-rows, same convention
    as every other upsert in this app (the real server predates Postgres
    9.5, no ON CONFLICT available)."""
    with _get_conn() as conn, conn.cursor() as cur:
        for variable_name, m in mappings.items():
            cur.execute(
                """
                UPDATE template_variable_mappings
                SET template_name = %s, source_type = %s, source_key = %s, updated_at = now()
                WHERE template_id = %s AND variable_name = %s
                """,
                (template_name, m["source_type"], m.get("source_key"), template_id, variable_name),
            )
            if cur.rowcount == 0:
                cur.execute(
                    """
                    INSERT INTO template_variable_mappings
                        (template_id, template_name, variable_name, source_type, source_key, created_by)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (template_id, template_name, variable_name, m["source_type"], m.get("source_key"), created_by),
                )
        conn.commit()


def resolve_recipient_variables(
    var_map_entries: list, mapping: dict, customer_row: dict,
    flat_values: dict = None, item_values: dict = None,
) -> dict:
    """Builds the actual {variable_name: value} to send for ONE recipient,
    given a saved mapping. `customer_row` is a dict of that customer's
    known attributes (whatever CUSTOMER_ATTRIBUTES keys are available --
    e.g. cusid/cusname/.../net_sales), `flat_values` is
    {variable_name: value} for whichever variables are mapped as
    'flat_value' (the same value for every recipient in one campaign, set
    once at send time -- see the build plan's Q13/header-image precedent
    for "same for everyone"). `item_values` is the same "asked once per
    campaign" shape for 'item_attribute' variables -- one product picked
    once (see ITEM_ATTRIBUTES above), not per recipient. A variable with no
    saved mapping at all resolves to '' rather than raising, so an
    incomplete mapping degrades visibly (a blank in the message) instead of
    crashing a whole campaign. This is what Phase 2's send engine (processing/
    wf_bulk_campaign.py) will eventually be fed by, once a real campaign
    is wired up to use a saved mapping instead of a bespoke handler."""
    flat_values = flat_values or {}
    item_values = item_values or {}
    out = {}
    for _, name in var_map_entries:
        m = mapping.get(name)
        if not m:
            out[name] = ""
        elif m["source_type"] == "flat_value":
            out[name] = flat_values.get(name, "")
        elif m["source_type"] == "item_attribute":
            out[name] = str(item_values.get(name, "") or "")
        else:
            out[name] = str(customer_row.get(m["source_key"], "") or "")
    return out


def build_item_values(var_map_entries: list, mapping: dict, item_row: dict) -> dict:
    """{variable_name: value} for every variable mapped as 'item_attribute',
    resolved against ONE picked item (item_row: {'item_code', 'item_name',
    'std_price', 'wh_price'} -- see views/marketing.py::_wf_item_price_catalog).
    Called once per campaign (or once per Template Mapping preview render),
    then handed to resolve_recipient_variables as `item_values` for every
    recipient -- the item itself doesn't vary by recipient, only the
    customer-attribute/flat_value variables do."""
    item_row = item_row or {}
    out = {}
    for _, name in var_map_entries:
        m = mapping.get(name)
        if m and m.get("source_type") == "item_attribute":
            out[name] = item_row.get(m["source_key"], "")
    return out
