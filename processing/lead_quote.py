# processing/lead_quote.py
"""
Lead Quote generator (Marketing -> Leads -> "Generate Quote"). Builds a
one-page PDF price quotation on an existing letterhead template, for a
hand-picked lead and a hand-picked list of items + quantities.

Pricing is ALWAYS sourced from zid=100007's own caitem.xstdprice and
opspprc tiers, regardless of which letterhead (brand) is chosen for the
quote's look -- confirmed explicitly: the letterhead choice only changes
which logo/branding prints and which subset of the 100007 catalog is
offered in the item picker (barcode prefix 03- = Zepto-sourced, 01-/02- =
HMBR/GI-sourced), never which ZID the price comes from.

opspprc.xqty/xqtypur define a closed, non-overlapping qty range per tier
(e.g. 1-1, 2-3, 4-99999) -- see scripts/upload_opspprc_100007.py for how
these were populated. resolve_tier_price below finds the tier whose range
contains the requested qty.
"""

from __future__ import annotations

from datetime import date
from io import BytesIO

import pandas as pd
import streamlit as st

from core.analytics import Analytics

_LETTERHEAD_PATHS = {
    "Zepto": "data/letterhead_zepto.pdf",
    "HMBR": "data/letterhead_hmbr.pdf",
}

# Barcode-prefix convention confirmed against real 100007 caitem data:
# 01- = HMBR(100001)-sourced, 02- = GI(100000)-sourced, 03- = Zepto(100005)-sourced.
# "HMBR" letterhead covers both 01- and 02- (100001 + 100000, per explicit ask).
# Source-ZID column (caitem.xitemnew, confirmed real) -- the authoritative
# way to tell which ZID a 100007 item was originally sourced from. "HMBR"
# letterhead covers both 100001 and 100000, per explicit ask.
_BRAND_SOURCE_ZIDS = {
    "Zepto": ("100005",),
    "HMBR": ("100001", "100000"),
}


@st.cache_data(show_spinner=False, ttl=900)
def load_quotable_items() -> pd.DataFrame:
    """One row per (item, tier) for every 100007 item that has at least one
    opspprc tier defined -- items with no tier at all aren't offered, since
    there'd be nothing to discount."""
    df = Analytics("item_price_tiers_100007", zid="100007", filters={}).data
    return df if df is not None else pd.DataFrame()


def items_for_brand(tiers_df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """Distinct item list (xitem, xdesc, xstdprice, xalias), filtered to the
    brand's own source ZID(s) via caitem.xitemnew -- one row per item, not
    per tier. An item with no xitemnew on file (either a genuinely new
    100007-only product, or a gap in the source data) is excluded from
    every brand rather than guessed into one."""
    if tiers_df.empty:
        return tiers_df
    source_zids = _BRAND_SOURCE_ZIDS.get(brand, ())
    mask = tiers_df["xitemnew"].astype(str).isin(source_zids)
    return (
        tiers_df.loc[mask, ["xitem", "xdesc", "xstdprice", "xalias"]]
        .drop_duplicates(subset=["xitem"])
        .sort_values("xdesc")
        .reset_index(drop=True)
    )


def resolve_tier_price(tiers_df: pd.DataFrame, xitem: str, qty: float) -> dict:
    """Finds the opspprc tier for this item whose [xqty, xqtypur] range
    contains qty, and returns the resulting unit price. Falls back to the
    highest tier at/below qty if qty falls in a gap, and to full xstdprice
    (no discount) if qty is below every defined tier -- never raises."""
    rows = tiers_df[tiers_df["xitem"] == xitem]
    if rows.empty:
        return {"xstdprice": 0.0, "xdisc": 0.0, "unit_price": 0.0}

    std_price = float(rows["xstdprice"].iloc[0])
    in_range = rows[(rows["xqty"] <= qty) & (qty <= rows["xqtypur"])]
    if not in_range.empty:
        disc = float(in_range.iloc[0]["xdisc"])
    else:
        below = rows[rows["xqty"] <= qty]
        disc = float(below.sort_values("xqty").iloc[-1]["xdisc"]) if not below.empty else 0.0

    unit_price = max(std_price - disc, 0.0)
    return {"xstdprice": std_price, "xdisc": disc, "unit_price": unit_price}


_LETTER_INTRO = (
    "Thank you for your interest in our products. As requested, please find "
    "below our price quotation for the items specified:"
)
_LETTER_CLOSING = (
    "This quotation is valid for 7 days from the date of issue. Prices are "
    "subject to change without prior notice thereafter. Please feel free to "
    "contact us for any clarification or to confirm your order."
)
_LETTER_SIGNOFF = "We look forward to serving you.\n\nSincerely,\n{company} Sales Team"


def build_quote_pdf(brand: str, lead_name: str, items: list[dict], quote_date: date | None = None) -> bytes:
    """items: [{"xitem","xdesc","qty","unit_price"}, ...]. Returns PDF bytes,
    one page, letterhead as background. Raises FileNotFoundError if that
    brand's letterhead file doesn't exist yet (HMBR, until uploaded)."""
    from pathlib import Path
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle
    from reportlab.pdfgen import canvas
    from pypdf import PdfReader, PdfWriter

    letterhead_path = Path(_LETTERHEAD_PATHS[brand])
    if not letterhead_path.exists():
        raise FileNotFoundError(
            f"{letterhead_path} not found -- upload the {brand} letterhead PDF to data/ first."
        )

    quote_date = quote_date or date.today()
    page_w, page_h = A4

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    margin_x = 50
    y = page_h - 160  # below the letterhead's header banner

    c.setFont("Helvetica", 10)
    c.drawRightString(page_w - margin_x, y, f"Date: {quote_date.strftime('%d %B %Y')}")
    y -= 30

    c.setFont("Helvetica", 11)
    c.drawString(margin_x, y, "Dear Sir/Madam,")
    y -= 14
    c.drawString(margin_x, y, f"(RE: {lead_name})" if lead_name else "")
    y -= 22

    c.setFont("Helvetica", 10)
    for line in _wrap(_LETTER_INTRO, 95):
        c.drawString(margin_x, y, line)
        y -= 14
    y -= 10

    table_data = [["Item", "Qty", "Unit Price (BDT)", "Line Total (BDT)"]]
    grand_total = 0.0
    for it in items:
        line_total = it["qty"] * it["unit_price"]
        grand_total += line_total
        table_data.append([
            it["xdesc"], str(it["qty"]),
            f"{it['unit_price']:,.2f}", f"{line_total:,.2f}",
        ])
    table_data.append(["", "", "Grand Total", f"{grand_total:,.2f}"])

    table = Table(table_data, colWidths=[230, 50, 110, 110])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -2), 0.5, colors.HexColor("#BDC3C7")),
        ("LINEABOVE", (0, -1), (-1, -1), 1, colors.HexColor("#2C3E50")),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    table_w, table_h = table.wrapOn(c, page_w - 2 * margin_x, y)
    y -= table_h
    table.drawOn(c, margin_x, y)
    y -= 24

    c.setFont("Helvetica", 10)
    for line in _wrap(_LETTER_CLOSING, 95):
        c.drawString(margin_x, y, line)
        y -= 14
    y -= 16

    for line in _LETTER_SIGNOFF.format(company=brand).split("\n"):
        c.drawString(margin_x, y, line)
        y -= 14

    c.save()
    buf.seek(0)

    overlay_reader = PdfReader(buf)
    letterhead_reader = PdfReader(str(letterhead_path))
    writer = PdfWriter()

    base_page = letterhead_reader.pages[0]
    base_page.merge_page(overlay_reader.pages[0])
    writer.add_page(base_page)

    out = BytesIO()
    writer.write(out)
    return out.getvalue()


def _wrap(text: str, width: int) -> list[str]:
    import textwrap
    return textwrap.wrap(text, width=width)
