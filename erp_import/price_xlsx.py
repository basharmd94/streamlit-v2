"""The price rule's before / after as an Excel file: `erp_import.py --job price --xlsx`. Reads only; needs openpyxl (pip install openpyxl)."""
import datetime as dt
from decimal import Decimal

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

FONT = "Arial"
APPLIED = ("raise", "lower", "new_item")        # the flags --job price --commit writes
SOURCE = {"100001": "HMBR (100001)", "100000": "GI (100000)", "100005": "Zepto (100005)", "FIXIT": "Fixit"}
RESULT = {"review": "HELD: move over the limit, keeps today's price", "lower": "Goes DOWN", "raise": "Goes UP",
          "new_item": "New price (none today)", "no_source_price": "No source price: unchanged",
          "skipped_exception": "Skipped (exceptions list)", "same": "No change"}
ORDER = list(RESULT)
HEAD = ["SKU", "100007 item", "Name", "Source", "Was (100007 now)", "Rule price", "Will be", "Rule change", "What happens", "Rule"]
WIDTH = [11, 12, 42, 15, 16, 12, 12, 12, 46, 36]
FILL = {"review": "FFF2CC", "lower": "FCE4E4"}


def _num(v):
    return None if v is None else float(v) if isinstance(v, Decimal) else v


def write(path, rows, review_pct):
    """rows = erp_import.price_rows(...). `Will be` is what --job price --commit leaves in 100007 once [pricing] apply = true."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Before and after"
    ws.append(HEAD)
    for c in ws[1]:
        c.font = Font(name=FONT, bold=True)
        c.fill = PatternFill("solid", fgColor="D9D9D9")
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for r in sorted(rows, key=lambda r: (ORDER.index(r["flag"]) if r["flag"] in ORDER else 99, str(r["sku"]))):
        will = r["new"] if r["flag"] in APPLIED else r["old"]
        ws.append([r["sku"], r["xitem"], r.get("name", ""), SOURCE.get(r["source"], r["source"]), _num(r["old"]), _num(r["new"]), _num(will),
                   None if r["change_pct"] is None else r["change_pct"] / 100.0, RESULT.get(r["flag"], r["flag"]), r["rule"]])
        fill = FILL.get(r["flag"])
        for c in ws[ws.max_row]:
            c.font = Font(name=FONT)
            if fill:
                c.fill = PatternFill("solid", fgColor=fill)
        for col in (5, 6, 7):
            ws.cell(ws.max_row, col).number_format = "#,##0.00"
        ws.cell(ws.max_row, 8).number_format = "0.0%"
    for i, w in enumerate(WIDTH, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    sm = wb.create_sheet("Summary")
    sm.append(["Prices of company 100007: what is today and what the rule will leave"])
    sm.append(["Made", dt.datetime.now().strftime("%Y-%m-%d %H:%M")])
    sm.append(["'Will be' assumes [pricing] apply = true and --job price --commit. A move bigger than %s%% is HELD and keeps today's price." % review_pct])
    sm.append([])
    sm.append(["Source"] + [RESULT[k] for k in ORDER] + ["Total"])
    seen = sorted({r["source"] for r in rows}, key=lambda s: list(SOURCE).index(s) if s in SOURCE else 99)
    tot = [0] * len(ORDER)
    for s in seen:
        counts = [sum(1 for r in rows if r["source"] == s and r["flag"] == k) for k in ORDER]
        tot = [a + b for a, b in zip(tot, counts)]
        sm.append([SOURCE.get(s, s)] + counts + [sum(counts)])
    sm.append(["Total"] + tot + [sum(tot)])
    for row in sm.iter_rows():
        for c in row:
            c.font = Font(name=FONT, bold=(c.row in (1, 5, sm.max_row)))
            c.alignment = Alignment(wrap_text=(c.row == 5), vertical="top")
    sm.column_dimensions["A"].width = 18
    for i in range(2, len(ORDER) + 3):
        sm.column_dimensions[get_column_letter(i)].width = 17
    wb.save(path)
    return path
