# processing/customer_tracking.py
# Pure data logic for the "Customer Tracking" commission section — two
# independent, manager-curated customer watchlists (Returns / Collection),
# per explicit ask 2026-10-10: "whatever customer I add in returns watch
# list, will check if there are any returns coming from them... whatever
# customer added to collection, they will notify me if that cust has not
# paid within 3 days."
#
# Same setup/results split as every other commission section (see
# views/commissions.py::_sections): setup (the two comma-list inputs) lives
# on the admin Commissions page, results live in Target Management ->
# Commission Results. JSON-backed, same convention as
# commission_product_watchlist.json — this isn't a payout mechanism, so it
# doesn't belong in the commission_campaigns table at all.

from __future__ import annotations

import json
from pathlib import Path

_DATA_DIR = Path(__file__).parent.parent / "data"
_WATCHLIST_FILE = _DATA_DIR / "customer_tracking_watchlist.json"

GROUPS = ("collection", "returns")


def _load_json(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def _save_json(path: Path, data: dict) -> None:
    _DATA_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def parse_codes(text: str) -> list:
    """Comma-separated free text -> cleaned, de-duped customer code list.
    Uppercased since every real cacus.xcus code in this ERP is uppercase
    (e.g. "CUS-005596") — guards against a manual paste with stray case."""
    if not text:
        return []
    seen = set()
    out = []
    for raw in text.split(","):
        code = raw.strip().upper()
        if code and code not in seen:
            seen.add(code)
            out.append(code)
    return out


def load_watchlist(zid: str) -> dict:
    """{"collection": [...], "returns": [...]} for one ZID — customer codes
    are only unique within one ZID, so scoped per ZID like every other
    watchlist in this app."""
    data = _load_json(_WATCHLIST_FILE)
    zid_data = data.get(str(zid), {})
    return {g: list(zid_data.get(g, [])) for g in GROUPS}


def save_group(zid: str, group: str, codes: list) -> None:
    if group not in GROUPS:
        raise ValueError(f"Unknown watchlist group: {group!r}")
    data = _load_json(_WATCHLIST_FILE)
    zid_key = str(zid)
    zid_data = dict(data.get(zid_key, {}))
    zid_data[group] = list(codes)
    data[zid_key] = zid_data
    _save_json(_WATCHLIST_FILE, data)
