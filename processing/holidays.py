# processing/holidays.py
"""
Working-day / public-holiday logic — extracted out of views/_tm_shared.py so
it's reusable from processing/ (pure pandas, no st.* calls), not just from
Target Management's own view code. views/_tm_shared.py re-exports these same
names for backward compatibility with existing call sites.

Friday is always off (hardcoded); data/public_holidays.json adds any other
non-Friday off-days (Eid, national holidays falling Sat-Thu).
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

_DATA_DIR = Path(__file__).parent.parent / "data"
_HOLIDAYS_FILE = _DATA_DIR / "public_holidays.json"


def _load_json(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def _save_json(path: Path, data: dict):
    _DATA_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def _get_holidays() -> set:
    """Return all saved public holidays as a set of 'YYYY-MM-DD' strings."""
    return set(_load_json(_HOLIDAYS_FILE).get("holidays", []))


def _prune_holidays():
    """Silently remove holidays from calendar years older than (current_year - 1)."""
    import pandas as pd
    data = _load_json(_HOLIDAYS_FILE)
    if not data:
        return
    keep_from = pd.Timestamp.today().year - 1
    holidays = data.get("holidays", [])
    pruned = [h for h in holidays if int(h[:4]) >= keep_from]
    if len(pruned) != len(holidays):
        data["holidays"] = sorted(pruned)
        _save_json(_HOLIDAYS_FILE, data)


def _toggle_holiday(date_str: str, add: bool):
    data = _load_json(_HOLIDAYS_FILE)
    holidays = set(data.get("holidays", []))
    if add:
        holidays.add(date_str)
    else:
        holidays.discard(date_str)
    data["holidays"] = sorted(holidays)
    _save_json(_HOLIDAYS_FILE, data)


def _is_working_day(d, holidays: set) -> bool:
    """Mon-Thu and Sat-Sun are working days; Friday and public holidays are off."""
    return d.weekday() != 4 and d.strftime("%Y-%m-%d") not in holidays


def _count_working_days(start_d, end_d, holidays: set) -> int:
    count = 0
    cur = start_d
    while cur <= end_d:
        if _is_working_day(cur, holidays):
            count += 1
        cur += timedelta(days=1)
    return count


def working_hours_elapsed(start_dt, end_dt, holidays: set) -> float:
    """Wall-clock hours between start_dt and end_dt, minus 24h for every
    non-working calendar date (Friday or a configured holiday) that falls
    anywhere in [start_dt.date(), end_dt.date()] -- e.g. a Thursday-evening
    order doesn't start accumulating overdue hours again until Saturday,
    since Friday contributes 0. Approximation, not an hour-boundary-exact
    business-hours model (this app doesn't track open/close times) -- good
    enough for a same-day/24h-style SLA threshold.
    Returns 0 if end_dt <= start_dt.
    """
    if end_dt <= start_dt:
        return 0.0
    total_hours = (end_dt - start_dt).total_seconds() / 3600.0
    cur = start_dt.date()
    non_working_days = 0
    while cur <= end_dt.date():
        if not _is_working_day(cur, holidays):
            non_working_days += 1
        cur += timedelta(days=1)
    return max(0.0, total_hours - non_working_days * 24.0)
