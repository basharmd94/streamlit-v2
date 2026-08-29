from __future__ import annotations

import pandas as pd
import streamlit as st


# ── Field Tracking ────────────────────────────────────────────────────────────

_TRACK_COLORS = [
    [0,   116, 217],
    [185,  66, 252],
    [255,  65,  54],
    [46,  204,  64],
    [1,   255, 112],
]
_ORDER_COLOR   = [255, 165,   0]
_CHECKIN_COLOR = [  0, 200, 100]

# Bangladesh bounding box — coords outside this are invalid/mock
_BD_LAT = (20.34, 26.63)
_BD_LON = (88.01, 92.67)

# Pin-shaped icon for order markers (white SVG with mask=True so get_color controls the tint)
import base64 as _b64
_PIN_ICON_URL = "data:image/svg+xml;base64," + _b64.b64encode(
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="64" height="64">'
    b'<path fill="white" d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13'
    b'c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5'
    b' 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/></svg>'
).decode()
_PIN_ICON = {"url": _PIN_ICON_URL, "width": 64, "height": 64, "anchorY": 64, "mask": True}

# 100001 and 100000 share the same field sales team — always query both together.
_FT_SHARED = frozenset({100001, 100000})


def _ft_zids(zid) -> list:
    """Return the ZID(s) to query for field tracking.
    When the active entity is one of the shared sales-team ZIDs, always
    return both so orders from either entity appear together."""
    z = int(zid)
    return sorted(_FT_SHARED) if z in _FT_SHARED else [z]


def _in_bangladesh(lat: float, lon: float) -> bool:
    return _BD_LAT[0] <= lat <= _BD_LAT[1] and _BD_LON[0] <= lon <= _BD_LON[1]


def _day_color(day_index: int, total_days: int) -> list:
    """Blue (day 1) → green (mid month) → red (last day) gradient."""
    t = day_index / max(total_days - 1, 1)
    if t <= 0.5:
        s = t / 0.5
        return [int(0 + s * 46), int(116 + s * 88), int(217 - s * 153)]
    else:
        s = (t - 0.5) / 0.5
        return [int(46 + s * 209), int(204 - s * 139), int(64 - s * 10)]


@st.cache_data(show_spinner=False, ttl=600)
def _load_tracking_salesmen(zids: tuple) -> pd.DataFrame:
    """Load distinct salesmen who have location records and opmob orders for
    any of the given ZIDs.  Pass a tuple (hashable) for cache compatibility."""
    from core.db import get_dataframe
    from core import queries
    sql, params = queries.get_field_tracking_salesmen(list(zids))
    df = get_dataframe(sql, params)
    return df if df is not None else pd.DataFrame()


def _render_field_tracking_monthly(ft_zids, sp_df, pdk):
    import calendar as _cal
    from core.db import get_dataframe
    from core import queries

    sp_labels = (sp_df["username"] + " — " + sp_df["display_name"]).tolist()
    sp_map    = dict(zip(sp_labels, sp_df["username"].tolist()))
    name_map  = dict(zip(sp_df["username"], sp_df["display_name"]))

    # ── Controls ──────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([3, 1, 2])
    with col1:
        sel_labels = st.multiselect("Salesman", sp_labels, default=sp_labels[:1], key="ft_mo_sp")
    with col2:
        today = pd.Timestamp.today()
        sel_mo = st.date_input(
            "Month", value=today.replace(day=1).date(), key="ft_mo_month",
            help="Pick any day — only year and month are used",
        )
    with col3:
        show_layers = st.multiselect(
            "Show layers",
            ["Movement lines", "Order locations"],
            default=["Movement lines", "Order locations"],
            key="ft_mo_layers",
        )
    show_lines  = "Movement lines"   in show_layers
    show_orders = "Order locations"  in show_layers

    if not sel_labels:
        st.info("Select at least one salesman.")
        return

    usernames = [sp_map[l] for l in sel_labels]
    sp_names  = [name_map.get(u, u) for u in usernames]
    year, month = sel_mo.year, sel_mo.month
    use_gradient = (len(usernames) == 1)   # date-gradient only for single; per-salesman colours for multi

    # ── Fetch data for each salesman ──────────────────────────────────────────
    all_track_dfs = {}
    all_order_dfs = {}
    for username in usernames:
        sql, params = queries.get_location_track_monthly(username, year, month)
        tdf = get_dataframe(sql, params)
        if tdf is not None and not tdf.empty:
            tdf["track_date"] = pd.to_datetime(tdf["track_date"]).dt.date
            all_track_dfs[username] = tdf

        sql2, params2 = queries.get_opmob_order_locations_monthly(list(ft_zids), username, year, month)
        odf = get_dataframe(sql2, params2)
        if odf is not None and not odf.empty:
            odf["order_date"] = pd.to_datetime(odf["xdate"]).dt.date
            all_order_dfs[username] = odf

    track_empty = not all_track_dfs
    order_empty = not all_order_dfs

    if track_empty and order_empty:
        st.info(f"No data for {', '.join(sp_names)} in {sel_mo.strftime('%B %Y')}.")
        return

    # ── Unified date list ─────────────────────────────────────────────────────
    all_dates_set = set()
    for tdf in all_track_dfs.values():
        all_dates_set.update(tdf["track_date"].unique())
    for odf in all_order_dfs.values():
        all_dates_set.update(odf["order_date"].unique())
    all_dates   = sorted(all_dates_set)
    n_days      = len(all_dates)
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    # ── No-data working-day warning (single salesman only) ────────────────────
    if use_gradient:
        username0   = usernames[0]
        track_dates0 = set(all_track_dfs[username0]["track_date"].unique()) if username0 in all_track_dfs else set()
        month_range = pd.date_range(
            start=pd.Timestamp(year=year, month=month, day=1),
            end=pd.Timestamp(year=year, month=month, day=_cal.monthrange(year, month)[1]),
            freq="D",
        )
        no_data_days = [
            d.strftime("%d %b")
            for d in month_range
            if d.weekday() != 4
            and d.date() not in track_dates0
            and d.date() <= pd.Timestamp.today().date()
        ]
        if no_data_days:
            st.caption("⚠ No GPS data (working days): " + ", ".join(no_data_days))

    # ── Build layers per salesman ─────────────────────────────────────────────
    path_data        = []
    point_data       = []
    order_data       = []
    all_coords       = []
    no_gps_orders_mo = []

    for sp_idx, username in enumerate(usernames):
        sp_name   = name_map.get(username, username)
        sp_color  = _TRACK_COLORS[sp_idx % len(_TRACK_COLORS)]

        tdf = all_track_dfs.get(username)
        if tdf is not None:
            for day in sorted(tdf["track_date"].unique()):
                color  = _day_color(date_to_idx[day], n_days) if use_gradient else sp_color
                day_df = tdf[tdf["track_date"] == day]
                coords = [(float(r["longitude"]), float(r["latitude"])) for _, r in day_df.iterrows()]
                if not coords:
                    continue
                all_coords.extend(coords)
                if len(coords) >= 2:
                    path_data.append({"path": coords, "color": color})
                for (lon, lat), (_, row) in zip(coords, day_df.iterrows()):
                    ts_str = pd.to_datetime(row["ts"]).strftime("%H:%M") if pd.notna(row.get("ts")) else ""
                    prefix = "" if use_gradient else f"[{sp_name}] "
                    point_data.append({
                        "coordinates": [lon, lat],
                        "color": color, "radius": 12,
                        "tooltip": f"{prefix}{day.strftime('%d %b')}  {ts_str}",
                    })

        odf = all_order_dfs.get(username)
        if odf is not None:
            for _, row in odf.iterrows():
                lat   = float(row["lat"] or 0)
                lon   = float(row["lon"] or 0)
                odate = row["order_date"]
                _is_no_stock = "no stock" in str(row.get("status", "")).lower()
                if not _in_bangladesh(lat, lon):
                    no_gps_orders_mo.append(row)
                    continue
                color = _day_color(date_to_idx.get(odate, 0), n_days) if use_gradient else sp_color
                all_coords.append((lon, lat))
                prefix = "" if use_gradient else f"[{sp_name}] "
                order_data.append({
                    "coordinates": [lon, lat],
                    "color": [180, 180, 180] if _is_no_stock else color,
                    "icon": _PIN_ICON,
                    "tooltip": (
                        f"{prefix}Order: {row['order_num']}\n"
                        f"Date: {odate.strftime('%d %b')}\n"
                        f"Customer: {row['cusname']}\n"
                        f"Status: {row['status']}\n"
                        f"Total: {int(row['total'] or 0):,}"
                    ),
                })

    if not all_coords:
        st.info(f"No valid GPS data for {', '.join(sp_names)} in {sel_mo.strftime('%B %Y')}.")
        return

    # ── Map ───────────────────────────────────────────────────────────────────
    lons = [c[0] for c in all_coords]
    lats = [c[1] for c in all_coords]
    span = max(max(lats) - min(lats), max(lons) - min(lons))
    zoom = 13 if span < 0.02 else (11 if span < 0.1 else (10 if span < 0.3 else 8))

    layers = []
    if show_lines and path_data:
        layers.append(pdk.Layer("PathLayer", data=path_data, get_path="path",
                                get_color="color", width_min_pixels=2,
                                width_max_pixels=4, pickable=True))
    if show_lines and point_data:
        layers.append(pdk.Layer("ScatterplotLayer", data=point_data,
                                get_position="coordinates", get_fill_color="color",
                                get_radius="radius", pickable=True,
                                auto_highlight=True, opacity=0.75))
    if show_orders and order_data:
        layers.append(pdk.Layer(
            "IconLayer", data=order_data,
            get_icon="icon", get_position="coordinates", get_color="color",
            get_size=40, size_min_pixels=24, size_max_pixels=60,
            pickable=True, auto_highlight=True,
        ))

    if not layers:
        st.info("Select at least one layer to display.")
        return

    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=(min(lats) + max(lats)) / 2,
            longitude=(min(lons) + max(lons)) / 2,
            zoom=zoom, pitch=0,
        ),
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"text": "{tooltip}"},
    ), use_container_width=True)

    # ── Legend ────────────────────────────────────────────────────────────────
    if use_gradient and all_dates:
        step    = max(1, n_days // 10)
        anchors = sorted(set(list(range(0, n_days, step)) + [n_days - 1]))
        leg_parts = []
        for idx in anchors:
            c     = _day_color(idx, n_days)
            hex_c = "#{:02x}{:02x}{:02x}".format(*c)
            leg_parts.append(f"<span style='color:{hex_c}'>●</span> {all_dates[idx].strftime('%d %b')}")
        st.markdown(
            "<small>" + "&nbsp;→&nbsp;".join(leg_parts)
            + "&nbsp;&nbsp;<b>·</b>&nbsp;&nbsp;"
            + "◯ line ping &nbsp; ● order (same gradient)</small>",
            unsafe_allow_html=True,
        )
    else:
        leg_items = []
        for sp_idx, (username, sp_name) in enumerate(zip(usernames, sp_names)):
            c = _TRACK_COLORS[sp_idx % len(_TRACK_COLORS)]
            hex_c = "#{:02x}{:02x}{:02x}".format(*c)
            leg_items.append(f"<span style='color:{hex_c}'>●</span> {sp_name}")
        leg_items.append("<span style='color:#b4b4b4'>●</span> No stock (not created)")
        st.markdown("<small>" + "&nbsp; &nbsp;".join(leg_items) + "</small>", unsafe_allow_html=True)

    # ── Caption ───────────────────────────────────────────────────────────────
    n_created  = sum(1 for o in order_data if o["color"] != [180, 180, 180])
    n_no_stock_map = len(order_data) - n_created
    _no_gps_count  = len(no_gps_orders_mo)
    _days_with_gps = len({d for tdf in all_track_dfs.values() for d in tdf["track_date"].unique()})
    _caption = (
        f"{', '.join(sp_names)} · {sel_mo.strftime('%B %Y')} · "
        f"{_days_with_gps} days with GPS · {len(point_data)} pings · "
        f"{n_created} orders created"
    )
    if n_no_stock_map:
        _caption += f" · {n_no_stock_map} not created (no stock)"
    if _no_gps_count:
        _caption += f" · {_no_gps_count} order(s) without GPS"
    st.caption(_caption)

    # ── Day-by-day breakdown table ────────────────────────────────────────────
    with st.expander("📋 Day-by-day breakdown", expanded=False):
        rows = []
        for day in all_dates:
            n_pings, first_s, last_s, checkins = 0, "—", "—", 0
            for username in usernames:
                tdf = all_track_dfs.get(username)
                if tdf is not None and day in tdf["track_date"].values:
                    day_df   = tdf[tdf["track_date"] == day]
                    n_pings += len(day_df)
                    ts_vals  = pd.to_datetime(day_df["ts"], errors="coerce").dropna()
                    if len(ts_vals):
                        fv = ts_vals.min().strftime("%H:%M")
                        lv = ts_vals.max().strftime("%H:%M")
                        first_s = fv if first_s == "—" else min(first_s, fv)
                        last_s  = lv if last_s  == "—" else max(last_s,  lv)
                    checkins += int(day_df["is_check_in"].sum()) if "is_check_in" in day_df.columns else 0

            n_orders = n_no_stock_d = ord_total = n_cust = 0
            for username in usernames:
                odf = all_order_dfs.get(username)
                if odf is not None and day in odf["order_date"].values:
                    day_ord    = odf[odf["order_date"] == day]
                    _ns_mask   = day_ord["status"].str.contains("no stock", case=False, na=False)
                    n_orders      += int((~_ns_mask).sum())
                    n_no_stock_d  += int(_ns_mask.sum())
                    ord_total     += int(day_ord.loc[~_ns_mask, "total"].sum())
                    n_cust        += int(day_ord["cusid"].nunique())

            row_d = {
                "Date":             day.strftime("%d %b %Y"),
                "Day":              day.strftime("%A"),
                "GPS Pings":        n_pings,
                "First Seen":       first_s,
                "Last Seen":        last_s,
                "Check-ins":        checkins,
                "Orders Created":   n_orders,
                "Unique Customers": n_cust,
                "Order Total":      ord_total,
            }
            if n_no_stock_d:
                row_d["No-Stock Attempts"] = n_no_stock_d
            rows.append(row_d)

        tbl = pd.DataFrame(rows)
        st.dataframe(
            tbl.style.format({"Order Total": "{:,.0f}"}),
            width="stretch",
            hide_index=True,
        )

        if no_gps_orders_mo:
            st.markdown(f"**{len(no_gps_orders_mo)} order(s) without GPS coordinates (not shown on map):**")
            no_gps_tbl = pd.DataFrame([{
                "Date":     r["order_date"].strftime("%d %b %Y"),
                "Order":    r["order_num"],
                "Customer": r["cusname"],
                "Status":   r["status"],
                "Total":    int(r["total"] or 0),
            } for r in no_gps_orders_mo])
            st.dataframe(no_gps_tbl, width="stretch", hide_index=True)

        # ── All orders this month (GPS + no-GPS, full detail) ─────────────────
        if all_order_dfs:
            all_ord_rows = []
            for _odf in all_order_dfs.values():
                for _, row in _odf.iterrows():
                    try:
                        loc = f"{float(row['lat']):.6f}, {float(row['lon']):.6f}"
                    except (TypeError, ValueError):
                        loc = ""
                    all_ord_rows.append({
                        "ZID":         row.get("zid", ""),
                        "Date":        row["order_date"].strftime("%d %b %Y"),
                        "Cust. Code":  row.get("cusid", ""),
                        "Customer":    row["cusname"],
                        "Lat, Lon":    loc,
                        "Order Value": int(row["total"] or 0),
                    })
            if all_ord_rows:
                st.markdown("**All orders this month:**")
                st.dataframe(
                    pd.DataFrame(all_ord_rows).style.format({"Order Value": "{:,.0f}"}),
                    width="stretch",
                    hide_index=True,
                )


def _render_field_tracking(zid):
    try:
        import pydeck as pdk
    except ImportError:
        st.error("pydeck is not installed. Run: pip install pydeck")
        return

    st.subheader("🗺️ Field Tracking")
    _ft_mode = st.radio(
        "View", ["📅 Daily", "📆 Monthly"], horizontal=True,
        key="ft_mode", label_visibility="collapsed",
    )

    ft_zids = tuple(_ft_zids(zid))
    sp_df = _load_tracking_salesmen(ft_zids)
    if sp_df.empty:
        st.info("No salesmen with location records found for this entity.")
        return

    if _ft_mode == "📆 Monthly":
        _render_field_tracking_monthly(ft_zids, sp_df, pdk)
        return

    sp_labels = (sp_df["username"] + " — " + sp_df["display_name"]).tolist()
    sp_map    = dict(zip(sp_labels, sp_df["username"].tolist()))

    col1, col2 = st.columns([3, 1])
    with col1:
        sel_labels = st.multiselect("Salesman", sp_labels, default=sp_labels[:1], key="ft_sp")
    with col2:
        sel_date = st.date_input("Date", value=pd.Timestamp.today().date(), key="ft_date")

    if not sel_labels:
        st.info("Select at least one salesman.")
        return

    from core.db import get_dataframe
    from core import queries

    date_str   = str(sel_date)

    # ── No-data warning note ──────────────────────────────────────────────────
    # Check which known salesmen have zero valid BD-coordinate rows on this date
    _cov_sql = """
        SELECT DISTINCT username FROM location_records
        WHERE DATE(COALESCE(timestamp, created_at)) = %s
          AND latitude  BETWEEN 20.34 AND 26.63
          AND longitude BETWEEN 88.01 AND 92.67
    """
    _cov_df = get_dataframe(_cov_sql, (date_str,))
    _active = set(_cov_df["username"].tolist()) if _cov_df is not None and not _cov_df.empty else set()
    _no_data = [
        row["display_name"]
        for _, row in sp_df.iterrows()
        if row["username"] not in _active
    ]
    if _no_data:
        st.caption(
            "⚠ No GPS data on this date for: "
            + ", ".join(_no_data)
        )
    path_data        = []
    point_data       = []
    order_data       = []
    no_gps_rows      = []   # orders with missing/invalid GPS — shown in table below map
    all_coords       = []
    stats            = []
    _stored_ord_dfs  = {}   # sp_name -> ord_df (kept for the orders table after the map)

    for i, label in enumerate(sel_labels):
        username  = sp_map[label]
        sp_name   = label.split(" — ", 1)[-1]
        color     = _TRACK_COLORS[i % len(_TRACK_COLORS)]

        # ── GPS track ─────────────────────────────────────────────────────────
        sql, params = queries.get_location_track(username, date_str)
        track_df    = get_dataframe(sql, params)

        n_pings = 0
        n_dropped_track = 0
        if track_df is not None and not track_df.empty:
            valid_coords = []
            for _, row in track_df.iterrows():
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                if not _in_bangladesh(lat, lon):
                    n_dropped_track += 1
                    continue
                valid_coords.append((lon, lat))
                ts_val  = row.get("ts")
                ts_str  = pd.to_datetime(ts_val).strftime("%H:%M") if pd.notna(ts_val) else ""
                addr    = str(row.get("formatted_address") or "").strip()
                is_ci   = bool(row.get("is_check_in"))
                is_mock = bool(row.get("is_mock_location"))
                tip     = f"{sp_name}  {ts_str}"
                if is_ci:
                    tip += "  [CHECK-IN]"
                if is_mock:
                    tip += "  ⚠ MOCK"
                if addr:
                    tip += f"\n{addr}"
                point_data.append({
                    "coordinates": [lon, lat],
                    "color":  _CHECKIN_COLOR if is_ci else color,
                    "radius": 35 if is_ci else 18,
                    "tooltip": tip,
                })

            if valid_coords:
                all_coords.extend(valid_coords)
                path_data.append({"path": valid_coords, "color": color})
            n_pings = len(valid_coords)

        # ── Order locations ───────────────────────────────────────────────────
        sql2, params2 = queries.get_opmob_order_locations(list(ft_zids), username, date_str)
        ord_df        = get_dataframe(sql2, params2)
        _stored_ord_dfs[sp_name] = ord_df

        n_orders   = 0
        n_no_stock = 0
        n_no_gps   = 0
        if ord_df is not None and not ord_df.empty:
            for _, row in ord_df.iterrows():
                lat = float(row["lat"] or 0)
                lon = float(row["lon"] or 0)
                _is_no_stock = "no stock" in str(row.get("status", "")).lower()
                if not _in_bangladesh(lat, lon):
                    n_no_gps += 1
                    no_gps_rows.append({
                        "Salesman":   sp_name,
                        "Order":      row["order_num"],
                        "Customer":   row["cusname"],
                        "Status":     row["status"],
                        "Total":      int(row["total"] or 0),
                    })
                    continue
                all_coords.append((lon, lat))
                if _is_no_stock:
                    n_no_stock += 1
                else:
                    n_orders += 1
                order_data.append({
                    "coordinates": [lon, lat],
                    "color":  [180, 180, 180] if _is_no_stock else _ORDER_COLOR,
                    "icon":   _PIN_ICON,
                    "tooltip": (
                        f"Order: {row['order_num']}\n"
                        f"Customer: {row['cusname']}\n"
                        f"Status: {row['status']}\n"
                        f"Total: {int(row['total'] or 0):,}"
                    ),
                })

        stat_line = f"**{sp_name}**: {n_pings} pings · {n_orders} orders created"
        if n_no_stock:
            stat_line += f" · {n_no_stock} not created (no stock)"
        if n_no_gps:
            stat_line += f" · {n_no_gps} order(s) without GPS"
        if n_dropped_track:
            stat_line += f" · ⚠ {n_dropped_track} invalid track coord(s)"
        stats.append(stat_line)

    if not all_coords:
        st.info(f"No location data for {sel_date.strftime('%d %b %Y')}.")
        return

    # ── Auto-center + zoom ────────────────────────────────────────────────────
    lons = [c[0] for c in all_coords]
    lats = [c[1] for c in all_coords]
    span = max(max(lats) - min(lats), max(lons) - min(lons))
    zoom = 14 if span < 0.01 else (12 if span < 0.05 else (10 if span < 0.2 else 8))

    # ── pydeck layers ─────────────────────────────────────────────────────────
    layers = []
    if path_data:
        layers.append(pdk.Layer(
            "PathLayer",
            data=path_data,
            get_path="path",
            get_color="color",
            width_min_pixels=3,
            width_max_pixels=5,
            pickable=False,
        ))
    if point_data:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=point_data,
            get_position="coordinates",
            get_fill_color="color",
            get_radius="radius",
            pickable=True,
            auto_highlight=True,
            opacity=0.85,
        ))
    if order_data:
        layers.append(pdk.Layer(
            "IconLayer",
            data=order_data,
            get_icon="icon",
            get_position="coordinates",
            get_color="color",
            get_size=40,
            size_min_pixels=24,
            size_max_pixels=60,
            pickable=True,
            auto_highlight=True,
        ))

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=(min(lats) + max(lats)) / 2,
            longitude=(min(lons) + max(lons)) / 2,
            zoom=zoom,
            pitch=0,
        ),
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"text": "{tooltip}"},
    )
    st.pydeck_chart(deck, use_container_width=True)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = []
    for i, label in enumerate(sel_labels):
        c = _TRACK_COLORS[i % len(_TRACK_COLORS)]
        hex_c = "#{:02x}{:02x}{:02x}".format(*c)
        legend_items.append(f"<span style='color:{hex_c}'>●</span> {label.split(' — ',1)[-1]}")
    legend_items.append("<span style='color:#ffa500'>●</span> Order location")
    legend_items.append("<span style='color:#00c864'>●</span> Check-in")
    st.markdown("  &nbsp;·&nbsp;  ".join(legend_items), unsafe_allow_html=True)
    st.caption("  ·  ".join(stats))

    # ── Orders table (all orders this date, GPS + no-GPS combined) ───────────
    _all_ord_rows = []
    for sp_name, _ord_df in _stored_ord_dfs.items():
        if _ord_df is None or _ord_df.empty:
            continue
        for _, row in _ord_df.iterrows():
            try:
                _lat = float(row["lat"] or 0)
                _lon = float(row["lon"] or 0)
                _loc = f"{_lat:.6f}, {_lon:.6f}" if _in_bangladesh(_lat, _lon) else "—"
            except (TypeError, ValueError):
                _loc = "—"
            _t = row.get("order_time")
            _time_str = pd.to_datetime(_t).strftime("%H:%M") if pd.notna(_t) else "—"
            _mob = str(row.get("cusmobile") or "").strip()
            _mob = _mob if _mob and _mob != "nan" else "—"
            _all_ord_rows.append({
                "Salesman":    sp_name,
                "Cus Code":   row.get("cusid", ""),
                "Customer":   row.get("cusname", ""),
                "Mobile":     _mob,
                "Status":     row.get("status", ""),
                "Time":       _time_str,
                "Order Value": int(row.get("total") or 0),
                "Lat, Lon":   _loc,
            })

    if _all_ord_rows:
        st.markdown("#### 📋 Orders — " + sel_date.strftime("%d %b %Y"))
        _ord_tbl = pd.DataFrame(_all_ord_rows)
        from processing.common import normalize_phone_cols
        _ord_tbl = normalize_phone_cols(_ord_tbl, extra_cols=["Mobile"])
        st.dataframe(
            _ord_tbl.style.format({"Order Value": "{:,.0f}"}),
            width="stretch",
            hide_index=True,
        )

    # ── Orders without GPS ────────────────────────────────────────────────────
    if no_gps_rows:
        with st.expander(f"📋 {len(no_gps_rows)} order(s) without GPS — not shown on map", expanded=False):
            st.dataframe(
                pd.DataFrame(no_gps_rows),
                width="stretch",
                hide_index=True,
            )


