import pandas as pd
import calendar
from datetime import date, timedelta


def default_end_date() -> date:
    """Returns today - 1 day."""
    return date.today() - timedelta(days=1)


def default_start_date(end: date) -> date:
    """Returns 3 calendar months before end, same day, clamped to last day of that month."""
    month = end.month - 3
    year = end.year
    if month <= 0:
        month += 12
        year -= 1
    last_day = calendar.monthrange(year, month)[1]
    day = min(end.day, last_day)
    return date(year, month, day)


def apply_date_window(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Filter df to rows where date is in [start, end]. Converts date column with errors='coerce'."""
    if df is None or df.empty or 'date' not in df.columns:
        return df if df is not None else pd.DataFrame()
    dates = pd.to_datetime(df['date'], errors='coerce')
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return df[mask].copy()


def apply_entity_filter(df: pd.DataFrame, col: str, codes: list) -> pd.DataFrame:
    """Filter df to rows where df[col].isin(codes). If codes is empty, return df unchanged."""
    if not codes or df is None or df.empty or col not in df.columns:
        return df if df is not None else pd.DataFrame()
    codes_str = [str(c) for c in codes]
    return df[df[col].astype(str).isin(codes_str)].copy()


def _apply_entity_filters_to_df(df: pd.DataFrame, sp_codes, item_codes, item_groups) -> pd.DataFrame:
    """Apply all three entity filters to a single DataFrame, silently skipping absent columns."""
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    if sp_codes and 'spid' in df.columns:
        df = apply_entity_filter(df, 'spid', sp_codes)
    if item_codes and 'itemcode' in df.columns:
        df = apply_entity_filter(df, 'itemcode', item_codes)
    if item_groups and 'itemgroup' in df.columns:
        df = apply_entity_filter(df, 'itemgroup', item_groups)
    return df


def compute_daily_metric(
    sales_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    collection_df: pd.DataFrame,
    metric: str,
    start,
    end,
    sp_codes: list,
    item_codes: list,
    item_groups: list,
) -> pd.DataFrame:
    """
    Returns DataFrame with columns [date_label (datetime), value (float)],
    one row per calendar day in the window, aggregated over all filtered entities.

    Net Sales       = sum(final_sales) - sum(treturnamt) per day
    Net Returns     = sum(treturnamt) per day
    Net Collections = sum(value) from collection_df per day
    """
    if metric == "Net Collections":
        c = apply_date_window(collection_df, start, end)
        if c is None or c.empty or 'value' not in c.columns:
            return pd.DataFrame(columns=['date_label', 'value'])
        c = c.copy()
        c['date'] = pd.to_datetime(c['date'], errors='coerce')
        # Apply salesman filter if sp_codes provided and collection has spid column
        if sp_codes and 'spid' in c.columns:
            c = apply_entity_filter(c, 'spid', sp_codes)
        if c.empty:
            return pd.DataFrame(columns=['date_label', 'value'])
        grp = c.groupby('date')['value'].sum().reset_index()
        grp.columns = ['date_label', 'value']
        grp['date_label'] = pd.to_datetime(grp['date_label'])
        return grp.sort_values('date_label').reset_index(drop=True)

    # Net Returns
    r = apply_date_window(returns_df, start, end)
    r = _apply_entity_filters_to_df(r, sp_codes, item_codes, item_groups)

    if metric == "Net Returns":
        if r is None or r.empty or 'treturnamt' not in r.columns:
            return pd.DataFrame(columns=['date_label', 'value'])
        r = r.copy()
        r['date'] = pd.to_datetime(r['date'], errors='coerce')
        grp = r.groupby('date')['treturnamt'].sum().reset_index()
        grp.columns = ['date_label', 'value']
        grp['date_label'] = pd.to_datetime(grp['date_label'])
        return grp.sort_values('date_label').reset_index(drop=True)

    # Net Sales
    s = apply_date_window(sales_df, start, end)
    s = _apply_entity_filters_to_df(s, sp_codes, item_codes, item_groups)

    s_grp = (
        s.assign(date=pd.to_datetime(s['date'], errors='coerce'))
        .groupby('date')['final_sales'].sum().reset_index()
        if (s is not None and not s.empty and 'final_sales' in s.columns)
        else pd.DataFrame(columns=['date', 'final_sales'])
    )
    r_grp = (
        r.assign(date=pd.to_datetime(r['date'], errors='coerce'))
        .groupby('date')['treturnamt'].sum().reset_index()
        if (r is not None and not r.empty and 'treturnamt' in r.columns)
        else pd.DataFrame(columns=['date', 'treturnamt'])
    )

    merged = pd.merge(s_grp, r_grp, on='date', how='outer')
    merged['final_sales'] = merged['final_sales'].fillna(0)
    merged['treturnamt'] = merged['treturnamt'].fillna(0)
    merged['value'] = merged['final_sales'] - merged['treturnamt']
    merged = merged.rename(columns={'date': 'date_label'})
    merged['date_label'] = pd.to_datetime(merged['date_label'])
    return merged[['date_label', 'value']].sort_values('date_label').reset_index(drop=True)


def compute_period_totals(
    sales_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    collection_df: pd.DataFrame,
    start,
    end,
    sp_codes: list,
    item_codes: list,
    item_groups: list,
) -> dict:
    """
    Returns {'net_sales': float, 'net_returns': float, 'net_collections': float}
    by summing compute_daily_metric for each metric over the full window.
    """
    kw = dict(sales_df=sales_df, returns_df=returns_df, collection_df=collection_df,
              start=start, end=end, sp_codes=sp_codes, item_codes=item_codes, item_groups=item_groups)
    ns = compute_daily_metric(metric="Net Sales", **kw)
    nr = compute_daily_metric(metric="Net Returns", **kw)
    nc = compute_daily_metric(metric="Net Collections", **kw)
    return {
        'net_sales':       float(ns['value'].sum()) if not ns.empty else 0.0,
        'net_returns':     float(nr['value'].sum()) if not nr.empty else 0.0,
        'net_collections': float(nc['value'].sum()) if not nc.empty else 0.0,
    }


def _entity_cols(entity: str):
    """Return (code_col, name_col) for entity name."""
    return {
        "Salesman":      ("spid",      "spname"),
        "Product":       ("itemcode",  "itemname"),
        "Product Group": ("itemgroup", None),
    }[entity]


def compute_daily_pivot(
    sales_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    collection_df: pd.DataFrame,
    entity: str,
    metric: str,
    start,
    end,
    sp_codes: list = None,
    item_codes: list = None,
    item_groups: list = None,
) -> pd.DataFrame:
    """
    Returns pivot: index=date (YYYY-MM-DD), columns=entity label, values=metric.

    Sidebar filters (sp_codes / item_codes / item_groups) narrow which rows are included
    before pivoting, so only the selected entities appear as columns.
    """
    if metric == "Net Collections":
        c = apply_date_window(collection_df, start, end)
        if c is None or c.empty or 'value' not in c.columns:
            return pd.DataFrame()
        c = c.copy()
        # Apply salesman filter so only selected sp shows as a column
        if sp_codes and 'spid' in c.columns:
            c = apply_entity_filter(c, 'spid', sp_codes)
        if c.empty:
            return pd.DataFrame()
        c['date'] = pd.to_datetime(c['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        # When entity is Salesman and spname is available, pivot by salesman
        if entity == "Salesman" and 'spname' in c.columns:
            grp = c.groupby(['date', 'spname'])['value'].sum().reset_index()
            pivot = grp.pivot_table(
                index='date', columns='spname', values='value', aggfunc='sum'
            ).fillna(0)
            pivot.columns.name = None
            return pivot.sort_index()
        # For Product / Product Group or when spname unavailable: single 'All' column
        grp = c.groupby('date')['value'].sum().reset_index()
        grp = grp.rename(columns={'value': 'All'}).set_index('date').sort_index()
        return grp

    code_col, name_col = _entity_cols(entity)

    s = apply_date_window(sales_df, start, end)
    r = apply_date_window(returns_df, start, end)

    # Apply entity filters so the pivot only shows selected entities as columns
    s = _apply_entity_filters_to_df(s, sp_codes or [], item_codes or [], item_groups or [])
    r = _apply_entity_filters_to_df(r, sp_codes or [], item_codes or [], item_groups or [])

    if metric == "Net Returns":
        if r is None or r.empty or code_col not in r.columns or 'treturnamt' not in r.columns:
            return pd.DataFrame()
        r = r.copy()
        r['date'] = pd.to_datetime(r['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        grp_cols = ['date', code_col] + ([name_col] if name_col and name_col in r.columns else [])
        grp = r.groupby(grp_cols)['treturnamt'].sum().reset_index()
        label_col = name_col if (name_col and name_col in grp.columns) else code_col
        pivot = grp.pivot_table(index='date', columns=label_col, values='treturnamt', aggfunc='sum').fillna(0)
        pivot.columns.name = None
        return pivot.sort_index()

    # Net Sales
    if (s is None or s.empty) and (r is None or r.empty):
        return pd.DataFrame()
    if s is None or s.empty or code_col not in s.columns or 'final_sales' not in s.columns:
        return pd.DataFrame()

    s = s.copy()
    s['date'] = pd.to_datetime(s['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    s_grp_cols = ['date', code_col] + ([name_col] if name_col and name_col in s.columns else [])
    s_agg = s.groupby(s_grp_cols)['final_sales'].sum().reset_index()

    if r is not None and not r.empty and code_col in r.columns and 'treturnamt' in r.columns:
        r = r.copy()
        r['date'] = pd.to_datetime(r['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        r_grp_cols = ['date', code_col] + ([name_col] if name_col and name_col in r.columns else [])
        r_agg = r.groupby(r_grp_cols)['treturnamt'].sum().reset_index()
        merge_keys = [c for c in s_grp_cols if c in r_grp_cols]
        merged = pd.merge(s_agg, r_agg, on=merge_keys, how='outer')
        merged['final_sales'] = merged['final_sales'].fillna(0)
        merged['treturnamt'] = merged['treturnamt'].fillna(0)
    else:
        merged = s_agg.copy()
        merged['treturnamt'] = 0.0

    merged['value'] = merged['final_sales'] - merged['treturnamt']
    label_col = name_col if (name_col and name_col in merged.columns) else code_col
    if label_col not in merged.columns:
        return pd.DataFrame()

    pivot = merged.pivot_table(index='date', columns=label_col, values='value', aggfunc='sum').fillna(0)
    pivot.columns.name = None
    return pivot.sort_index()


def compute_moving_avg_table(
    sales_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    entity: str,
    metric: str,
    end_date,
    collection_df: pd.DataFrame = None,
    sp_codes: list = None,
    item_codes: list = None,
    item_groups: list = None,
) -> pd.DataFrame:
    """
    Returns DataFrame: index=entity names, columns=MA window labels, values=average daily metric.

    Lookback windows (all end at end_date):
        12M = 365 days, 9M = 274 days, 6M = 182 days, 3M = 91 days

    Sidebar filters (sp_codes / item_codes / item_groups) are applied inside each window so
    only the selected entities appear as rows.
    Net Collections is supported only when entity == 'Salesman' (requires spname in collection_df).
    """
    code_col, name_col = _entity_cols(entity)
    idx_col = name_col if name_col else code_col

    windows = {
        "MA 12M Daily Avg": 365,
        "MA 9M Daily Avg":  274,
        "MA 6M Daily Avg":  182,
        "MA 3M Daily Avg":  91,
    }

    end = pd.Timestamp(end_date)
    col_series = {}

    for label, days in windows.items():
        start = (end - pd.Timedelta(days=days)).date()
        end_d = end.date()

        if metric == "Net Collections":
            if collection_df is None or collection_df.empty:
                col_series[label] = pd.Series(dtype=float, name=label)
                continue
            c = apply_date_window(collection_df, start, end_d)
            if c is None or c.empty or 'value' not in c.columns:
                col_series[label] = pd.Series(dtype=float, name=label)
                continue
            # Apply salesman filter before grouping
            if sp_codes and 'spid' in c.columns:
                c = apply_entity_filter(c, 'spid', sp_codes)
            if c.empty:
                col_series[label] = pd.Series(dtype=float, name=label)
                continue
            # Group by spname if available, else spid
            grp_key = 'spname' if 'spname' in c.columns else ('spid' if 'spid' in c.columns else None)
            if grp_key is None:
                col_series[label] = pd.Series(dtype=float, name=label)
                continue
            col_series[label] = c.groupby(grp_key)['value'].sum() / days
            continue

        s = apply_date_window(sales_df, start, end_d)
        r = apply_date_window(returns_df, start, end_d)

        # Apply entity filters so only selected rows contribute to the averages
        s = _apply_entity_filters_to_df(s, sp_codes or [], item_codes or [], item_groups or [])
        r = _apply_entity_filters_to_df(r, sp_codes or [], item_codes or [], item_groups or [])

        if metric == "Net Returns":
            if r is None or r.empty or code_col not in r.columns or 'treturnamt' not in r.columns:
                col_series[label] = pd.Series(dtype=float, name=label)
                continue
            # Group directly by the display label; avoids duplicate-index crash from set_index
            grp_key = idx_col if idx_col in r.columns else code_col
            col_series[label] = r.groupby(grp_key)['treturnamt'].sum() / days

        else:  # Net Sales
            if s is None or s.empty or code_col not in s.columns or 'final_sales' not in s.columns:
                col_series[label] = pd.Series(dtype=float, name=label)
                continue
            grp_key = idx_col if idx_col in s.columns else code_col
            s_agg = s.groupby(grp_key)['final_sales'].sum()

            if r is not None and not r.empty and code_col in r.columns and 'treturnamt' in r.columns:
                r_grp_key = idx_col if idx_col in r.columns else code_col
                r_agg = r.groupby(r_grp_key)['treturnamt'].sum()
                net = s_agg.subtract(r_agg, fill_value=0)
            else:
                net = s_agg

            col_series[label] = net / days

    if not col_series:
        return pd.DataFrame()

    result = pd.DataFrame(col_series).fillna(0).round(2)
    result.index.name = entity
    return result


def _codes_by_col(code_col: str, selected_codes: list):
    """Map selected_codes to (sp_codes, item_codes, item_groups) based on code_col."""
    if not selected_codes:
        return [], [], []
    if code_col == 'spid':
        return selected_codes, [], []
    if code_col == 'itemcode':
        return [], selected_codes, []
    if code_col == 'itemgroup':
        return [], [], selected_codes
    return [], [], []


def compute_yoy_daily(
    sales_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    collection_df: pd.DataFrame,
    metric: str,
    end_date,
    code_col: str,
    selected_codes: list,
) -> pd.DataFrame:
    """
    Current period:  [end_date - 3 months, end_date]
    Prior period:    same 3-month window one year earlier.

    Returns long-format DataFrame: [date_label (MM-DD string), value (float), period (str)].
    period values: "Current Year" | "Prior Year".
    """
    end = end_date if isinstance(end_date, date) else end_date.date()
    start = default_start_date(end)

    prior_year = end.year - 1
    prior_month = end.month
    prior_day = min(end.day, calendar.monthrange(prior_year, prior_month)[1])
    prior_end = date(prior_year, prior_month, prior_day)
    prior_start = default_start_date(prior_end)

    sp, item, groups = _codes_by_col(code_col, [str(c) for c in selected_codes] if selected_codes else [])

    def _window(s, e, period_label):
        daily = compute_daily_metric(sales_df, returns_df, collection_df, metric, s, e, sp, item, groups)
        if daily.empty:
            return pd.DataFrame(columns=['date_label', 'value', 'period'])
        out = daily.copy()
        out['period'] = period_label
        out['date_label'] = pd.to_datetime(out['date_label']).dt.strftime('%m-%d')
        return out[['date_label', 'value', 'period']]

    current = _window(start, end, "Current Year")
    prior   = _window(prior_start, prior_end, "Prior Year")
    return pd.concat([current, prior], ignore_index=True)


def compute_mom_daily(
    sales_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    collection_df: pd.DataFrame,
    metric: str,
    selected_months: list,
    code_col: str,
    selected_codes: list,
) -> pd.DataFrame:
    """
    For each selected month (MM-YYYY string), show daily values.
    Returns long-format: [day (int 1-31), value (float), month_label (str)].
    """
    sp, item, groups = _codes_by_col(code_col, [str(c) for c in selected_codes] if selected_codes else [])
    rows = []

    for month_label in selected_months:
        try:
            mm, yyyy = month_label.split('-')
            month_int, year_int = int(mm), int(yyyy)
        except ValueError:
            continue

        last_day = calendar.monthrange(year_int, month_int)[1]
        start_w = date(year_int, month_int, 1)
        end_w   = date(year_int, month_int, last_day)

        daily = compute_daily_metric(sales_df, returns_df, collection_df, metric, start_w, end_w, sp, item, groups)
        if daily.empty:
            continue
        out = daily.copy()
        out['day']         = pd.to_datetime(out['date_label']).dt.day
        out['month_label'] = month_label
        rows.append(out[['day', 'value', 'month_label']])

    if not rows:
        return pd.DataFrame(columns=['day', 'value', 'month_label'])
    return pd.concat(rows, ignore_index=True)


def compute_metric_comparison(
    sales_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    entity: str,
    selected_codes: list,
    metrics: list,
    start,
    end,
) -> pd.DataFrame:
    """
    For each selected entity and each metric, compute the total over [start, end].
    Returns DataFrame: rows=entity names, columns=metric names.
    """
    code_col, name_col = _entity_cols(entity)
    idx_col = name_col if name_col else code_col

    s = apply_date_window(sales_df, start, end)
    r = apply_date_window(returns_df, start, end)

    if selected_codes:
        s = apply_entity_filter(s, code_col, selected_codes)
        if code_col in r.columns:
            r = apply_entity_filter(r, code_col, selected_codes)

    if s is None or s.empty or code_col not in s.columns:
        return pd.DataFrame()

    grp_cols = [code_col] + ([idx_col] if idx_col != code_col and idx_col in s.columns else [])
    result_dict = {}

    for metric in metrics:
        if metric == "Net Returns":
            if r is None or r.empty or code_col not in r.columns or 'treturnamt' not in r.columns:
                continue
            r_idx = idx_col if (idx_col in r.columns) else code_col
            agg = r.groupby(r_idx)['treturnamt'].sum()
            result_dict[metric] = agg
        else:
            if s.empty or 'final_sales' not in s.columns:
                continue
            s_agg = s.groupby(grp_cols)['final_sales'].sum().reset_index()
            if r is not None and not r.empty and code_col in r.columns and 'treturnamt' in r.columns:
                r_grp = [code_col] + ([idx_col] if idx_col != code_col and idx_col in r.columns else [])
                r_agg = r.groupby(r_grp)['treturnamt'].sum().reset_index()
                merge_keys = [c for c in grp_cols if c in r_grp]
                merged = pd.merge(s_agg, r_agg, on=merge_keys, how='outer')
                merged['final_sales'] = merged['final_sales'].fillna(0)
                merged['treturnamt']  = merged['treturnamt'].fillna(0)
            else:
                merged = s_agg.copy()
                merged['treturnamt'] = 0.0
            merged['net'] = merged['final_sales'] - merged['treturnamt']
            idx = idx_col if idx_col in merged.columns else code_col
            result_dict[metric] = merged.set_index(idx)['net']

    if not result_dict:
        return pd.DataFrame()

    result = pd.DataFrame(result_dict).fillna(0).round(2)
    result.index.name = entity
    return result
