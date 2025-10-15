# procedures.py
import pandas as pd
import numpy as np
from collections import defaultdict, deque

# --- Global state for Welford + robust histories (reset each run) -------------
cat_counts = defaultdict(int)
cat_mean   = defaultdict(float)
cat_M2     = defaultdict(float)
cat_values = defaultdict(list)

def reset_state():
    """Reset global running-stat containers to avoid cross-run contamination."""
    global cat_counts, cat_mean, cat_M2, cat_values
    cat_counts = defaultdict(int)
    cat_mean   = defaultdict(float)
    cat_M2     = defaultdict(float)
    cat_values = defaultdict(list)

# --- Basic IO ------------------------------------------------------------------
def read_df(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    df['inflow']  = df['amount'].where(df['amount'] > 0, 0.0)
    df['outflow'] = (-df['amount']).where(df['amount'] < 0, 0.0)
    df['year_month'] = df['date'].dt.to_period('M')
    # handy features (not strictly necessary but harmless)
    df['weekday'] = df['date'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6])
    df['week_of_month'] = df['date'].dt.day.map(lambda d: (d-1)//7 + 1)
    return df

# --- Core math -----------------------------------------------------------------
def current_std(c):
    n = cat_counts[c]
    if n < 2:
        return np.nan
    return np.sqrt(cat_M2[c] / (n - 1))

def modified_z(x, median, mad):
    if mad is None or mad == 0 or np.isnan(mad):
        return np.nan
    return 0.6745 * (x - median) / mad

# --- Build df_tx with outlier flags + refunds + effective_outflow --------------
def df_tx(df):
    reset_state()

    df_sorted = df.sort_values(['date']).reset_index(drop=True)
    records = []

    for _, row in df_sorted.iterrows():
        c = row.get('category', None)
        outflow = float(row['outflow'])
        n_prev = cat_counts[c]

        # classical running stats (before current row)
        mean_prev = cat_mean[c] if n_prev > 0 else np.nan
        std_prev  = current_std(c)
        cv_prev   = (std_prev/mean_prev) if (n_prev >= 2 and mean_prev and mean_prev > 0) else np.nan

        # robust/IQR (need >=5)
        if n_prev >= 5:
            hist = np.array(cat_values[c], dtype=float)
            med_prev = np.median(hist)
            mad_prev = np.median(np.abs(hist - med_prev))
            q1_prev, q3_prev = np.quantile(hist, [0.25, 0.75])
            iqr_prev = q3_prev - q1_prev
            upper_iqr_prev = q3_prev + 1.5 * iqr_prev
        else:
            med_prev = mad_prev = q1_prev = q3_prev = iqr_prev = upper_iqr_prev = np.nan

        # z / modz
        z  = (outflow - mean_prev)/std_prev if (n_prev >= 2 and std_prev and std_prev > 0 and not np.isnan(mean_prev)) else np.nan
        mz = modified_z(outflow, med_prev, mad_prev) if n_prev >= 5 else np.nan

        # adaptive z threshold
        if np.isnan(cv_prev):
            thr_z = 2.5
        elif cv_prev > 1.5:
            thr_z = 4.0
        elif cv_prev > 1.0:
            thr_z = 3.5
        else:
            thr_z = 2.5

        # decision
        if outflow <= 0:
            outlier = False; outlier_status = 'NotApplicable'
        elif n_prev < 5:
            outlier = False; outlier_status = 'Insufficient'
        else:
            cond_z   = (not np.isnan(z))  and (abs(z)  > thr_z)
            cond_mz  = (not np.isnan(mz)) and (abs(mz) > 3.5)
            cond_iqr = (not np.isnan(upper_iqr_prev)) and (outflow > upper_iqr_prev)
            outlier = bool(cond_z or cond_mz or cond_iqr)
            outlier_status = 'Valid'

        records.append({
            'date': row['date'],
            'category': row.get('category', None),
            'amount': row['amount'],
            'inflow': row['inflow'],
            'outflow': row['outflow'],
            'year_month': row['year_month'],
            'z_score': z,
            'modz_score': mz,
            'thr_z_adaptive': thr_z,
            'upper_iqr_prev': upper_iqr_prev,
            'outlier_gross': outlier,
            'outlier_status': outlier_status,
        })

        # update running stats with positive outflow
        if outflow > 0:
            n = cat_counts[c] + 1
            delta = outflow - cat_mean[c]
            mean_new = cat_mean[c] + delta / n
            delta2 = outflow - mean_new
            cat_counts[c] = n
            cat_mean[c] = mean_new
            cat_M2[c] += delta * delta2
            cat_values[c].append(outflow)

    dftx = pd.DataFrame.from_records(records)

    # refund pairing (same logic, minimal)
    dftx['amount_abs'] = dftx[['inflow','outflow']].max(axis=1)
    queues = defaultdict(deque)
    refund_pair_id = np.full(len(dftx), np.nan)
    refund_alloc_to_outflow = np.zeros(len(dftx), dtype=float)

    pair_counter = 0
    window_days = 7  # your latest code

    for i, row in dftx.iterrows():
        key = (row['category'], row['amount_abs']); d = row['date']
        if row['outflow'] > 0:
            queues[key].append((d, i))
        elif row['inflow'] > 0:
            dq = queues.get(key, deque())
            while dq and (d - dq[0][0]).days > window_days:
                dq.popleft()
            if dq:
                prior_date, prior_idx = dq.popleft()
                pair_counter += 1
                refund_pair_id[i] = pair_counter
                refund_pair_id[prior_idx] = pair_counter
                refund_alloc_to_outflow[prior_idx] += row['inflow']

    dftx['refund_pair_id'] = refund_pair_id
    dftx['refund_alloc_to_outflow'] = refund_alloc_to_outflow
    dftx['effective_outflow'] = np.maximum(dftx['outflow'] - dftx['refund_alloc_to_outflow'], 0.0)

    # reversed_quickly (≥90% refund within 14 days)
    dftx['reversed_quickly'] = False
    for pid in dftx['refund_pair_id'].dropna().unique():
        pid = int(pid)
        rows = dftx.index[dftx['refund_pair_id'] == pid].tolist()
        if len(rows) == 2:
            a, b = rows
            idx_out, idx_ref = (a, b) if dftx.loc[a, 'outflow'] > 0 else (b, a)
            out_amt = dftx.loc[idx_out, 'outflow']; ref_amt = dftx.loc[idx_ref, 'inflow']
            days = (dftx.loc[idx_ref, 'date'] - dftx.loc[idx_out, 'date']).days
            if out_amt > 0 and ref_amt / out_amt >= 0.9 and 0 <= days <= 14:
                dftx.loc[[idx_out, idx_ref], 'reversed_quickly'] = True

    return dftx

# --- Latest-month anomalies (exactly as notebook filter) ----------------------
def latest_month_anomalies(dftx, df_raw=None):
    """
    Return (anoms, last_month_str) where:
      - anoms is the latest-month anomalous transactions (exact notebook filters)
      - last_month_str is 'YYYY-MM'
    If df_raw is provided and 'description' is missing, we merge it using robust keys
    with a temporary month string to avoid Period[str]/object mismatches.
    """
    # latest month
    last_month = dftx['year_month'].max()        # Period[M]
    last_month_str = str(last_month)             # 'YYYY-MM'
    tx_last = dftx[dftx['year_month'] == last_month].copy()

    # EXACT notebook filters
    mask = (
        (tx_last['outlier_status'] == 'Valid') &
        (tx_last['outlier_gross']) &
        (~tx_last['reversed_quickly']) &
        (tx_last['outflow'] > 0)
    )
    anoms = tx_last[mask].copy()

    # If nothing to do, return early
    if anoms.empty:
        # Ensure columns exist for the JSON builder
        for col in ['date','description','amount','z_score','modz_score','thr_z_adaptive','upper_iqr_prev','outflow']:
            if col not in anoms.columns:
                anoms[col] = np.nan
        return anoms, last_month_str

    # If description is missing, try to merge from df_raw
    if df_raw is not None and ('description' not in anoms.columns or anoms['description'].isna().all()):
        raw = df_raw.copy()

        # Parse/align types
        if 'date' in raw.columns:
            raw['date'] = pd.to_datetime(raw['date'], errors='coerce')
        if 'year_month' not in raw.columns and 'date' in raw.columns:
            raw['year_month'] = raw['date'].dt.to_period('M')

        # Build a common month string key on BOTH frames to avoid Period/object mismatch
        anoms['__ym_str__'] = anoms['year_month'].astype(str)
        raw['__ym_str__'] = raw['year_month'].astype(str)

        # Restrict raw to the same month to reduce accidental m:m matches
        raw_last = raw[raw['__ym_str__'] == last_month_str].copy()

        # Ensure there is a description column in raw
        if 'description' not in raw_last.columns:
            for alt in ['merchant', 'memo', 'details']:
                if alt in raw_last.columns:
                    raw_last = raw_last.rename(columns={alt: 'description'})
                    break
        if 'description' not in raw_last.columns:
            raw_last['description'] = np.nan  # still proceed gracefully

        # Coerce numeric columns (helps merging)
        for col in ['amount', 'inflow', 'outflow']:
            if col in raw_last.columns:
                raw_last[col] = pd.to_numeric(raw_last[col], errors='coerce')

        # Choose robust keys; avoid 'year_month' to sidestep dtype issues, use __ym_str__ instead
        base_keys = ['date', 'category', 'amount']  # strong identifiers
        keys = [k for k in base_keys if k in anoms.columns and k in raw_last.columns]
        keys.append('__ym_str__')  # month context

        # Build the merge
        anoms = (
            anoms.reset_index()
                 .merge(raw_last[keys + ['description']], on=keys, how='left', validate='m:m')
                 .sort_values('index')
                 .drop_duplicates('index', keep='first')
                 .set_index('index')
        )

    # Ensure required columns exist for JSON payload
    for col in ['date','description','amount','z_score','modz_score','thr_z_adaptive','upper_iqr_prev','outflow']:
        if col not in anoms.columns:
            anoms[col] = np.nan

    return anoms, last_month_str


# --- Metric selection used in JSON payload ------------------------------------
def choose_metric_value(row):
    z   = row.get('z_score', np.nan)
    mz  = row.get('modz_score', np.nan)
    thr = row.get('thr_z_adaptive', np.nan)
    upper = row.get('upper_iqr_prev', np.nan)
    outflow = row.get('outflow', np.nan)

    cond_z   = pd.notna(z) and pd.notna(thr) and abs(z) > float(thr)
    cond_mz  = pd.notna(mz) and abs(mz) > 3.5
    cond_iqr = pd.notna(upper) and pd.notna(outflow) and (outflow > upper)

    if cond_z:
        return float(abs(z))
    if cond_mz:
        return float(abs(mz))
    if cond_iqr and upper > 0:
        return float(outflow / upper)  # ratio beyond upper fence
    return np.nan

def build_payload(anoms):
    values = anoms.apply(choose_metric_value, axis=1).round(4).tolist()
    tx_strings = []
    for _, r in anoms.iterrows():
        date_str = r['date'].date().isoformat() if pd.notna(r['date']) else ''
        desc = r['description'] if pd.notna(r['description']) and str(r['description']).strip() else r.get('category','Transaction')
        amt = float(r['amount']) if pd.notna(r['amount']) else np.nan
        tx_strings.append(f"{date_str} — {desc} — {amt:.2f}")
    return {
        "metric_name": "adaptive_outlier_rule",
        "value": values,
        "transaction": tx_strings
    }
