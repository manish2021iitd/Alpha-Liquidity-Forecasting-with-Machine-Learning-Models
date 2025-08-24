import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from scipy.stats import spearmanr

def set_seed(seed: int = 42):
    import random, os
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
    except Exception:
        pass

def time_splits(df: pd.DataFrame, val_start: str, test_start: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df['date'] < val_start].copy()
    val = df[(df['date'] >= val_start) & (df['date'] < test_start)].copy()
    test = df[df['date'] >= test_start].copy()
    return train, val, test

def make_lagged(df: pd.DataFrame, cols: List[str], lags: int = 5) -> pd.DataFrame:
    df = df.sort_values(['ticker','date']).copy()
    for c in cols:
        for l in range(1, lags+1):
            df[f'{c}_lag{l}'] = df.groupby('ticker')[c].shift(l)
    return df

def dropna_by_lookback(df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    # Remove rows with insufficient lag history
    valid = df.groupby('ticker').cumcount() >= lookback_days
    return df[valid].dropna()

def daily_ic(df: pd.DataFrame, pred_col: str, target_col: str = 'ret_fwd_1d') -> float:
    ics = []
    for d, g in df.groupby('date'):
        if g[pred_col].nunique() > 1 and g[target_col].nunique() > 1:
            ic = spearmanr(g[pred_col], g[target_col], nan_policy='omit').correlation
            if np.isfinite(ic):
                ics.append(ic)
    return float(np.nanmean(ics)) if len(ics) else np.nan

def sharpe_ratio(returns: pd.Series, ann_factor: int = 252) -> float:
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return np.nan
    daily_sr = mu / sigma
    return float(np.sqrt(ann_factor) * daily_sr)

def oos_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # 1 - SSE/SST with respect to out-of-sample mean of y_true
    sse = np.sum((y_true - y_pred)**2)
    sst = np.sum((y_true - np.mean(y_true))**2)
    return float(1 - sse / sst) if sst != 0 else np.nan

def long_short_backtest(df: pd.DataFrame, pred_col: str, q: float = 0.2, target_col: str = 'ret_fwd_1d') -> pd.DataFrame:
    # Form daily equal-weight long-short portfolio by quantiles of prediction
    pnl = []
    for d, g in df.groupby('date'):
        if len(g) < 5:
            continue
        lo = g[pred_col].quantile(q)
        hi = g[pred_col].quantile(1-q)
        longs = g[g[pred_col] >= hi]
        shorts = g[g[pred_col] <= lo]
        if len(longs)==0 or len(shorts)==0:
            continue
        ret_long = longs[target_col].mean()
        ret_short = shorts[target_col].mean()
        pnl.append({'date': d, 'ret': ret_long - ret_short})
    pnl = pd.DataFrame(pnl).sort_values('date')
    pnl['cumret'] = (1 + pnl['ret']).cumprod()
    return pnl
