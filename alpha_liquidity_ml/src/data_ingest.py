import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime

def download_ohlcv(tickers: list, start: str, end: str) -> pd.DataFrame:
    frames = []
    for t in tickers:
        df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            continue
        df = df.reset_index().rename(columns=str.lower)
        df['ticker'] = t
        frames.append(df[['date','ticker','open','high','low','close','volume']])
    out = pd.concat(frames, ignore_index=True).sort_values(['ticker','date'])
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", type=str, required=True, help="Comma-separated tickers, e.g. RELIANCE.NS,TCS.NS")
    p.add_argument("--start", type=str, default="2018-01-01")
    p.add_argument("--end", type=str, default="2025-08-01")
    p.add_argument("--output", type=str, default="data/raw/ohlcv.csv")
    args = p.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    ohlcv = download_ohlcv(tickers, args.start, args.end)

    # Placeholder joins for fundamentals/order-book:
    # Expect user-provided CSVs with columns: date,ticker,pe_ratio,pb_ratio,div_yield,depth_imbalance
    # If present in data/raw, we'll outer-join.
    import os
    fundamentals_fp = "data/raw/fundamentals.csv"
    orderbook_fp = "data/raw/orderbook.csv"

    df = ohlcv.copy()
    if os.path.exists(fundamentals_fp):
        f = pd.read_csv(fundamentals_fp, parse_dates=['date'])
        df = df.merge(f, on=['date','ticker'], how='left')
    else:
        df['pe_ratio'] = df.groupby('ticker')['close'].transform(lambda s: (s / s.rolling(60).mean()))
        df['pb_ratio'] = df['pe_ratio'] * 0.5
        df['div_yield'] = 0.01

    if os.path.exists(orderbook_fp):
        ob = pd.read_csv(orderbook_fp, parse_dates=['date'])
        df = df.merge(ob, on=['date','ticker'], how='left')
    else:
        df['depth_imbalance'] = 0.0

    df.to_csv(args.output, index=False)
    print(f"Saved OHLCV+ to {args.output}, rows={len(df)}")
