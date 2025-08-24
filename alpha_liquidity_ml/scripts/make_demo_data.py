import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def make_panel(tickers=20, days=600, seed=42):
    random.seed(seed); np.random.seed(seed)
    start = datetime(2022,1,1)
    dates = [start + timedelta(days=i) for i in range(days)]
    dates = [d for d in dates if d.weekday() < 5]  # weekdays only

    rows = []
    for k in range(tickers):
        tkr = f"TICK{k:03d}"
        price = 100 + np.cumsum(np.random.normal(0, 1, size=len(dates)))
        volume = np.random.lognormal(mean=12, sigma=0.4, size=len(dates)).astype(int)
        high = price * (1 + np.random.uniform(0.0, 0.01, size=len(dates)))
        low = price * (1 - np.random.uniform(0.0, 0.01, size=len(dates)))
        close = price + np.random.normal(0, 0.5, size=len(dates))
        # fundamentals (slow-moving)
        pe = 15 + np.random.normal(0, 0.3, size=len(dates)).cumsum()/100
        pb = 3 + np.random.normal(0, 0.2, size=len(dates)).cumsum()/100
        dy = 0.01 + np.random.normal(0, 0.0005, size=len(dates))
        # order book depth imbalance proxy
        depth_imb = np.random.normal(0, 0.1, size=len(dates))
        for d, o, h, l, c, v, pe_i, pb_i, dy_i, di in zip(dates, price, high, low, close, volume, pe, pb, dy, depth_imb):
            rows.append([d, tkr, o, h, l, c, v, pe_i, pb_i, dy_i, di])
    df = pd.DataFrame(rows, columns=['date','ticker','open','high','low','close','volume','pe_ratio','pb_ratio','div_yield','depth_imbalance'])
    return df

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", type=int, default=20)
    p.add_argument("--days", type=int, default=600)
    p.add_argument("--output", type=str, default="data/raw/panel_demo.csv")
    args = p.parse_args()

    df = make_panel(args.tickers, args.days, seed=42)
    df.to_csv(args.output, index=False)
    print(f"Demo panel saved to {args.output}, rows={len(df)}")
