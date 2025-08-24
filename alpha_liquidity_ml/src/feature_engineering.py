import argparse
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from ta.trend import MACD

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker','date']).copy()

    # Basic spreads & proxies
    df['hl_spread'] = (df['high'] - df['low']) / df['close'].replace(0,np.nan)
    df['vwap_proxy'] = (df['high'] + df['low'] + df['close']) / 3.0

    # Momentum
    for w in [5,10,20]:
        df[f'mom_{w}'] = df.groupby('ticker')['close'].pct_change(w)

    # Volatility
    for w in [10,20]:
        df[f'vol_{w}'] = df.groupby('ticker')['close'].pct_change().rolling(w).std()

    # RSI & ATR
    rsi = df.groupby('ticker', group_keys=False).apply(lambda g: RSIIndicator(g['close'], window=14).rsi())
    atr = df.groupby('ticker', group_keys=False).apply(lambda g: AverageTrueRange(g['high'], g['low'], g['close'], window=14).average_true_range())
    df['rsi_14'] = rsi.values
    df['atr_14'] = atr.values

    # OBV
    obv = df.groupby('ticker', group_keys=False).apply(lambda g: OnBalanceVolumeIndicator(close=g['close'], volume=g['volume']).on_balance_volume())
    df['obv'] = obv.values

    # Liquidity proxies
    df['turnover'] = df['close'] * df['volume']
    df['illiq_amihud'] = (df['hl_spread'].abs()) / (df['turnover'].replace(0,np.nan))
    df['illiq_amihud'] = df['illiq_amihud'].replace([np.inf,-np.inf], np.nan)

    # Forward returns (target)
    df['ret_fwd_1d'] = df.groupby('ticker')['close'].pct_change().shift(-1)

    # Clean up
    return df

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="data/raw/panel_demo.csv")
    p.add_argument("--output", type=str, default="data/processed/features.csv")
    args = p.parse_args()

    df = pd.read_csv(args.input, parse_dates=['date'])
    df = engineer_features(df)
    df.to_csv(args.output, index=False)
    print(f"Saved engineered features to {args.output}, rows={len(df)}")
