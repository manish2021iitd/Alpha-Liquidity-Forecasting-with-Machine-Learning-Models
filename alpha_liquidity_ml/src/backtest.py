import argparse, json, joblib
import numpy as np
import pandas as pd
from utils import time_splits, long_short_backtest, sharpe_ratio


def load_predict(model_file, model_type, X, df_eval):
    if model_type == "xgb":
        art = joblib.load(model_file)
        model = art["model"] if isinstance(art, dict) else art
        return model.predict(X)

    elif model_type == "lstm":
        import torch
        from models.lstm_model import PanelLSTM
        art = torch.load(model_file, map_location="cpu")
        m = PanelLSTM(n_features=X.shape[1])
        m.load_state_dict(art["state_dict"]); m.eval()
        preds = pd.Series(np.nan, index=df_eval.index)
        # Align predictions by ticker sequences (30 lookback assumed)
        for tkr, g in df_eval.sort_values(['ticker','date']).groupby('ticker'):
            g = g.reset_index()
            for i in range(30, len(g)):
                x = g.loc[i-30:i-1, art["features"]].values.astype('float32')
                with torch.no_grad():
                    pr = m(torch.tensor(x).unsqueeze(0)).item()
                preds.iloc[g.loc[i,'index']] = pr
        return preds.values

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="data/processed/features_dr.csv")
    p.add_argument("--model_file", type=str, required=True)
    p.add_argument("--model", choices=["xgb","lstm"], required=True)
    p.add_argument("--quantile", type=float, default=0.2)
    p.add_argument("--report", type=str, default="bt_report.json")
    args = p.parse_args()

    df = pd.read_csv(args.input, parse_dates=['date']).dropna(subset=['ret_fwd_1d'])
    _, _, test = time_splits(df, val_start="2022-01-01", test_start="2024-01-01")
    feature_cols = [c for c in df.columns if c not in ['date','ticker','ret_fwd_1d'] and df[c].dtype != 'O']

    df_eval = test.copy()
    yhat = load_predict(args.model_file, args.model, df_eval[feature_cols].values, df_eval)
    df_eval = df_eval.assign(pred=yhat).dropna(subset=['pred'])

    pnl = long_short_backtest(df_eval, 'pred', q=args.quantile, target_col='ret_fwd_1d')
    sr = sharpe_ratio(pnl['ret']) if len(pnl) else float("nan")

    out = {"model": args.model, "quantile": args.quantile, "days": int(len(pnl)), "sharpe": float(sr)}
    with open(args.report, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
