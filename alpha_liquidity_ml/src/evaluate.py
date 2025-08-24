import argparse
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import pmdarima as pm

warnings.filterwarnings("ignore")

# ------------------------
# XGB Prediction
# ------------------------
def predict_xgb(model_art, X_eval):
    if isinstance(model_art, dict):
        model = model_art["model"]
        # Align features in evaluation with training features
        X_eval = X_eval[model_art["features"]]
    else:
        model = model_art
    return model.predict(X_eval)



# ------------------------
# LSTM Prediction
# ------------------------
def predict_lstm(model_file, df, feature_cols, lookback=30):
    import torch
    from models.lstm_model import PanelLSTM

    # Load saved model artifact
    art = torch.load(model_file, map_location="cpu")
    state = art["state_dict"]
    feats = art["features"]

    assert feats == feature_cols, "Feature mismatch between training and inference."

    m = PanelLSTM(n_features=len(feature_cols))
    m.load_state_dict(state)
    m.eval()

    # Build sequences aligned to df rows
    df = df.sort_values(['ticker', 'date']).copy()
    preds = pd.Series(np.nan, index=df.index)

    for tkr, g in df.groupby('ticker'):
        g = g.reset_index()
        for i in range(lookback, len(g)):
            x = g.loc[i-lookback:i-1, feature_cols].values.astype('float32')
            with torch.no_grad():
                pr = m(torch.tensor(x).unsqueeze(0)).item()
            preds.iloc[g.loc[i, 'index']] = pr

    return preds.values


# ------------------------
# Baseline ARIMA (Optional Sanity Check)
# ------------------------
def baseline_arima_r2(df_test):
    r2s = []
    for tkr, g in df_test.groupby('ticker'):
        g = g.sort_values('date')
        y = g['ret_fwd_1d'].values
        if len(y) < 50 or np.isnan(y).any():
            continue
        try:
            model = pm.auto_arima(y[:-1], seasonal=False,
                                  suppress_warnings=True, error_action='ignore')
            yhat = model.predict(n_periods=1)[0]
            r2s.append(r2_score([y[-1]], [yhat]))
        except Exception:
            continue
    return float(np.nanmean(r2s)) if r2s else np.nan


# ------------------------
# Main
# ------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["xgb", "lstm"])
    parser.add_argument("--report", type=str, required=True)
    args = parser.parse_args()

    # Load data
    df_eval = pd.read_csv(args.input)
    X = df_eval.drop(columns=["target"], errors="ignore")
    y = df_eval["target"] if "target" in df_eval.columns else None

    # Predict depending on model type
    if args.model == "xgb":
        model = joblib.load(args.model_file)
        yhat = predict_xgb(model, X)

    elif args.model == "lstm":
        feature_cols = [c for c in df_eval.columns if c not in ["target", "ticker", "date"]]
        yhat = predict_lstm(args.model_file, df_eval, feature_cols)

    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    print(f"[INFO] Predictions shape: {yhat.shape}, Data shape: {X.shape}")

    # Ensure predictions align with dataframe length
    if len(yhat) != len(df_eval):
        raise ValueError(f"Mismatch: predictions {len(yhat)} vs data {len(df_eval)}")

    df_eval = df_eval.copy()
    df_eval["pred"] = yhat

    # Save metrics report
    metrics = {}
    if y is not None and not np.all(np.isnan(yhat)):
        metrics["mse"] = mean_squared_error(y, yhat)
        metrics["r2"] = r2_score(y, yhat)

    with open(args.report, "w") as f:
        json.dump(metrics, f, indent=4)

    print("[INFO] Evaluation report saved:", args.report)
