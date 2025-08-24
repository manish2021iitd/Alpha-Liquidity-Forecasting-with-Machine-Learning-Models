import argparse
import pandas as pd
import joblib
import torch
from models.xgb_model import train_xgb
from models.lstm_model import train_lstm

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, choices=["xgb", "lstm"], required=True)
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--optuna", type=int, default=0)
    args = p.parse_args()

    # Load dataset
    df = pd.read_csv(args.input, parse_dates=["date"])

    if args.model == "xgb":
        # Train XGB model
        model, X_train = train_xgb(df, args.optuna)

        # Save both model and features
        joblib.dump(
            {"model": model, "features": X_train.columns.tolist()},
            args.output
        )
        print(
            f"[INFO] Saved XGB model to {args.output} "
            f"with {len(X_train.columns)} features"
        )

    elif args.model == "lstm":
        # Train LSTM model
        model = train_lstm(df, args.optuna)

        # Save only weights for now
        torch.save(model.state_dict(), args.output)
        print(f"[INFO] Saved LSTM model to {args.output}")


    