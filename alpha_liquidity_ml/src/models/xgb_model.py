import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_xgb(df, optuna_trials=20):
    # Features and target
    X = df.drop(columns=["ret_fwd_1d", "date", "ticker"])
    y = df["ret_fwd_1d"]

    # Clean data
    mask = y.notnull() & np.isfinite(y)
    X, y = X[mask], y[mask]
    X = X.fillna(0)

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "eta": trial.suggest_float("eta", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "eval_metric": "rmse",
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        return r2_score(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=optuna_trials)

    best_params = study.best_params
    model = xgb.XGBRegressor(**best_params)
    model.fit(X, y)
    return model, X_train