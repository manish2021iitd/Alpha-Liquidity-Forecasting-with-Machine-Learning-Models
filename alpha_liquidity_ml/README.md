# Alpha Liquidity Forecasting with Machine Learning Models


End-to-end project to predict **daily liquidity-adjusted returns** of liquid assets (e.g., NIFTY 50/S&P 500 constituents)
from **fundamentals + market microstructure + technical signals**. It includes feature engineering, dimensionality
reduction (PCA + Autoencoder), models (XGBoost, LSTM, optional **Temporal Fusion Transformer (TFT)**), Bayesian
hyperparameter optimization (Optuna), evaluation (Out-of-sample R², Information Coefficient, Sharpe), and
a simple long-short **alpha backtest**.

## Highlights
- **Data**: OHLCV (via `yfinance` or CSVs), synthetic order-book depth proxy & fundamentals (for demo).
- **Feature Engineering**: momentum/volatility/liquidity ratios via `ta`, lagged features & target.
- **Dimensionality Reduction**: PCA + PyTorch Autoencoder over hundreds of signals.
- **Models**: XGBoost, LSTM, and TFT (via `darts`, optional).
- **Bayesian Optimization**: Optuna for XGB & LSTM hyperparams.
- **Evaluation**: OOS R², daily **Spearman IC**, and **Sharpe** for a simple long-short portfolio.
- **Baseline**: ARIMA (via `pmdarima`) for comparison.
- **Backtest**: Equal-weight long-short (top/bottom quantile by predicted return).

---

## Quickstart

### 0) Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> If `torch` wheels fail on your OS, install the correct ones from PyTorch website for your Python/CUDA.
> The project will **gracefully skip TFT** if `darts` (and torch) are unavailable.

### 1) Get Data
**Option A: Demo (no internet needed)**
```bash
python scripts/make_demo_data.py --tickers 20 --days 600
```
This writes `data/raw/panel_demo.csv` which powers the entire pipeline.

**Option B: Real (needs internet)**
```bash
python src/data_ingest.py --tickers "RELIANCE.NS,TCS.NS,INFY.NS" --start 2018-01-01 --end 2025-08-01
```
This saves `data/raw/ohlcv.csv`. You can join your own **fundamentals** & **order-book** CSVs by matching `date,ticker`.

### 2) Feature Engineering
```bash
python src/feature_engineering.py --input data/raw/panel_demo.csv --output data/processed/features.csv
```
(Replace input with your joined real dataset if using Option B.)

### 3) Dimensionality Reduction (PCA + Autoencoder)
```bash
python src/signals_dimred.py --input data/processed/features.csv --output data/processed/features_dr.csv
```
This adds `pca_*` and `ae_*` columns.

### 4) Train Models + Hyperparameter Search
```bash
# Train and tune XGBoost
python src/train.py --model xgb --input data/processed/features_dr.csv --output models_xgb.pkl --optuna 50

# Train and tune LSTM (sequence model)
python src/train.py --model lstm --input data/processed/features_dr.csv --output models_lstm.pth --optuna 20

# Train TFT (optional, requires darts/torch)
python src/train.py --model tft --input data/processed/features_dr.csv --output models_tft.pth
```

### 5) Evaluate + Backtest
```bash
# Evaluate predictive metrics + baseline ARIMA
python src/evaluate.py --input data/processed/features_dr.csv --model_file models_xgb.pkl --model xgb --report reports_xgb.json

# Backtest long-short alpha
python src/backtest.py --input data/processed/features_dr.csv --model_file models_xgb.pkl --model xgb --quantile 0.2 --report bt_xgb.json
```

Open the JSON reports to see **OOS R²**, **IC**, **Sharpe**, and baseline comparisons. Replace with LSTM/TFT as desired.

---

## Repo Structure
```
alpha_liquidity_ml/
├─ data/
│  ├─ raw/                  # input panel or ohlcv/fundamentals/orderbook
│  └─ processed/            # engineered features & DR outputs
├─ notebooks/               # EDA & training notebook templates
├─ scripts/
│  └─ make_demo_data.py     # synthetic, reproducible demo panel
├─ src/
│  ├─ models/               # xgb/lstm/tft definitions
│  ├─ data_ingest.py        # yfinance download & joins
│  ├─ feature_engineering.py# technicals & liquidity features
│  ├─ signals_dimred.py     # PCA + Autoencoder
│  ├─ train.py              # trainers + Optuna BO
│  ├─ evaluate.py           # metrics incl. baseline ARIMA
│  ├─ backtest.py           # long-short backtest by quantiles
│  └─ utils.py              # helpers: splits, metrics, seeds
├─ config.yaml              # general settings
├─ requirements.txt
└─ README.md
```

---

## Notes
- **Targets**: uses `ret_fwd_1d` (1-day forward return). You can change the horizon in config and regenerate.
- **Cross-section**: IC is computed **by day** across all tickers.
- **Leakage**: time-based splits with forward gaps; model pipelines include lookback/lagging to avoid peeking.
- **Baseline**: ARIMA is trained per ticker on returns and aggregated for comparison.
- **Reproducibility**: set seeds in `config.yaml`.
