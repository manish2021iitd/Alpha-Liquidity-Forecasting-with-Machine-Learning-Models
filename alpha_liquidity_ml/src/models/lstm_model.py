import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import optuna

# ------------------------
# Define LSTM Model
# ------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch, seq_len, features]
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # take last timestep
        return self.fc(out)

# ------------------------
# Training Loop
# ------------------------
def train_loop(model, train_loader, val_loader, epochs=20, lr=1e-3, device="cpu"):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds.squeeze(), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds.squeeze(), yb)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
              f"Val Loss={val_loss/len(val_loader):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    return model

# ------------------------
# Main Train Function
# ------------------------
def train_lstm(df, n_trials=10):
    # Drop non-numeric columns (like date, ticker)
    df = df.drop(columns=["date", "ticker"], errors="ignore")

    # Ensure all numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Features and target
    X = df.drop(columns=["ret_fwd_1d"])
    y = df["ret_fwd_1d"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1)  # [B,1,F]
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    input_size = X_train.shape[1]

    def objective(trial):
        hidden_size = trial.suggest_int("hidden_size", 32, 128)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        model = LSTMModel(input_size, hidden_size, num_layers, dropout)
        model = train_loop(model, train_loader, val_loader, epochs=10, lr=lr)

        # Evaluate
        criterion = nn.MSELoss()
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                val_loss += criterion(preds.squeeze(), yb).item()

        return val_loss / len(val_loader)

    # Run Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("Best Params:", study.best_params)

    # Train final model
    best_params = study.best_params
    final_model = LSTMModel(input_size, 
                            best_params["hidden_size"], 
                            best_params["num_layers"], 
                            best_params["dropout"])
    final_model = train_loop(final_model, train_loader, val_loader, epochs=30, lr=best_params["lr"])

    return final_model
