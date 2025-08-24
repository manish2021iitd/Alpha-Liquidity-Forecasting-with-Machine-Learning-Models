import argparse
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

AE_LATENT = 8

class AutoEncoder(nn.Module):
    def __init__(self, n_features: int, latent: int = AE_LATENT):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 64),
            nn.ReLU(),
            nn.Linear(64, n_features),
        )
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

def fit_autoencoder(X: np.ndarray, epochs: int = 20, lr: float = 1e-3, seed: int = 42):
    torch.manual_seed(seed)
    model = AutoEncoder(X.shape[1], AE_LATENT)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X_t = torch.tensor(X, dtype=torch.float32)
    for ep in range(epochs):
        opt.zero_grad()
        x_hat, _ = model(X_t)
        loss = loss_fn(x_hat, X_t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        _, Z = model(X_t)
    return model, Z.numpy()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="data/processed/features.csv")
    p.add_argument("--output", type=str, default="data/processed/features_dr.csv")
    args = p.parse_args()

    df = pd.read_csv(args.input, parse_dates=['date'])
    feature_cols = [c for c in df.columns if c not in ['date','ticker','ret_fwd_1d'] and df[c].dtype != 'O']

    # ---- CLEAN DATA ----
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)   # or X.fillna(X.mean()) if you prefer mean imputation

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values.astype(float))

    # PCA
    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    for i in range(X_pca.shape[1]):
        df[f'pca_{i+1}'] = X_pca[:, i]

    # Autoencoder latent
    try:
        _, Z = fit_autoencoder(X_scaled, epochs=25, lr=1e-3, seed=42)
        for i in range(Z.shape[1]):
            df[f'ae_{i+1}'] = Z[:, i]
    except Exception as e:
        print("Autoencoder failed (likely torch issue). Proceeding with PCA only.", e)

    df.to_csv(args.output, index=False)
    print(f"Saved features with PCA/AE to {args.output}, rows={len(df)}")
