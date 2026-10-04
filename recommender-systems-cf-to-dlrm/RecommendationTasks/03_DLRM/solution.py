"""Task 3: DLRM-style CTR predictor."""
from __future__ import annotations
import argparse
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, f1_score

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class DLRM(nn.Module):
    def __init__(self, vocab_sizes, n_numeric, emb_dim=16, interactions=True):
        super().__init__()
        self.interactions = interactions
        self.emb = nn.ModuleList([nn.Embedding(s, emb_dim) for s in vocab_sizes])
        self.bottom = nn.Sequential(nn.Linear(n_numeric, 64), nn.ReLU(), nn.Linear(64, emb_dim))
        feature_count = 1 + len(vocab_sizes)
        pair_count = feature_count * (feature_count - 1) // 2 if interactions else 0
        self.top = nn.Sequential(
            nn.Linear(emb_dim + pair_count, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x_num, x_cat):
        dense = self.bottom(x_num)
        features = [dense] + [layer(x_cat[:, i]) for i, layer in enumerate(self.emb)]
        if self.interactions:
            inter = [
                (features[i] * features[j]).sum(1, keepdim=True)
                for i in range(len(features))
                for j in range(i + 1, len(features))
            ]
            z = torch.cat([dense, *inter], dim=1)
        else:
            z = dense
        return self.top(z).squeeze(1)

def prepare(train, valid, numeric, categorical):
    tr = train[numeric].fillna(0).astype("float32")
    va = valid[numeric].fillna(0).astype("float32")
    mean, std = tr.mean(), tr.std().replace(0, 1)
    Xn = ((tr - mean) / std).to_numpy("float32")
    Vn = ((va - mean) / std).to_numpy("float32")
    Xc, Vc, sizes = [], [], []
    for c in categorical:
        values = train[c].fillna("<NA>").astype(str)
        vocab = {v: i + 1 for i, v in enumerate(values.unique())}
        sizes.append(len(vocab) + 1)
        Xc.append(values.map(vocab).fillna(0).to_numpy("int64"))
        Vc.append(valid[c].fillna("<NA>").astype(str).map(vocab).fillna(0).to_numpy("int64"))
    return torch.tensor(Xn), torch.tensor(np.stack(Xc, 1)), torch.tensor(Vn), torch.tensor(np.stack(Vc, 1)), sizes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--emb-dim", type=int, default=16)
    ap.add_argument("--no-interactions", action="store_true")
    args = ap.parse_args()
    seed_all()

    df = pd.read_csv(args.train)
    numeric = [c for c in df if c != "label" and pd.api.types.is_numeric_dtype(df[c])][:13]
    categorical = [c for c in df if c not in numeric + ["label"]][:26]
    train, valid = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    Xn, Xc, Vn, Vc, sizes = prepare(train, valid, numeric, categorical)

    y = torch.tensor(train["label"].to_numpy("float32"))
    model = DLRM(sizes, len(numeric), args.emb_dim, not args.no_interactions)
    print("parameters:", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    started = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(Xn, Xc), y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Vn, Vc)).numpy()
        print({
            "epoch": epoch + 1,
            "loss": float(loss),
            "ROC-AUC": roc_auc_score(valid["label"], p),
            "PR-AUC": average_precision_score(valid["label"], p),
            "LogLoss": log_loss(valid["label"], p),
            "F1": f1_score(valid["label"], p >= 0.5),
        })

    print("elapsed_sec:", round(time.perf_counter() - started, 3))

if __name__ == "__main__":
    main()
