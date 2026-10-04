"""Task 2: embedding-based neural CTR predictor.
Input CSV must contain label plus at least 13 numerical and 26 categorical fields.
"""
from __future__ import annotations
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, accuracy_score, f1_score

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_columns(df):
    numeric = [c for c in df if c != "label" and pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in df if c not in set(numeric + ["label"])]
    if len(numeric) < 13 or len(categorical) < 26:
        raise ValueError(f"Expected at least 13 numeric + 26 categorical fields, got {len(numeric)} + {len(categorical)}")
    return numeric[:13], categorical[:26]

def prepare(train, valid, numeric, categorical):
    trn = train[numeric].fillna(0).astype("float32")
    van = valid[numeric].fillna(0).astype("float32")
    mean = trn.mean()
    std = trn.std().replace(0, 1)
    Xn = ((trn - mean) / std).to_numpy("float32")
    Vn = ((van - mean) / std).to_numpy("float32")

    Xc, Vc, sizes = [], [], []
    for c in categorical:
        vals = train[c].fillna("<NA>").astype(str)
        vocab = {v: i + 1 for i, v in enumerate(vals.unique())}
        sizes.append(len(vocab) + 1)
        Xc.append(vals.map(vocab).fillna(0).to_numpy("int64"))
        Vc.append(valid[c].fillna("<NA>").astype(str).map(vocab).fillna(0).to_numpy("int64"))
    return Xn, np.stack(Xc, 1), Vn, np.stack(Vc, 1), sizes

class CTRModel(nn.Module):
    def __init__(self, sizes, n_numeric, emb_dim=16, hidden=(256, 64)):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(s, emb_dim) for s in sizes])
        in_dim = n_numeric + len(sizes) * emb_dim
        layers, cur = [], in_dim
        for h in hidden:
            layers += [nn.Linear(cur, h), nn.ReLU(), nn.Dropout(0.2)]
            cur = h
        layers.append(nn.Linear(cur, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        embeds = [layer(x_cat[:, i]) for i, layer in enumerate(self.emb)]
        return self.mlp(torch.cat([x_num, *embeds], dim=1)).squeeze(1)

def report_metrics(y, p):
    return {
        "ROC-AUC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
        "LogLoss": log_loss(y, p),
        "Accuracy": accuracy_score(y, p >= 0.5),
        "F1": f1_score(y, p >= 0.5),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--emb-dim", type=int, default=16)
    args = ap.parse_args()
    seed_all()

    df = pd.read_csv(args.train)
    if "label" not in df.columns:
        raise ValueError("label column is required")
    numeric, categorical = get_columns(df)
    train, valid = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    Xn, Xc, Vn, Vc, sizes = prepare(train, valid, numeric, categorical)

    Xn, Xc, Vn, Vc = [torch.tensor(v) for v in (Xn, Xc, Vn, Vc)]
    y = torch.tensor(train["label"].to_numpy("float32"))
    model = CTRModel(sizes, len(numeric), args.emb_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(y))
        for start in range(0, len(y), args.batch_size):
            idx = order[start:start + args.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xn[idx], Xc[idx]), y[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Vn, Vc)).numpy()
        print({"epoch": epoch + 1, **report_metrics(valid["label"].to_numpy(), p)})

if __name__ == "__main__":
    main()
