"""Task 1: item-item collaborative filtering and matrix factorization.
Input CSV columns: user_id,item_id,rating
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

def encode(df):
    users = {v: i for i, v in enumerate(df["user_id"].drop_duplicates())}
    items = {v: i for i, v in enumerate(df["item_id"].drop_duplicates())}
    out = df.copy()
    out["u"] = out["user_id"].map(users).astype(int)
    out["i"] = out["item_id"].map(items).astype(int)
    return out, users, items

def item_cf(train, test, k=30):
    n_users = int(train.u.max()) + 1
    n_items = int(max(train.i.max(), test.i.max())) + 1
    R = np.zeros((n_users, n_items), dtype=np.float32)
    for r in train.itertuples():
        R[r.u, r.i] = r.rating
    norms = np.linalg.norm(R, axis=0) + 1e-8
    S = (R.T @ R) / (norms[:, None] * norms[None, :])
    fallback = float(train.rating.mean())
    means = train.groupby("i").rating.mean().to_dict()
    preds = []
    for r in test.itertuples():
        rated = np.flatnonzero(R[r.u] != 0)
        rated = rated[rated != r.i]
        if len(rated) == 0:
            preds.append(means.get(r.i, fallback))
            continue
        scores = S[r.i, rated]
        take = np.argsort(np.abs(scores))[-min(k, len(scores)):]
        s = scores[take]
        ratings = R[r.u, rated[take]]
        denom = np.abs(s).sum()
        preds.append(float(np.dot(s, ratings) / denom) if denom > 1e-8 else fallback)
    return np.asarray(preds)

@dataclass
class MF:
    mu: float
    bu: np.ndarray
    bi: np.ndarray
    P: np.ndarray
    Q: np.ndarray

def train_mf(train, n_users, n_items, dim=32, epochs=25, lr=0.01, reg=0.02, seed=42):
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 0.05, (n_users, dim)).astype(np.float32)
    Q = rng.normal(0, 0.05, (n_items, dim)).astype(np.float32)
    bu = np.zeros(n_users, dtype=np.float32)
    bi = np.zeros(n_items, dtype=np.float32)
    mu = float(train.rating.mean())
    rows = list(train.itertuples(index=False))
    for _ in range(epochs):
        rng.shuffle(rows)
        for r in rows:
            pred = mu + bu[r.u] + bi[r.i] + float(P[r.u] @ Q[r.i])
            err = float(r.rating - pred)
            pu, qi = P[r.u].copy(), Q[r.i].copy()
            P[r.u] += lr * (err * qi - reg * pu)
            Q[r.i] += lr * (err * pu - reg * qi)
            bu[r.u] += lr * (err - reg * bu[r.u])
            bi[r.i] += lr * (err - reg * bi[r.i])
    return MF(mu, bu, bi, P, Q)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--k", type=int, default=30)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    required = {"user_id", "item_id", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df, users, items = encode(df[["user_id", "item_id", "rating"]].dropna())
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    p_cf = item_cf(train, test, args.k)
    model = train_mf(train, len(users), len(items), args.dim, args.epochs)

    p_mf = np.asarray([
        model.mu + model.bu[r.u] + model.bi[r.i] + model.P[r.u] @ model.Q[r.i]
        for r in test.itertuples()
    ])

    out = pd.DataFrame([
        ["item-CF", mean_squared_error(test.rating, p_cf) ** 0.5, mean_absolute_error(test.rating, p_cf)],
        ["matrix-factorization", mean_squared_error(test.rating, p_mf) ** 0.5, mean_absolute_error(test.rating, p_mf)],
    ], columns=["model", "RMSE", "MAE"])
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
