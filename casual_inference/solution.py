"""Causal Inference task: IHDP naive ATE, IPW, propensity matching and causal forest.

Supports the standard CEVAE IHDP CSV layout:
    treatment, y_factual, y_counterfactual, mu0, mu1, x1...x25

The loader also accepts named columns commonly found in processed IHDP files.

Example:
    python solution.py --data ihdp_npci_1.csv --output results
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


def load_ihdp(path: str | Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Load common IHDP formats and return frame, X, mu0, mu1, y_cf."""
    df = pd.read_csv(path)

    # Standard CEVAE/NPCI replication: t, y, y_cf, mu0, mu1, then 25 covariates.
    if df.shape[1] >= 30 and not {"treatment", "y_factual"}.issubset(df.columns):
        tcol = df.columns[0]
        ycol = df.columns[1]
        ycfcol = df.columns[2]
        mu0col = df.columns[3]
        mu1col = df.columns[4]
        xcols = list(df.columns[5:30])
        out = df.copy()
        out = out.rename(columns={tcol:"treatment", ycol:"y_factual", ycfcol:"y_counterfactual",
                                  mu0col:"mu0", mu1col:"mu1"})
        X = out[xcols].to_numpy(float)
        return out, X, out["mu0"].to_numpy(float), out["mu1"].to_numpy(float), out["y_counterfactual"].to_numpy(float)

    lower = {c.lower(): c for c in df.columns}
    tname = next((lower[c] for c in ["treatment","t","w"] if c in lower), None)
    yname = next((lower[c] for c in ["y_factual","outcome_factual","outcome","y"] if c in lower), None)
    mu0name = next((lower[c] for c in ["mu0","y0","outcome_t0"] if c in lower), None)
    mu1name = next((lower[c] for c in ["mu1","y1","outcome_t1"] if c in lower), None)
    ycfname = next((lower[c] for c in ["y_counterfactual","y_cfactual","y_cf"] if c in lower), None)

    if tname is None or yname is None:
        raise ValueError("Could not identify treatment/outcome columns.")

    excluded = {tname, yname}
    for c in [mu0name, mu1name, ycfname]:
        if c: excluded.add(c)

    xcols = [c for c in df.columns if c not in excluded]
    # Prefer x1..x25 style columns when present.
    named_x = [c for c in df.columns if str(c).lower().startswith("x") and str(c)[1:].isdigit()]
    if len(named_x) >= 5:
        xcols = named_x

    X = df[xcols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
    mu0 = df[mu0name].to_numpy(float) if mu0name else None
    mu1 = df[mu1name].to_numpy(float) if mu1name else None
    ycf = df[ycfname].to_numpy(float) if ycfname else None
    out = df.rename(columns={tname:"treatment", yname:"y_factual"}).copy()
    return out, X, mu0, mu1, ycf


def potential_outcomes(treatment, y_factual, y_cf=None, mu0=None, mu1=None):
    """Recover y0/y1 when counterfactual outcomes are available."""
    t = np.asarray(treatment).astype(int).ravel()
    y = np.asarray(y_factual, dtype=float).ravel()
    if mu0 is not None and mu1 is not None:
        return np.asarray(mu0), np.asarray(mu1)
    if y_cf is None:
        return None, None
    ycf = np.asarray(y_cf, dtype=float).ravel()
    y0 = np.where(t == 1, ycf, y)
    y1 = np.where(t == 1, y, ycf)
    return y0, y1


def propensity_scores(X, T, seed=42):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    pipe.fit(X, T)
    return pipe.predict_proba(X)[:, 1], pipe


def naive_ate(Y, T):
    return float(np.mean(Y[T == 1]) - np.mean(Y[T == 0]))


def stabilized_ipw_ate(Y, T, e, clip=.01):
    p = np.clip(e, clip, 1 - clip)
    p_treat = float(T.mean())
    wt = np.where(T == 1, p_treat / p, (1 - p_treat) / (1 - p))
    treated_mean = np.sum(wt[T == 1] * Y[T == 1]) / np.sum(wt[T == 1])
    control_mean = np.sum(wt[T == 0] * Y[T == 0]) / np.sum(wt[T == 0])
    return float(treated_mean - control_mean)


def propensity_match_ate(Y, T, e):
    """1:1 nearest-neighbor matching on the propensity score, both directions."""
    treated = np.where(T == 1)[0]
    control = np.where(T == 0)[0]
    if len(treated) == 0 or len(control) == 0:
        raise ValueError("Both treatment groups are required.")

    nt = min(len(treated), len(control))
    nn_c = NearestNeighbors(n_neighbors=1).fit(e[control, None])
    _, idx_c = nn_c.kneighbors(e[treated, None])
    t_effects = Y[treated[:nt]] - Y[control[idx_c[:nt, 0]]]

    nn_t = NearestNeighbors(n_neighbors=1).fit(e[treated, None])
    _, idx_t = nn_t.kneighbors(e[control, None])
    c_effects = Y[treated[idx_t[:nt, 0]]] - Y[control[:nt]]

    return float((np.mean(t_effects) + np.mean(c_effects)) / 2)


def plot_overlap(e, T, path):
    plt.figure(figsize=(8, 5))
    plt.hist(e[T == 0], bins=30, alpha=.6, density=True, label="Control")
    plt.hist(e[T == 1], bins=30, alpha=.6, density=True, label="Treated")
    plt.xlabel("Estimated propensity P(T=1|X)")
    plt.ylabel("Density")
    plt.title("Propensity-score overlap")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_ite(tau_true, tau_hat, path):
    plt.figure(figsize=(6, 6))
    plt.scatter(tau_true, tau_hat, alpha=.65, s=20)
    lo = min(np.min(tau_true), np.min(tau_hat))
    hi = max(np.max(tau_true), np.max(tau_hat))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("True ITE")
    plt.ylabel("Estimated ITE")
    plt.title("Causal forest: estimated vs ground-truth ITE")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def fit_causal_forest(X, T, Y, seed=42):
    try:
        from econml.dml import CausalForestDML
    except ImportError as exc:
        raise RuntimeError("Install EconML first: pip install econml") from exc

    # Flexible nuisance models; the forest itself handles heterogeneity.
    model_y = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=-1
    )
    model_t = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=-1
    )
    cf = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        n_estimators=400,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1,
    )
    cf.fit(Y, T, X=X)
    tau_hat = cf.effect(X)
    return cf, np.asarray(tau_hat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    df, X, mu0, mu1, y_cf = load_ihdp(args.data)
    T = df["treatment"].to_numpy(int).ravel()
    Y = df["y_factual"].to_numpy(float).ravel()

    y0, y1 = potential_outcomes(T, Y, y_cf=y_cf, mu0=mu0, mu1=mu1)
    true_ate = float(np.mean(y1 - y0)) if y0 is not None else None

    e, _ = propensity_scores(X, T, args.seed)
    plot_overlap(e, T, outdir / "propensity_overlap.png")

    estimates = {"naive": naive_ate(Y, T)}
    estimates["ipw"] = stabilized_ipw_ate(Y, T, e)
    estimates["propensity_matching"] = propensity_match_ate(Y, T, e)

    cf, tau_hat = fit_causal_forest(X, T, Y, args.seed)
    estimates["causal_forest"] = float(np.mean(tau_hat))

    table = pd.DataFrame(
        [{"method": k, "estimated_ATE": v,
          "absolute_ATE_error": abs(v - true_ate) if true_ate is not None else np.nan}
         for k, v in estimates.items()]
    )

    print("\nATE comparison")
    print(table.to_string(index=False))

    if true_ate is not None:
        tau_true = y1 - y0
        pehe = float(np.sqrt(mean_squared_error(tau_true, tau_hat)))
        corr = float(np.corrcoef(tau_true, tau_hat)[0, 1])
        print(f"ground_truth_ATE={true_ate:.6f}")
        print(f"causal_forest_PEHE={pehe:.6f}")
        print(f"causal_forest_ITE_correlation={corr:.6f}")
        plot_ite(tau_true, tau_hat, outdir / "ite_true_vs_estimated.png")

    overlap = pd.DataFrame({
        "propensity": e,
        "treatment": T,
        "factual_outcome": Y,
        "true_ite": (y1-y0) if y0 is not None else np.nan,
        "estimated_ite": tau_hat,
    })
    overlap.to_csv(outdir / "unit_level_estimates.csv", index=False)
    table.to_csv(outdir / "ate_comparison.csv", index=False)

    print("\nSaved results to", outdir.resolve())


if __name__ == "__main__":
    main()
