"""
Train and evaluate regression models that predict hydrogel compressive modulus
from formulation and processing parameters.

Workflow
    1. Load data, validate columns.
    2. Stratified (by polymer) 80/20 train/test split.
    3. Compare models with 5-fold cross-validation on the training set:
       mean baseline, linear, ridge, random forest, gradient boosting.
    4. Refit the best model on the full training set and evaluate once on the
       held-out test set (R^2, MAE, RMSE, MAPE in kPa).
    5. Save the model, metrics and diagnostic figures.

The target is modelled on a log10 scale because hydrogel stiffness spans
orders of magnitude and measurement error is roughly multiplicative.

Usage:
    python src/train.py --data data/hydrogel_formulations.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from features import FEATURES, TARGET, build_preprocessor, pow10, validate_columns
from plotting import POLYMER_COLORS, style_axes, setup_style

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def make_model(estimator) -> TransformedTargetRegressor:
    """Wrap an estimator with preprocessing and a log10 target transform."""
    pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log10, inverse_func=pow10)


def candidate_models() -> dict:
    return {
        "Mean baseline": DummyRegressor(),
        "Linear regression": LinearRegression(),
        "Ridge (alpha=1)": Ridge(alpha=1.0),
        "Random forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
        "Gradient boosting": GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=3,
                                                       subsample=0.8, random_state=SEED),
    }


def log_r2(y_true, y_pred) -> float:
    return r2_score(np.log10(y_true), np.log10(y_pred))


def evaluate(y_true, y_pred) -> dict:
    return {
        "r2_kpa": float(r2_score(y_true, y_pred)),
        "r2_log10": float(log_r2(y_true, y_pred)),
        "mae_kpa": float(mean_absolute_error(y_true, y_pred)),
        "rmse_kpa": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape_pct": float(100 * mean_absolute_percentage_error(y_true, y_pred)),
    }


def cross_validate_models(X, y) -> pd.DataFrame:
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, est in candidate_models().items():
        res = cross_validate(
            make_model(est), X, y, cv=cv,
            scoring={"r2": "r2", "mae": "neg_mean_absolute_error",
                     "mape": "neg_mean_absolute_percentage_error"},
        )
        rows.append({
            "model": name,
            "cv_r2_mean": res["test_r2"].mean(), "cv_r2_std": res["test_r2"].std(),
            "cv_mae_kpa": -res["test_mae"].mean(),
            "cv_mape_pct": -100 * res["test_mape"].mean(),
        })
        print(f"  {name:<20} R2 = {rows[-1]['cv_r2_mean']:.3f} ± {rows[-1]['cv_r2_std']:.3f}   "
              f"MAPE = {rows[-1]['cv_mape_pct']:.1f} %")
    return pd.DataFrame(rows).sort_values("cv_mape_pct").reset_index(drop=True)


# ----------------------------------------------------------------- figures --

def plot_parity(test_df, y_pred, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    lo, hi = 0.5, test_df[TARGET].max() * 2
    ax.plot([lo, hi], [lo, hi], color="#8a8985", lw=1, ls="--", zorder=1, label="Perfect prediction")
    for polymer, colour in POLYMER_COLORS.items():
        m = test_df["polymer"].values == polymer
        ax.scatter(test_df[TARGET].values[m], y_pred[m], s=42, color=colour,
                   edgecolor="#fcfcfb", linewidth=1.2, label=polymer, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Measured compressive modulus (kPa)")
    ax.set_ylabel("Predicted compressive modulus (kPa)")
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, loc="upper left")
    style_axes(ax)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_residuals(test_df, y_pred, path: Path) -> None:
    pct_err = 100 * (y_pred - test_df[TARGET].values) / test_df[TARGET].values
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="#8a8985", lw=1)
    for polymer, colour in POLYMER_COLORS.items():
        m = test_df["polymer"].values == polymer
        ax.scatter(test_df[TARGET].values[m], pct_err[m], s=36, color=colour,
                   edgecolor="#fcfcfb", linewidth=1.2, label=polymer)
    ax.set_xscale("log")
    ax.set_xlabel("Measured compressive modulus (kPa)")
    ax.set_ylabel("Prediction error (%)")
    ax.set_title("Residuals on held-out test set", loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    style_axes(ax)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_model_comparison(cv_df: pd.DataFrame, path: Path) -> None:
    d = cv_df.sort_values("cv_r2_mean")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.barh(d["model"], d["cv_r2_mean"].clip(lower=0), xerr=d["cv_r2_std"],
                   color="#2a78d6", height=0.55, error_kw={"ecolor": "#52514e", "lw": 1})
    for b, v in zip(bars, d["cv_r2_mean"]):
        ax.text(max(v, 0) + 0.03, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                va="center", color="#0b0b0b", fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Cross-validated R² (kPa scale, 5-fold)")
    ax.set_title("Model comparison", loc="left")
    style_axes(ax, grid_axis="x")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_importance(model, X_test, y_test, path: Path) -> pd.DataFrame:
    r = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=SEED,
                               scoring="neg_mean_absolute_percentage_error")
    imp = pd.DataFrame({"feature": FEATURES, "importance": r.importances_mean * 100,
                        "std": r.importances_std * 100}).sort_values("importance")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(imp["feature"], imp["importance"], xerr=imp["std"], color="#2a78d6", height=0.55,
            error_kw={"ecolor": "#52514e", "lw": 1})
    ax.set_xlabel("Increase in MAPE when feature is shuffled (percentage points)")
    ax.set_title("Permutation feature importance", loc="left")
    style_axes(ax, grid_axis="x")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return imp.sort_values("importance", ascending=False)


# -------------------------------------------------------------------- main --

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "hydrogel_formulations.csv")
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--model-out", type=Path, default=ROOT / "models" / "stiffness_model.joblib")
    a = p.parse_args()
    setup_style()
    a.results.mkdir(parents=True, exist_ok=True)
    a.model_out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.data)
    validate_columns(df)
    print(f"Loaded {len(df)} samples from {a.data}")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["polymer"])
    X_tr, y_tr = train_df[FEATURES], train_df[TARGET]
    X_te, y_te = test_df[FEATURES], test_df[TARGET]

    print("\n5-fold cross-validation on training set:")
    cv_df = cross_validate_models(X_tr, y_tr)
    cv_df.to_csv(a.results / "cv_model_comparison.csv", index=False)
    plot_model_comparison(cv_df, a.results / "model_comparison.png")

    best_name = cv_df.iloc[0]["model"]
    best = make_model(candidate_models()[best_name]).fit(X_tr, y_tr)
    y_pred = best.predict(X_te)
    test_metrics = evaluate(y_te, y_pred)
    per_polymer = {
        pol: evaluate(y_te[test_df["polymer"] == pol], y_pred[(test_df["polymer"] == pol).values])
        for pol in sorted(df["polymer"].unique())
    }

    print(f"\nBest model: {best_name}")
    print("Held-out test set: " + ", ".join(f"{k} = {v:.3f}" for k, v in test_metrics.items()))

    plot_parity(test_df, y_pred, a.results / "parity_plot.png",
                f"{best_name}: test R² = {test_metrics['r2_kpa']:.2f}, MAPE = {test_metrics['mape_pct']:.0f} %")
    plot_residuals(test_df, y_pred, a.results / "residuals.png")
    imp = plot_importance(best, X_te, y_te, a.results / "feature_importance.png")
    imp.to_csv(a.results / "feature_importance.csv", index=False)

    out = test_df[["sample_id", "polymer", TARGET]].copy()
    out["predicted_kpa"] = y_pred.round(2)
    out.to_csv(a.results / "test_predictions.csv", index=False)

    with open(a.results / "metrics.json", "w") as f:
        json.dump({"best_model": best_name, "n_train": len(train_df), "n_test": len(test_df),
                   "test": test_metrics, "test_by_polymer": per_polymer}, f, indent=2)
    joblib.dump({"model": best, "features": FEATURES, "model_name": best_name}, a.model_out)
    print(f"\nSaved model -> {a.model_out}\nSaved figures and metrics -> {a.results}")


if __name__ == "__main__":
    main()
