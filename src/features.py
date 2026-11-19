"""Feature definitions and preprocessing for hydrogel stiffness modelling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "compressive_modulus_kpa"

CATEGORICAL = ["polymer", "crosslinker_type"]
NUMERIC = [
    "polymer_conc_wv",
    "macromer_mw_kda",
    "degree_of_methacrylation_pct",
    "crosslinker_conc",
    "crosslink_time_s",
    "test_temperature_c",
]
# Columns that only apply to some polymers (NaN = "not applicable").
SPARSE_NUMERIC = ["macromer_mw_kda", "degree_of_methacrylation_pct"]
FEATURES = CATEGORICAL + NUMERIC


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-informed features.

    Network theory predicts power-law dependence of modulus on concentration,
    so log(concentration) lets even a linear model capture E ~ c^n.
    Crosslinking time and initiator dose act multiplicatively, so their logs
    are also useful.
    """
    out = df.copy()
    out["log_conc"] = np.log(out["polymer_conc_wv"])
    out["log_crosslinker"] = np.log(out["crosslinker_conc"])
    out["log_time"] = np.log(out["crosslink_time_s"])
    return out


ENGINEERED = ["log_conc", "log_crosslinker", "log_time"]


def build_preprocessor() -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    ct = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", numeric, NUMERIC + ENGINEERED),
    ])
    return Pipeline([
        ("physics", FunctionTransformer(add_physics_features)),
        ("columns", ct),
    ])


def validate_columns(df: pd.DataFrame, need_target: bool = True) -> None:
    need = FEATURES + ([TARGET] if need_target else [])
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")
    if (df["polymer_conc_wv"] <= 0).any() or (df["crosslinker_conc"] <= 0).any() \
            or (df["crosslink_time_s"] <= 0).any():
        raise ValueError("Concentrations and crosslinking time must be > 0.")


def pow10(y):
    """Inverse of log10 target transform (module-level so models can be pickled)."""
    return np.power(10.0, y)
