"""
Predict compressive modulus for new hydrogel formulations with a trained model.

Examples:
    # Batch prediction from a CSV with the same feature columns as the training data
    python src/predict.py --input data/example_new_formulations.csv --output results/new_predictions.csv

    # Single formulation from the command line
    python src/predict.py --polymer GelMA --conc 10 --dom 80 --crosslinker 0.25 --time 60 --temp 37
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import FEATURES, validate_columns

ROOT = Path(__file__).resolve().parents[1]
CROSSLINKER_TYPE = {"GelMA": "LAP_wv", "PEGDA": "LAP_wv", "Alginate": "CaCl2_mM"}


def load_model(path: Path):
    bundle = joblib.load(path)
    return bundle["model"], bundle.get("model_name", "model")


def predict(df: pd.DataFrame, model) -> np.ndarray:
    validate_columns(df, need_target=False)
    return model.predict(df[FEATURES])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=ROOT / "models" / "stiffness_model.joblib")
    p.add_argument("--input", type=Path, help="CSV of formulations")
    p.add_argument("--output", type=Path, help="Where to write predictions (CSV)")
    p.add_argument("--polymer", choices=list(CROSSLINKER_TYPE))
    p.add_argument("--conc", type=float, help="polymer concentration, %% w/v")
    p.add_argument("--mw", type=float, default=np.nan, help="PEGDA macromer MW, kDa")
    p.add_argument("--dom", type=float, default=np.nan, help="GelMA degree of methacrylation, %%")
    p.add_argument("--crosslinker", type=float, help="LAP %% w/v (GelMA/PEGDA) or CaCl2 mM (alginate)")
    p.add_argument("--time", type=float, help="UV exposure or gelation time, s")
    p.add_argument("--temp", type=float, default=25, help="test temperature, C")
    a = p.parse_args()

    model, name = load_model(a.model)

    if a.input:
        df = pd.read_csv(a.input)
    elif a.polymer:
        df = pd.DataFrame([{
            "polymer": a.polymer, "polymer_conc_wv": a.conc, "macromer_mw_kda": a.mw,
            "degree_of_methacrylation_pct": a.dom, "crosslinker_conc": a.crosslinker,
            "crosslinker_type": CROSSLINKER_TYPE[a.polymer], "crosslink_time_s": a.time,
            "test_temperature_c": a.temp,
        }])
    else:
        p.error("Provide --input CSV or a single formulation via --polymer ...")

    df["predicted_modulus_kpa"] = predict(df, model).round(2)
    print(f"Model: {name}")
    print(df.to_string(index=False))
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.output, index=False)
        print(f"\nSaved -> {a.output}")


if __name__ == "__main__":
    main()
