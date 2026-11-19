"""
Generate a literature-informed synthetic hydrogel formulation dataset.

Why synthetic?
    Open, machine-readable datasets that pair hydrogel formulations with
    measured stiffness are rare (usually scattered across paper tables).
    This script produces a *reproducible stand-in* whose trends follow
    well-known polymer-network behaviour, so the modelling pipeline can be
    developed and tested end to end.

Physical trends encoded:
    * Modulus scales with polymer concentration as a power law,
      E ~ c^n with n ~ 2-2.5 for semi-dilute crosslinked networks
      (de Gennes scaling; commonly observed for GelMA and PEGDA).
    * Crosslinking is a saturating process: E rises with exposure / gelation
      time as 1 - exp(-t / tau) and with initiator / crosslinker
      concentration as a Langmuir-type term.
    * For PEGDA, higher macromer molecular weight gives a looser network
      (longer chains between crosslinks), so lower modulus.
    * For GelMA, a higher degree of methacrylation (DoM) gives more
      crosslinkable sites, so higher modulus.
    * Measurement temperature has a small effect (gelatin's physical
      network partially melts at 37 C).
    * Multiplicative log-normal noise (~12 %) mimics batch-to-batch and
      instrument variability.

Typical magnitudes produced (compressive modulus):
    GelMA 5-15 % w/v      ~ 1-60 kPa
    PEGDA 5-20 % w/v      ~ 5-600 kPa (strongly MW-dependent)
    Alginate 1-3 % w/v    ~ 2-80 kPa (CaCl2 crosslinked)

Usage:
    python data/generate_dataset.py --n 450 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RNG_DEFAULT_SEED = 42


def _saturating(x: np.ndarray, half: float) -> np.ndarray:
    """Langmuir-style saturation: 0 at x=0, 0.5 at x=half, -> 1 as x grows."""
    return x / (x + half)


def _sample_gelma(n: int, rng: np.random.Generator) -> pd.DataFrame:
    conc = rng.uniform(5, 15, n)                  # % w/v
    dom = rng.choice([40, 60, 80, 95], n)         # degree of methacrylation, %
    pi = rng.uniform(0.05, 0.5, n)                # LAP photoinitiator, % w/v
    t = rng.uniform(10, 300, n)                   # UV exposure, s
    temp = rng.choice([25, 37], n)
    e = (
        0.15 * conc ** 2.3
        * (0.35 + 0.65 * dom / 100)
        * _saturating(pi, 0.08)
        * (1 - np.exp(-t / 45))
        * np.where(temp == 37, 0.85, 1.0)
    )
    return pd.DataFrame({
        "polymer": "GelMA", "polymer_conc_wv": conc, "macromer_mw_kda": np.nan,
        "degree_of_methacrylation_pct": dom, "crosslinker_conc": pi,
        "crosslinker_type": "LAP_wv", "crosslink_time_s": t,
        "test_temperature_c": temp, "modulus_true": e,
    })


def _sample_pegda(n: int, rng: np.random.Generator) -> pd.DataFrame:
    conc = rng.uniform(5, 20, n)
    mw = rng.choice([0.7, 3.4, 6.0, 10.0], n)    # kDa
    pi = rng.uniform(0.05, 0.5, n)
    t = rng.uniform(10, 300, n)
    temp = rng.choice([25, 37], n)
    e = (
        0.9 * conc ** 2.1
        * (3.4 / mw) ** 0.55
        * _saturating(pi, 0.06)
        * (1 - np.exp(-t / 30))
    )
    return pd.DataFrame({
        "polymer": "PEGDA", "polymer_conc_wv": conc, "macromer_mw_kda": mw,
        "degree_of_methacrylation_pct": np.nan, "crosslinker_conc": pi,
        "crosslinker_type": "LAP_wv", "crosslink_time_s": t,
        "test_temperature_c": temp, "modulus_true": e,
    })


def _sample_alginate(n: int, rng: np.random.Generator) -> pd.DataFrame:
    conc = rng.uniform(1, 3, n)
    ca = rng.uniform(20, 200, n)                  # CaCl2, mM
    t = rng.uniform(60, 1800, n)                  # ionic gelation time, s
    temp = rng.choice([25, 37], n)
    e = (
        6.5 * conc ** 2.0
        * _saturating(ca, 40)
        * (1 - np.exp(-t / 400))
        * 1.6
    )
    return pd.DataFrame({
        "polymer": "Alginate", "polymer_conc_wv": conc, "macromer_mw_kda": np.nan,
        "degree_of_methacrylation_pct": np.nan, "crosslinker_conc": ca,
        "crosslinker_type": "CaCl2_mM", "crosslink_time_s": t,
        "test_temperature_c": temp, "modulus_true": e,
    })


def generate(n: int = 450, seed: int = RNG_DEFAULT_SEED, noise: float = 0.12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = [
        _sample_gelma(int(n * 0.45), rng),
        _sample_pegda(int(n * 0.35), rng),
        _sample_alginate(n - int(n * 0.45) - int(n * 0.35), rng),
    ]
    df = pd.concat(parts, ignore_index=True)
    df["compressive_modulus_kpa"] = df["modulus_true"] * rng.lognormal(0, noise, len(df))
    df = df.drop(columns="modulus_true")
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "sample_id", [f"HG{i:04d}" for i in range(1, len(df) + 1)])
    num = df.select_dtypes("number").columns
    df[num] = df[num].round(3)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=450)
    p.add_argument("--seed", type=int, default=RNG_DEFAULT_SEED)
    p.add_argument("--noise", type=float, default=0.12, help="log-normal sigma")
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "hydrogel_formulations.csv")
    a = p.parse_args()
    df = generate(a.n, a.seed, a.noise)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df)} rows -> {a.out}")
    print(df.groupby("polymer")["compressive_modulus_kpa"].describe().round(1))


if __name__ == "__main__":
    main()
