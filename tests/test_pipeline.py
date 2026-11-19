import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from features import FEATURES, TARGET, validate_columns  # noqa: E402
from generate_dataset import generate  # noqa: E402
from train import candidate_models, make_model  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return generate(n=300, seed=0)


def test_dataset_schema_and_ranges(data):
    validate_columns(data)
    assert set(data["polymer"]) == {"GelMA", "PEGDA", "Alginate"}
    assert (data[TARGET] > 0).all()


def test_stiffness_increases_with_concentration(data):
    g = data[data["polymer"] == "GelMA"]
    assert np.corrcoef(np.log(g["polymer_conc_wv"]), np.log(g[TARGET]))[0, 1] > 0.6


def test_model_beats_baseline(data):
    train, test = data.iloc[:240], data.iloc[240:]
    gb = make_model(candidate_models()["Gradient boosting"]).fit(train[FEATURES], train[TARGET])
    base = make_model(candidate_models()["Mean baseline"]).fit(train[FEATURES], train[TARGET])
    err = lambda m: np.mean(np.abs(m.predict(test[FEATURES]) / test[TARGET] - 1))
    assert err(gb) < 0.3
    assert err(gb) < err(base) / 3


def test_validate_columns_rejects_missing():
    with pytest.raises(ValueError):
        validate_columns(pd.DataFrame({"polymer": ["GelMA"]}))
