"""Tests de base pour la CI (en attendant les tests complets du membre 5)."""

import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent

POSITIVE = "This movie was absolutely wonderful, brilliant acting and a great story. I loved it!"
NEGATIVE = "This movie was terrible, boring and a complete waste of time. I hated it."


def test_prepare_data_convertit_le_parquet(tmp_path):
    """prepare_data.py doit transformer text/label (0/1) en review/sentiment."""
    parquet = tmp_path / "mini.parquet"
    pd.DataFrame({"text": ["good film", "bad film"], "label": [1, 0]}).to_parquet(parquet)
    output = tmp_path / "out.csv"

    result = subprocess.run(
        [sys.executable, str(ROOT / "prepare_data.py"),
         "--inputs", str(parquet), "--output", str(output)],
        capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    df = pd.read_csv(output)
    assert list(df.columns) == ["review", "sentiment"]
    assert df["sentiment"].tolist() == ["positive", "negative"]


def test_model_pkl_se_charge_et_predit():
    """Le modèle utilisé par l'application (membre 1) doit se charger et prédire."""
    model = joblib.load(ROOT / "model.pkl")
    predictions = model.predict([POSITIVE, NEGATIVE])
    assert len(predictions) == 2


def test_modele_entraine_en_ci():
    """Le modèle entraîné par la CI doit reconnaître un avis positif et un négatif."""
    path = ROOT / "model_from_csv.pkl"
    if not path.exists():
        pytest.skip("model_from_csv.pkl absent : lancer train_model.py d'abord")
    model = joblib.load(path)
    assert list(model.predict([POSITIVE, NEGATIVE])) == ["positive", "negative"]