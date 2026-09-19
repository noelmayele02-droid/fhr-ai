"""Tests unitaires du pipeline de données/modèle réel."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.load_real_data import load_raw, load_features_and_labels, train_val_test_split, FEATURE_COLS


def test_real_data_loads_and_has_expected_size():
    df = load_raw()
    assert len(df) == 2126, "Le dataset UCI Cardiotocography doit contenir 2126 cas réels"
    assert set(df["NSP"].unique()) == {1, 2, 3}


def test_feature_extraction_shapes():
    X, y, cols = load_features_and_labels()
    assert X.shape == (2126, len(FEATURE_COLS))
    assert y.shape == (2126,)
    assert set(np.unique(y)) == {0, 1, 2}
    assert cols == FEATURE_COLS


def test_split_preserves_class_proportions_roughly():
    X, y, _ = load_features_and_labels()
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

    assert len(X_train) + len(X_val) + len(X_test) == len(X)

    full_ratio = np.bincount(y) / len(y)
    train_ratio = np.bincount(y_train) / len(y_train)
    # tolérance de 5 points de pourcentage par classe (split stratifié)
    assert np.allclose(full_ratio, train_ratio, atol=0.05)


def test_trained_model_exists_and_predicts():
    import joblib
    model_path = Path(__file__).resolve().parent.parent / "models" / "artifacts" / "random_forest.joblib"
    if not model_path.exists():
        pytest.skip("Modèle non entraîné — lancer models/train_real.py d'abord")

    model = joblib.load(model_path)
    X, y, _ = load_features_and_labels()
    preds = model.predict(X[:10])
    assert len(preds) == 10
    assert set(preds).issubset({0, 1, 2})
