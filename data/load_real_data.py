"""
load_real_data.py
==================
Charge le VRAI dataset clinique UCI Cardiotocography (Campos & Bernardes, 2000) :
2126 cardiotocogrammes réels, mesurés par le système SisPorto sur des patientes
réelles, et classés par 3 obstétriciens experts + consensus.

Source : UCI Machine Learning Repository, dataset ID 193
https://archive.ics.uci.edu/dataset/193/cardiotocography
Licence : CC BY 4.0 — Campos, D. & Bernardes, J. (2000)

Fichier local : data/real/CTG.csv (miroir GitHub, téléchargé car mon
environnement n'a pas d'accès réseau direct à archive.ics.uci.edu ni
physionet.org — voir README pour les détails).

Les colonnes du fichier sont déjà les indicateurs FHRMA calculés
cliniquement par SisPorto :
    LB    = baseline FHR (bpm)                    -> équivalent baseline_fhr
    AC    = nombre d'accélérations                -> équivalent n_accelerations
    ASTV  = % de temps en variabilité anormale ct  -> lié à la STV
    MSTV  = variabilité court terme moyenne (bpm)  -> équivalent stv
    ALTV  = % de temps en variabilité anormale lt  -> lié à la LTV
    MLTV  = variabilité long terme moyenne (bpm)   -> équivalent ltv
    DL/DS/DP = décélérations légères/sévères/prolongées -> équivalent n_decelerations
    UC    = contractions utérines
    NSP   = label cible : 1=Normal, 2=Suspect, 3=Pathologique (classification FIGO)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path(__file__).resolve().parent / "real" / "CTG.csv"

# Colonnes de métadonnées à exclure (pas des features cliniques)
METADATA_COLS = ["FileName", "Date", "SegFile", "b", "e", "LBE", "CLASS"]

# Colonnes de features cliniques utilisées pour l'entraînement
FEATURE_COLS = [
    "LB", "AC", "FM", "UC", "ASTV", "MSTV", "ALTV", "MLTV",
    "DL", "DS", "DP", "DR", "Width", "Min", "Max", "Nmax", "Nzeros",
    "Mode", "Mean", "Median", "Variance", "Tendency",
]

CLASS_NAMES = {1: "Normal", 2: "Suspect", 3: "Pathologique"}


def load_raw() -> pd.DataFrame:
    """Charge le CSV brut et retire les lignes sans label NSP valide."""
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["NSP"])
    df = df[df["NSP"].isin([1, 2, 3])]
    return df.reset_index(drop=True)


def load_features_and_labels() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Retourne (X, y, feature_names) prêts pour l'entraînement.
    y est réindexé en 0/1/2 (Normal/Suspect/Pathologique) pour PyTorch/sklearn.
    """
    df = load_raw()
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

    X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    y = df["NSP"].astype(int) - 1  # 1,2,3 -> 0,1,2

    valid = ~X.isna().any(axis=1)
    X, y = X[valid], y[valid]

    return X.values.astype(np.float32), y.values.astype(np.int64), FEATURE_COLS


def train_val_test_split(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """Split stratifié 70/15/15, en conservant la proportion des 3 classes."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        train_idx.extend(cls_idx[:n_train])
        val_idx.extend(cls_idx[n_train:n_train + n_val])
        test_idx.extend(cls_idx[n_train + n_val:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)

    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])


if __name__ == "__main__":
    X, y, cols = load_features_and_labels()
    print(f"Dataset réel chargé : {X.shape[0]} échantillons, {X.shape[1]} features")
    print(f"Colonnes : {cols}")
    labels, counts = np.unique(y, return_counts=True)
    for lbl, cnt in zip(labels, counts):
        print(f"  {CLASS_NAMES[lbl + 1]:<15} {cnt:>5} ({100 * cnt / len(y):.1f}%)")
