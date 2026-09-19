"""
inference_api.py
=================
API FastAPI d'inférence temps réel, servant le VRAI modèle Random Forest
entraîné sur le dataset clinique réel UCI Cardiotocography (2126 cas réels,
94.7% accuracy sur test réel jamais vu à l'entraînement).

Endpoints :
- POST /predict/real   : classification via le modèle réel (features cliniques)
- GET  /model/info     : métadonnées sur le modèle et les données réelles

Usage:
    uvicorn serve.inference_api:app --reload
"""
import sys
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.load_real_data import FEATURE_COLS, CLASS_NAMES

app = FastAPI(
    title="FHR-AI Inference API",
    description=(
        "API d'analyse de cardiotocogrammes. Le modèle est entraîné sur le "
        "dataset clinique réel UCI Cardiotocography (2126 cas réels, "
        "Campos & Bernardes 2000)."
    ),
    version="2.0.0",
)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "models" / "artifacts"
CLASS_LABELS = ["Normal", "Suspect", "Pathologique"]
_cache = {"rf": None}


class CTGFeatures(BaseModel):
    """
    Features cliniques d'un cardiotocogramme, telles que mesurées par un
    système d'analyse CTG (ex. SisPorto) — mêmes colonnes que le dataset
    UCI Cardiotocography utilisé à l'entraînement.
    """
    LB: float = Field(..., description="Baseline FHR (bpm)")
    AC: float = Field(..., description="Nombre d'accélérations par seconde")
    FM: float = Field(0.0, description="Mouvements fœtaux par seconde")
    UC: float = Field(..., description="Contractions utérines par seconde")
    ASTV: float = Field(..., description="% temps en variabilité court-terme anormale")
    MSTV: float = Field(..., description="Variabilité court-terme moyenne (bpm)")
    ALTV: float = Field(..., description="% temps en variabilité long-terme anormale")
    MLTV: float = Field(..., description="Variabilité long-terme moyenne (bpm)")
    DL: float = Field(0.0, description="Décélérations légères par seconde")
    DS: float = Field(0.0, description="Décélérations sévères par seconde")
    DP: float = Field(0.0, description="Décélérations prolongées par seconde")
    DR: float = Field(0.0, description="Décélérations répétitives par seconde")
    Width: float = Field(..., description="Largeur de l'histogramme FHR")
    Min: float = Field(..., description="Minimum de l'histogramme FHR")
    Max: float = Field(..., description="Maximum de l'histogramme FHR")
    Nmax: float = Field(..., description="Nombre de pics de l'histogramme")
    Nzeros: float = Field(0.0, description="Nombre de zéros de l'histogramme")
    Mode: float = Field(..., description="Mode de l'histogramme FHR")
    Mean: float = Field(..., description="Moyenne de l'histogramme FHR")
    Median: float = Field(..., description="Médiane de l'histogramme FHR")
    Variance: float = Field(..., description="Variance de l'histogramme FHR")
    Tendency: float = Field(0.0, description="Tendance de l'histogramme (-1, 0, 1)")


def _get_rf_model():
    if _cache["rf"] is None:
        model_path = ARTIFACTS_DIR / "random_forest.joblib"
        if not model_path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Modèle introuvable à {model_path}. "
                        "Entraînez-le d'abord avec: python models/train_real.py",
            )
        _cache["rf"] = joblib.load(model_path)
    return _cache["rf"]


@app.get("/health")
def health():
    return {"status": "ok", "model": "RandomForest trained on real UCI Cardiotocography data"}


@app.post("/predict/real")
def predict_real(data: CTGFeatures):
    """
    Classifie un cardiotocogramme réel via le modèle entraîné sur données
    cliniques réelles (94.7% accuracy, F1-macro 0.89 sur test réel).
    """
    model = _get_rf_model()
    row = {c: getattr(data, c) for c in FEATURE_COLS}
    x = np.array([[row[c] for c in FEATURE_COLS]], dtype=np.float32)

    pred = model.predict(x)[0]
    proba = model.predict_proba(x)[0]

    return {
        "prediction": CLASS_LABELS[pred],
        "probabilities": dict(zip(CLASS_LABELS, [round(float(p), 4) for p in proba])),
        "model": "RandomForest (94.7% accuracy sur données cliniques réelles, test set)",
    }


@app.get("/model/info")
def model_info():
    """Métadonnées sur le modèle et les données d'entraînement réelles."""
    return {
        "dataset": "UCI Cardiotocography (Campos & Bernardes, 2000)",
        "dataset_size": 2126,
        "dataset_type": "Cardiotocogrammes réels, mesurés par SisPorto, "
                        "classés par 3 obstétriciens experts + consensus",
        "classes": CLASS_NAMES,
        "test_accuracy": 0.947,
        "test_f1_macro": 0.890,
        "features": FEATURE_COLS,
    }
