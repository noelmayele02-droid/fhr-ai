"""
dashboard.py
============
Dashboard Streamlit — explore le VRAI dataset clinique UCI Cardiotocography
et le modèle Random Forest réellement entraîné dessus (94.7% accuracy test).

Permet de :
- explorer la distribution des vrais cas cliniques (Normal/Suspect/Pathologique)
- sélectionner un vrai cas du dataset et voir la prédiction du modèle vs. le vrai label
- tester le modèle avec des valeurs de features personnalisées

Usage:
    streamlit run app/dashboard.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.load_real_data import load_raw, FEATURE_COLS, CLASS_NAMES

st.set_page_config(page_title="FHR-AI — Dashboard CTG (données réelles)", layout="wide")

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "models" / "artifacts"
CLASS_LABELS = ["Normal", "Suspect", "Pathologique"]

st.title("🫀 FHR-AI — Analyse du Rythme Cardiaque Fœtal")
st.caption(
    "Modèle entraîné sur le **vrai** dataset clinique UCI Cardiotocography "
    "(Campos & Bernardes, 2000) — 2126 cardiotocogrammes réels, mesurés par le "
    "système SisPorto et classés par 3 obstétriciens experts."
)


@st.cache_data
def get_data():
    return load_raw()


@st.cache_resource
def get_model():
    model_path = ARTIFACTS_DIR / "random_forest.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


df = get_data()
model = get_model()

if model is None:
    st.error(
        "Modèle non trouvé. Entraînez-le d'abord avec :\n\n"
        "`python models/train_real.py`"
    )
    st.stop()

tab1, tab2, tab3 = st.tabs(["📊 Exploration du dataset réel", "🔍 Tester un vrai cas", "✍️ Saisie manuelle"])

with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Répartition des classes (réelles)")
        counts = df["NSP"].map(CLASS_NAMES).value_counts()
        fig = px.pie(values=counts.values, names=counts.index,
                     color=counts.index,
                     color_discrete_map={"Normal": "#2a9d8f", "Suspect": "#e9c46a", "Pathologique": "#e63946"})
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Nombre total de cas réels", len(df))

    with col2:
        st.subheader("Distribution des indicateurs cliniques par classe")
        feature_to_plot = st.selectbox(
            "Indicateur FHRMA",
            ["LB", "ASTV", "MSTV", "ALTV", "MLTV", "AC", "DP"],
            help="LB=baseline, ASTV/MSTV=variabilité court-terme, ALTV/MLTV=variabilité long-terme",
        )
        df_plot = df.copy()
        df_plot["Classe"] = df_plot["NSP"].map(CLASS_NAMES)
        fig2 = px.box(df_plot, x="Classe", y=feature_to_plot, color="Classe",
                      color_discrete_map={"Normal": "#2a9d8f", "Suspect": "#e9c46a", "Pathologique": "#e63946"})
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Performance du modèle (mesurée sur données réelles jamais vues à l'entraînement)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy (test réel)", "94.7 %")
    c2.metric("F1-macro (test réel)", "0.890")
    c3.metric("Cas d'entraînement", "1487 / 2126")

with tab2:
    st.subheader("Sélectionner un vrai cas clinique du dataset")
    idx = st.slider("Index du cas (0 à %d)" % (len(df) - 1), 0, len(df) - 1, 10)
    row = df.iloc[idx]
    true_label = CLASS_NAMES[int(row["NSP"])]

    x = row[FEATURE_COLS].values.astype(np.float32).reshape(1, -1)
    pred = model.predict(x)[0]
    proba = model.predict_proba(x)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Vrai label (expert obstétricien)", true_label)
        color = "🟢" if pred == int(row["NSP"]) - 1 else "🔴"
        st.metric(f"{color} Prédiction du modèle", CLASS_LABELS[pred])

    with col2:
        fig3 = go.Figure(go.Bar(x=CLASS_LABELS, y=proba,
                                  marker_color=["#2a9d8f", "#e9c46a", "#e63946"]))
        fig3.update_layout(title="Probabilités du modèle", yaxis_range=[0, 1], height=250)
        st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Voir les features cliniques de ce cas"):
        st.dataframe(row[FEATURE_COLS].to_frame("Valeur"))

with tab3:
    st.subheader("Tester le modèle avec vos propres valeurs")
    st.caption("Valeurs par défaut = moyennes du dataset réel")

    defaults = df[FEATURE_COLS].mean()
    cols = st.columns(3)
    values = {}
    for i, feat in enumerate(FEATURE_COLS):
        with cols[i % 3]:
            values[feat] = st.number_input(feat, value=float(round(defaults[feat], 2)))

    if st.button("🔮 Prédire", use_container_width=True):
        x = np.array([[values[c] for c in FEATURE_COLS]], dtype=np.float32)
        pred = model.predict(x)[0]
        proba = model.predict_proba(x)[0]
        st.success(f"Prédiction : **{CLASS_LABELS[pred]}**")
        fig4 = go.Figure(go.Bar(x=CLASS_LABELS, y=proba,
                                  marker_color=["#2a9d8f", "#e9c46a", "#e63946"]))
        fig4.update_layout(yaxis_range=[0, 1], height=250)
        st.plotly_chart(fig4, use_container_width=True)

st.divider()
with st.expander("ℹ️ À propos des données et du modèle"):
    st.markdown("""
    **Source des données** : [UCI Cardiotocography Dataset](https://archive.ics.uci.edu/dataset/193/cardiotocography)
    (Campos, D. & Bernardes, J., 2000, licence CC BY 4.0) — 2126 cardiotocogrammes
    réels, mesurés automatiquement par le système SisPorto et classés par consensus
    de 3 obstétriciens experts selon la nomenclature FIGO (Normal/Suspect/Pathologique).

    **Modèle** : Random Forest (300 arbres, pondération équilibrée des classes),
    entraîné sur 1487 cas réels, validé sur 318, testé sur 321 cas jamais vus.

    **Résultats sur le jeu de test (données réelles, jamais vues à l'entraînement)** :
    accuracy = 94.7%, F1-macro = 0.890.

    Ce projet ne contient **aucune donnée simulée** : chaque ligne provient d'un
    vrai examen CTG réalisé sur une vraie patiente.
    """)
