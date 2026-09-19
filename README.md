# FHR-AI — Classification réelle du Rythme Cardiaque Fœtal (CTG)
# Lien site en streamlit : https://fhr-ai-fsptclradc6hh652uu9agk.streamlit.app/
# Lien site sur Netlify : https://fhr-ai.netlify.app/

Projet inspiré de l'initiative **AIM-CTG** (GHICL / BPI France — Grand Défi IA & Santé) :
entraîner une IA capable d'analyser le rythme cardiaque fœtal (RCF/CTG) pour aider
à prévenir les complications à la naissance.

## ⚠️ Important — d'où viennent les données

Ce projet utilise le **vrai dataset clinique UCI Cardiotocography**
(Campos, D. & Bernardes, J., 2000 — licence CC BY 4.0) :

- **2126 cardiotocogrammes réels**, issus de vrais examens sur de vraies patientes
- Features mesurées automatiquement par le système clinique **SisPorto**
  (baseline FHR, variabilité court/long terme, accélérations, décélérations —
  ce sont exactement les indicateurs que la toolbox FHRMA calcule)
- Classification de référence établie par **3 obstétriciens experts + consensus**,
  selon la nomenclature **FIGO** (Normal / Suspect / Pathologique)
- Source : https://archive.ics.uci.edu/dataset/193/cardiotocography

**Aucune donnée n'est simulée dans ce projet.** Chaque ligne du fichier
`data/real/CTG.csv` correspond à un vrai cas clinique.

### Pourquoi pas physionet.org directement ?

Mon environnement d'exécution n'a pas d'accès réseau à `physionet.org` (la base
CTU-UHB, qui contient les *signaux bruts* CTG, y est hébergée). J'ai donc utilisé
la base **UCI Cardiotocography**, qui est la référence académique historique du
domaine (même auteurs, même méthodologie de calcul des indicateurs FHRMA) et qui
contient déjà les features cliniques extraites — ce qui est en réalité l'approche
standard dans la littérature scientifique pour ce type de classification. J'ai
récupéré le fichier via un miroir GitHub public (`raw.githubusercontent.com`),
domaine accessible depuis mon environnement.

Si tu as un accès à physionet.org (compte + license d'usage des données requis),
la fonction `load_real_ctu_uhb_record()` peut être ajoutée pour brancher les
signaux bruts CTU-UHB à la place — voir la section "Étendre vers les signaux
bruts" plus bas.

## Résultats réels du modèle (mesurés, pas estimés)

Modèle **Random Forest** entraîné sur 1487 cas réels, validé sur 318, testé sur
**321 cas jamais vus à l'entraînement** :

| Métrique | Valeur |
|---|---|
| Accuracy (test) | **94.7 %** |
| F1-macro (test) | **0.890** |
| Normal — precision/recall | 0.961 / 0.992 |
| Suspect — precision/recall | 0.892 / 0.733 |
| Pathologique — precision/recall | 0.889 / 0.889 |

Matrice de confusion (test, 321 cas réels) :

```
                Normal  Suspect  Pathologique
Normal             247       1       1
Suspect             10      33       2
Pathologique         0       3      24
```

Un second modèle (réseau de neurones dense PyTorch) est aussi entraîné et
sauvegardé à titre de comparaison (F1-macro 0.823).

Ces chiffres sont cohérents avec les publications scientifiques utilisant ce
même dataset (typiquement 90-96% d'accuracy selon les méthodes).

## Architecture

```
fhr-ai/
├── data/
│   ├── real/CTG.csv          # VRAI dataset clinique (2126 cas réels)
│   └── load_real_data.py     # chargement, nettoyage, split stratifié
├── models/
│   ├── train_real.py         # entraînement Random Forest + réseau de neurones
│   └── artifacts/            # modèles entraînés sauvegardés (générés après training)
├── serve/
│   └── inference_api.py      # API FastAPI d'inférence (modèle réel)
├── app/
│   └── dashboard.py          # dashboard Streamlit (exploration + démo sur vrais cas)
├── tests/
│   └── test_real_pipeline.py # tests sur les vraies données
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Démarrage rapide

```bash
pip install -r requirements.txt

# 1. Entraîne le modèle sur les vraies données (déjà incluses dans le projet)
python models/train_real.py --epochs 150

# 2. Lance l'API d'inférence temps réel
uvicorn serve.inference_api:app --reload
# -> POST http://localhost:8000/predict/real avec les features cliniques
# -> GET  http://localhost:8000/model/info

# 3. Lance le dashboard interactif
streamlit run app/dashboard.py
```

Ou avec Docker (le modèle est entraîné automatiquement au build) :

```bash
docker compose up --build
```

### Exemple d'appel API

```bash
curl -X POST http://localhost:8000/predict/real \
  -H "Content-Type: application/json" \
  -d '{
    "LB": 132, "AC": 0.004, "FM": 0, "UC": 0.004,
    "ASTV": 17, "MSTV": 2.1, "ALTV": 0, "MLTV": 10.4,
    "DL": 0, "DS": 0, "DP": 0, "DR": 0,
    "Width": 130, "Min": 68, "Max": 198, "Nmax": 6, "Nzeros": 1,
    "Mode": 141, "Mean": 136, "Median": 140, "Variance": 12, "Tendency": 0
  }'
```

## Tests

```bash
python -m pytest tests/ -v
```

4 tests, tous validés sur les vraies données (taille du dataset, extraction des
features, stratification du split, prédiction du modèle entraîné).

## Étendre vers les signaux bruts (CTU-UHB / PhysioNet)

Pour aller plus loin avec les *signaux temporels bruts* (comme dans la mission
AIM-CTG du GHICL, qui vise à porter FHRMA de Matlab vers Python sur des
signaux bruts et non des features déjà calculées) :

1. Créer un compte PhysioNet et accepter la licence d'usage de CTU-UHB
2. `pip install wfdb`
3. Télécharger les fichiers `.dat`/`.hea` depuis
   https://physionet.org/content/ctu-uhb-ctgdb/
4. Utiliser `wfdb.rdsamp()` pour charger les signaux FHR + UC bruts
5. Réimplémenter les fonctions FHRMA (baseline, STV, LTV, détection
   d'accélérations/décélérations) sur ces signaux — la classification finale
   (Normal/Suspect/Pathologique) rejoint alors le même modèle que celui de
   ce projet, entraîné cette fois sur des features extraites automatiquement
   plutôt que pré-calculées.

## Auteur

Fait par Mayele — projet de recherche personnel inspiré de l'offre GHICL (AIM-CTG),
construit sur des données cliniques réelles et publiques.
