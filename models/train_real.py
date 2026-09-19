"""
train_real.py
==============
Entraîne un VRAI modèle de classification sur le dataset clinique réel
UCI Cardiotocography (2126 cardiotocogrammes réels).

Compare deux approches :
1. Random Forest (baseline solide, standard sur ce type de données tabulaires)
2. Réseau de neurones dense (PyTorch) — pondéré pour gérer le déséquilibre
   des classes (77.8% Normal / 13.9% Suspect / 8.3% Pathologique)

Usage:
    python models/train_real.py --epochs 100
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.load_real_data import load_features_and_labels, train_val_test_split, CLASS_NAMES

CLASS_LABELS = ["Normal", "Suspect", "Pathologique"]


class CTGTabularNet(nn.Module):
    """Réseau dense pour classification à partir des features cliniques FHRMA."""

    def __init__(self, n_features: int, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def print_confusion(y_true, y_pred, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print(f"\nMatrice de confusion — {title} (lignes=vrai, colonnes=prédit):")
    header = "".ljust(14) + "".join(f"{c[:6]:>8}" for c in CLASS_LABELS)
    print(header)
    for i, row in enumerate(cm):
        print(f"{CLASS_LABELS[i][:12]:<14}" + "".join(f"{v:>8}" for v in row))


def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n" + "=" * 60)
    print("MODÈLE 1 : Random Forest")
    print("=" * 60)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    print(f"\nAccuracy validation : {(val_pred == y_val).mean():.4f}")
    print(f"Accuracy test       : {(test_pred == y_test).mean():.4f}")
    print(f"F1-macro test       : {f1_score(y_test, test_pred, average='macro'):.4f}")
    print("\nRapport de classification (test) :")
    print(classification_report(y_test, test_pred, target_names=CLASS_LABELS, digits=3))
    print_confusion(y_test, test_pred, "Random Forest")

    importances = clf.feature_importances_
    return clf, importances


def train_neural_net(X_train, y_train, X_val, y_val, X_test, y_test, epochs: int, feature_names):
    print("\n" + "=" * 60)
    print("MODÈLE 2 : Réseau de neurones dense (PyTorch)")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CTGTabularNet(n_features=X_train.shape[1]).to(device)

    class_counts = np.bincount(y_train, minlength=3)
    weights = torch.tensor(1.0 / np.maximum(class_counts, 1), dtype=torch.float32)
    weights = (weights / weights.sum() * 3).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)

    best_val_f1, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_pred = val_out.argmax(dim=1).cpu().numpy()
            val_f1 = f1_score(y_val, val_pred, average="macro")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"Époque {epoch:>3}/{epochs} — loss={loss.item():.4f}  val_f1_macro={val_f1:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t).argmax(dim=1).cpu().numpy()

    print(f"\nMeilleur val_f1_macro : {best_val_f1:.4f}")
    print(f"Accuracy test         : {(test_pred == y_test).mean():.4f}")
    print(f"F1-macro test         : {f1_score(y_test, test_pred, average='macro'):.4f}")
    print("\nRapport de classification (test) :")
    print(classification_report(y_test, test_pred, target_names=CLASS_LABELS, digits=3))
    print_confusion(y_test, test_pred, "Réseau de neurones")

    return model, scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--out_dir", type=str, default="models/artifacts")
    args = parser.parse_args()

    print("Chargement du dataset réel UCI Cardiotocography (2126 cas cliniques réels)...")
    X, y, feature_names = load_features_and_labels()
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)
    print(f"Split : train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")

    rf_model, importances = train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test)

    print("\nImportance des features (Random Forest) :")
    order = np.argsort(importances)[::-1]
    for i in order[:10]:
        print(f"  {feature_names[i]:<10} {importances[i]:.4f}")

    nn_model, scaler = train_neural_net(
        X_train, y_train, X_val, y_val, X_test, y_test, args.epochs, feature_names
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    joblib.dump(rf_model, out_dir / "random_forest.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    torch.save(nn_model.state_dict(), out_dir / "neural_net.pt")

    print(f"\nModèles sauvegardés dans {out_dir}/")
    print("  - random_forest.joblib")
    print("  - neural_net.pt")
    print("  - scaler.joblib")


if __name__ == "__main__":
    main()
