"""Vérifie que les métriques du modèle respectent les seuils minimaux."""

import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("metrics_path", type=Path, help="Fichier metrics.json")
parser.add_argument("--min-accuracy", type=float, default=0.85)
parser.add_argument("--min-f1", type=float, default=0.85)
args = parser.parse_args()

if not args.metrics_path.is_file():
    raise SystemExit(f"Fichier de métriques introuvable : {args.metrics_path}")

metrics = json.loads(args.metrics_path.read_text(encoding="utf-8"))

checks = [
    ("Accuracy", metrics["accuracy"], args.min_accuracy),
    ("F1 macro", metrics["f1_macro"], args.min_f1),
]

failed = False
lines = ["| Métrique | Valeur | Seuil | Statut |", "|---|---|---|---|"]
for name, value, threshold in checks:
    ok = value >= threshold
    failed = failed or not ok
    status = "✅" if ok else "❌"
    print(f"{status} {name} : {value:.4f} (seuil : {threshold:.2f})")
    lines.append(f"| {name} | {value:.4f} | {threshold:.2f} | {status} |")

# Affiche un tableau dans le résumé du run GitHub Actions
summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
if summary_file:
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write("## 📊 Métriques du modèle\n\n" + "\n".join(lines) + "\n")

if failed:
    raise SystemExit("❌ Le modèle ne respecte pas les seuils de qualité : pipeline arrêté.")
print("✅ Le modèle respecte tous les seuils de qualité.")