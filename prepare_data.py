"""Convertit les fichiers Parquet IMDb (raw/) en CSV compatible avec train_model.py."""

import argparse
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--inputs", nargs="+", type=Path,
                    default=[Path("data/raw/train.parquet"), Path("data/raw/test.parquet")])
parser.add_argument("--output", type=Path, default=Path("data/imdb.csv"))
args = parser.parse_args()

# 1. Lecture et fusion des fichiers Parquet
frames = []
for path in args.inputs:
    if not path.is_file():
        raise SystemExit(f"Fichier introuvable : {path}")
    frames.append(pd.read_parquet(path))
df = pd.concat(frames, ignore_index=True)

# 2. Vérification du format d'origine
if not {"text", "label"}.issubset(df.columns):
    raise SystemExit(f"Colonnes attendues : text, label ; trouvées : {list(df.columns)}")

# 3. Conversion vers le format attendu par train_model.py
df = pd.DataFrame({
    "review": df["text"],
    "sentiment": df["label"].map({0: "negative", 1: "positive"}),
})
if df["sentiment"].isna().any():
    raise SystemExit("Certaines valeurs de 'label' ne sont ni 0 ni 1.")

# 4. Écriture du CSV
args.output.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(args.output, index=False)
print(f"{args.output} créé : {len(df):,} avis")
print(df["sentiment"].value_counts())