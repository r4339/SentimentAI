"""Entraîne et évalue le pipeline IMDb TF-IDF + Logistic Regression."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def build_pipeline() -> Pipeline:
    """Construit un pipeline complet afin que l'application reçoive du texte brut."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=30_000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1_000, solver="liblinear")),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Chemin du fichier CSV IMDb")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "model_from_csv.pkl",
        help="Chemin de sortie (par défaut : model_from_csv.pkl à côté du script)",
    )
    args = parser.parse_args()

    if not args.csv_path.is_file():
        raise SystemExit(f"Dataset introuvable : {args.csv_path}")

    data = pd.read_csv(args.csv_path)
    required_columns = {"review", "sentiment"}
    if not required_columns.issubset(data.columns):
        raise SystemExit("Le CSV doit contenir les colonnes 'review' et 'sentiment'.")
    if data["review"].isna().any() or data["sentiment"].isna().any():
        raise SystemExit("Le dataset contient des avis ou labels manquants ; corrigez-les avant l'entraînement.")

    data["review"] = data["review"].astype(str)
    data["sentiment"] = data["sentiment"].astype(str).str.strip().str.lower()
    labels = set(data["sentiment"].unique())
    if labels != {"positive", "negative"}:
        raise SystemExit(f"Labels attendus : positive et negative ; labels trouvés : {sorted(labels)}")

    X_train, X_test, y_train, y_test = train_test_split(
        data["review"],
        data["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=data["sentiment"],
    )

    # Mesure la performance sur une partie tenue à l'écart de l'entraînement.
    evaluation_pipeline = build_pipeline()
    evaluation_pipeline.fit(X_train, y_train)
    predictions = evaluation_pipeline.predict(X_test)
    print(f"Avis : {len(data):,} | Entraînement : {len(X_train):,} | Test : {len(X_test):,}")
    print(f"Accuracy (test) : {accuracy_score(y_test, predictions):.4f}")
    print("Rapport de classification (test) :")
    print(classification_report(y_test, predictions, digits=4))

    # Pour l'artefact final, réentraîne le même pipeline sur toutes les données.
    final_pipeline = build_pipeline()
    final_pipeline.fit(data["review"], data["sentiment"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, args.output)
    print(f"Pipeline complet enregistré dans : {args.output.resolve()}")


if __name__ == "__main__":
    main()
