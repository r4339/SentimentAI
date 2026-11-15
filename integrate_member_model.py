"""Réunit les artefacts séparés du membre 1 en pipeline utilisé par l'app."""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from model_pipeline import clean_reviews


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
OUTPUT_PATH = APP_DIR / "model.pkl"


def main() -> None:
    vectorizer_path = MODEL_DIR / "tfidf.joblib"
    classifier_path = MODEL_DIR / "logreg.joblib"
    if not vectorizer_path.is_file() or not classifier_path.is_file():
        raise SystemExit(
            "Artefacts du membre 1 introuvables. Placez tfidf.joblib et logreg.joblib "
            "dans le dossier models/."
        )

    vectorizer = joblib.load(vectorizer_path)
    classifier = joblib.load(classifier_path)
    if not isinstance(vectorizer, TfidfVectorizer):
        raise SystemExit("tfidf.joblib ne contient pas un TfidfVectorizer scikit-learn.")
    if not isinstance(classifier, LogisticRegression):
        raise SystemExit("logreg.joblib ne contient pas une LogisticRegression scikit-learn.")

    # Le nettoyage fait partie du pipeline pour que l'app envoie le texte brut.
    pipeline = Pipeline(
        [
            ("cleaner", FunctionTransformer(clean_reviews, validate=False)),
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ]
    )
    joblib.dump(pipeline, OUTPUT_PATH)
    print(f"Pipeline du membre 1 enregistré : {OUTPUT_PATH}")
    print(f"Classes du modèle : {list(pipeline.classes_)}")


if __name__ == "__main__":
    main()
