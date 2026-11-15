"""Prétraitement repris du projet du membre 1 pour l'inférence SentimentAI."""

import re
from collections.abc import Iterable


def clean_text(text: str) -> str:
    """Applique exactement le nettoyage utilisé par src/preprocess.py du modèle."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_reviews(reviews: Iterable[str]) -> list[str]:
    """Nettoie une séquence d'avis dans une étape compatible avec scikit-learn."""
    return [clean_text(str(review)) for review in reviews]
