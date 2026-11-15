"""Interface Streamlit pour analyser de vrais avis avec le pipeline SentimentAI."""

from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.pkl"
LOW_CONFIDENCE_THRESHOLD = 0.60

# Adaptez cette table si le modèle renvoie d'autres labels (par exemple 1/0).
LABEL_MAP = {
    "positive": "Positive",
    "pos": "Positive",
    "1": "Positive",
    "negative": "Negative",
    "neg": "Negative",
    "0": "Negative",
}


def normalize_label(label: Any, label_map: dict[str, str] | None = None) -> str:
    """Convertit les encodages courants en label lisible sans deviner les autres."""
    mapping = LABEL_MAP if label_map is None else label_map
    key = str(label).strip().lower()
    result = mapping.get(key)
    if result not in {"Positive", "Negative"}:
        raise ValueError(
            f"Le modèle a renvoyé le label « {label} », qui n'est pas configuré. "
            "Adaptez LABEL_MAP dans app.py au format de sortie du modèle."
        )
    return result


@st.cache_resource(show_spinner="Chargement du modèle…")
def load_model(model_path: str = str(MODEL_PATH)) -> Any:
    """Charge une seule fois le pipeline scikit-learn fourni par l'équipe ML."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Le fichier du modèle est absent : {path.name}")
    return joblib.load(path)


def predict_review(model: Any, review: str) -> tuple[str, float | None]:
    """Prédit directement sur le texte brut et renvoie le label et sa confiance."""
    prediction = model.predict([review])[0]
    sentiment = normalize_label(prediction)
    confidence = None

    if callable(getattr(model, "predict_proba", None)):
        probabilities = model.predict_proba([review])[0]
        classes = getattr(model, "classes_", None)
        if classes is not None:
            for index, class_label in enumerate(classes):
                if str(class_label) == str(prediction):
                    confidence = float(probabilities[index])
                    break
        elif len(probabilities) == 2:
            # Pour un pipeline qui ne publie pas classes_, ordre conventionnel sklearn.
            confidence = float(max(probabilities))
    return sentiment, confidence


def add_analysis(review: str, sentiment: str, confidence: float | None) -> None:
    """Ajoute une analyse réussie à l'historique de la session."""
    st.session_state.history.insert(
        0,
        {
            "Avis": review,
            "Sentiment": sentiment,
            "Confiance": confidence,
            "Date et heure": datetime.now().astimezone().strftime("%d/%m/%Y %H:%M:%S %Z"),
        },
    )


def show_dashboard(history: list[dict[str, Any]]) -> None:
    st.subheader("Vue d'ensemble")
    total = len(history)
    positive = sum(item["Sentiment"] == "Positive" for item in history)
    negative = sum(item["Sentiment"] == "Negative" for item in history)
    positive_pct = positive / total * 100 if total else 0

    columns = st.columns(4)
    columns[0].metric("Avis analysés", total)
    columns[1].metric("Positifs", positive)
    columns[2].metric("Négatifs", negative)
    columns[3].metric("Avis positifs", f"{positive_pct:.1f}%")

    st.markdown("#### Répartition des sentiments")
    if total:
        chart_data = pd.DataFrame(
            {"Nombre d'avis": [positive, negative]},
            index=["Positif", "Négatif"],
        )
        st.bar_chart(chart_data, color=["#16a36a"])
    else:
        st.info("Les statistiques apparaîtront après votre première analyse.")


def main() -> None:
    st.set_page_config(page_title="SentimentAI", page_icon="💬", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {max-width: 1100px; padding-top: 2rem;}
        [data-testid="stMetric"] {
            background: #202a36 !important;
            border: 1px solid #3b4756 !important;
            padding: 1rem 1.1rem !important;
            border-radius: 12px !important;
            min-height: 112px;
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
            color: #d1d9e2 !important;
            opacity: 1 !important;
        }
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] div {
            color: #ffffff !important;
        }
        [data-testid="stMetricDelta"] {color: #d1d9e2 !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.title("💬 SentimentAI")
        st.caption("Analyse de sentiment des avis clients")
        page = st.radio("Navigation", ["Accueil", "Analyser un avis", "Tableau de bord", "Historique"])
        st.divider()
        st.caption("Modèle : `model.pkl` · Pipeline TF-IDF + Logistic Regression")

    st.title("SentimentAI")
    st.caption("Comprenez rapidement le ressenti exprimé dans un avis client.")

    if page == "Accueil":
        st.markdown("### Une lecture simple des retours clients")
        st.write(
            "SentimentAI classe un avis comme **positif** ou **négatif** à l'aide du pipeline "
            "de Machine Learning entraîné par l'équipe. Le texte est transmis directement au "
            "pipeline, qui inclut la vectorisation TF-IDF et la régression logistique."
        )
        st.info("Choisissez **Analyser un avis** dans le menu pour commencer.")
        show_dashboard(st.session_state.history)
    elif page == "Analyser un avis":
        st.subheader("Analyse d'un avis client")
        review = st.text_area("Votre avis", placeholder="Ex. : La livraison était rapide et le produit est excellent.", height=160)
        if st.button("Analyser le sentiment", type="primary", use_container_width=True):
            if not review.strip():
                st.warning("Saisissez un avis avant de lancer l'analyse.")
            else:
                try:
                    model = load_model()
                except FileNotFoundError:
                    st.error("Le modèle `model.pkl` est introuvable. Ajoutez le pipeline entraîné dans le dossier de l'application.")
                except Exception:
                    st.error("Le modèle n'a pas pu être chargé. Vérifiez que `model.pkl` est un pipeline Joblib compatible.")
                else:
                    try:
                        sentiment, confidence = predict_review(model, review.strip())
                    except Exception:
                        st.error("L'analyse a échoué. Vérifiez le format des labels du modèle et réessayez.")
                    else:
                        add_analysis(review.strip(), sentiment, confidence)
                        if confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD:
                            st.warning(
                                f"⚠️ Résultat incertain : le modèle penche vers « {sentiment.lower()} », "
                                "mais sa confiance est faible. Interprétez cette prédiction avec prudence."
                            )
                        elif sentiment == "Positive":
                            st.success("✅ Avis positif")
                        else:
                            st.error("❌ Avis négatif")
                        if confidence is not None:
                            st.metric("Score de confiance", f"{confidence:.1%}")
    elif page == "Tableau de bord":
        show_dashboard(st.session_state.history)
    else:
        st.subheader("Historique de la session")
        history = st.session_state.history
        if history:
            st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
            if st.button("Supprimer l'historique", type="secondary"):
                st.session_state.history = []
                st.rerun()
        else:
            st.info("Aucune analyse pour le moment. Les analyses réussies apparaîtront ici.")


if __name__ == "__main__":
    main()
