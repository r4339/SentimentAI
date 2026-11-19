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
    st.markdown('<div class="eyebrow">VOTRE ACTIVITÉ</div>', unsafe_allow_html=True)
    st.subheader("Vue d’ensemble")
    st.caption("Un aperçu des avis analysés pendant cette session.")
    total = len(history)
    positive = sum(item["Sentiment"] == "Positive" for item in history)
    negative = sum(item["Sentiment"] == "Negative" for item in history)
    positive_pct = positive / total * 100 if total else 0

    columns = st.columns(4)
    columns[0].metric("Avis analysés", total, help="Nombre d’avis traités dans cette session.")
    columns[1].metric("Positifs", positive)
    columns[2].metric("Négatifs", negative)
    columns[3].metric("Part positive", f"{positive_pct:.1f}%")

    st.markdown("#### Répartition des sentiments")
    if total:
        chart_data = pd.DataFrame(
            {"Positifs": [positive], "Négatifs": [negative]},
            index=["Avis analysés"],
        )
        st.bar_chart(chart_data, color=["#167d72", "#c65a70"])
    else:
        st.info("Votre tableau de bord se remplira dès la première analyse.")


def main() -> None:
    st.set_page_config(page_title="SentimentAI", page_icon="💬", layout="wide")
    st.markdown(
        """
        <style>
        :root {
            --ink: #17243a;
            --muted: #65738a;
            --line: #e2e8f0;
            --accent: #4968d8;
        }
        [data-testid="stAppViewContainer"] {background: #f5f7fb; color: var(--ink);}
        [data-testid="stHeader"] {background: transparent;}
        .block-container {max-width: 1240px; padding-top: 2.4rem; padding-bottom: 3rem;}
        h1, h2, h3, h4 {color: var(--ink) !important; letter-spacing: -0.025em;}
        p, label, [data-testid="stCaptionContainer"] {color: var(--muted);}
        .eyebrow {
            color: #4968d8; font-size: .72rem; font-weight: 750;
            letter-spacing: .14em; margin: 0 0 .45rem 0;
        }
        .hero-card {
            background: linear-gradient(120deg, #182b49 0%, #283f69 58%, #4968d8 100%);
            border-radius: 24px; padding: 2rem 2.2rem; margin: .7rem 0 1.5rem;
            color: #fff; box-shadow: 0 18px 42px rgba(26, 45, 78, .16);
        }
        .hero-card .hero-kicker {color: #c5d4ff; font-size: .75rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase;}
        .hero-card h2 {color: #fff !important; font-size: 2rem; margin: .55rem 0 .5rem;}
        .hero-card p {color: #e2eaff; font-size: 1rem; margin: 0; max-width: 720px; line-height: 1.65;}
        .tip-card {
            background: #edf2ff; border: 1px solid #dbe5ff; border-radius: 18px;
            padding: 1.15rem 1.25rem; color: #33466c; line-height: 1.6;
        }
        .tip-card strong {color: #253a62;}
        .brand-lockup {display: flex; align-items: center; gap: .72rem; margin: .65rem 0 .55rem;}
        .brand-mark {
            display: inline-flex; align-items: center; justify-content: center;
            width: 2.65rem; height: 2.65rem; border-radius: 13px;
            background: linear-gradient(145deg, #8474f2, #5345c2);
            box-shadow: 0 7px 18px rgba(83,69,194,.35); font-size: 1.32rem;
        }
        .brand-name {color: #fff; font-size: 1.48rem; font-weight: 800; letter-spacing: -.045em; line-height: 1;}
        .brand-name span {color: #aeb8ff;}
        [data-testid="stSidebar"] {background: linear-gradient(180deg, #17243a 0%, #202f4a 100%);}
        [data-testid="stSidebar"] * {color: #e8eef9;}
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color: #aebbd0;}
        [data-testid="stSidebar"] [data-testid="stRadio"] label p {color: #e8eef9;}
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            padding: .5rem .65rem; border: 1px solid transparent; border-radius: 11px;
            transition: background .15s ease, border-color .15s ease;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: rgba(255,255,255,.08);
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            background: rgba(111,133,238,.22); border-color: rgba(164,180,255,.42);
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
            color: #fff !important; font-weight: 700;
        }
        [data-testid="stSidebar"] hr {border-color: rgba(255,255,255,.16);}
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] code {color: #d8e2fa;}
        [data-testid="stMetric"] {
            background: #fff !important; border: 1px solid var(--line) !important;
            padding: 1.1rem 1.2rem !important; border-radius: 18px !important;
            min-height: 112px; box-shadow: 0 5px 18px rgba(26,45,78,.045);
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
            color: #65738a !important; opacity: 1 !important;
        }
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] div {
            color: #17243a !important;
        }
        [data-testid="stMetricDelta"] {color: #65738a !important;}
        [data-testid="stTextArea"] textarea {
            background: #fff !important; color: #17243a !important;
            border: 1px solid #d8e0eb !important; border-radius: 15px !important;
            min-height: 190px; line-height: 1.65;
        }
        [data-testid="stTextArea"] textarea:focus {border-color: #7189e2 !important; box-shadow: 0 0 0 3px rgba(73,104,216,.13) !important;}
        .stButton > button[kind="primary"] {
            min-height: 3.1rem; border-radius: 13px; font-weight: 700;
            color: #fff !important; background: linear-gradient(100deg, #4536b8, #6554d8); border: 0;
            box-shadow: 0 8px 18px rgba(73,104,216,.22); transition: transform .15s ease, box-shadow .15s ease;
        }
        .stButton > button[kind="primary"] * {color: #fff !important;}
        .stButton > button[kind="primary"]:hover {background: linear-gradient(100deg, #3c2ea8, #5948ca); transform: translateY(-1px); box-shadow: 0 11px 22px rgba(73,104,216,.3);}
        [data-testid="stVerticalBlockBorderWrapper"] {background: #fff; border-color: var(--line) !important; border-radius: 20px !important;}
        .result-card {padding: 1.35rem 1.5rem; border-radius: 18px; margin: 1rem 0 .7rem; border: 1px solid;}
        .result-positive {background: #edf8f4; border-color: #c5e8db; color: #17634f;}
        .result-negative {background: #fff1f2; border-color: #f2cbd1; color: #94394a;}
        .result-card .result-label {font-size: .75rem; font-weight: 750; letter-spacing: .12em; text-transform: uppercase; opacity: .8;}
        .result-card .result-value {font-size: 1.65rem; font-weight: 760; margin-top: .25rem;}
        @media (max-width: 700px) { .block-container {padding: 1.1rem 1rem 2rem;} .hero-card {padding: 1.4rem;} .hero-card h2 {font-size: 1.55rem;} }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.markdown(
            "<div class='brand-lockup'><span class='brand-mark'>💬</span>"
            "<span class='brand-name'>Sentiment<span>AI</span></span></div>",
            unsafe_allow_html=True,
        )
        st.caption("CUSTOMER VOICE ANALYTICS")
        st.markdown("Analysez le ressenti, simplement.")
        st.markdown("<br>", unsafe_allow_html=True)
        page = st.radio(
            "ESPACE DE TRAVAIL",
            ["Accueil", "Analyser un avis", "Tableau de bord", "Historique"],
            key="page_navigation",
        )
        st.divider()
        st.caption("MODÈLE ACTIF")
        st.markdown("**TF‑IDF + Logistic Regression**")
        st.caption("Les résultats sont indicatifs et dépendent de la langue et du domaine des avis.")

    st.markdown('<div class="eyebrow">SENTIMENTAI · CUSTOMER INSIGHTS</div>', unsafe_allow_html=True)

    if page == "Accueil":
        st.markdown(
            "<div class='hero-card'><div class='hero-kicker'>Écoutez la voix de vos clients</div>"
            "<h2>Transformez les avis en signaux utiles.</h2>"
            "<p>Une lecture rapide du sentiment exprimé dans chaque commentaire, avec un indicateur de confiance pour guider l’interprétation.</p></div>",
            unsafe_allow_html=True,
        )
        left, right = st.columns([1.35, 1])
        with left:
            st.markdown("### Commencez une analyse")
            st.write("Collez un avis, lancez l’analyse et consultez le résultat en quelques secondes.")
            st.button(
                "Analyser un avis  →",
                type="primary",
                use_container_width=True,
                on_click=lambda: st.session_state.update({"page_navigation": "Analyser un avis"}),
            )
        with right:
            st.markdown(
                "<div class='tip-card'><strong>À propos du résultat</strong><br>"
                "SentimentAI classe les avis en positif ou négatif. Le score de confiance mesure la certitude du modèle, pas la qualité réelle de l’avis."
                "<br><br><strong>Langue</strong><br>Pour un modèle entraîné sur IMDb, les prédictions sont à interpréter avec prudence pour les avis rédigés dans une autre langue que celle des données d’entraînement.</div>",
                unsafe_allow_html=True,
            )
        st.markdown("### Activité de la session")
        show_dashboard(st.session_state.history)
    elif page == "Analyser un avis":
        st.markdown("## Analyse d’un avis")
        st.caption("Collez un commentaire ci-dessous pour obtenir une prédiction de sentiment.")
        input_col, guide_col = st.columns([1.65, 1])
        with input_col:
            with st.container(border=True):
                st.markdown("#### Texte à analyser")
                review = st.text_area(
                    "Votre avis",
                    placeholder="Ex. : The product exceeded my expectations. Great quality and fast delivery!",
                    height=190,
                    label_visibility="collapsed",
                )
                st.caption("Conseil : collez un avis complet pour donner davantage de contexte au modèle.")
                analyze_clicked = st.button("Analyser le sentiment  →", type="primary", use_container_width=True)
        with guide_col:
            st.markdown(
                "<div class='tip-card'><strong>Comment lire le résultat</strong><br>"
                "La classe indique le sentiment prédit. Le score reflète la confiance estimée par le modèle ; une confiance faible invite à vérifier le résultat manuellement."
                "<br><br><strong>À garder en tête</strong><br>Un modèle entraîné sur des avis IMDb en anglais peut être moins fiable sur des textes français ou hors du domaine du cinéma.</div>",
                unsafe_allow_html=True,
            )
        if analyze_clicked:
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
                        result_class = "result-positive" if sentiment == "Positive" else "result-negative"
                        sentiment_icon = "✓" if sentiment == "Positive" else "−"
                        st.markdown(
                            f"<div class='result-card {result_class}'><div class='result-label'>Sentiment détecté</div>"
                            f"<div class='result-value'>{sentiment_icon} &nbsp;{sentiment}</div></div>",
                            unsafe_allow_html=True,
                        )
                        if confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD:
                            st.warning("Confiance faible : cette prédiction est à interpréter avec prudence.")
                        if confidence is not None:
                            confidence_col, note_col = st.columns([1, 2])
                            with confidence_col:
                                st.metric("Confiance estimée", f"{confidence:.1%}")
                            with note_col:
                                st.markdown("##### Niveau de confiance")
                                st.progress(max(0.0, min(confidence, 1.0)))
                                st.caption("Indicateur d’incertitude du modèle — ce n’est pas une garantie d’exactitude.")
    elif page == "Tableau de bord":
        show_dashboard(st.session_state.history)
    else:
        st.markdown("## Historique de la session")
        st.caption("Les analyses effectuées pendant cette session sont listées ici.")
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
