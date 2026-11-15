# SentimentAI

Application web Streamlit qui classe les avis clients en **Positive** ou **Negative** avec le pipeline scikit-learn entraîné par l'équipe (TF-IDF + Logistic Regression). L'application ne crée, n'entraîne ni ne simule aucun modèle.

## Technologies

Python 3.10+, Streamlit, Pandas, Joblib et scikit-learn.

## Intégrer les artefacts du membre ML

Le projet du membre 1 fournit `models/tfidf.joblib` et `models/logreg.joblib` séparément. Placez-les dans le sous-dossier `models/`, puis créez le pipeline complet attendu par l'application :

```powershell
python integrate_member_model.py
```

Le pipeline utilise le nettoyage de `src/preprocess.py` du projet ML, puis TF-IDF et la régression logistique. Il est enregistré dans `model.pkl`, que l'application charge et que le Dockerfile copie dans l'image. Le modèle du membre 1 annonce une accuracy de test de 0,9054 dans son README.

## Entraîner un modèle depuis le CSV IMDb brut

Le script alternatif `train_model.py` entraîne un pipeline sur un CSV IMDb avec les colonnes `review` et `sentiment`, évalue une séparation de test stratifiée (20 %, graine 42), puis réentraîne le pipeline sur toutes les lignes et enregistre `model_from_csv.pkl`. Il ne remplace donc pas le `model.pkl` intégré depuis les artefacts du membre 1.

```powershell
python train_model.py "C:\chemin\vers\IMDB Dataset.csv"
```

Le script affiche l'accuracy et le rapport de classification sur le jeu de test. Le modèle exporté est un véritable pipeline TF-IDF + régression logistique ; aucune prédiction n'est simulée.

## Installation et lancement

Depuis ce dossier, créez un environnement virtuel puis installez les dépendances :

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux : source .venv/bin/activate
python -m pip install -r requirements.txt
```

Ajoutez le véritable fichier `model.pkl` à côté de `app.py`, puis lancez :

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

Ouvrez ensuite http://localhost:8501.

## Intégration du modèle

Le fichier `model.pkl` n'est pas inclus : le membre chargé du Machine Learning doit fournir un pipeline complet sérialisé avec Joblib, avec vectorisation TF-IDF et classifieur Logistic Regression. Copiez-le à côté de `app.py`. Il sera chargé via `joblib.load` et mis en cache avec `st.cache_resource`. Aucun prétraitement n'est refait dans l'application ; la prédiction est appelée directement avec `model.predict([review])`.

Les labels usuels `positive`/`negative`, `pos`/`neg` et `1`/`0` sont pris en charge (`1` positif, `0` négatif). Si le modèle emploie un autre encodage, adaptez `LABEL_MAP` dans `app.py` en fonction de son contrat. Si `predict_proba` est disponible, le score associé à la classe prédite est affiché.

Sans `model.pkl`, l'application démarre et explique comment l'ajouter ; la classification n'est pas disponible tant que le modèle réel n'a pas été fourni.

## Docker

Après avoir ajouté `model.pkl` dans ce dossier :

```bash
docker build -t sentimentai .
docker run --rm -p 8501:8501 sentimentai
```

L'application écoute sur `0.0.0.0:8501` dans le conteneur. Le modèle est exclu du contexte Git par `.gitignore` ; vérifiez qu'il est présent au moment du build Docker.

## Structure

```text
sentimentai/
├── app.py           # Interface, prédiction, statistiques et historique
├── train_model.py   # Entraînement alternatif depuis le CSV brut
├── integrate_member_model.py # Assemble les artefacts du membre 1
├── model_pipeline.py # Nettoyage texte partagé avec le pipeline sérialisé
├── models/          # tfidf.joblib et logreg.joblib reçus du membre 1
├── model.pkl        # Pipeline du membre 1 intégré pour l'application
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore
```

L'historique est conservé dans `st.session_state` pour la session en cours. Le tableau de bord calcule ses statistiques à partir des analyses réellement effectuées.
