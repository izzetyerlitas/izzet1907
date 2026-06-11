
# Système d’aide à la décision — Sélection d’un prestataire IA santé

Prototype Streamlit pour un projet de fin d’études : sélection d’un prestataire de conseil en intelligence artificielle dans le secteur de la santé avec une approche hybride **Fuzzy AHP Buckley + Fuzzy TOPSIS**.

## Fonctionnalités

- Ajout/suppression d’alternatives
- Questionnaire Fuzzy AHP pour critères principaux et sous-critères
- Calcul des poids par la méthode de Buckley
- Défuzzification par centroïde et normalisation
- Contrôle de cohérence CR
- Évaluation linguistique des performances des alternatives
- Classement Fuzzy TOPSIS avec D+, D− et CCi
- Analyse de sensibilité OAT ±20%
- Thème visuel rouge–or inspiré de l’Université Galatasaray

## Installation locale

```bash
python -m venv venv
```

Windows :

```bash
venv\Scripts\activate
```

Mac/Linux :

```bash
source venv/bin/activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Lancer le site :

```bash
streamlit run app.py
```

Puis ouvrir :

```text
http://localhost:8501
```

## Déploiement Streamlit Community Cloud

1. Créer un repository GitHub.
2. Ajouter `app.py`, `requirements.txt` et `README.md`.
3. Aller sur Streamlit Community Cloud.
4. Connecter le repo GitHub.
5. Sélectionner `app.py` comme fichier principal.
6. Cliquer sur Deploy.

## Note méthodologique

Les notes de performance TOPSIS sont interprétées comme des scores de qualité/performance. Pour les critères économiques, une note élevée signifie donc une meilleure performance économique, par exemple un coût plus avantageux.
