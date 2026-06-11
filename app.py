
import itertools
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# =========================================================
# GSÜ STYLE
# =========================================================
GS_RED = "#A70301"
GS_GOLD = "#D6B663"
BG = "#FAF8F3"

st.set_page_config(
    page_title="Sélection de prestataire IA santé | Fuzzy AHP–TOPSIS",
    page_icon="GSÜ",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; }}
    h1, h2, h3 {{ color: {GS_RED}; }}
    .main-header {{
        background: linear-gradient(90deg, {GS_RED}, #7d0201);
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        color: white;
        border-bottom: 6px solid {GS_GOLD};
        margin-bottom: 1rem;
    }}
    .main-header h1 {{ color: white; margin: 0; font-size: 2rem; }}
    .main-header p {{ color: #f4ebd3; margin: .35rem 0 0 0; }}
    .gsu-card {{
        background-color: white;
        border: 1.5px solid {GS_GOLD};
        padding: 1rem 1.1rem;
        border-radius: 16px;
        margin-bottom: .8rem;
    }}
    .metric-card {{
        background-color: #fff;
        border-left: 7px solid {GS_RED};
        padding: .8rem 1rem;
        border-radius: 12px;
        box-shadow: 0 1px 5px rgba(0,0,0,.06);
    }}
    div.stButton > button:first-child {{
        background-color: {GS_RED};
        color: white;
        border: 1px solid {GS_RED};
        border-radius: 10px;
        font-weight: 600;
    }}
    div.stButton > button:hover {{
        background-color: {GS_GOLD};
        color: {GS_RED};
        border: 1px solid {GS_GOLD};
    }}
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA MODEL
# =========================================================
CRITERIA = {
    "C1": {"name": "Compétence technique", "subs": {
        "C1.1": "Compétence en développement d’algorithmes",
        "C1.2": "Évolutivité",
        "C1.3": "Maintenance et mise à jour des modèles"}},
    "C2": {"name": "Compétences en gestion des données", "subs": {
        "C2.1": "Gestion de la qualité des données",
        "C2.2": "Intégration des données",
        "C2.3": "Confidentialité et sécurité des données"}},
    "C3": {"name": "Conformité sectorielle et clinique", "subs": {
        "C3.1": "Adéquation au flux de travail clinique",
        "C3.2": "Connaissance de la réglementation",
        "C3.3": "Projets de référence / expérience"}},
    "C4": {"name": "Coûts et facteurs économiques", "subs": {
        "C4.1": "Coût d’installation",
        "C4.2": "Coût total de possession (TCO)",
        "C4.3": "Retour sur investissement (ROI)"}},
    "C5": {"name": "Alignement stratégique", "subs": {
        "C5.1": "Alignement de la vision et des objectifs",
        "C5.2": "Culture d’entreprise",
        "C5.3": "Flexibilité et personnalisation"}},
    "C6": {"name": "Confiance, éthique et conformité", "subs": {
        "C6.1": "Explicabilité (XAI)",
        "C6.2": "Contrôle des biais",
        "C6.3": "Conformité réglementaire"}},
}
SUBS = {sk: sv for c in CRITERIA.values() for sk, sv in c["subs"].items()}
MAIN_CODES = list(CRITERIA.keys())
SUB_CODES = list(SUBS.keys())

AHP_BASE = {
    "Également important": (1, 1, 1),
    "Modérément plus important": (2, 3, 4),
    "Très plus important": (4, 5, 6),
    "Fortement plus important": (6, 7, 8),
    "Extrêmement plus important": (8, 9, 10),
}
TOPSIS_SCALE = {
    "Très faible": (1, 1, 3),
    "Faible": (1, 3, 5),
    "Moyen": (3, 5, 7),
    "Élevé": (5, 7, 9),
    "Très élevé": (7, 9, 9),
}
RI = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

# =========================================================
# FUZZY MATH FUNCTIONS
# =========================================================
def reciprocal(tfn):
    l, m, u = tfn
    return (1 / u, 1 / m, 1 / l)

def defuzzify(tfn):
    return sum(tfn) / 3

def selection_to_tfn(selection: str):
    if selection == "Également important":
        return AHP_BASE[selection]
    if selection.startswith("Premier critère"):
        degree = selection.replace("Premier critère ", "")
        return AHP_BASE[degree]
    if selection.startswith("Second critère"):
        degree = selection.replace("Second critère ", "")
        return reciprocal(AHP_BASE[degree])
    return (1, 1, 1)

def build_fuzzy_pairwise_matrix(items, prefix):
    n = len(items)
    M = np.empty((n, n, 3), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i, j] = (1, 1, 1)
    for i, j in itertools.combinations(range(n), 2):
        key = f"{prefix}_{items[i]}_{items[j]}"
        val = st.session_state.get(key, "Également important")
        tfn = selection_to_tfn(val)
        M[i, j] = tfn
        M[j, i] = reciprocal(tfn)
    return M

def buckley_weights(M):
    n = M.shape[0]
    r = np.zeros((n, 3), dtype=float)
    for i in range(n):
        r[i] = (
            np.prod(M[i, :, 0]) ** (1 / n),
            np.prod(M[i, :, 1]) ** (1 / n),
            np.prod(M[i, :, 2]) ** (1 / n),
        )
    R = np.sum(r, axis=0)
    R_inv = np.array([1 / R[2], 1 / R[1], 1 / R[0]])
    fuzzy_w = np.zeros((n, 3), dtype=float)
    for i in range(n):
        fuzzy_w[i] = (r[i, 0] * R_inv[0], r[i, 1] * R_inv[1], r[i, 2] * R_inv[2])
    crisp = np.array([defuzzify(tuple(w)) for w in fuzzy_w])
    crisp = crisp / crisp.sum()
    return fuzzy_w, crisp

def consistency_ratio(M):
    n = M.shape[0]
    if n <= 2:
        return {"lambda_max": n, "CI": 0.0, "RI": RI[n], "CR": 0.0, "consistent": True}
    A = M[:, :, 1].astype(float)
    eigvals = np.linalg.eigvals(A)
    lambda_max = float(np.max(eigvals.real))
    CI = (lambda_max - n) / (n - 1)
    random_index = RI.get(n, 1.49)
    CR = CI / random_index if random_index else 0.0
    return {"lambda_max": lambda_max, "CI": CI, "RI": random_index, "CR": CR, "consistent": CR < 0.10}

def fuzzy_topsis(performance, weights):
    alternatives = list(performance.keys())
    criteria = list(weights.keys())
    X = np.array([[performance[a][c] for c in criteria] for a in alternatives], dtype=float)
    max_u = X[:, :, 2].max(axis=0)
    R = X / max_u.reshape(1, -1, 1)
    w = np.array([weights[c] for c in criteria], dtype=float)
    V = R * w.reshape(1, -1, 1)
    FPIS = V.max(axis=0)
    FNIS = V.min(axis=0)
    def vertex_distance(a, b):
        return np.sqrt(((a - b) ** 2).sum(axis=1) / 3.0)
    d_plus, d_minus = [], []
    for i in range(len(alternatives)):
        d_plus.append(vertex_distance(V[i], FPIS).sum())
        d_minus.append(vertex_distance(V[i], FNIS).sum())
    d_plus = np.array(d_plus)
    d_minus = np.array(d_minus)
    cci = d_minus / (d_plus + d_minus + 1e-12)
    result = pd.DataFrame({"Alternative": alternatives, "D+": d_plus, "D-": d_minus, "CCi": cci})
    result = result.sort_values("CCi", ascending=False).reset_index(drop=True)
    result.index = result.index + 1
    return result, V, FPIS, FNIS

def compute_all_weights():
    M_main = build_fuzzy_pairwise_matrix(MAIN_CODES, "main")
    fuzzy_main, main_w = buckley_weights(M_main)
    cr_main = consistency_ratio(M_main)
    main_df = pd.DataFrame({
        "Code": MAIN_CODES,
        "Critère": [CRITERIA[c]["name"] for c in MAIN_CODES],
        "Poids": main_w,
        "Flou (l,m,u)": [tuple(np.round(x, 4)) for x in fuzzy_main],
    })
    local_rows, global_weights, cr_subs = [], {}, []
    for idx, c in enumerate(MAIN_CODES):
        sub_codes = list(CRITERIA[c]["subs"].keys())
        M_sub = build_fuzzy_pairwise_matrix(sub_codes, f"sub_{c}")
        fuzzy_sub, local_w = buckley_weights(M_sub)
        cr_sub = consistency_ratio(M_sub)
        cr_subs.append({"Groupe": c, **cr_sub})
        for s_idx, sc in enumerate(sub_codes):
            gw = main_w[idx] * local_w[s_idx]
            global_weights[sc] = gw
            local_rows.append({
                "Critère principal": c,
                "Sous-critère": sc,
                "Nom": CRITERIA[c]["subs"][sc],
                "Poids local": local_w[s_idx],
                "Poids global": gw,
                "Flou local (l,m,u)": tuple(np.round(fuzzy_sub[s_idx], 4)),
            })
    total = sum(global_weights.values())
    global_weights = {k: v / total for k, v in global_weights.items()}
    local_df = pd.DataFrame(local_rows)
    local_df["Poids global"] = local_df["Sous-critère"].map(global_weights)
    cr_sub_df = pd.DataFrame(cr_subs)
    return main_df, local_df, global_weights, cr_main, cr_sub_df

def oat_analysis(performance, weights, selected_criterion, variation_pct):
    factor = 1 + variation_pct / 100
    new_weights = weights.copy()
    new_weights[selected_criterion] = max(new_weights[selected_criterion] * factor, 1e-9)
    total = sum(new_weights.values())
    new_weights = {k: v / total for k, v in new_weights.items()}
    base, *_ = fuzzy_topsis(performance, weights)
    scenario, *_ = fuzzy_topsis(performance, new_weights)
    return base, scenario, new_weights

# =========================================================
# SESSION STATE
# =========================================================
if "alternatives" not in st.session_state:
    st.session_state.alternatives = ["A1", "A2", "A3"]

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="main-header">
    <h1>Système d’aide à la décision — Sélection d’un prestataire de conseil en IA</h1>
    <p>Approche hybride Fuzzy AHP–Fuzzy TOPSIS pour le secteur de la santé | Université Galatasaray</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
st.sidebar.markdown("**Méthode :** Fuzzy AHP Buckley + Fuzzy TOPSIS")
st.sidebar.markdown("**Couleurs :** thème GSÜ rouge–or")

TABS = st.tabs(["1. Accueil", "2. Alternatives", "3. Questionnaire AHP", "4. Performance TOPSIS", "5. Résultats", "6. Sensibilité OAT"])

with TABS[0]:
    st.markdown("### Objectif du prototype")
    st.markdown("""
    <div class="gsu-card">
    Ce site permet de sélectionner un prestataire de conseil en intelligence artificielle dans le secteur de la santé.
    L’utilisateur ajoute des alternatives, remplit les comparaisons linguistiques pour Fuzzy AHP, évalue la performance
    des alternatives, puis obtient un classement final par Fuzzy TOPSIS.
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown("<div class='metric-card'><b>Étape 1</b><br>Définir les alternatives</div>", unsafe_allow_html=True)
    with c2: st.markdown("<div class='metric-card'><b>Étape 2</b><br>Calculer les poids Fuzzy AHP</div>", unsafe_allow_html=True)
    with c3: st.markdown("<div class='metric-card'><b>Étape 3</b><br>Classer avec Fuzzy TOPSIS</div>", unsafe_allow_html=True)
    st.markdown("### Hiérarchie des critères")
    crit_table = []
    for c, data in CRITERIA.items():
        for sc, name in data["subs"].items():
            crit_table.append({"Critère": f"{c} — {data['name']}", "Sous-critère": f"{sc} — {name}"})
    st.dataframe(pd.DataFrame(crit_table), use_container_width=True, hide_index=True)
    st.info("Les notes de performance TOPSIS sont interprétées comme des scores de qualité/performance. Pour les critères économiques, une note élevée signifie une meilleure performance économique, par exemple un coût plus avantageux.")

with TABS[1]:
    st.markdown("### Ajouter / gérer les alternatives")
    with st.form("add_alt_form", clear_on_submit=True):
        new_alt = st.text_input("Nom de l’alternative", placeholder="Ex. A4 ou Nom du fournisseur")
        submitted = st.form_submit_button("+ Ajouter une alternative")
        if submitted:
            if new_alt.strip() and new_alt.strip() not in st.session_state.alternatives:
                st.session_state.alternatives.append(new_alt.strip())
                st.success(f"Alternative ajoutée : {new_alt.strip()}")
            else:
                st.warning("Veuillez entrer un nom unique.")
    st.markdown("#### Alternatives actuelles")
    for alt in list(st.session_state.alternatives):
        col1, col2 = st.columns([5, 1])
        col1.write(f"• {alt}")
        if col2.button("Supprimer", key=f"del_{alt}"):
            st.session_state.alternatives.remove(alt)
            st.rerun()
    if len(st.session_state.alternatives) < 2:
        st.error("Ajoutez au moins deux alternatives pour pouvoir lancer TOPSIS.")

def directional_options(left, right):
    return [
        "Également important",
        "Premier critère Modérément plus important",
        "Premier critère Très plus important",
        "Premier critère Fortement plus important",
        "Premier critère Extrêmement plus important",
        "Second critère Modérément plus important",
        "Second critère Très plus important",
        "Second critère Fortement plus important",
        "Second critère Extrêmement plus important",
    ]

with TABS[2]:
    st.markdown("### Questionnaire Fuzzy AHP")
    st.markdown("Sélectionnez des termes linguistiques pour comparer les critères deux à deux.")
    with st.expander("Comparaison des critères principaux C1–C6", expanded=True):
        for i, j in itertools.combinations(range(len(MAIN_CODES)), 2):
            left, right = MAIN_CODES[i], MAIN_CODES[j]
            label = f"{left} — {CRITERIA[left]['name']}  VS  {right} — {CRITERIA[right]['name']}"
            st.selectbox(label, directional_options(left, right), key=f"main_{left}_{right}")
    for c in MAIN_CODES:
        with st.expander(f"Comparaison des sous-critères de {c} — {CRITERIA[c]['name']}"):
            sub_codes = list(CRITERIA[c]["subs"].keys())
            for i, j in itertools.combinations(range(len(sub_codes)), 2):
                left, right = sub_codes[i], sub_codes[j]
                label = f"{left} — {CRITERIA[c]['subs'][left]}  VS  {right} — {CRITERIA[c]['subs'][right]}"
                st.selectbox(label, directional_options(left, right), key=f"sub_{c}_{left}_{right}")
    if st.button("Calculer les poids Fuzzy AHP"):
        main_df, local_df, global_weights, cr_main, cr_sub_df = compute_all_weights()
        st.session_state["main_df"] = main_df
        st.session_state["local_df"] = local_df
        st.session_state["global_weights"] = global_weights
        st.session_state["cr_main"] = cr_main
        st.session_state["cr_sub_df"] = cr_sub_df
        st.success("Poids calculés avec succès.")

with TABS[3]:
    st.markdown("### Évaluation des performances pour Fuzzy TOPSIS")
    st.markdown("Pour chaque alternative et chaque sous-critère, choisissez une évaluation linguistique.")
    if len(st.session_state.alternatives) < 2:
        st.warning("Ajoutez au moins deux alternatives avant de remplir cette section.")
    for alt in st.session_state.alternatives:
        with st.expander(f"Évaluations de {alt}", expanded=False):
            for sc in SUB_CODES:
                label = f"{sc} — {SUBS[sc]}"
                st.selectbox(label, list(TOPSIS_SCALE.keys()), index=2, key=f"perf_{alt}_{sc}")
    if st.button("Enregistrer les performances"):
        performance = {}
        for alt in st.session_state.alternatives:
            performance[alt] = {}
            for sc in SUB_CODES:
                label = st.session_state.get(f"perf_{alt}_{sc}", "Moyen")
                performance[alt][sc] = TOPSIS_SCALE[label]
        st.session_state["performance"] = performance
        st.success("Performances enregistrées.")

with TABS[4]:
    st.markdown("### Résultats")
    if "global_weights" not in st.session_state:
        st.warning("Veuillez d’abord calculer les poids Fuzzy AHP dans l’onglet 3.")
    else:
        main_df = st.session_state["main_df"]
        local_df = st.session_state["local_df"]
        cr_main = st.session_state["cr_main"]
        cr_sub_df = st.session_state["cr_sub_df"]
        global_weights = st.session_state["global_weights"]
        st.markdown("#### Poids des critères principaux")
        st.dataframe(main_df, use_container_width=True, hide_index=True)
        fig = px.bar(main_df, x="Code", y="Poids", text="Poids", color_discrete_sequence=[GS_RED])
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, title="Poids Fuzzy AHP des critères principaux")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### Poids locaux et globaux des sous-critères")
        st.dataframe(local_df, use_container_width=True, hide_index=True)
        st.markdown("#### Contrôle de cohérence")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("λmax", f"{cr_main['lambda_max']:.4f}")
        col2.metric("CI / IC", f"{cr_main['CI']:.4f}")
        col3.metric("RI", f"{cr_main['RI']:.2f}")
        col4.metric("CR", f"{cr_main['CR']:.4f}", "Cohérent" if cr_main["consistent"] else "À revoir")
        st.dataframe(cr_sub_df, use_container_width=True, hide_index=True)
    if "performance" not in st.session_state:
        st.warning("Veuillez enregistrer les performances dans l’onglet 4.")
    elif "global_weights" in st.session_state:
        if st.button("Calculer le classement Fuzzy TOPSIS"):
            result, V, FPIS, FNIS = fuzzy_topsis(st.session_state["performance"], st.session_state["global_weights"])
            st.session_state["topsis_result"] = result
            st.success("Classement calculé.")
        if "topsis_result" in st.session_state:
            result = st.session_state["topsis_result"]
            st.markdown("#### Classement final des alternatives")
            st.dataframe(result, use_container_width=True)
            fig2 = px.bar(result, x="Alternative", y="CCi", text="CCi", color_discrete_sequence=[GS_RED])
            fig2.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig2.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, title="Coefficient de proximité CCi")
            st.plotly_chart(fig2, use_container_width=True)
            best = result.iloc[0]
            st.success(f"Alternative recommandée : {best['Alternative']} avec CCi = {best['CCi']:.4f}")
            csv = result.to_csv(index=True).encode("utf-8")
            st.download_button("Télécharger les résultats TOPSIS en CSV", csv, "resultats_topsis.csv", "text/csv")

with TABS[5]:
    st.markdown("### Analyse de sensibilité OAT")
    st.markdown("Modifiez le poids d’un seul sous-critère et observez l’effet sur le classement.")
    if "global_weights" not in st.session_state or "performance" not in st.session_state:
        st.warning("Veuillez d’abord calculer les poids AHP et enregistrer les performances.")
    else:
        selected = st.selectbox("Sous-critère à modifier", SUB_CODES, format_func=lambda x: f"{x} — {SUBS[x]}")
        variation = st.slider("Variation du poids", min_value=-20, max_value=20, value=10, step=5)
        if st.button("Lancer l’analyse OAT"):
            base, scenario, new_weights = oat_analysis(st.session_state["performance"], st.session_state["global_weights"], selected, variation)
            st.markdown("#### Classement initial")
            st.dataframe(base, use_container_width=True)
            st.markdown(f"#### Classement après variation de {variation}% sur {selected}")
            st.dataframe(scenario, use_container_width=True)
            compare = base[["Alternative", "CCi"]].rename(columns={"CCi": "CCi initial"}).merge(
                scenario[["Alternative", "CCi"]].rename(columns={"CCi": "CCi scénario"}), on="Alternative")
            compare["Variation CCi"] = compare["CCi scénario"] - compare["CCi initial"]
            st.markdown("#### Comparaison")
            st.dataframe(compare, use_container_width=True, hide_index=True)
            fig3 = px.bar(compare, x="Alternative", y="Variation CCi", color_discrete_sequence=[GS_GOLD])
            fig3.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, title="Impact de la variation OAT sur CCi")
            st.plotly_chart(fig3, use_container_width=True)

st.caption("Prototype académique — Fuzzy AHP Buckley + Fuzzy TOPSIS | Université Galatasaray")
