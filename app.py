import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)
from xgboost import XGBClassifier
import io
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, silhouette_score

# ══════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="Fidélisation Client — Télécom",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════
#  DESIGN FUTURISTE — CSS GLOBAL
# ══════════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main { background: #07080f; }
  .block-container { padding: 2rem 2.5rem; }

  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0f1e 0%, #111328 100%);
    border-right: 1px solid rgba(99,102,241,0.2);
  }
  [data-testid="stSidebar"] * { color: #c7d2fe !important; }
  [data-testid="stSidebar"] hr { border-color: rgba(99,102,241,0.3); }

  .page-header {
    font-size: 2.2rem; font-weight: 900; letter-spacing: -0.5px;
    background: linear-gradient(90deg, #818cf8, #38bdf8, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
  }
  .page-sub { color: #64748b; font-size: 1rem; margin-bottom: 1.5rem; }

  .glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px; padding: 22px 26px;
    backdrop-filter: blur(12px);
    transition: transform .2s, box-shadow .2s;
    text-align: center;
  }
  .glass-card:hover { transform: translateY(-3px); box-shadow: 0 8px 32px rgba(99,102,241,.2); }
  .glass-card .val { font-size: 2.4rem; font-weight: 800; line-height: 1.1; }
  .glass-card .lbl { color: #94a3b8; font-size: 0.85rem; margin-top: 4px; }
  .glass-card.blue  .val { color: #818cf8; }
  .glass-card.red   .val { color: #f87171; }
  .glass-card.green .val { color: #34d399; }
  .glass-card.amber .val { color: #fbbf24; }

  .sec-title {
    font-size: 1.1rem; font-weight: 700; color: #e2e8f0;
    border-left: 4px solid #818cf8; padding-left: 10px;
    margin: 20px 0 14px 0;
  }

  .rec-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 14px 18px; margin-bottom: 10px;
  }
  .rec-card b { color: #e2e8f0; }
  .rec-card span { color: #94a3b8; font-size: 0.9rem; }

  .badge { display: inline-block; border-radius: 20px; padding: 3px 12px; font-size: .78rem; font-weight: 600; margin-left: 6px; }
  .badge-red   { background: rgba(239,68,68,.15);  color: #f87171; }
  .badge-amber { background: rgba(251,191,36,.15); color: #fbbf24; }
  .badge-green { background: rgba(52,211,153,.15); color: #34d399; }
  .badge-blue  { background: rgba(129,140,248,.15); color: #818cf8; }

  .alert-ok   { background: rgba(52,211,153,.1);  border: 1px solid rgba(52,211,153,.3);  border-radius: 10px; padding: 12px 16px; color: #34d399; margin-bottom: 14px; }
  .alert-err  { background: rgba(239,68,68,.1);   border: 1px solid rgba(239,68,68,.3);   border-radius: 10px; padding: 12px 16px; color: #f87171; margin-bottom: 14px; }
  .alert-warn { background: rgba(251,191,36,.1);  border: 1px solid rgba(251,191,36,.3);  border-radius: 10px; padding: 12px 16px; color: #fbbf24; margin-bottom: 14px; }
  .alert-info { background: rgba(56,189,248,.08); border: 1px solid rgba(56,189,248,.25); border-radius: 10px; padding: 12px 16px; color: #7dd3fc; margin-bottom: 14px; }

  [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
  button[data-baseweb="tab"] { color: #94a3b8 !important; }
  button[data-baseweb="tab"][aria-selected="true"] { color: #818cf8 !important; border-bottom-color: #818cf8 !important; }
  [data-testid="stFileUploader"] { border: 2px dashed rgba(129,140,248,.4) !important; border-radius: 14px !important; background: rgba(129,140,248,.04) !important; }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0d0f1e; }
  ::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  COLONNES OBLIGATOIRES
# ══════════════════════════════════════════════
REQUIRED_COLS = {
    'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn', 'Cluster',
    'Contract', 'PaymentMethod', 'InternetService', 'MultipleLines',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'SeniorCitizen', 'PhoneService',
    'PaperlessBilling', 'gender', 'Partner', 'Dependents',
}

FRIENDLY_NAMES = {
    'tenure': 'Ancienneté du client',
    'MonthlyCharges': 'Montant mensuel',
    'TotalCharges': 'Total dépensé',
    'Churn': 'Résiliation',
    'Cluster': 'Groupe de clients',
    'Contract': 'Type de contrat',
    'PaymentMethod': 'Moyen de paiement',
    'InternetService': 'Type de connexion internet',
    'MultipleLines': 'Plusieurs lignes téléphoniques',
    'OnlineSecurity': 'Sécurité en ligne',
    'OnlineBackup': 'Sauvegarde en ligne',
    'DeviceProtection': 'Protection des appareils',
    'TechSupport': 'Support technique',
    'StreamingTV': 'Streaming TV',
    'StreamingMovies': 'Streaming Films',
    'SeniorCitizen': 'Senior',
    'PhoneService': 'Téléphonie',
    'PaperlessBilling': 'Facture numérique',
    'gender': 'Sexe',
    'Partner': 'En couple',
    'Dependents': 'Avec famille à charge',
}

def validate_file(df: pd.DataFrame):
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        return False, sorted(missing)
    return True, []

# ══════════════════════════════════════════════
#  PRÉPARATION & CACHE
# ══════════════════════════════════════════════
@st.cache_data
def prepare_data(df_raw: pd.DataFrame):
    df = df_raw.copy()
    df['charge_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    df['is_high_value'] = (df['MonthlyCharges'] >= df['MonthlyCharges'].quantile(0.75)).astype(int)
    service_raw = ['PhoneService', 'PaperlessBilling', 'SeniorCitizen']
    service_cat = ['MultipleLines','OnlineSecurity','OnlineBackup',
                   'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    df['service_count'] = (df[service_raw].sum(axis=1) +
                           df[service_cat].apply(lambda c: (c == 'Yes').astype(int)).sum(axis=1))
    return df

@st.cache_resource
def train_models(_df):
    multi_cat = ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
                 'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
                 'Contract','PaymentMethod']
    df_model = pd.get_dummies(_df, columns=multi_cat, drop_first=True)
    X = df_model.drop(columns=['Churn'])
    y = df_model['Churn']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_tes = scaler.transform(X_te)

    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(X_trs, y_tr)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1)
    rf.fit(X_tr, y_tr)

    spw = (y_tr == 0).sum() / (y_tr == 1).sum()
    gs = GridSearchCV(
        XGBClassifier(scale_pos_weight=spw, random_state=42, eval_metric='logloss', verbosity=0),
        {'n_estimators':[100,200], 'max_depth':[3,5], 'learning_rate':[0.05,0.1]},
        cv=3, scoring='roc_auc', n_jobs=-1)
    gs.fit(X_tr, y_tr)
    xgb = gs.best_estimator_

    results = {}
    fn = X.columns.tolist()
    for name, model, Xeval in [
        ('Approche simple',         lr,  X_tes),
        ('Forêt de décisions',      rf,  X_te),
        ('Modèle optimisé',         xgb, X_te),
    ]:
        pred  = model.predict(Xeval)
        proba = model.predict_proba(Xeval)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, proba)
        results[name] = dict(
            pred=pred, proba=proba,
            acc=accuracy_score(y_te,pred), prec=precision_score(y_te,pred),
            rec=recall_score(y_te,pred),   f1=f1_score(y_te,pred),
            auc=roc_auc_score(y_te,proba), cm=confusion_matrix(y_te,pred),
            fpr=fpr, tpr=tpr,
            fi=model.feature_importances_ if hasattr(model,'feature_importances_') else None,
        )

    df_sc = pd.get_dummies(_df, columns=multi_cat, drop_first=True)
    all_proba = xgb.predict_proba(df_sc.drop(columns=['Churn']))[:, 1]
    return results, y_te, fn, all_proba, gs.best_params_

# ══════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════
if 'df_raw' not in st.session_state:
    st.session_state['df_raw'] = None
if 'data_source' not in st.session_state:
    st.session_state['data_source'] = None
if 's3_df' not in st.session_state:
    st.session_state['s3_df'] = None
if 's3_k' not in st.session_state:
    st.session_state['s3_k'] = 4
if 's2_df' not in st.session_state:
    st.session_state['s2_df'] = None
if 's2_labels' not in st.session_state:
    st.session_state['s2_labels'] = None
if 's2_scaler' not in st.session_state:
    st.session_state['s2_scaler'] = None

# ══════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.sidebar.markdown("---")
    sujet_actif = st.sidebar.radio(
        "📚 Choisir un sujet",
        options=["📱 Sujet 1 — Télécom Churn", 
                 "🛍️ Sujet 2 — Segmentation Mall",
                 "🎓 Sujet 3 — Réussite Étudiante"],
        key="sujet_selector"
    )
    st.sidebar.markdown("---")
    if sujet_actif == "📱 Sujet 1 — Télécom Churn":
        st.markdown("## 🔮 Fidélisation Client")
        st.markdown("*Plateforme d'analyse des départs*")
        st.markdown("---")
        page = st.radio("Navigation", [
            "📂 Importer mes données",
            "🏠 Tableau de bord",
            "🔍 Explorer les données",
            "👥 Mes groupes de clients",
            "🤖 Prédire les départs",
            "📊 Comparer les résultats",
            "💡 Que faire maintenant ?",
        ], label_visibility="collapsed")
        st.markdown("---")
        if st.session_state['df_raw'] is not None:
            d = st.session_state['df_raw']
            st.markdown("**Fichier chargé ✅**")
            st.caption(f"{len(d):,} clients · {d.shape[1]} colonnes")
            st.caption(f"Source : {st.session_state['data_source']}")
            if st.button("🗑️ Changer de fichier"):
                st.session_state['df_raw'] = None
                st.session_state['data_source'] = None
                st.rerun()
        else:
            st.markdown('<div class="alert-warn">⚠️ Aucune donnée chargée</div>', unsafe_allow_html=True)

    elif sujet_actif == "🛍️ Sujet 2 — Segmentation Mall":
        st.markdown("## 🛍️ Segmentation Mall")
        st.markdown("*Analyse et ciblage clients*")
        st.markdown("---")
        page_s2 = st.radio("Navigation", [
            "📂 Importer mes données",
            "🔍 Explorer les données",
            "👥 Segmentation clients",
            "💡 Stratégie marketing",
        ], label_visibility="collapsed", key="s2_nav")
        st.markdown("---")
        if st.session_state.get('s2_df') is not None:
            d_s2 = st.session_state['s2_df']
            st.markdown("**Fichier chargé ✅**")
            st.caption(f"{len(d_s2):,} clients · {d_s2.shape[1]} colonnes")
            if st.button("🗑️ Changer de fichier", key="s2_clear"):
                st.session_state['s2_df'] = None
                st.session_state['s2_labels'] = None
                st.session_state['s2_scaler'] = None
                st.rerun()
        else:
            st.markdown('<div class="alert-warn">⚠️ Aucune donnée chargée</div>', unsafe_allow_html=True)

    elif sujet_actif == "🎓 Sujet 3 — Réussite Étudiante":
        st.markdown("## 🎓 Réussite Étudiante")
        st.markdown("*Analyse des performances étudiantes*")
        st.markdown("---")
        page_s3 = st.radio("Navigation", [
            "📂 Importer mes données",
            "🔍 Explorer les données",
            "🔑 Facteurs de réussite",
            "👥 Profils étudiants (Clustering)",
            "🤖 Prédire la note finale",
            "💡 Recommandations pédagogiques",
        ], label_visibility="collapsed", key="s3_nav")
        st.markdown("---")
        if st.session_state.get('s3_df') is not None:
            d_s3 = st.session_state['s3_df']
            st.markdown("**Fichier chargé ✅**")
            st.caption(f"{len(d_s3):,} étudiants · {d_s3.shape[1]} colonnes")
            if st.button("🗑️ Changer de fichier", key="s3_clear"):
                st.session_state['s3_df'] = None
                st.rerun()
        else:
            st.markdown('<div class="alert-warn">⚠️ Aucune donnée chargée</div>', unsafe_allow_html=True)


def require_data():
    if st.session_state['df_raw'] is None:
        st.markdown('<div class="alert-warn">⚠️ Veuillez d\'abord importer vos données dans la section <b>📂 Importer mes données</b>.</div>', unsafe_allow_html=True)
        st.stop()

PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font_color='#e2e8f0', margin=dict(t=40, b=20),
)
GRID = dict(gridcolor='rgba(255,255,255,0.06)')
COLORS = ['#818cf8','#f97316','#8b5cf6']

# ══════════════════════════════════════════════
#  SUJET 2 — FONCTIONS
# ══════════════════════════════════════════════

S2_REQUIRED_COLS = {'CustomerID', 'Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'}

def validate_file_s2(df: pd.DataFrame):
    missing = S2_REQUIRED_COLS - set(df.columns)
    if missing:
        return False, sorted(missing)
    return True, []

def require_data_s2():
    if st.session_state.get('s2_df') is None:
        st.markdown('<div class="alert-warn">⚠️ Veuillez d\'abord importer vos données dans la section <b>📂 Importer mes données</b>.</div>', unsafe_allow_html=True)
        st.stop()

def generate_demo_s2():
    rng = np.random.default_rng(42)
    n = 200
    # 5 natural clusters: (income_mean, income_std, score_mean, score_std, count)
    clusters_def = [
        (25, 7, 20, 8, 40),   # Low income + Low spending
        (25, 7, 78, 9, 40),   # Low income + High spending
        (55, 8, 50, 9, 40),   # Average income + Average spending
        (90, 8, 18, 8, 40),   # High income + Low spending
        (90, 8, 82, 8, 40),   # High income + High spending
    ]
    incomes, scores, ages, genres = [], [], [], []
    for inc_m, inc_s, sc_m, sc_s, cnt in clusters_def:
        incomes.append(np.clip(rng.normal(inc_m, inc_s, cnt), 15, 137))
        scores.append(np.clip(rng.normal(sc_m, sc_s, cnt), 1, 99))
        ages.append(np.clip(rng.normal(38, 13, cnt), 18, 70))
        # 56% female overall
        genres.append(rng.choice(['Female', 'Male'], size=cnt, p=[0.56, 0.44]))
    income_arr = np.concatenate(incomes).round(0).astype(int)
    score_arr  = np.concatenate(scores).round(0).astype(int)
    age_arr    = np.concatenate(ages).round(0).astype(int)
    genre_arr  = np.concatenate(genres)
    idx = rng.permutation(n)
    return pd.DataFrame({
        'CustomerID': range(1, n + 1),
        'Genre': genre_arr[idx],
        'Age': age_arr[idx],
        'Annual Income (k$)': income_arr[idx],
        'Spending Score (1-100)': score_arr[idx],
    })

def name_cluster_s2(income_mean, score_mean):
    high_inc = income_mean > 70
    low_inc  = income_mean < 40
    high_sc  = score_mean > 60
    low_sc   = score_mean < 40
    if high_inc and high_sc:
        return "🟢 Cibles Premium"
    if high_inc and low_sc:
        return "🔴 Riches Economes"
    if low_inc and high_sc:
        return "🟡 Depensiers Limites"
    if low_inc and low_sc:
        return "⚫ Economes Modestes"
    return "🔵 Classe Moyenne"

PRIORITY_S2 = {
    "🟢 Cibles Premium":    ("HAUTE",   "badge-green"),
    "🔴 Riches Economes":   ("HAUTE",   "badge-red"),
    "🟡 Depensiers Limites":("MOYENNE", "badge-amber"),
    "⚫ Economes Modestes":  ("BASSE",   "badge-blue"),
    "🔵 Classe Moyenne":    ("MOYENNE", "badge-blue"),
}

RECO_S2 = {
    "🟢 Cibles Premium": (
        "Programme de fidelite VIP, invitations evenements exclusifs, early access nouveaux produits. "
        "Ces clients generent le plus de valeur — les choyer est prioritaire."
    ),
    "🔴 Riches Economes": (
        "Campagnes de reactivation ciblees, offres premium sur-mesure, demonstrations produits haut de gamme. "
        "Fort potentiel inexploite — comprendre leurs freins a l'achat."
    ),
    "🟡 Depensiers Limites": (
        "Promotions, programmes de points, facilites de paiement. "
        "Ces clients aiment depenser — les aider a le faire sans se mettre en difficulte financiere."
    ),
    "⚫ Economes Modestes": (
        "Offres d'entree de gamme, soldes et promotions saisonnieres. "
        "Ne pas investir trop de ressources marketing sur ce segment."
    ),
    "🔵 Classe Moyenne": (
        "Fidelisation classique, newsletters, offres saisonnieres. "
        "Segment stable — maintenir l'engagement sans sur-investir."
    ),
}

@st.cache_data
def compute_elbow_s2(df: pd.DataFrame):
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
    Xs = StandardScaler().fit_transform(X)
    inertias, sils = [], []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labs = km.fit_predict(Xs)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Xs, labs))
    return list(range(2, 11)), inertias, sils

@st.cache_data
def run_kmeans_s2(df: pd.DataFrame, k: int):
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(Xs)
    return labels, sc, km.cluster_centers_

# ══════════════════════════════════════════════
#  SUJET 3 — FONCTIONS
# ══════════════════════════════════════════════

S3_REQUIRED_COLS = {
    'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
    'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures',
    'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
    'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences',
    'G1', 'G2', 'G3'
}

def validate_file_s3(df: pd.DataFrame):
    missing = S3_REQUIRED_COLS - set(df.columns)
    if missing:
        return False, sorted(missing)
    return True, []

def require_data_s3():
    if st.session_state.get('s3_df') is None:
        st.markdown('<div class="alert-warn">⚠️ Veuillez d\'abord importer vos données dans la section <b>📂 Importer mes données</b>.</div>', unsafe_allow_html=True)
        st.stop()

def label_cluster_s3(g3_mean):
    if g3_mean >= 15:
        return "🟢 Excellents"
    elif g3_mean >= 12:
        return "🔵 Bons élèves"
    elif g3_mean >= 10:
        return "🟡 Fragiles"
    else:
        return "🔴 En difficulté"

@st.cache_data
def preprocess_s3(df_raw: pd.DataFrame):
    df = df_raw.copy()
    for col, mapping in {'sex': {'M': 1, 'F': 0}, 'address': {'U': 1, 'R': 0},
                          'famsize': {'GT3': 1, 'LE3': 0}, 'Pstatus': {'T': 1, 'A': 0}}.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    for col in ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']:
        if col in df.columns:
            df[col] = (df[col] == 'yes').astype(int)
    cat_cols = [c for c in ['school', 'Mjob', 'Fjob', 'reason', 'guardian'] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

@st.cache_data
def compute_elbow_s3(df: pd.DataFrame):
    feats = [c for c in ['studytime', 'failures', 'absences', 'Dalc', 'Walc', 'G1', 'G2'] if c in df.columns]
    X = df[feats].fillna(df[feats].median())
    Xs = StandardScaler().fit_transform(X)
    ks_list = list(range(2, 9))
    inertias_list, sils_list = [], []
    for k in ks_list:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labs = km.fit_predict(Xs)
        inertias_list.append(km.inertia_)
        sils_list.append(silhouette_score(Xs, labs))
    return ks_list, inertias_list, sils_list

@st.cache_data
def cluster_students_s3(df: pd.DataFrame, k: int):
    feats = [c for c in ['studytime', 'failures', 'absences', 'Dalc', 'Walc', 'G1', 'G2'] if c in df.columns]
    X = df[feats].fillna(df[feats].median())
    Xs = StandardScaler().fit_transform(X)
    return KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xs)

@st.cache_resource
def get_feature_importance_s3(_df):
    df_proc = preprocess_s3(_df)
    drop_cols = [c for c in ['G3', 'G1', 'G2'] if c in df_proc.columns]
    X = df_proc.drop(columns=drop_cols)
    y = _df['G3']
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False).reset_index(drop=True)

@st.cache_resource
def train_models_s3(_df):
    df_proc = preprocess_s3(_df)
    drop_cols = [c for c in ['G3', 'G1', 'G2'] if c in df_proc.columns]
    X = df_proc.drop(columns=drop_cols)
    y = _df['G3'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    models_def = {
        'Régression Linéaire': LinearRegression(),
        'Forêt Aléatoire': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    res_s3 = {}
    for name, model in models_def.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        res_s3[name] = {
            'model': model, 'pred': pred, 'actual': y_te,
            'rmse': float(np.sqrt(mean_squared_error(y_te, pred))),
            'mae': float(mean_absolute_error(y_te, pred)),
            'r2': float(r2_score(y_te, pred)),
        }
    best_s3 = max(res_s3, key=lambda x: res_s3[x]['r2'])
    all_pred_s3 = res_s3[best_s3]['model'].predict(X)
    return res_s3, best_s3, all_pred_s3, X.columns.tolist()

# ══════════════════════════════════════════════
#  PAGE 1 — IMPORTER LES DONNÉES
# ══════════════════════════════════════════════
if sujet_actif == "📱 Sujet 1 — Télécom Churn":
    if page == "📂 Importer mes données":
        st.markdown('<div class="page-header">📂 Importer vos données clients</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-sub">Chargez votre fichier pour démarrer l\'analyse. Nous vérifions automatiquement la compatibilité.</div>', unsafe_allow_html=True)

        col_upload, col_demo = st.columns([3, 2], gap="large")

        with col_upload:
            st.markdown('<div class="sec-title">Charger votre propre fichier</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Glissez-déposez votre fichier CSV ici, ou cliquez pour parcourir",
                type=['csv'],
            )
            if uploaded is not None:
                try:
                    df_up = pd.read_csv(uploaded)
                    ok, missing = validate_file(df_up)
                    if ok:
                        st.markdown(f'<div class="alert-ok">✅ <b>Fichier compatible !</b> {len(df_up):,} clients détectés avec toutes les informations nécessaires.</div>', unsafe_allow_html=True)
                        st.session_state['df_raw'] = df_up
                        st.session_state['data_source'] = uploaded.name
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown(f'<div class="glass-card blue"><div class="val">{len(df_up):,}</div><div class="lbl">Clients dans le fichier</div></div>', unsafe_allow_html=True)
                        with c2:
                            st.markdown(f'<div class="glass-card red"><div class="val">{df_up["Churn"].mean()*100:.1f}%</div><div class="lbl">Ont résilié</div></div>', unsafe_allow_html=True)
                        with c3:
                            st.markdown(f'<div class="glass-card amber"><div class="val">{df_up["Cluster"].nunique()}</div><div class="lbl">Groupes identifiés</div></div>', unsafe_allow_html=True)
                        st.markdown("<br>**Aperçu :**")
                        st.dataframe(df_up.head(5), use_container_width=True)
                    else:
                        friendly_missing = [FRIENDLY_NAMES.get(c, c) for c in missing]
                        badges = ''.join(f'<code style="background:rgba(239,68,68,.15);color:#fca5a5;border-radius:4px;padding:2px 8px;margin:3px;display:inline-block">{n}</code>' for n in friendly_missing)
                        st.markdown(f"""
                        <div class="alert-err">
                            ❌ <b>Ce fichier n'est pas compatible.</b><br><br>
                            Il manque les informations suivantes :<br><br>
                            {badges}
                            <br><br>
                            <b>Que faire ?</b> Assurez-vous que votre fichier provient bien du pipeline complet d'analyse qui inclut la segmentation des clients en groupes (colonne <code>Cluster</code>).
                        </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="alert-err">❌ Impossible de lire le fichier. Vérifiez qu\'il s\'agit bien d\'un fichier CSV valide.<br><small>{e}</small></div>', unsafe_allow_html=True)

        with col_demo:
            st.markdown('<div class="sec-title">Utiliser les données de démonstration</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="glass-card" style="text-align:left">
                <b style="color:#e2e8f0">Fichier inclus</b><br><br>
                <span style="color:#94a3b8">Données réelles d'un opérateur télécom :</span><br><br>
                📊 7 043 clients analysés<br>
                🏷️ 21 informations par client<br>
                👥 3 groupes de clients identifiés<br>
                ✅ Statut de résiliation connu<br><br>
                <span style="color:#34d399">✅ Fichier validé et prêt à l'emploi</span>
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Utiliser les données de démonstration", use_container_width=True):
                try:
                    df_demo = pd.read_csv('telco_with_clusters.csv')
                    ok, missing = validate_file(df_demo)
                    if ok:
                        st.session_state['df_raw'] = df_demo
                        st.session_state['data_source'] = 'Données de démonstration'
                        st.markdown('<div class="alert-ok">✅ Chargé ! Allez sur le <b>Tableau de bord</b> pour commencer.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="alert-err">❌ Fichier de démonstration incomplet. Vérifiez que <code>telco_with_clusters.csv</code> est présent dans le même dossier.</div>', unsafe_allow_html=True)
                except FileNotFoundError:
                    st.markdown('<div class="alert-err">❌ Fichier <code>telco_with_clusters.csv</code> introuvable. Placez-le dans le même dossier que l\'application.</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="sec-title">Quelles informations sont attendues dans le fichier ?</div>', unsafe_allow_html=True)
        col_i1, col_i2, col_i3 = st.columns(3)
        with col_i1:
            st.markdown("""<div class="glass-card" style="text-align:left">
                <b style="color:#818cf8">👤 Profil du client</b><br><br>
                <span style="color:#94a3b8">• Sexe, senior ou non<br>• Situation familiale<br>• Ancienneté (en mois)<br>• Groupe de clients (Cluster)</span>
            </div>""", unsafe_allow_html=True)
        with col_i2:
            st.markdown("""<div class="glass-card" style="text-align:left">
                <b style="color:#38bdf8">📦 Services souscrits</b><br><br>
                <span style="color:#94a3b8">• Type de connexion internet<br>• Téléphonie et options<br>• Streaming, sécurité en ligne<br>• Support technique</span>
            </div>""", unsafe_allow_html=True)
        with col_i3:
            st.markdown("""<div class="glass-card" style="text-align:left">
                <b style="color:#34d399">💳 Contrat & Facturation</b><br><br>
                <span style="color:#94a3b8">• Type de contrat<br>• Mode de paiement<br>• Montant mensuel et total<br>• Statut de résiliation</span>
            </div>""", unsafe_allow_html=True)


    # ══════════════════════════════════════════════
    #  PAGE 2 — TABLEAU DE BORD
    # ══════════════════════════════════════════════
    elif page == "🏠 Tableau de bord":
        require_data()
        df = prepare_data(st.session_state['df_raw'])
        st.markdown('<div class="page-header">🏠 Tableau de Bord</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-sub">Vue globale de la situation — d\'un seul coup d\'œil.</div>', unsafe_allow_html=True)

        total   = len(df)
        churned = int(df['Churn'].sum())
        active  = total - churned
        pct     = churned / total * 100

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f'<div class="glass-card blue"><div class="val">{total:,}</div><div class="lbl">Clients au total</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="glass-card red"><div class="val">{churned:,}</div><div class="lbl">Ont quitté l\'opérateur</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="glass-card green"><div class="val">{active:,}</div><div class="lbl">Clients encore actifs</div></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="glass-card amber"><div class="val">{pct:.1f}%</div><div class="lbl">Taux de départ global</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if pct > 30:
            st.markdown(f'<div class="alert-err">🚨 <b>Situation préoccupante :</b> plus d\'1 client sur 3 a résilié ({pct:.1f}%). Une action rapide est nécessaire, notamment sur les contrats courts et la qualité du service perçue.</div>', unsafe_allow_html=True)
        elif pct > 20:
            st.markdown(f'<div class="alert-warn">⚠️ <b>À surveiller :</b> {pct:.1f}% de départs, soit {churned:,} clients perdus. Des campagnes de fidélisation ciblées peuvent faire la différence.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-ok">✅ <b>Bonne maîtrise :</b> le taux de départ est de {pct:.1f}%. Continuez les efforts de fidélisation pour maintenir ce niveau.</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="sec-title">Actifs vs. Résiliés</div>', unsafe_allow_html=True)
            fig = px.pie(values=[active, churned], names=['Clients actifs','Ont résilié'],
                         color_discrete_sequence=['#34d399','#f87171'], hole=0.55)
            fig.update_traces(textposition='outside', textinfo='percent+label', textfont_size=13)
            fig.update_layout(**PLOT_LAYOUT, showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown('<div class="sec-title">Taux de départ par groupe de clients</div>', unsafe_allow_html=True)
            cc = df.groupby('Cluster')['Churn'].mean().mul(100).reset_index()
            cc['color'] = cc['Churn'].apply(lambda x: '#f87171' if x>35 else '#fbbf24' if x>25 else '#34d399')
            cc['label'] = cc['Cluster'].apply(lambda x: f'Groupe {x}')
            fig2 = go.Figure()
            for _, row in cc.iterrows():
                fig2.add_trace(go.Bar(x=[row['label']], y=[row['Churn']], marker_color=row['color'],
                                       text=f"{row['Churn']:.1f}%", textposition='outside',
                                       name=row['label'], showlegend=False))
            fig2.update_layout(**PLOT_LAYOUT, height=300,
                                yaxis=dict(title='% de départs', range=[0,60], **GRID), xaxis=dict(title=''))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="sec-title">Type de contrat et résiliations</div>', unsafe_allow_html=True)
        cc2 = df.groupby('Contract').agg(Nb=('Churn','count'), Taux=('Churn', lambda x: round(x.mean()*100,1))).reset_index().sort_values('Taux', ascending=False)
        fig3 = px.bar(cc2, x='Contract', y='Taux', color='Taux', text='Taux',
                      color_continuous_scale='RdYlGn_r', custom_data=['Nb'],
                      labels={'Contract':'Type de contrat','Taux':'Taux de départ (%)'},
                      title='')
        fig3.update_traces(texttemplate='%{text}%', textposition='outside',
                            hovertemplate='<b>%{x}</b><br>Taux de départ : %{y}%<br>Clients : %{customdata[0]:,}<extra></extra>')
        fig3.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, height=320,
                            yaxis=dict(range=[0,65], **GRID))
        st.plotly_chart(fig3, use_container_width=True)

        worst, best = cc2.iloc[0], cc2.iloc[-1]
        st.markdown(f'<div class="alert-info">💡 <b>Ce que cela signifie :</b> les clients avec un contrat <b>{worst["Contract"]}</b> partent {worst["Taux"]:.0f}% du temps, contre seulement {best["Taux"]:.0f}% pour ceux avec un contrat <b>{best["Contract"]}</b>. Encourager la migration vers des contrats longue durée est l\'un des leviers les plus efficaces.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE 3 — EXPLORER LES DONNÉES
# ══════════════════════════════════════════════
    elif page == "🔍 Explorer les données":
        require_data()
        df = prepare_data(st.session_state['df_raw'])
        st.markdown('<div class="page-header">🔍 Explorer les Données</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-sub">Analysez chaque dimension pour comprendre ce qui pousse un client à partir.</div>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📈 Chiffres & Montants", "🏷️ Comportements & Services", "🔗 Liens entre facteurs"])

        NUM_OPTS = {'tenure':'Ancienneté (mois)','MonthlyCharges':'Montant mensuel ($)','TotalCharges':'Total dépensé ($)',
                    'charge_per_tenure':'Dépense mensuelle relative','service_count':'Nombre de services activés'}
        CAT_OPTS = {'Contract':'Type de contrat','PaymentMethod':'Moyen de paiement',
                    'InternetService':'Type de connexion internet','TechSupport':'Assistance technique',
                    'OnlineSecurity':'Sécurité en ligne','MultipleLines':'Plusieurs lignes',
                    'StreamingTV':'Streaming TV','StreamingMovies':'Streaming Films'}

        with tab1:
            var = st.selectbox("Quelle information analyser ?", list(NUM_OPTS.keys()), format_func=lambda x: NUM_OPTS[x])
            c_l, c_r = st.columns(2)
            churn_map = df['Churn'].map({0:'Client actif',1:'A résilié'})
            cmap = {'Client actif':'#34d399','A résilié':'#f87171'}
            with c_l:
                fig = px.histogram(df, x=var, color=churn_map, barmode='overlay', opacity=0.75,
                                   nbins=40, color_discrete_map=cmap,
                                   labels={'color':'Statut', var: NUM_OPTS[var]},
                                   title=f'Répartition : {NUM_OPTS[var]}')
                fig.update_layout(**PLOT_LAYOUT, height=350, yaxis=GRID, legend_title='Statut')
                st.plotly_chart(fig, use_container_width=True)
            with c_r:
                fig2 = px.box(df, x=churn_map, y=var, color=churn_map, color_discrete_map=cmap,
                              labels={'x':'Statut','color':'Statut', var: NUM_OPTS[var]},
                              title=f'Comparaison : {NUM_OPTS[var]}')
                fig2.update_layout(**PLOT_LAYOUT, height=350, yaxis=GRID)
                st.plotly_chart(fig2, use_container_width=True)

            m0 = df[df['Churn']==0][var].mean()
            m1 = df[df['Churn']==1][var].mean()
            direction = "plus élevée" if m1 > m0 else "plus basse"
            st.markdown(f'<div class="alert-info">💡 <b>En clair :</b> les clients qui ont résilié avaient en moyenne <b>{NUM_OPTS[var]}</b> à <b>{m1:.1f}</b>, contre <b>{m0:.1f}</b> pour les clients fidèles. La valeur est <b>{direction}</b> chez ceux qui partent.</div>', unsafe_allow_html=True)

        with tab2:
            var_c = st.selectbox("Quelle caractéristique analyser ?", list(CAT_OPTS.keys()), format_func=lambda x: CAT_OPTS[x])
            cr = df.groupby(var_c)['Churn'].mean().mul(100).reset_index()
            cr.columns = [var_c,'Taux (%)']
            cr = cr.sort_values('Taux (%)', ascending=False)
            c_l2, c_r2 = st.columns(2)
            with c_l2:
                fig3 = px.bar(cr, x=var_c, y='Taux (%)', color='Taux (%)',
                              color_continuous_scale='RdYlGn_r',
                              text=cr['Taux (%)'].round(1).astype(str)+'%',
                              title=f'Taux de départ selon {CAT_OPTS[var_c]}')
                fig3.update_traces(textposition='outside')
                fig3.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, xaxis_tickangle=-15,
                                    height=360, yaxis=dict(range=[0,70], **GRID))
                st.plotly_chart(fig3, use_container_width=True)
            with c_r2:
                cnt = df.groupby([var_c, df['Churn'].map({0:'Client actif',1:'A résilié'})]).size().reset_index()
                cnt.columns = [var_c,'Statut','Nombre']
                fig4 = px.bar(cnt, x=var_c, y='Nombre', color='Statut',
                              color_discrete_map={'Client actif':'#34d399','A résilié':'#f87171'},
                              barmode='stack', title=f'Répartition selon {CAT_OPTS[var_c]}')
                fig4.update_layout(**PLOT_LAYOUT, xaxis_tickangle=-15, height=360, yaxis=GRID)
                st.plotly_chart(fig4, use_container_width=True)

            mx, mn = cr.iloc[0], cr.iloc[-1]
            st.markdown(f'<div class="alert-info">💡 <b>En clair :</b> parmi les clients avec <b>{CAT_OPTS[var_c]}</b> = <b>{mx[var_c]}</b>, <b>{mx["Taux (%)"]:.1f}%</b> ont résilié. À l\'opposé, ceux avec <b>{mn[var_c]}</b> ne partent que dans <b>{mn["Taux (%)"]:.1f}%</b> des cas. Cet écart est actionnable.</div>', unsafe_allow_html=True)

        with tab3:
            nf = ['tenure','MonthlyCharges','TotalCharges','charge_per_tenure','service_count','Cluster','Churn']
            rn = {'tenure':'Ancienneté','MonthlyCharges':'Montant mensuel','TotalCharges':'Total dépensé',
                  'charge_per_tenure':'Dépense relative','service_count':'Nb services',
                  'Cluster':'Groupe client','Churn':'Résiliation'}
            corr = df[nf].corr().rename(index=rn, columns=rn)
            fig5 = px.imshow(corr.round(2), text_auto=True, color_continuous_scale='RdBu_r',
                             zmin=-1, zmax=1, title='Liens entre les différentes informations clients')
            fig5.update_layout(**{**PLOT_LAYOUT, 'height': 480, 'margin': dict(t=50)})
            st.plotly_chart(fig5, use_container_width=True)
            st.markdown('<div class="alert-info">💡 <b>Comment lire ce tableau ?</b> Les cases <span style="color:#f87171">rouges</span> = deux facteurs évoluent ensemble. Les cases <span style="color:#818cf8">bleues</span> = ils vont en sens inverse. Plus la couleur est intense, plus le lien est fort. La colonne <b>Résiliation</b> montre quels facteurs sont le plus associés aux départs.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE 4 — GROUPES DE CLIENTS
# ══════════════════════════════════════════════
    elif page == "👥 Mes groupes de clients":
        require_data()
        df = prepare_data(st.session_state['df_raw'])
        st.markdown('<div class="page-header">👥 Vos Groupes de Clients</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-sub">L\'algorithme de segmentation a identifié automatiquement des groupes de clients aux comportements similaires.</div>', unsafe_allow_html=True)

        cp = df.groupby('Cluster').agg(
            Nb=('Churn','count'), Depart=('Churn', lambda x: round(x.mean()*100,1)),
            Anciennete=('tenure', lambda x: round(x.mean(),1)),
            Mensuel=('MonthlyCharges', lambda x: round(x.mean(),1)),
            Services=('service_count', lambda x: round(x.mean(),1)),
            Seniors=('SeniorCitizen', lambda x: round(x.mean()*100,1)),
        ).reset_index()

        cols_g = st.columns(len(cp))
        for i, (_, row) in enumerate(cp.iterrows()):
            badge_cls = 'badge-red' if row['Depart']>35 else 'badge-amber' if row['Depart']>25 else 'badge-green'
            risque    = 'Risque élevé' if row['Depart']>35 else 'Risque modéré' if row['Depart']>25 else 'Risque faible'
            with cols_g[i]:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.04);border:1px solid {COLORS[i]}44;
                            border-top:4px solid {COLORS[i]};border-radius:16px;padding:20px">
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
                        <b style="color:{COLORS[i]};font-size:1.1rem">Groupe {int(row['Cluster'])}</b>
                        <span class="badge {badge_cls}">{risque}</span>
                    </div>
                    <div style="color:#e2e8f0;font-size:1.9rem;font-weight:800">{int(row['Nb']):,}</div>
                    <div style="color:#94a3b8;font-size:0.82rem;margin-bottom:14px">clients dans ce groupe</div>
                    <hr style="border-color:rgba(255,255,255,0.08)">
                    <table style="width:100%;color:#94a3b8;font-size:0.85rem;border-spacing:0 6px">
                        <tr><td>Taux de départ</td><td style="text-align:right;color:#f87171;font-weight:700">{row['Depart']}%</td></tr>
                        <tr><td>Ancienneté moyenne</td><td style="text-align:right;color:#e2e8f0">{row['Anciennete']} mois</td></tr>
                        <tr><td>Facture mensuelle moy.</td><td style="text-align:right;color:#e2e8f0">${row['Mensuel']}</td></tr>
                        <tr><td>Services actifs moy.</td><td style="text-align:right;color:#e2e8f0">{row['Services']}</td></tr>
                        <tr><td>Part de seniors</td><td style="text-align:right;color:#e2e8f0">{row['Seniors']}%</td></tr>
                    </table>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        tab_a, tab_b, tab_c = st.tabs(["📊 Comparaison","📋 Contrats & Internet","🕸️ Vue radar"])

        with tab_a:
            METRICS = {'Depart':'Taux de départ (%)','Anciennete':'Ancienneté (mois)',
                       'Mensuel':'Facture mensuelle ($)','Services':'Nb services actifs','Seniors':'Part de seniors (%)'}
            mc = st.selectbox("Comparer sur :", list(METRICS.keys()), format_func=lambda x: METRICS[x])
            cp2 = cp.copy(); cp2['Groupe'] = cp2['Cluster'].apply(lambda x: f'Groupe {x}')
            fig_m = px.bar(cp2, x='Groupe', y=mc, color='Groupe',
                           color_discrete_sequence=COLORS, text=mc,
                           labels={mc: METRICS[mc]})
            fig_m.update_traces(textposition='outside')
            fig_m.update_layout(**PLOT_LAYOUT, showlegend=False, height=360, yaxis=GRID)
            st.plotly_chart(fig_m, use_container_width=True)

        with tab_b:
            c1, c2 = st.columns(2)
            with c1:
                cd = df.groupby(['Cluster','Contract']).size().reset_index(name='N')
                cd['Groupe'] = cd['Cluster'].apply(lambda x: f'Groupe {x}')
                cd['%'] = (cd['N'] / cd.groupby('Groupe')['N'].transform('sum') * 100).round(1)
                fig_c1 = px.bar(cd, x='Groupe', y='%', color='Contract', barmode='stack',
                                title='Types de contrats par groupe', labels={'%':'% des clients'})
                fig_c1.update_layout(**PLOT_LAYOUT, height=360, yaxis=GRID)
                st.plotly_chart(fig_c1, use_container_width=True)
            with c2:
                ci = df.groupby(['Cluster','InternetService']).size().reset_index(name='N')
                ci['Groupe'] = ci['Cluster'].apply(lambda x: f'Groupe {x}')
                ci['%'] = (ci['N'] / ci.groupby('Groupe')['N'].transform('sum') * 100).round(1)
                fig_c2 = px.bar(ci, x='Groupe', y='%', color='InternetService', barmode='stack',
                                title='Types de connexion par groupe', labels={'%':'% des clients'})
                fig_c2.update_layout(**PLOT_LAYOUT, height=360, yaxis=GRID)
                st.plotly_chart(fig_c2, use_container_width=True)

        with tab_c:
            metrics_r = ['Depart','Anciennete','Mensuel','Services','Seniors']
            labels_r  = ['Taux de\ndépart','Ancienneté','Facture\nmensuelle','Nb services','Seniors']
            rd = cp[metrics_r].values.astype(float)
            rn = (rd - rd.min(0)) / (rd.max(0) - rd.min(0) + 1e-9)
            fig_r = go.Figure()
            for i, (row, color) in enumerate(zip(rn, COLORS)):
                vals = list(row) + [row[0]]
                fig_r.add_trace(go.Scatterpolar(r=vals, theta=labels_r+[labels_r[0]], fill='toself',
                                                 name=f'Groupe {i}', line=dict(color=color, width=2.5),
                                                 fillcolor=color, opacity=0.12))
            fig_r.update_layout(
                polar=dict(bgcolor='rgba(0,0,0,0)',
                           radialaxis=dict(visible=True, range=[0,1], color='#475569', gridcolor='rgba(255,255,255,0.08)'),
                           angularaxis=dict(color='#94a3b8')),
                paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', height=450,
                title='Profil comparatif des groupes', legend=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig_r, use_container_width=True)
            st.markdown('<div class="alert-info">💡 <b>Comment lire ce graphique :</b> chaque axe est une caractéristique. Plus la forme d\'un groupe s\'étend vers l\'extérieur sur un axe, plus ses clients ont une valeur élevée pour cette caractéristique.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE 5 — PRÉDIRE LES DÉPARTS
# ══════════════════════════════════════════════
    elif page == "🤖 Prédire les départs":
        require_data()
        df = prepare_data(st.session_state['df_raw'])
        st.markdown('<div class="page-header">🤖 Prédire les Départs</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-sub">Trois approches pour identifier les clients les plus susceptibles de partir avant qu\'ils ne le fassent.</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card" style="text-align:left;margin-bottom:24px">
            <b style="color:#e2e8f0;font-size:1.05rem">Comment ça marche ?</b><br><br>
            <span style="color:#94a3b8">L'outil apprend des habitudes des anciens clients (partis et restés) pour repérer des schémas récurrents, puis teste trois "cerveaux" différents :</span><br><br>
            🔹 <b style="color:#818cf8">Approche simple</b> — calcule la probabilité de départ via une formule directe, facile à interpréter.<br>
            🔹 <b style="color:#f97316">Forêt de décisions</b> — consulte 200 "arbres de règles" et prend la décision majoritaire.<br>
            🔹 <b style="color:#34d399">Modèle optimisé</b> — apprentissage progressif, paramètres ajustés automatiquement.
        </div>""", unsafe_allow_html=True)

        with st.spinner("⏳ Apprentissage en cours… (30 à 60 secondes)"):
            results, y_te, feature_names, all_proba, best_params = train_models(df)
        st.markdown('<div class="alert-ok">✅ Apprentissage terminé ! Les trois approches ont été évaluées.</div>', unsafe_allow_html=True)

        st.markdown('<div class="sec-title">Niveau de risque de chaque client (Modèle optimisé)</div>', unsafe_allow_html=True)
        df_risk = df.copy()
        df_risk['Probabilité de départ'] = all_proba
        df_risk['Niveau de risque'] = pd.cut(all_proba, bins=[0,.3,.6,1.],
                                              labels=['🟢 Faible risque','🟡 Risque modéré','🔴 Risque élevé'])
        c1, c2, c3 = st.columns(3)
        for col, lbl, clr in zip([c1,c2,c3],
                                   ['🟢 Faible risque','🟡 Risque modéré','🔴 Risque élevé'],
                                   ['green','amber','red']):
            n = (df_risk['Niveau de risque']==lbl).sum()
            with col:
                st.markdown(f'<div class="glass-card {clr}"><div class="val">{n:,}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c_l, c_r = st.columns(2)
        with c_l:
            rbc = df_risk.groupby(['Cluster','Niveau de risque']).size().reset_index(name='N')
            rbc['Groupe'] = rbc['Cluster'].apply(lambda x: f'Groupe {x}')
            fig_rs = px.bar(rbc, x='Groupe', y='N', color='Niveau de risque',
                            color_discrete_map={'🟢 Faible risque':'#34d399','🟡 Risque modéré':'#fbbf24','🔴 Risque élevé':'#f87171'},
                            barmode='stack', title='Répartition du risque par groupe',
                            labels={'N':'Nombre de clients'})
            fig_rs.update_layout(**PLOT_LAYOUT, height=350, yaxis=GRID)
            st.plotly_chart(fig_rs, use_container_width=True)
        with c_r:
            fig_d = px.histogram(df_risk, x='Probabilité de départ',
                                 color=df_risk['Cluster'].apply(lambda x: f'Groupe {x}'),
                                 color_discrete_sequence=COLORS, nbins=40, opacity=0.75,
                                 barmode='overlay', title='Distribution des probabilités par groupe')
            fig_d.add_vline(x=0.5, line_dash='dash', line_color='#f87171',
                             annotation_text='Seuil de départ', annotation_font_color='#f87171')
            fig_d.update_layout(**PLOT_LAYOUT, height=350, yaxis=GRID, legend_title='Groupe')
            st.plotly_chart(fig_d, use_container_width=True)

        st.markdown('<div class="sec-title">Quels facteurs influencent le plus la décision de partir ?</div>', unsafe_allow_html=True)
        mc = st.radio("Voir les facteurs selon :", ['Forêt de décisions','Modèle optimisé'], horizontal=True)
        mk = 'Forêt de décisions' if mc == 'Forêt de décisions' else 'Modèle optimisé'
        fi = results[mk]['fi']
        if fi is not None:
            fi_df = pd.DataFrame({'Facteur': feature_names, 'Importance': fi})
            fi_df['Facteur'] = fi_df['Facteur'].str.replace('_',' ').str.replace('tenure','Ancienneté').str.replace('MonthlyCharges','Montant mensuel').str.replace('TotalCharges','Total dépensé').str.replace('Cluster','Groupe client').str.replace('Contract','Contrat')
            fi_df = fi_df.sort_values('Importance', ascending=False).head(15)
            fig_fi = px.bar(fi_df, x='Importance', y='Facteur', orientation='h',
                            color='Importance', color_continuous_scale='Purples',
                            title=f'Top 15 facteurs — {mc}',
                            labels={'Importance':"Niveau d'influence"})
            fig_fi.update_layout(yaxis={'autorange':'reversed'}, coloraxis_showscale=False,
                                  **PLOT_LAYOUT, height=480, xaxis=GRID)
            st.plotly_chart(fig_fi, use_container_width=True)


# ══════════════════════════════════════════════
#  PAGE 6 — COMPARER LES RÉSULTATS
# ══════════════════════════════════════════════
    elif page == "📊 Comparer les résultats":
        require_data()
        df = prepare_data(st.session_state['df_raw'])
        st.markdown('<div class="page-header">📊 Comparer les Résultats</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-sub">Quelle approche prédit le mieux les clients qui vont partir ?</div>', unsafe_allow_html=True)

        with st.spinner("⏳ Chargement des résultats…"):
            results, y_te, feature_names, all_proba, best_params = train_models(df)

        rows = []
        for name, r in results.items():
            rows.append({
                'Approche': name,
                'Bonne réponse globale': f"{r['acc']*100:.1f}%",
                'Précision (quand il dit oui)': f"{r['prec']*100:.1f}%",
                'Partants détectés': f"{r['rec']*100:.1f}%",
                'Score global': f"{r['f1']*100:.1f}%",
                'Fiabilité générale': f"{r['auc']*100:.1f}%",
            })
        comp_df = pd.DataFrame(rows).set_index('Approche')
        st.markdown('<div class="sec-title">Tableau comparatif des trois approches</div>', unsafe_allow_html=True)
        st.dataframe(comp_df, use_container_width=True)

        best_name = max(results, key=lambda x: results[x]['auc'])
        br = results[best_name]
        st.markdown(f"""
        <div class="alert-info">
            💡 <b>Comment lire ce tableau ?</b><br><br>
            • <b>Bonne réponse globale</b> : sur 100 clients, combien sont correctement classés.<br>
            • <b>Partants détectés</b> : parmi tous les clients qui vont partir, quelle proportion est repérée — <b>c'est la mesure la plus importante !</b> Manquer un départ coûte plus cher qu'une fausse alerte.<br>
            • <b>Fiabilité générale</b> : capacité à distinguer un partant d'un client fidèle (100% = parfait, 50% = hasard).<br><br>
            🏆 <b>Meilleure approche : {best_name}</b> — {br['auc']*100:.1f}% de fiabilité et {br['rec']*100:.1f}% des partants détectés.
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="sec-title">Courbes de performance</div>', unsafe_allow_html=True)
            fig_roc = go.Figure()
            c_roc = ['#818cf8','#f97316','#34d399']
            for (name, r), color in zip(results.items(), c_roc):
                fig_roc.add_trace(go.Scatter(x=r['fpr'], y=r['tpr'], mode='lines',
                                              name=f"{name} ({r['auc']*100:.1f}%)",
                                              line=dict(color=color, width=2.5)))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                          line=dict(dash='dash', color='#475569'), name='Au hasard'))
            fig_roc.update_layout(**PLOT_LAYOUT, height=380,
                                   xaxis=dict(title='Fausses alertes', **GRID),
                                   yaxis=dict(title='Partants détectés', **GRID),
                                   legend=dict(bgcolor='rgba(0,0,0,0)', x=0.38, y=0.08))
            st.plotly_chart(fig_roc, use_container_width=True)
            st.markdown('<div class="alert-info" style="font-size:.85rem">💡 Plus la courbe se rapproche du coin supérieur gauche, meilleure est l\'approche. La diagonale = prédiction au hasard.</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="sec-title">Détail des prédictions</div>', unsafe_allow_html=True)
            model_cm = st.selectbox("Approche à analyser", list(results.keys()))
            cm = results[model_cm]['cm']
            tn, fp, fn, tp = cm.ravel()
            fig_cm = px.imshow(cm,
                               labels=dict(x='Ce que le modèle a prédit', y='Réalité', color='Nombre'),
                               x=['Prédit : Va rester','Prédit : Va partir'],
                               y=['A réellement resté','A réellement parti'],
                               text_auto=True, color_continuous_scale='Purples',
                               title=f'Résultats — {model_cm}')
            fig_cm.update_layout(**PLOT_LAYOUT, height=360, margin=dict(t=50))
            st.plotly_chart(fig_cm, use_container_width=True)
            st.markdown(f"""
            <div class="alert-info" style="font-size:.85rem">
                💡 <b>Lecture :</b>
                <b style="color:#34d399">{tp:,} partants correctement détectés ✅</b> ·
                <b style="color:#fbbf24">{fn:,} partants manqués ⚠️</b> ·
                <b style="color:#f87171">{fp:,} fausses alertes</b> ·
                <b style="color:#818cf8">{tn:,} clients fidèles bien identifiés</b>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE 7 — QUE FAIRE MAINTENANT
# ══════════════════════════════════════════════
    elif page == "💡 Que faire maintenant ?":
        require_data()
        df = prepare_data(st.session_state['df_raw'])
        st.markdown('<div class="page-header">💡 Que Faire Maintenant ?</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-sub">Recommandations concrètes, groupe par groupe, pour réduire les départs et améliorer la fidélisation.</div>', unsafe_allow_html=True)

        cluster_churn = df.groupby('Cluster')['Churn'].mean() * 100
        cluster_size  = df['Cluster'].value_counts().sort_index()

        c1, c2, c3 = st.columns(3)
        for col, cid, title, clr in [(c1,0,'Groupe 0 — Action urgente','red'),
                                       (c2,1,'Groupe 1 — À engager','amber'),
                                       (c3,2,'Groupe 2 — Clients fidèles','green')]:
            with col:
                st.markdown(f'<div class="glass-card {clr}"><div class="val">{cluster_churn.get(cid,0):.1f}%</div><div class="lbl">{title}<br>{cluster_size.get(cid,0):,} clients</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        recos = [
            dict(id=0, title='🔴 Groupe 0 — Clients à risque élevé',
                 desc=f"{cluster_size.get(0,0):,} clients · {cluster_churn.get(0,0):.1f}% de départs · Agir en priorité", color='#f87171',
                 profil="Ce groupe présente le taux de départ le plus élevé. Ces clients ont souvent des contrats courts, une faible ancienneté, et la valeur perçue de leur abonnement n'est pas au rendez-vous.",
                 actions=[("🎁 Offre personnalisée de fidélisation","Proposer une remise ou un avantage exclusif lors d'une migration vers un contrat annuel ou biannuel.","⭐⭐⭐ Prioritaire"),
                          ("📞 Prise de contact proactive","Appeler le client dans les 3 premiers mois de son abonnement pour vérifier sa satisfaction, avant qu'il ne songe à partir.","⭐⭐⭐ Prioritaire"),
                          ("🔒 Offre groupée de services","Proposer un pack combiné (connexion + support + sécurité) à prix avantageux pour augmenter l'engagement.","⭐⭐ Important")]),
            dict(id=1, title='🟡 Groupe 1 — Clients à surveiller',
                 desc=f"{cluster_size.get(1,0):,} clients · {cluster_churn.get(1,0):.1f}% de départs · Engager et fidéliser", color='#fbbf24',
                 profil="Segment intermédiaire avec un potentiel d'amélioration. Ces clients utilisent les services mais pourraient être plus attachés à l'opérateur avec les bons incitatifs.",
                 actions=[("🌟 Programme de fidélité","Mettre en place des récompenses cumulables (points, avantages) après 12 et 24 mois d'abonnement.","⭐⭐⭐ Prioritaire"),
                          ("📦 Enrichissement de l'offre","Proposer un mois gratuit de services Streaming pour encourager la découverte et augmenter l'attachement.","⭐⭐ Important"),
                          ("📧 Bilan mensuel personnalisé","Envoyer un résumé mensuel montrant ce que le client a consommé et les économies réalisées grâce à son contrat.","⭐⭐ Important")]),
            dict(id=2, title='🟢 Groupe 2 — Clients fidèles',
                 desc=f"{cluster_size.get(2,0):,} clients · {cluster_churn.get(2,0):.1f}% de départs · Valoriser et mobiliser", color='#34d399',
                 profil="Vos meilleurs clients. Ils sont là depuis longtemps, paient régulièrement et partent rarement. L'objectif est de les récompenser et d'en faire des ambassadeurs de votre marque.",
                 actions=[("👑 Statut client privilégié","Créer un statut VIP avec accès prioritaire au support, lignes dédiées et invitations à des événements exclusifs.","⭐⭐⭐ Prioritaire"),
                          ("🔄 Renouvellement anticipé avantageux","Proposer le renouvellement 3 mois avant l'échéance avec un cadeau ou une amélioration de l'offre.","⭐⭐ Important"),
                          ("💬 Programme de parrainage","Ces clients sont les meilleurs ambassadeurs : récompensez le parrainage pour attirer de nouveaux clients à faible coût.","⭐⭐ Important")]),
        ]

        for r in recos:
            with st.expander(f"**{r['title']}** — {r['desc']}", expanded=(r['id']==0)):
                st.markdown(f"<span style='color:{r['color']};font-weight:600'>Profil du groupe :</span> <span style='color:#94a3b8'>{r['profil']}</span>", unsafe_allow_html=True)
                st.markdown("**Actions recommandées :**")
                for action, detail, priority in r['actions']:
                    st.markdown(f"""
                    <div class="rec-card">
                        <div style="display:flex;justify-content:space-between;align-items:flex-start">
                            <b>{action}</b><span class="badge badge-blue">{priority}</span>
                        </div>
                        <span>{detail}</span>
                    </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="sec-title">Les 8 signaux d\'alarme les plus fréquents avant un départ</div>', unsafe_allow_html=True)
        signals = pd.DataFrame({
            'Situation observée': ['Contrat mois par mois (sans engagement)','Client récent (moins de 12 mois)',
                                   'Connexion fibre sans support technique','Pas de sécurité en ligne activée',
                                   'Paiement par chèque électronique','Montant mensuel élevé pour un nouveau client',
                                   'Client vivant seul (sans famille à charge)','Pas de sauvegarde en ligne activée'],
            'Niveau d\'alerte': [95,88,72,65,55,50,45,40],
            'Ce que vous pouvez faire': ['Proposer une migration vers un contrat longue durée',
                                          'Mettre en place un suivi actif dans les 3 premiers mois',
                                          'Offrir le support technique à prix réduit',
                                          'Inclure la sécurité dans les offres de bienvenue',
                                          'Proposer des modes de paiement automatiques avec remise',
                                          'Adapter l\'offre au profil ou proposer un essai de services',
                                          'Offres parrainage pour attirer l\'entourage du client',
                                          'Inclure la sauvegarde dans les packs d\'entrée de gamme']
        })
        fig_s = px.bar(signals, x='Niveau d\'alerte', y='Situation observée', orientation='h',
                       color='Niveau d\'alerte', color_continuous_scale='RdYlGn_r',
                       title='Signaux d\'alarme (plus le score est élevé, plus c\'est préoccupant)')
        fig_s.update_layout(yaxis={'autorange':'reversed'}, coloraxis_showscale=False,
                             **PLOT_LAYOUT, height=400, xaxis=GRID)
        st.plotly_chart(fig_s, use_container_width=True)

        st.markdown('<div class="sec-title">Plan d\'actions résumé</div>', unsafe_allow_html=True)
        st.dataframe(signals[['Situation observée','Ce que vous pouvez faire']], use_container_width=True, hide_index=True)

        st.markdown('<div class="sec-title">Indicateurs à suivre pour mesurer vos progrès</div>', unsafe_allow_html=True)
        kpis = pd.DataFrame({
            'Ce qu\'on mesure': ['Part de contrats courts convertis en contrats longs',
                                 'Taux de départ réel vs. taux prédit par l\'outil',
                                 'Retour sur investissement des campagnes Groupe 0',
                                 'Part des partants effectivement détectés à l\'avance',
                                 'Satisfaction client (score NPS) par groupe'],
            'Objectif': ['+5% par trimestre','Écart < 2%','Bénéfice/coût > 3:1','Détecter 85% des partants','Suivi trimestriel'],
            'Fréquence': ['Trimestrielle','Mensuelle','Trimestrielle','Mensuelle','Trimestrielle']
        })
        st.dataframe(kpis, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
#  SUJET 2 — PAGES
# ══════════════════════════════════════════════
elif sujet_actif == "🛍️ Sujet 2 — Segmentation Mall":

    # ── PAGE S2-1 : IMPORTER ─────────────────────
    if page_s2 == "📂 Importer mes données":
        st.markdown('<div class="page-header">📂 Importer vos données clients</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-sub">Chargez votre fichier CSV ou utilisez les données de démonstration pour démarrer la segmentation.</div>', unsafe_allow_html=True)

        col_up, col_demo = st.columns([3, 2], gap="large")

        with col_up:
            st.markdown('<div class="sec-title">Charger votre propre fichier</div>', unsafe_allow_html=True)
            uploaded_s2 = st.file_uploader(
                "Glissez-déposez votre fichier CSV ici",
                type=['csv'], key='s2_uploader'
            )
            if uploaded_s2 is not None:
                try:
                    raw_s2 = uploaded_s2.getvalue()
                    df_up_s2 = None
                    for sep in [',', ';']:
                        tmp = pd.read_csv(io.BytesIO(raw_s2), sep=sep)
                        if tmp.shape[1] >= 4:
                            df_up_s2 = tmp
                            break
                    if df_up_s2 is None:
                        df_up_s2 = pd.read_csv(io.BytesIO(raw_s2))
                    ok_s2, miss_s2 = validate_file_s2(df_up_s2)
                    if ok_s2:
                        st.session_state['s2_df'] = df_up_s2
                        st.session_state['s2_labels'] = None
                        st.markdown(f'<div class="alert-ok">✅ <b>Fichier compatible !</b> {len(df_up_s2):,} clients détectés.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="alert-err">❌ <b>Colonnes manquantes :</b> {", ".join(miss_s2)}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="alert-err">❌ Erreur de lecture : {e}</div>', unsafe_allow_html=True)

        with col_demo:
            st.markdown('<div class="sec-title">Données de démonstration</div>', unsafe_allow_html=True)
            st.markdown("""
<div class="glass-card" style="text-align:left">
    <b style="color:#e2e8f0">Fichier inclus</b><br><br>
    <span style="color:#94a3b8">Données simulées d'un centre commercial :</span><br><br>
    👥 200 clients analysés<br>
    📊 5 variables : identité, âge, revenu, score de dépense<br>
    🏷️ 5 segments naturels identifiables<br>
    ✅ Données idéales pour la segmentation marketing<br><br>
    <span style="color:#34d399">✅ Prêt à l'emploi</span>
</div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Utiliser les données de démonstration", use_container_width=True, key="s2_demo_btn"):
                st.session_state['s2_df'] = generate_demo_s2()
                st.session_state['s2_labels'] = None
                st.markdown('<div class="alert-ok">✅ Données chargées ! Allez sur <b>Explorer les données</b>.</div>', unsafe_allow_html=True)

        if st.session_state.get('s2_df') is not None:
            df_s2 = st.session_state['s2_df']
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("👥 Total clients", f"{len(df_s2):,}")
            c2.metric("🎂 Âge moyen", f"{df_s2['Age'].mean():.1f} ans")
            c3.metric("💰 Revenu moyen", f"{df_s2['Annual Income (k$)'].mean():.1f} k$")
            c4.metric("🛍️ Score moyen", f"{df_s2['Spending Score (1-100)'].mean():.1f}")
            st.markdown("<br>**Aperçu des données :**")
            st.dataframe(df_s2.head(10), use_container_width=True)

    # ── PAGE S2-2 : EXPLORER ──────────────────────
    elif page_s2 == "🔍 Explorer les données":
        require_data_s2()
        df_s2 = st.session_state['s2_df']
        st.markdown('<div class="page-header">🔍 Explorer les Données</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-sub">Visualisez les distributions et les relations entre les variables clés.</div>', unsafe_allow_html=True)

        # Row 1: 3 histograms
        col_h1, col_h2, col_h3 = st.columns(3)
        with col_h1:
            st.markdown('<div class="sec-title">Distribution — Âge</div>', unsafe_allow_html=True)
            fig_age = px.histogram(df_s2, x='Age', nbins=20, color_discrete_sequence=['#818cf8'])
            fig_age.add_vline(x=df_s2['Age'].mean(), line_dash='dash', line_color='#34d399',
                               annotation_text=f"Moy. {df_s2['Age'].mean():.1f}", annotation_font_color='#34d399')
            fig_age.update_layout(**PLOT_LAYOUT, height=280, yaxis=GRID, showlegend=False)
            st.plotly_chart(fig_age, use_container_width=True)

        with col_h2:
            st.markdown('<div class="sec-title">Distribution — Revenu (k$)</div>', unsafe_allow_html=True)
            fig_inc = px.histogram(df_s2, x='Annual Income (k$)', nbins=20, color_discrete_sequence=['#f97316'])
            fig_inc.add_vline(x=df_s2['Annual Income (k$)'].mean(), line_dash='dash', line_color='#34d399',
                               annotation_text=f"Moy. {df_s2['Annual Income (k$)'].mean():.1f}", annotation_font_color='#34d399')
            fig_inc.update_layout(**PLOT_LAYOUT, height=280, yaxis=GRID, showlegend=False)
            st.plotly_chart(fig_inc, use_container_width=True)

        with col_h3:
            st.markdown('<div class="sec-title">Distribution — Score dépense</div>', unsafe_allow_html=True)
            fig_sc = px.histogram(df_s2, x='Spending Score (1-100)', nbins=20, color_discrete_sequence=['#8b5cf6'])
            fig_sc.add_vline(x=df_s2['Spending Score (1-100)'].mean(), line_dash='dash', line_color='#34d399',
                              annotation_text=f"Moy. {df_s2['Spending Score (1-100)'].mean():.1f}", annotation_font_color='#34d399')
            fig_sc.update_layout(**PLOT_LAYOUT, height=280, yaxis=GRID, showlegend=False)
            st.plotly_chart(fig_sc, use_container_width=True)

        # Row 2: pie + scatter + box
        col_pie2, col_scat2 = st.columns(2)
        with col_pie2:
            st.markdown('<div class="sec-title">Répartition par Genre</div>', unsafe_allow_html=True)
            genre_cnt = df_s2['Genre'].value_counts().reset_index()
            genre_cnt.columns = ['Genre', 'Nombre']
            fig_g = px.pie(genre_cnt, values='Nombre', names='Genre',
                           color_discrete_sequence=['#818cf8', '#f97316'], hole=0.45)
            fig_g.update_traces(textposition='outside', textinfo='percent+label', textfont_size=13)
            fig_g.update_layout(**PLOT_LAYOUT, showlegend=False, height=300)
            st.plotly_chart(fig_g, use_container_width=True)

        with col_scat2:
            st.markdown('<div class="sec-title">Revenu vs Score de dépense</div>', unsafe_allow_html=True)
            fig_sc2 = px.scatter(df_s2, x='Annual Income (k$)', y='Spending Score (1-100)',
                                  color='Genre', hover_data=['Age'],
                                  color_discrete_map={'Male': '#818cf8', 'Female': '#f97316'},
                                  opacity=0.75,
                                  labels={'Annual Income (k$)': 'Revenu annuel (k$)',
                                          'Spending Score (1-100)': 'Score de dépense'})
            fig_sc2.update_layout(**PLOT_LAYOUT, height=300, xaxis=GRID, yaxis=GRID,
                                   legend=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig_sc2, use_container_width=True)

        st.markdown('<div class="sec-title">Score de dépense par Genre</div>', unsafe_allow_html=True)
        fig_box = px.box(df_s2, x='Genre', y='Spending Score (1-100)', color='Genre',
                          color_discrete_map={'Male': '#818cf8', 'Female': '#f97316'},
                          labels={'Spending Score (1-100)': 'Score de dépense'})
        fig_box.update_layout(**PLOT_LAYOUT, height=320, yaxis=GRID, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

        st.info("Les clients à fort revenu ne sont pas nécessairement les plus dépensiers. On observe deux groupes distincts parmi les hauts revenus : les économes et les dépensiers.")

    # ── PAGE S2-3 : SEGMENTATION ──────────────────
    elif page_s2 == "👥 Segmentation clients":
        require_data_s2()
        df_s2 = st.session_state['s2_df']
        st.markdown('<div class="page-header">👥 Segmentation Clients</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-sub">K-Means sur le revenu et le score de dépense pour identifier des segments marketing actionnables.</div>', unsafe_allow_html=True)

        with st.spinner("Calcul des courbes Elbow et Silhouette…"):
            ks_s2, inertias_s2, sils_s2 = compute_elbow_s2(df_s2)

        best_k_s2 = ks_s2[int(np.argmax(sils_s2))]

        col_el2, col_sil2 = st.columns(2)
        with col_el2:
            st.markdown('<div class="sec-title">Méthode du Coude — Choix du k optimal</div>', unsafe_allow_html=True)
            fig_elb2 = px.line(x=ks_s2, y=inertias_s2, markers=True,
                                labels={'x': 'Nombre de segments (k)', 'y': 'Inertie'},
                                color_discrete_sequence=['#818cf8'])
            fig_elb2.add_vline(x=5, line_dash='dash', line_color='#f87171',
                                annotation_text='k=5', annotation_font_color='#f87171')
            fig_elb2.update_layout(**PLOT_LAYOUT, height=300, xaxis=GRID, yaxis=GRID)
            st.plotly_chart(fig_elb2, use_container_width=True)

        with col_sil2:
            st.markdown('<div class="sec-title">Score de Silhouette par k</div>', unsafe_allow_html=True)
            sil_colors = ['#f87171' if k == best_k_s2 else '#818cf8' for k in ks_s2]
            fig_sil2 = go.Figure(go.Bar(
                x=ks_s2, y=[round(s, 3) for s in sils_s2],
                marker_color=sil_colors,
                text=[f"{s:.3f}" for s in sils_s2], textposition='outside',
            ))
            fig_sil2.update_layout(**PLOT_LAYOUT, height=300,
                                    xaxis=dict(title='k', **GRID),
                                    yaxis=dict(title='Score Silhouette', **GRID))
            st.plotly_chart(fig_sil2, use_container_width=True)

        k_s2 = st.slider("Choisir le nombre de segments", min_value=2, max_value=10,
                          value=5, key="s2_k_slider", label_visibility="visible")

        with st.spinner(f"Segmentation en {k_s2} groupes…"):
            s2_labels, s2_sc, s2_centers = run_kmeans_s2(df_s2, k_s2)

        st.session_state['s2_labels'] = s2_labels
        st.session_state['s2_scaler'] = s2_sc

        df_s2_cl = df_s2.copy()
        df_s2_cl['Cluster'] = s2_labels

        # Assign business names from original-scale centroids
        inc_arr = df_s2['Annual Income (k$)'].values
        sc_arr  = df_s2['Spending Score (1-100)'].values
        cl_inc  = {c: inc_arr[s2_labels == c].mean() for c in range(k_s2)}
        cl_sc   = {c: sc_arr[s2_labels == c].mean()  for c in range(k_s2)}
        cl_name = {c: name_cluster_s2(cl_inc[c], cl_sc[c]) for c in range(k_s2)}
        df_s2_cl['Segment'] = df_s2_cl['Cluster'].map(cl_name)

        st.markdown('<div class="sec-title">Segmentation clients — Revenu vs Score de dépense</div>', unsafe_allow_html=True)
        s2_pal = ['#818cf8','#f97316','#8b5cf6','#34d399','#38bdf8','#fbbf24','#f87171','#a78bfa','#fb923c','#4ade80']
        fig_main_s2 = px.scatter(df_s2_cl, x='Annual Income (k$)', y='Spending Score (1-100)',
                                  color='Segment', size='Age',
                                  hover_data={'CustomerID': True, 'Genre': True, 'Age': True,
                                              'Annual Income (k$)': True, 'Spending Score (1-100)': True},
                                  color_discrete_sequence=s2_pal[:k_s2],
                                  labels={'Annual Income (k$)': 'Revenu annuel (k$)',
                                          'Spending Score (1-100)': 'Score de dépense'},
                                  title='Segmentation clients — Revenu vs Score de dépense',
                                  opacity=0.85)
        fig_main_s2.update_layout(**PLOT_LAYOUT, height=480, xaxis=GRID, yaxis=GRID,
                                   legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig_main_s2, use_container_width=True)

        st.markdown('<div class="sec-title">Profil de chaque segment</div>', unsafe_allow_html=True)
        profile_rows = []
        for c in sorted(range(k_s2), key=lambda x: cl_inc[x], reverse=True):
            mask = s2_labels == c
            pct_f = (df_s2.loc[mask, 'Genre'] == 'Female').mean() * 100
            profile_rows.append({
                'Cluster': c,
                'Nom métier': cl_name[c],
                'Nb clients': int(mask.sum()),
                'Revenu moyen (k$)': round(cl_inc[c], 1),
                'Score moyen': round(cl_sc[c], 1),
                'Age moyen': round(df_s2.loc[mask, 'Age'].mean(), 1),
                '% Femmes': round(pct_f, 1),
            })
        st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)

    # ── PAGE S2-4 : STRATEGIE ─────────────────────
    elif page_s2 == "💡 Stratégie marketing":
        require_data_s2()
        df_s2 = st.session_state['s2_df']
        st.markdown('<div class="page-header">💡 Stratégie Marketing</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-sub">Recommandations concrètes par segment pour maximiser le retour sur investissement marketing.</div>', unsafe_allow_html=True)

        k_strat = 5
        s2_labs_strat, _, _ = run_kmeans_s2(df_s2, k_strat)
        df_strat = df_s2.copy()
        df_strat['Cluster'] = s2_labs_strat

        inc_arr_st = df_s2['Annual Income (k$)'].values
        sc_arr_st  = df_s2['Spending Score (1-100)'].values
        cl_inc_st  = {c: inc_arr_st[s2_labs_strat == c].mean() for c in range(k_strat)}
        cl_sc_st   = {c: sc_arr_st[s2_labs_strat == c].mean()  for c in range(k_strat)}
        cl_name_st = {c: name_cluster_s2(cl_inc_st[c], cl_sc_st[c]) for c in range(k_strat)}
        df_strat['Segment'] = df_strat['Cluster'].map(cl_name_st)

        cl_size_st = df_strat['Cluster'].value_counts()

        for c in sorted(range(k_strat), key=lambda x: cl_inc_st[x], reverse=True):
            lbl   = cl_name_st[c]
            nb    = int(cl_size_st.get(c, 0))
            reco  = RECO_S2.get(lbl, "Suivi personnalisé recommandé.")
            prio_txt, prio_cls = PRIORITY_S2.get(lbl, ("MOYENNE", "badge-blue"))
            mask_c = s2_labs_strat == c
            pct_f  = (df_s2.loc[mask_c, 'Genre'] == 'Female').mean() * 100
            color_map = {'badge-green': '#34d399', 'badge-red': '#f87171',
                          'badge-amber': '#fbbf24', 'badge-blue': '#818cf8'}
            clr = color_map.get(prio_cls, '#818cf8')
            st.markdown(f"""
<div style="background:rgba(255,255,255,0.04);border:1px solid {clr}44;
            border-left:5px solid {clr};border-radius:14px;padding:18px 22px;margin-bottom:16px">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
        <b style="color:{clr};font-size:1.15rem">{lbl}</b>
        <span class="badge {prio_cls}">Priorité : {prio_txt}</span>
    </div>
    <div style="display:flex;gap:24px;color:#94a3b8;font-size:0.85rem;margin-bottom:12px">
        <span>👥 <b style="color:#e2e8f0">{nb}</b> clients</span>
        <span>💰 Revenu moy. <b style="color:#e2e8f0">{cl_inc_st[c]:.1f} k$</b></span>
        <span>🛍️ Score moy. <b style="color:#e2e8f0">{cl_sc_st[c]:.1f}</b></span>
        <span>🎂 Âge moy. <b style="color:#e2e8f0">{df_s2.loc[mask_c, 'Age'].mean():.1f}</b></span>
        <span>♀ <b style="color:#e2e8f0">{pct_f:.0f}%</b> femmes</span>
    </div>
    <div class="rec-card"><b>📋 Recommandation</b><br><span>{reco}</span></div>
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        premium_mask = np.array([cl_name_st[c] == "🟢 Cibles Premium" for c in s2_labs_strat])
        nb_premium   = int(premium_mask.sum())
        rev_potential = nb_premium * int(df_s2.loc[premium_mask, 'Annual Income (k$)'].mean()) if nb_premium > 0 else 0

        c1s, c2s = st.columns(2)
        c1s.metric("🟢 Clients Premium", f"{nb_premium:,}", help="Segment à plus fort potentiel de valeur")
        c2s.metric("💰 Potentiel revenu Premium", f"{rev_potential:,} k$",
                   help="Nb clients Premium × revenu moyen du segment")

        df_export = df_strat[['CustomerID', 'Genre', 'Age', 'Annual Income (k$)',
                               'Spending Score (1-100)', 'Cluster', 'Segment']].copy()
        csv_s2 = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Exporter les clients segmentés (CSV)",
            data=csv_s2,
            file_name="clients_segmentes.csv",
            mime="text/csv",
            use_container_width=True,
            key="s2_download_btn",
        )

# ══════════════════════════════════════════════
#  SUJET 3 — PAGES
# ══════════════════════════════════════════════
elif sujet_actif == "🎓 Sujet 3 — Réussite Étudiante":

    # ── PAGE S3-1 : IMPORTER ─────────────────────
    if page_s3 == "📂 Importer mes données":
        st.markdown('<div class="page-header">📂 Importer vos données étudiantes</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-sub">Chargez votre fichier CSV pour démarrer l\'analyse des performances.</div>', unsafe_allow_html=True)
        uploaded_s3 = st.file_uploader(
            "Glissez-déposez votre fichier CSV (séparateurs ';' ou ',' acceptés)",
            type=['csv'], key='s3_uploader'
        )
        if uploaded_s3 is not None:
            try:
                raw_s3 = uploaded_s3.getvalue()
                df_up_s3 = None
                for sep in [';', ',']:
                    tmp = pd.read_csv(io.BytesIO(raw_s3), sep=sep)
                    if tmp.shape[1] >= 5:
                        df_up_s3 = tmp
                        break
                if df_up_s3 is None:
                    df_up_s3 = pd.read_csv(io.BytesIO(raw_s3))
                ok_s3, miss_s3 = validate_file_s3(df_up_s3)
                if ok_s3:
                    st.session_state['s3_df'] = df_up_s3
                    df_s3 = df_up_s3
                    st.markdown(f'<div class="alert-ok">✅ <b>Fichier compatible !</b> {len(df_s3):,} étudiants détectés.</div>', unsafe_allow_html=True)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("👤 Total étudiants", f"{len(df_s3):,}")
                    c2.metric("📊 Note G3 moyenne", f"{df_s3['G3'].mean():.1f} / 20")
                    c3.metric("✅ Taux de réussite", f"{(df_s3['G3']>=10).mean()*100:.1f}%")
                    c4.metric("❌ Taux d'échec", f"{(df_s3['G3']<10).mean()*100:.1f}%")
                    st.markdown("<br>**Aperçu des données :**")
                    st.dataframe(df_s3.head(10), use_container_width=True)
                else:
                    st.markdown(f'<div class="alert-err">❌ <b>Colonnes manquantes :</b> {", ".join(miss_s3)}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="alert-err">❌ Erreur de lecture : {e}</div>', unsafe_allow_html=True)
        elif st.session_state.get('s3_df') is not None:
            df_s3 = st.session_state['s3_df']
            st.markdown('<div class="alert-ok">✅ Données déjà chargées.</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("👤 Total étudiants", f"{len(df_s3):,}")
            c2.metric("📊 Note G3 moyenne", f"{df_s3['G3'].mean():.1f} / 20")
            c3.metric("✅ Taux de réussite", f"{(df_s3['G3']>=10).mean()*100:.1f}%")
            c4.metric("❌ Taux d'échec", f"{(df_s3['G3']<10).mean()*100:.1f}%")
            st.dataframe(df_s3.head(10), use_container_width=True)
        else:
            st.markdown('<div class="alert-info">ℹ️ Aucune donnée chargée. Importez un fichier CSV avec les colonnes requises.</div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">Colonnes attendues</div>', unsafe_allow_html=True)
            st.code(", ".join(sorted(S3_REQUIRED_COLS)))

    # ── PAGE S3-2 : EXPLORER ──────────────────────
    elif page_s3 == "🔍 Explorer les données":
        require_data_s3()
        df_s3 = st.session_state['s3_df']
        st.markdown('<div class="page-header">🔍 Explorer les Données</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-sub">Visualisez les distributions et les relations entre facteurs et performances.</div>', unsafe_allow_html=True)

        st.markdown('<div class="sec-title">Distribution de la note finale (G3)</div>', unsafe_allow_html=True)
        fig_g3 = px.histogram(df_s3, x='G3', nbins=21, color_discrete_sequence=['#818cf8'],
                               labels={'G3': 'Note finale G3'})
        fig_g3.add_vline(x=10, line_dash='dash', line_color='#f87171',
                          annotation_text='Seuil de réussite (10)', annotation_font_color='#f87171')
        fig_g3.add_vline(x=df_s3['G3'].mean(), line_dash='dot', line_color='#34d399',
                          annotation_text=f"Moyenne ({df_s3['G3'].mean():.1f})", annotation_font_color='#34d399')
        fig_g3.update_layout(**PLOT_LAYOUT, height=320, yaxis=GRID)
        st.plotly_chart(fig_g3, use_container_width=True)

        col_pie, col_std = st.columns(2)
        with col_pie:
            st.markdown('<div class="sec-title">Répartition par niveau</div>', unsafe_allow_html=True)
            levels_s3 = pd.cut(df_s3['G3'], bins=[-1, 9.5, 11.5, 14.5, 20],
                                labels=['Échec (<10)', 'Passable (10-11)', 'Bien (12-14)', 'Très bien (>=15)'])
            lv_counts = levels_s3.value_counts().reset_index()
            lv_counts.columns = ['Niveau', 'Nombre']
            fig_pie_s3 = px.pie(lv_counts, values='Nombre', names='Niveau',
                                color_discrete_sequence=['#f87171', '#fbbf24', '#818cf8', '#34d399'], hole=0.45)
            fig_pie_s3.update_traces(textposition='outside', textinfo='percent+label', textfont_size=12)
            fig_pie_s3.update_layout(**PLOT_LAYOUT, showlegend=False, height=320)
            st.plotly_chart(fig_pie_s3, use_container_width=True)

        with col_std:
            st.markdown('<div class="sec-title">Note moyenne selon le temps d\'etude</div>', unsafe_allow_html=True)
            std_df = df_s3.groupby('studytime')['G3'].mean().reset_index()
            std_df.columns = ["Temps d'etude", 'G3 moyen']
            fig_std = px.bar(std_df, x="Temps d'etude", y='G3 moyen',
                             color='G3 moyen', color_continuous_scale='Purples', text='G3 moyen')
            fig_std.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig_std.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, height=320,
                                   yaxis=dict(range=[0, 22], **GRID))
            st.plotly_chart(fig_std, use_container_width=True)

        col_fail, col_alc = st.columns(2)
        with col_fail:
            st.markdown('<div class="sec-title">Note moyenne selon les echecs passes</div>', unsafe_allow_html=True)
            fail_df = df_s3.groupby('failures')['G3'].mean().reset_index()
            fail_df.columns = ['Echecs anterieurs', 'G3 moyen']
            fig_fail = px.bar(fail_df, x='Echecs anterieurs', y='G3 moyen',
                              color='G3 moyen', color_continuous_scale='RdYlGn', text='G3 moyen')
            fig_fail.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig_fail.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, height=320,
                                    yaxis=dict(range=[0, 22], **GRID))
            st.plotly_chart(fig_fail, use_container_width=True)

        with col_alc:
            st.markdown('<div class="sec-title">Note selon la consommation d\'alcool</div>', unsafe_allow_html=True)
            walc_df = df_s3.groupby('Walc')['G3'].mean().reset_index().rename(columns={'Walc': 'Niveau', 'G3': 'G3 moyen'})
            walc_df['Type'] = 'Week-end (Walc)'
            dalc_df = df_s3.groupby('Dalc')['G3'].mean().reset_index().rename(columns={'Dalc': 'Niveau', 'G3': 'G3 moyen'})
            dalc_df['Type'] = 'Semaine (Dalc)'
            alc_df = pd.concat([walc_df, dalc_df], ignore_index=True)
            fig_alc = px.line(alc_df, x='Niveau', y='G3 moyen', color='Type', markers=True,
                              color_discrete_sequence=['#818cf8', '#f97316'],
                              labels={'Niveau': 'Niveau de consommation (1=faible, 5=eleve)'})
            fig_alc.update_layout(**PLOT_LAYOUT, height=320, yaxis=dict(range=[0, 22], **GRID),
                                   legend=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig_alc, use_container_width=True)

        st.markdown('<div class="sec-title">Absences vs Note finale (par sexe)</div>', unsafe_allow_html=True)
        sex_lbl = df_s3['sex'].map({'M': 'Masculin', 'F': 'Feminin'}).fillna(df_s3['sex'])
        fig_abs = px.scatter(df_s3, x='absences', y='G3', color=sex_lbl,
                             color_discrete_map={'Masculin': '#818cf8', 'Feminin': '#f97316'},
                             opacity=0.7, labels={'absences': 'Absences', 'G3': 'Note finale G3', 'color': 'Sexe'})
        fig_abs.update_layout(**PLOT_LAYOUT, height=360, xaxis=GRID, yaxis=GRID,
                               legend=dict(bgcolor='rgba(0,0,0,0)', title='Sexe'))
        st.plotly_chart(fig_abs, use_container_width=True)

    # ── PAGE S3-3 : FACTEURS ──────────────────────
    elif page_s3 == "🔑 Facteurs de réussite":
        require_data_s3()
        df_s3 = st.session_state['s3_df']
        st.markdown('<div class="page-header">🔑 Facteurs de Reussite</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-sub">Identification des variables les plus influentes sur G3 (hors G1 et G2).</div>', unsafe_allow_html=True)

        with st.spinner("Calcul des importances de variables..."):
            fi_s3 = get_feature_importance_s3(df_s3)

        top15_s3 = fi_s3.head(15).copy()
        top15_s3['Feature'] = top15_s3['Feature'].str.replace('_', ' ', regex=False)
        fig_fi_s3 = px.bar(top15_s3, x='Importance', y='Feature', orientation='h',
                            color='Importance', color_continuous_scale='Purples',
                            title="Top 15 - Facteurs d'influence sur la note finale G3",
                            labels={'Importance': "Niveau d'influence", 'Feature': 'Facteur'})
        fig_fi_s3.update_layout(yaxis={'autorange': 'reversed'}, coloraxis_showscale=False,
                                  **PLOT_LAYOUT, height=500, xaxis=GRID)
        st.plotly_chart(fig_fi_s3, use_container_width=True)

        top3_s3 = fi_s3.head(3)['Feature'].str.replace('_', ' ', regex=False).tolist()
        st.info(
            "**Les 3 principaux facteurs de reussite :**\n\n"
            f"1. **{top3_s3[0]}** - Ce facteur a la plus forte influence sur la note finale. "
            "Il reflete directement l'investissement ou les conditions d'apprentissage de l'etudiant.\n\n"
            f"2. **{top3_s3[1]}** - En deuxieme position, ce facteur modifie significativement "
            "les chances de reussite. Une action pedagogique ciblee sur ce point peut faire la difference.\n\n"
            f"3. **{top3_s3[2]}** - Troisieme facteur cle. Sa correlation avec G3 suggere qu'un suivi "
            "personnalise sur cette dimension ameliorerait les resultats globaux."
        )

        st.markdown('<div class="sec-title">Matrice de correlation - Variables numeriques</div>', unsafe_allow_html=True)
        num_cols_s3 = [c for c in ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                                    'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
                                    'absences', 'G1', 'G2', 'G3'] if c in df_s3.columns]
        corr_s3 = df_s3[num_cols_s3].corr().round(2)
        fig_corr_s3 = px.imshow(corr_s3, text_auto=True, color_continuous_scale='RdBu_r',
                                  zmin=-1, zmax=1, title='Matrice de correlation')
        fig_corr_s3.update_layout(**{**PLOT_LAYOUT, 'height': 600, 'margin': dict(t=50)})
        st.plotly_chart(fig_corr_s3, use_container_width=True)
        st.markdown('<div class="alert-info">Cases rouges = correlation positive. Cases bleues = correlation negative. La colonne G3 montre les facteurs les plus lies a la note finale.</div>', unsafe_allow_html=True)

    # ── PAGE S3-4 : CLUSTERING ────────────────────
    elif page_s3 == "👥 Profils étudiants (Clustering)":
        require_data_s3()
        df_s3 = st.session_state['s3_df']
        st.markdown('<div class="page-header">👥 Profils Etudiants</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-sub">Segmentation automatique des etudiants en groupes aux comportements similaires.</div>', unsafe_allow_html=True)

        with st.spinner("Calcul des courbes Elbow et Silhouette..."):
            ks_s3, inertias_s3, sils_s3 = compute_elbow_s3(df_s3)

        col_elb, col_sil = st.columns(2)
        with col_elb:
            st.markdown('<div class="sec-title">Courbe Elbow</div>', unsafe_allow_html=True)
            fig_elb = px.line(x=ks_s3, y=inertias_s3, markers=True,
                              labels={'x': 'Nombre de groupes (k)', 'y': 'Inertie'},
                              color_discrete_sequence=['#818cf8'])
            fig_elb.update_layout(**PLOT_LAYOUT, height=280, xaxis=GRID, yaxis=GRID)
            st.plotly_chart(fig_elb, use_container_width=True)

        with col_sil:
            st.markdown('<div class="sec-title">Score de Silhouette</div>', unsafe_allow_html=True)
            fig_sil_s3 = px.line(x=ks_s3, y=sils_s3, markers=True,
                                  labels={'x': 'Nombre de groupes (k)', 'y': 'Score Silhouette'},
                                  color_discrete_sequence=['#34d399'])
            fig_sil_s3.update_layout(**PLOT_LAYOUT, height=280, xaxis=GRID, yaxis=GRID)
            st.plotly_chart(fig_sil_s3, use_container_width=True)

        k_val = st.slider("Choisir le nombre de groupes", min_value=2, max_value=8,
                           value=st.session_state.get('s3_k', 4), key='s3_k_slider')
        st.session_state['s3_k'] = k_val

        with st.spinner(f"Segmentation en {k_val} groupes..."):
            s3_labels = cluster_students_s3(df_s3, k_val)

        df_s3_cl = df_s3.copy()
        df_s3_cl['Cluster'] = s3_labels
        df_s3_cl['Groupe'] = pd.Series(s3_labels).apply(lambda x: f'Groupe {x}').values

        st.markdown('<div class="sec-title">Carte des etudiants : Absences vs Note finale</div>', unsafe_allow_html=True)
        s3_pal = ['#818cf8','#f97316','#8b5cf6','#34d399','#38bdf8','#fbbf24','#f87171','#a78bfa']
        fig_cl_sc = px.scatter(df_s3_cl, x='absences', y='G3', color='Groupe',
                                color_discrete_sequence=s3_pal[:k_val], opacity=0.75,
                                labels={'absences': 'Absences', 'G3': 'Note finale G3'})
        fig_cl_sc.update_layout(**PLOT_LAYOUT, height=380, xaxis=GRID, yaxis=GRID,
                                  legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig_cl_sc, use_container_width=True)

        st.markdown('<div class="sec-title">Profil de chaque groupe</div>', unsafe_allow_html=True)
        prof_cols_s3 = [c for c in ['studytime','failures','absences','Dalc','Walc','G1','G2','G3'] if c in df_s3_cl.columns]
        prof_s3 = df_s3_cl.groupby('Cluster')[prof_cols_s3].mean().round(2).reset_index()
        prof_s3['Etiquette'] = prof_s3['G3'].apply(label_cluster_s3)
        sz_map = df_s3_cl['Cluster'].value_counts().to_dict()
        prof_s3['Nb etudiants'] = prof_s3['Cluster'].map(sz_map)
        ren = {'studytime':'Studytime moy.','failures':'Failures moy.','absences':'Absences moy.',
               'Dalc':'Dalc moy.','Walc':'Walc moy.','G1':'G1 moy.','G2':'G2 moy.','G3':'G3 moyen'}
        prof_s3 = prof_s3.rename(columns=ren)
        disp = ['Cluster','Etiquette','Nb etudiants','G3 moyen','Studytime moy.','Failures moy.',
                'Absences moy.','Dalc moy.','Walc moy.']
        st.dataframe(prof_s3[[c for c in disp if c in prof_s3.columns]], use_container_width=True, hide_index=True)

    # ── PAGE S3-5 : PREDIRE ───────────────────────
    elif page_s3 == "🤖 Prédire la note finale":
        require_data_s3()
        df_s3 = st.session_state['s3_df']
        st.markdown('<div class="page-header">Predire la Note Finale</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-sub">Trois modeles de machine learning pour estimer G3 sans utiliser G1 et G2.</div>', unsafe_allow_html=True)

        with st.expander("Pourquoi ces modeles ?"):
            st.markdown("""
- **Regression Lineaire** : Modele de reference pour comprendre les relations lineaires entre les facteurs et la note finale.
- **Foret Aleatoire** : Capture les relations non lineaires entre variables. Robuste aux outliers et aux variables peu importantes.
- **Gradient Boosting** : Optimise les erreurs successivement. Tres performant sur des donnees tabulaires de taille moyenne.
- **Note** : On exclut G1 et G2 car ce sont des notes intermediaires, pas des facteurs d'entree disponibles avant le cours.
""")

        with st.spinner("Entrainement des modeles..."):
            res_s3, best_s3, all_pred_s3, _ = train_models_s3(df_s3)

        st.markdown(f'<div class="alert-ok">Entrainement termine ! Meilleur modele : <b>{best_s3}</b> R2 = {res_s3[best_s3]["r2"]:.3f}</div>', unsafe_allow_html=True)

        st.markdown('<div class="sec-title">Performances des modeles</div>', unsafe_allow_html=True)
        met_rows = [{'Modele': n, 'RMSE': f"{r['rmse']:.3f}", 'MAE': f"{r['mae']:.3f}", 'R2': f"{r['r2']:.3f}"}
                    for n, r in res_s3.items()]
        st.dataframe(pd.DataFrame(met_rows).set_index('Modele'), use_container_width=True)

        st.markdown('<div class="sec-title">Predictions vs Realite</div>', unsafe_allow_html=True)
        pred_cols_s3 = st.columns(len(res_s3))
        for col_p, (name, r) in zip(pred_cols_s3, res_s3.items()):
            with col_p:
                max_v = float(max(np.max(r['actual']), np.max(r['pred']))) + 1
                fig_p = px.scatter(x=r['actual'], y=r['pred'], opacity=0.6,
                                   labels={'x': 'Note reelle', 'y': 'Note predite'},
                                   title=name, color_discrete_sequence=['#818cf8'])
                fig_p.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode='lines',
                                            line=dict(color='#f87171', dash='dash'), showlegend=False))
                fig_p.update_layout(**PLOT_LAYOUT, height=320, xaxis=GRID, yaxis=GRID)
                st.plotly_chart(fig_p, use_container_width=True)
                st.markdown(
                    f'<div class="glass-card" style="text-align:left;padding:10px 14px">'
                    f'RMSE <b style="color:#f87171">{r["rmse"]:.3f}</b> &nbsp;|&nbsp; '
                    f'MAE <b style="color:#34d399">{r["mae"]:.3f}</b> &nbsp;|&nbsp; '
                    f'R2 <b style="color:#fbbf24">{r["r2"]:.3f}</b>'
                    f'</div>', unsafe_allow_html=True)

    # ── PAGE S3-6 : RECOMMANDATIONS ───────────────
    elif page_s3 == "💡 Recommandations pédagogiques":
        require_data_s3()
        df_s3 = st.session_state['s3_df']
        st.markdown('<div class="page-header">Recommandations Pedagogiques</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-sub">Actions concretes par profil etudiant pour ameliorer les resultats.</div>', unsafe_allow_html=True)

        k_r = st.session_state.get('s3_k', 4)
        s3_labs_r = cluster_students_s3(df_s3, k_r)
        df_r = df_s3.copy()
        df_r['Cluster'] = s3_labs_r

        with st.spinner("Chargement des recommandations..."):
            res_r, best_r, all_pred_r, _ = train_models_s3(df_s3)
            fi_r = get_feature_importance_s3(df_s3)

        df_r['G3_predit'] = np.clip(all_pred_r, 0, 20).round(1)
        cl_g3_r = df_r.groupby('Cluster')['G3'].mean()
        df_r['Profil'] = df_r['Cluster'].map(cl_g3_r).apply(label_cluster_s3)
        cl_size_r = df_r['Cluster'].value_counts()

        RECO_S3 = {
            "🟢 Excellents": ("Proposer des programmes enrichis et des projets avances. Ces etudiants peuvent servir de tuteurs pour leurs camarades.", '#34d399', 'green'),
            "🔵 Bons élèves": ("Maintenir l'encadrement actuel. Encourager la participation a des activites parascolaires.", '#818cf8', 'blue'),
            "🟡 Fragiles": ("Mettre en place un suivi hebdomadaire personnalise. Identifier les lacunes specifiques avant qu'elles s'aggravent.", '#fbbf24', 'amber'),
            "🔴 En difficulté": ("Intervention immediate : tutorat, soutien psychologique, reduction des absences. Contacter les parents.", '#f87171', 'red'),
        }

        for cid in cl_g3_r.sort_values(ascending=False).index:
            g3m = cl_g3_r[cid]
            lbl = label_cluster_s3(g3m)
            reco_txt, clr, cls = RECO_S3.get(lbl, ("Suivi personnalise recommande.", '#818cf8', 'blue'))
            nb = int(cl_size_r.get(cid, 0))
            st.markdown(f"""
<div style="background:rgba(255,255,255,0.04);border:1px solid {clr}44;
            border-left:5px solid {clr};border-radius:14px;padding:18px 22px;margin-bottom:14px">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
        <b style="color:{clr};font-size:1.1rem">Cluster {cid} - {lbl}</b>
        <span class="badge badge-{cls}">{nb} etudiants - G3 moy. {g3m:.1f}/20</span>
    </div>
    <div class="rec-card"><b>Recommandation</b><br><span>{reco_txt}</span></div>
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        at_risk_s3 = int((df_s3['G3'] < 10).sum())
        pct_risk_s3 = at_risk_s3 / len(df_s3) * 100
        top_fact_s3 = fi_r.iloc[0]['Feature'].replace('_', ' ')
        c1r, c2r, c3r = st.columns(3)
        c1r.metric("Etudiants a risque (G3 < 10)", f"{at_risk_s3:,}")
        c2r.metric("Part du total", f"{pct_risk_s3:.1f}%")
        c3r.metric("Facteur le plus impactant", top_fact_s3)

        csv_s3 = df_r.to_csv(index=False, sep=';').encode('utf-8')
        st.download_button(
            label="Exporter les profils etudiants (CSV)",
            data=csv_s3,
            file_name="etudiants_profils.csv",
            mime="text/csv",
            use_container_width=True,
        )
