"""
Amazon Alexa Sentiment Analysis — SVM App
==========================================
Run the notebook ONCE to generate the Models/ folder.
After that, this app loads instantly with no dataset upload needed.

Folder structure expected:
  Models/
    model_svm.pkl
    tfidf_vectorizer.pkl
    scaler.pkl
    threshold.json
    metrics.json
    data_processed.csv
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import re, json, pickle, os
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics import (roc_curve, precision_recall_curve,
                             confusion_matrix, ConfusionMatrixDisplay)
from wordcloud import WordCloud
import warnings
warnings.filterwarnings("ignore")

# ── NLTK ──────────────────────────────────────────────────────────────────────
import nltk

try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))

STOPWORDS = STOPWORDS - {"not", "no", "never", "n't"}
STEMMER   = PorterStemmer()
PALETTE   = ["#232F3E", "#146EB4", "#FF9900", "#d62728", "#2ca02c"]

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alexa · SVM Sentiment",
    page_icon="https://ds6yc8t7pnx74.cloudfront.net/content/dam/alexa/alexa-brand-guidelines-2021-refresh-/Alexa_Logo_RGB_BLUE.png/_jcr_content/renditions/cq5dam.web.1280.1280.webp",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono&display=swap');
html,body,[class*="css"]{ font-family:'DM Sans',sans-serif; }

.banner{
    background:linear-gradient(135deg,#0f1923 0%,#1a2f4e 50%,#0f1923 100%);
    border-radius:18px; padding:2rem 2.5rem; margin-bottom:1.5rem;
    border:1px solid #2a4a6e;
    box-shadow:0 8px 32px rgba(0,0,0,0.4);
}
.banner h1{
    margin:0; font-size:2.2rem; font-weight:700; letter-spacing:-0.5px;
    background:linear-gradient(90deg,#fff 0%,#7ec8f7 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.banner p{ margin:0.4rem 0 0; color:#8aaec8; font-size:0.95rem; }

.kpi{
    background:#0f1923; border:1px solid #1e3a56;
    border-radius:14px; padding:1.2rem 1rem;
    text-align:center; color:white; margin:0.2rem;
    box-shadow:0 2px 12px rgba(0,0,0,0.3);
}
.kpi .v{ font-size:2rem; font-weight:700; color:#7ec8f7; }
.kpi .l{ font-size:0.7rem; color:#8aaec8; text-transform:uppercase;
          letter-spacing:1px; margin-top:0.3rem; }

.best-badge{
    background:linear-gradient(135deg,#146EB4,#0d5a9e);
    border-radius:14px; padding:1.3rem 1.5rem; color:white; margin:0.3rem 0;
    box-shadow:0 4px 16px rgba(20,110,180,0.3);
}
.best-badge h4{ margin:0 0 0.4rem; font-size:0.8rem; opacity:0.7;
                text-transform:uppercase; letter-spacing:1px; }
.best-badge .v{ font-size:2rem; font-weight:700; }

.pos-pill{
    background:#d4edda; color:#155724;
    padding:0.5rem 1.6rem; border-radius:30px;
    font-weight:700; font-size:1.1rem; display:inline-block;
}
.neg-pill{
    background:#f8d7da; color:#721c24;
    padding:0.5rem 1.6rem; border-radius:30px;
    font-weight:700; font-size:1.1rem; display:inline-block;
}
.neu-pill{
    background:#fff3cd;
    color:#856404;
    padding:0.5rem 1.6rem;
    border-radius:30px;
    font-weight:700;
    font-size:1.1rem;
    display:inline-block;
}
.mono{
    font-family:'DM Mono', monospace;
    font-size:0.85rem;
    background:linear-gradient(135deg,#0f1923,#1a2f4e);
    color:#c9e6ff;
    border-left:4px solid #7ec8f7;
    border-radius:8px;
    padding:0.7rem 1rem;
    box-shadow:0 2px 10px rgba(0,0,0,0.3);
}

.stTabs [data-baseweb="tab-list"]{ 
    gap:6px; 
}

/* Inactive tabs */
.stTabs [data-baseweb="tab"]{
    background:#1a2f4e;              /* dark blue */
    color:#c9e6ff;                   /* light text */
    border-radius:8px 8px 0 0;
    padding:8px 20px;
    font-weight:500;
    transition: all 0.3s ease;
}

/* Hover effect */
.stTabs [data-baseweb="tab"]:hover{
    background:#243f66;
    color:#ffffff;
}

/* Active tab */
.stTabs [aria-selected="true"]{
    background:#0f1923 !important;   /* your theme dark */
    color:white !important;
    font-weight:600;
    box-shadow:0 3px 10px rgba(0,0,0,0.4);
}

/* Expander (optional dark tweak) */
div[data-testid="stExpander"]{
    border:1px solid #2a4a6e;
    border-radius:10px;
    background:#0f1923;
}
</style>
""", unsafe_allow_html=True)


# ── Model loading ──────────────────────────────────────────────────────────────
MODELS_DIR = "Models"

@st.cache_resource
def load_models():
    required = ["model_svm.pkl", "tfidf_vectorizer.pkl", "scaler.pkl",
                "metrics.json", "data_processed.csv", "threshold.json"]
    missing = [f for f in required
               if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        return None, None, None, None, None, None, missing

    model  = pickle.load(open(f"{MODELS_DIR}/model_svm.pkl",        "rb"))
    tfidf  = pickle.load(open(f"{MODELS_DIR}/tfidf_vectorizer.pkl", "rb"))
    scaler = pickle.load(open(f"{MODELS_DIR}/scaler.pkl",           "rb"))
    with open(f"{MODELS_DIR}/metrics.json") as f:
        metrics = json.load(f)
    with open(f"{MODELS_DIR}/threshold.json") as f:
        threshold = json.load(f)["best_threshold"]
    df = pd.read_csv(f"{MODELS_DIR}/data_processed.csv")
    return model, tfidf, scaler, metrics, df, threshold, []

model, tfidf, scaler, metrics, df, THRESHOLD, missing_files = load_models()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://ds6yc8t7pnx74.cloudfront.net/content/dam/alexa/alexa-brand-guidelines-2021-refresh-/Alexa_Logo_RGB_BLUE.png/_jcr_content/renditions/cq5dam.web.1280.1280.webp", width=140)
    st.markdown("---")
    if model:
        bp = metrics["best_params"]
        thresh = metrics.get("best_threshold", 0.5)
        st.markdown("### Models Loaded")
        st.markdown(f"""
        <div class="mono">
        kernel = <b>{bp['kernel']}</b><br>
        C      = <b>{bp['C']}</b><br>
        gamma  = <b>{bp.get('gamma','scale')}</b><br>
        class_weight = <b>balanced</b><br>
        threshold = <b>{thresh}</b>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(f"**Test Accuracy:** `{metrics['test_accuracy']*100:.2f}%`")
        st.markdown(f"**ROC-AUC:** `{metrics['roc_auc']:.4f}`")
        st.markdown(f"**Best CV F1:** `{metrics['best_cv_f1']*100:.2f}%`")
        if "macro_f1" in metrics:
            st.markdown(f"**Macro F1:** `{metrics['macro_f1']*100:.2f}%`")
        st.markdown("---")
        st.markdown("**Pipeline**")
        st.code("TF-IDF (3k, bigrams)\n→ MaxAbsScaler (sparse)\n→ SMOTE (train only)\n→ LinearSVC + CalibratedCV")
        # SMOTE balance info
        if "smote_balance" in metrics:
            sb = metrics["smote_balance"]
            st.markdown("**SMOTE Resampling**")
            st.markdown(f"Before: Neg `{sb['before_neg']}` / Pos `{sb['before_pos']}`")
            st.markdown(f"After:  Neg `{sb['after_neg']}` / Pos `{sb['after_pos']}`")
    else:
        st.error("Models not found")
        st.markdown("Run the notebook first to generate `Models/`")


# ── Banner ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="banner" style="display:flex; align-items:center; gap:15px;">
  
  <img src="https://ds6yc8t7pnx74.cloudfront.net/content/dam/alexa/alexa-brand-guidelines-2021-refresh-/Alexa_Logo_RGB_BLUE.png/_jcr_content/renditions/cq5dam.web.1280.1280.webp" 
       width="350" style="border-radius:10px;">

  <div>
    <h1>Amazon Alexa Sentiment Analysis</h1>
    <p>
        Support Vector Machine · TF-IDF (Bigrams) · SMOTE · Calibrated SVM · Threshold Optimization
    </p>
  </div>

</div>
""", unsafe_allow_html=True)


# ── Missing models guard ───────────────────────────────────────────────────────
if missing_files:
    st.error(f"**Missing model files:** {', '.join(missing_files)}")
    st.markdown("""
    ### How to set up
    1. Download [amazon_alexa.tsv from Kaggle](https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews)
       and place it in a `Data/` folder next to this app.
    2. Open **`Data_Exploration_Modelling_SVM.ipynb`** in Jupyter.
    3. Run all cells — this will create the `Models/` folder with all saved files.
    4. Come back here and refresh. No more waits! ⚡
    
    **Expected `Models/` contents after notebook run:**
    ```
    Models/
    ├── model_svm.pkl
    ├── tfidf_vectorizer.pkl
    ├── scaler.pkl
    ├── threshold.json
    ├── metrics.json
    └── data_processed.csv
    ```
    """)
    st.stop()


# ── Prediction helper ──────────────────────────────────────────────────────────
def preprocess(text):
    text = text.lower()

    # Expand contractions
    text = text.replace("don't", "do not")
    text = text.replace("doesn't", "does not")
    text = text.replace("didn't", "did not")
    text = text.replace("can't", "can not")
    text = text.replace("won't", "will not")

    words = re.sub("[^a-zA-Z]", " ", text).split()

    tokens = []
    negate = False

    for word in words:
        if word in ["not", "no", "never"]:
            negate = True
            continue

        if word not in STOPWORDS:
            stem = STEMMER.stem(word)

            if negate:
                tokens.append("not_" + stem)
                negate = False
            else:
                tokens.append(stem)

    return tokens
def predict(text: str):
    rev = preprocess(text)

    # TF-IDF → sparse → MaxAbsScaler → dense for inference
    vec_sparse = tfidf.transform([" ".join(rev)])          # sparse (1, 3000)
    vec_scl    = scaler.transform(vec_sparse).toarray()    # MaxAbsScaler, then dense
    vec_s      = vec_scl                                   # alias used below

    prob    = model.predict_proba(vec_s)[0]
    classes = model.classes_

    pos_index = list(classes).index(1)
    neg_index = list(classes).index(0)

    pos_prob = prob[pos_index]
    neg_prob = prob[neg_index]

    # Use the saved threshold (default 0.5 if not found)
    pred = 1 if pos_prob >= THRESHOLD else 0

    return pred, [neg_prob, pos_prob]

def make_wc(text, title, cmap):
    wc = WordCloud(background_color="white", max_words=60,
                   width=700, height=380, colormap=cmap)
    wc.generate(text)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  EDA",
    "⚙️  Pipeline & SVM Theory",
    "🤖  Model & Tuning",
    "🔍  Predict",
    "📈  Metrics",
])


# ══════════════════════════════════════════════
# TAB 1 · EDA
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Exploratory Data Analysis")

    # KPI row
    c1,c2,c3,c4 = st.columns(4)
    kpis = [
        (metrics["dataset_shape"][0], "Total Reviews"),
        (metrics["positive_count"],   "Positive Reviews"),
        (f"{round(metrics['negative_count']/metrics['dataset_shape'][0]*100,1)}%", "Negative Reviews"),
        (metrics["num_variations"],   "Product Variations"),
    ]
    for col,(val,lbl) in zip([c1,c2,c3,c4],kpis):
        col.markdown(f'<div class="kpi"><div class="v">{val}</div><div class="l">{lbl}</div></div>',
                     unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Dataset Preview")
    st.dataframe(df.head(12), width='stretch')

    l, r = st.columns(2)
    with l:
        fig = px.bar(df["rating"].value_counts().reset_index(),
                     x="rating", y="count", color="count",
                     color_continuous_scale="Reds", title="Rating Distribution",
                     labels={"rating":"Rating","count":"Count"})
        fig.update_layout(coloraxis_showscale=False, template="simple_white")
        st.plotly_chart(fig, width='stretch', key="rating_dist")

        fb = df["feedback"].value_counts().rename({0:"Negative",1:"Positive"})
        fig2 = px.pie(values=fb.values, names=fb.index,
                      color_discrete_sequence=["#dc3545","#28a745"],
                      title="Feedback Distribution", hole=0.4)
        st.plotly_chart(fig2, width='stretch', key="feedback_pie")

    with r:
        var_df = df["variation"].value_counts().reset_index()
        fig3 = px.bar(var_df, x="count", y="variation", orientation="h",
                      color="count", color_continuous_scale="Blues",
                      title="Reviews per Variation")
        fig3.update_layout(coloraxis_showscale=False, yaxis_title="", template="simple_white")
        st.plotly_chart(fig3, width='stretch', key="variation_bar")

        mr = df.groupby("variation")["rating"].mean().sort_values().reset_index()
        fig4 = px.bar(mr, x="rating", y="variation", orientation="h",
                      color="rating", color_continuous_scale="Greens",
                      title="Mean Rating by Variation")
        fig4.update_layout(coloraxis_showscale=False, yaxis_title="", template="simple_white")
        st.plotly_chart(fig4, width='stretch', key="mean_rating")

    st.markdown("---")
    st.markdown("#### Review Length Analysis")
    c1, c2 = st.columns(2)
    with c1:
        fig5 = px.histogram(df, x="length",
                            color=df["feedback"].map({0:"Negative",1:"Positive"}),
                            barmode="overlay", nbins=60,
                            color_discrete_map={"Negative":"#dc3545","Positive":"#28a745"},
                            title="Review Length by Sentiment")
        fig5.update_layout(template="simple_white")
        st.plotly_chart(fig5, width='stretch', key="length_hist")
    with c2:
        fig6 = px.box(df, x=df["feedback"].map({0:"Negative",1:"Positive"}), y="length",
                      color=df["feedback"].map({0:"Negative",1:"Positive"}),
                      color_discrete_map={"Negative":"#dc3545","Positive":"#28a745"},
                      title="Length Box Plot by Sentiment")
        fig6.update_layout(xaxis_title="Sentiment", showlegend=False, template="simple_white")
        st.plotly_chart(fig6, width='stretch', key="length_box")

    st.markdown("---")
    st.markdown("#### Word Clouds")
    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        st.pyplot(make_wc(" ".join(df["verified_reviews"]), "All Reviews", "Blues"),
                  width='stretch')
    with wc2:
        st.pyplot(make_wc(" ".join(df[df["feedback"]==0]["verified_reviews"]),
                           "Negative Reviews", "Reds"), width='stretch')
    with wc3:
        st.pyplot(make_wc(" ".join(df[df["feedback"]==1]["verified_reviews"]),
                           "Positive Reviews", "Greens"), width='stretch')

    st.markdown("---")
    st.markdown("#### Rating vs Feedback Cross-analysis")
    cross = df.groupby(["rating","feedback"]).size().reset_index(name="count")
    cross["Sentiment"] = cross["feedback"].map({0:"Negative",1:"Positive"})
    fig7 = px.bar(cross, x="rating", y="count", color="Sentiment",
                  color_discrete_map={"Negative":"#dc3545","Positive":"#28a745"},
                  barmode="group", title="Rating vs Sentiment Count",
                  labels={"rating":"Rating","count":"Count"})
    fig7.update_layout(template="simple_white")
    st.plotly_chart(fig7, width='stretch', key="rating_vs_sentiment")


# ══════════════════════════════════════════════
# TAB 2 · Pipeline & SVM Theory
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Preprocessing Pipeline")

    pipe_steps = [
        ("1️⃣  Drop Nulls",        f"Removed null rows → **{metrics['dataset_shape'][0]} rows** retained"),
        ("2️⃣  Regex Cleaning",    "Strip all non-alphabetic characters: `re.sub('[^a-zA-Z]', ' ', text)`"),
        ("3️⃣  Lowercase",         "Convert all text to lowercase for uniformity"),
        ("4️⃣  Stopword Removal",  f"Removed **{len(STOPWORDS)}** English stopwords using NLTK corpus"),
        ("5️⃣  Porter Stemming",   "Reduce words to root: *loving → love*, *plays → play*, *running → run*"),
        ("6️⃣  TF-IDF Vectorizer", "`max_features=3000, ngram_range=(1,2), sublinear_tf=True` — captures unigrams & bigrams, penalises common words"),
        ("7️⃣  Stratified Split",  "70% train · 30% test · `stratify=y` to preserve class ratio in both sets"),
        ("8️⃣  MaxAbsScaler",      "Scales each feature by its maximum absolute value → **[0, 1]** for TF-IDF. **Preserves sparsity** (no centering) — ideal for sparse matrices. Dense conversion happens only when needed for SMOTE."),
        ("9️⃣  SMOTE (train only)", "Synthetic Minority Over-sampling creates new Negative samples in feature space until classes are balanced. **Test set is never touched** — no data leakage."),
    ]
    for title, desc in pipe_steps:
        with st.expander(title):
            st.markdown(desc)

    st.markdown("---")
    st.markdown("### SVM Theory")

    col_l, col_r = st.columns([1.2, 1])
    with col_l:
        st.markdown("""
**Support Vector Machine** solves the following optimisation:

> Maximise the **margin** between two class hyperplanes, subject to all points being correctly classified (with slack for soft-margin).

| Concept | Meaning |
|---|---|
| **Hyperplane** | Decision boundary: `w·x + b = 0` |
| **Support Vectors** | Points closest to the hyperplane — they *define* the margin |
| **Margin** | `2 / ‖w‖` — SVM maximises this distance |
| **C** | Regularisation: low C → wide margin, more tolerance; high C → tight margin, fewer errors |
| **Kernel** | Maps features to higher-dim space implicitly via `K(xᵢ, xⱼ)` |
| **SMOTE** | Balances training data synthetically before fitting — removes need for `class_weight` alone |
| **class_weight='balanced'** | Extra safety net alongside SMOTE to handle any residual imbalance |

**Why TF-IDF over raw Bag-of-Words for SVM?**
- Penalises very common words (*"the", "and"*) that carry no sentiment signal
- Rewards rare, informative words (*"broken", "amazing"*)
- Bigrams capture *"not good"*, *"very bad"* that unigrams miss
- `sublinear_tf=True` applies log scaling → prevents length bias
""")

    with col_r:
        # 2-D SVM illustration
        from sklearn.svm import SVC as _SVC
        np.random.seed(42)
        n = 45
        Xp = np.random.randn(n,2) + [2.2, 2.2]
        Xn = np.random.randn(n,2) + [-2.2,-2.2]
        Xt = np.vstack([Xp, Xn])
        yt = np.array([1]*n + [0]*n)
        toy = _SVC(kernel="linear", C=1)
        toy.fit(Xt, yt)
        xx, yy = np.meshgrid(np.linspace(-6,6,250), np.linspace(-6,6,250))
        Z = toy.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        fig_sv, ax = plt.subplots(figsize=(5.5, 5))
        ax.contourf(xx, yy, Z, levels=[-4,-1,0,1,4],
                    colors=["#f8d7da","#fff9e6","#d4edda"], alpha=0.55)
        ax.contour(xx, yy, Z, levels=[-1,0,1],
                   linestyles=["--","-","--"],
                   colors=["#dc3545","#232F3E","#28a745"],
                   linewidths=[1.8, 2.8, 1.8])
        ax.scatter(Xp[:,0], Xp[:,1], c="#28a745", edgecolors="white",
                   s=60, label="Positive", zorder=3)
        ax.scatter(Xn[:,0], Xn[:,1], c="#dc3545", edgecolors="white",
                   s=60, label="Negative", zorder=3)
        sv = toy.support_vectors_
        ax.scatter(sv[:,0], sv[:,1], s=200, facecolors="none",
                   edgecolors="#232F3E", linewidths=2.2,
                   label="Support Vectors", zorder=5)
        # Annotate margin arrows
        ax.annotate("", xy=(0.6, 0.6), xytext=(-0.6,-0.6),
                    arrowprops=dict(arrowstyle="<->", color="#FF9900", lw=1.8))
        ax.text(0.1, 0.1, "Margin", fontsize=9, color="#FF9900",
                rotation=45, ha="center", va="center")
        ax.set_title("SVM — Optimal Hyperplane (2D Demo)",
                     fontweight="bold", fontsize=11)
        ax.legend(fontsize=9, loc="upper left")
        ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
        ax.set_xlim(-6,6); ax.set_ylim(-6,6)
        plt.tight_layout()
        st.pyplot(fig_sv, width='stretch')

    st.markdown("---")
    st.markdown("### Kernel Comparison")
    kern_df = pd.DataFrame({
        "Kernel":   ["Linear","RBF (Gaussian)","Polynomial","Sigmoid"],
        "Formula":  ["K = xᵀy",
                     "K = exp(−γ‖x−y‖²)",
                     "K = (γxᵀy + r)ᵈ",
                     "K = tanh(γxᵀy + r)"],
        "Best For": ["Text, linear data",
                     "Non-linear, complex boundaries",
                     "Polynomial feature interactions",
                     "Neural-net-like activation"],
        "Speed":    ["⚡ Fastest","🔵 Moderate","🟡 Slower","🔵 Moderate"],
        "NLP Rating":["⭐⭐⭐⭐⭐","⭐⭐⭐⭐","⭐⭐⭐","⭐⭐"],
    })
    st.dataframe(kern_df, width='stretch', hide_index=True)

    # Feature stats
    st.markdown("---")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Samples", metrics["dataset_shape"][0])
    c2.metric("TF-IDF Features", "3 000")
    c3.metric("Bigrams (n=2)", "✅ Enabled")
    c4.metric("sublinear_tf", "✅ Enabled")

    st.markdown("---")

    # ─────────────────────────────────────────
    # LIVE EXAMPLE WALKTHROUGH
    # ─────────────────────────────────────────
    st.markdown("### Step-by-Step Example (Live)")

    example = st.text_input(
        "Try your own sentence:",
        "do not buy this product"
    )

    if example:

        # Step 1: Preprocess
        tokens = preprocess(example)

        st.markdown("####  Step 1: Preprocessing")
        st.write("**Original:**", example)
        st.write("**Processed Tokens:**", tokens)

        # Step 2: TF-IDF
        vec_sparse = tfidf.transform([" ".join(tokens)])           # sparse
        vec = vec_sparse.toarray()[0]                              # dense for display

        st.markdown("#### Step 2: TF-IDF Vector (Top Features)")
        nz = np.where(vec > 0)[0]

        if len(nz) > 0:
            for i in nz[:10]:
                st.write(f"• {tfidf.get_feature_names_out()[i]} → {vec[i]:.4f}")
        else:
            st.write("No important words found")

        # Step 3: Scaling
        vec_scaled = scaler.transform(vec_sparse).toarray()[0]   # MaxAbsScaler on sparse

        st.markdown("#### Step 3: Feature Scaling (MaxAbsScaler)")

        st.latex(r"x' = \frac{x}{\max|x|}")

        if len(nz) > 0:
            idx = nz[0]
            original = vec[idx]
            max_abs_val = scaler.max_abs_[idx]

            scaled_val = original / max_abs_val if max_abs_val != 0 else 0

            st.write(f"Feature: **{tfidf.get_feature_names_out()[idx]}**")
            st.write(f"Original TF-IDF: {original:.4f}")
            st.write(f"Max Absolute: {max_abs_val:.4f}")
            st.write(f"Scaled Value: {scaled_val:.4f}")

        # Step 4: Decision / Probability
        st.markdown("#### Step 4: Probability Score (CalibratedClassifierCV)")
        st.latex(r"P(\hat{y}=1 \mid x)")

        vec_s_1d = scaler.transform(vec_sparse).toarray()[0]
        prob_step = model.predict_proba([vec_s_1d])[0]
        pos_prob_step = prob_step[list(model.classes_).index(1)]
        st.write(f"Positive Probability: **{pos_prob_step:.4f}**")
        st.write(f"Decision Threshold:   **{THRESHOLD}**")

        # Final prediction
        if pos_prob_step >= THRESHOLD:
            st.success("Prediction → Positive ✅")
        else:
            st.error("Prediction → Negative ❌")

    st.markdown("---")

    # ─────────────────────────────────────────
    # SVM MATH
    # ─────────────────────────────────────────
    st.markdown("### SVM Mathematical Formulation")

    st.latex(r"\min_{w,b,\xi} \frac{1}{2}||w||^2 + C \sum \xi_i")

    st.markdown("Subject to:")

    st.latex(r"y_i(w \cdot x_i + b) \geq 1 - \xi_i")

    st.markdown("""
    - **w** → weight vector  
    - **b** → bias  
    - **C** → regularisation parameter  
    - **ξ (xi)** → slack variable (error tolerance)  

    👉 Objective: maximize margin + minimize classification error
    """)

    st.markdown("---")

    # ─────────────────────────────────────────
    # MARGIN INTUITION
    # ─────────────────────────────────────────
    st.markdown("### 📏 Margin Concept")

    st.latex(r"\text{Margin} = \frac{2}{||w||}")

    st.markdown("""
    - Larger margin → better generalisation  
    - Smaller margin → overfitting  

    SVM always selects the hyperplane with **maximum margin**
    """)

    st.markdown("---")

    # ─────────────────────────────────────────
    # WHY SVM FOR TEXT
    # ─────────────────────────────────────────
    st.markdown("###  Why SVM Works Well for NLP")

    st.markdown("""
    ✔ Handles high-dimensional sparse data (TF-IDF)  
    ✔ Effective in linear separation problems  
    ✔ Uses only support vectors → memory efficient  
    ✔ Strong performance on small & medium datasets  
    ✔ Works well with class imbalance using `class_weight='balanced'`
    """)

    st.markdown("---")
# ══════════════════════════════════════════════
# TAB 3 · Model & Tuning
# ══════════════════════════════════════════════
with tab3:
    st.subheader("🤖 Hyperparameter Tuning & Model Analysis")

    # Best params
    bp = metrics["best_params"]
    st.markdown("### GridSearchCV Best Parameters")
    b1,b2,b3,b4 = st.columns(4)
    with b1:
        st.markdown(f'<div class="best-badge"><h4>Kernel</h4><div class="v">{bp["kernel"].upper()}</div></div>',
                    unsafe_allow_html=True)
    with b2:
        st.markdown(f'<div class="best-badge"><h4>C</h4><div class="v">{bp["C"]}</div></div>',
                    unsafe_allow_html=True)
    with b3:
        st.markdown(f'<div class="best-badge"><h4>Gamma</h4><div class="v">{bp.get("gamma","scale")}</div></div>',
                    unsafe_allow_html=True)
    with b4:
        st.markdown(f'<div class="best-badge"><h4>Best CV F1</h4><div class="v">{metrics["best_cv_f1"]*100:.1f}%</div></div>',
                    unsafe_allow_html=True)

    st.markdown(f"""
    <div class="mono">
    GridSearchCV: 9 C values × 5-fold CV = <b>45 fits</b> on SMOTE-balanced data<br>
    Scoring metric: <b>F1 Macro</b> (better for imbalanced data)<br>
    Best: kernel=<b>{bp['kernel']}</b> | C=<b>{bp['C']}</b> | threshold=<b>{metrics.get('best_threshold', 0.5)}</b> | class_weight=<b>balanced</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Performance Summary")
    perf_df = pd.DataFrame([{"Metric": "Train Accuracy",  "Value": f"{metrics['train_accuracy']*100:.2f}%"},
        {"Metric": "Test Accuracy",  "Value": f"{metrics['test_accuracy']*100:.2f}%"},
        {"Metric": "Macro F1",       "Value": f"{metrics.get('macro_f1', 0)*100:.2f}%"},
        {"Metric": "ROC-AUC",        "Value": f"{metrics['roc_auc']:.4f}"},
        {"Metric": "Avg Precision",  "Value": f"{metrics['avg_precision']:.4f}"},
        {"Metric": "CV F1 Mean",     "Value": f"{metrics['cv_f1_mean']*100:.2f}%"},
        {"Metric": "CV F1 Std",      "Value": f"±{metrics['cv_f1_std']*100:.2f}%"},
    ])
    st.dataframe(perf_df, width='stretch', hide_index=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Confusion Matrix")
        cm = np.array(metrics["confusion_matrix"])
        fig_cm, ax = plt.subplots(figsize=(5,4))
        ConfusionMatrixDisplay(cm, display_labels=["Negative","Positive"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Best SVM ({bp['kernel'].upper()} kernel)", fontweight="bold")
        # annotate TN/FP/FN/TP
        labels = [["TN","FP"],["FN","TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(j+0.5, i+0.72, labels[i][j],
                        ha="center", va="center", fontsize=8,
                        color="white" if cm[i,j]>cm.max()*0.5 else "gray",
                        transform=ax.transData)
        plt.tight_layout()
        st.pyplot(fig_cm, width='stretch')

    with col_b:
        st.markdown("#### Classification Report")
        rpt = metrics["classification_report"]
        rpt_df = pd.DataFrame(rpt).transpose().round(3)
        st.dataframe(rpt_df, width='stretch')
        st.markdown(f"""
        **Key insight:**  
        - Precision for *Positive* class: **{rpt['Positive']['precision']*100:.1f}%**  
        - Recall for *Negative* class: **{rpt['Negative']['recall']*100:.1f}%** ← improved by SMOTE  
        - SMOTE balanced training data synthetically; threshold `{metrics.get('best_threshold', 0.5)}` further optimises minority-class recall
        """)

    st.markdown("---")
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("#### ROC Curve")
        fpr = metrics["roc_fpr"]
        tpr = metrics["roc_tpr"]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"Best SVM (AUC = {metrics['roc_auc']:.3f})",
            line=dict(color="#146EB4", width=2.5),
            fill="tozeroy", fillcolor="rgba(20,110,180,0.1)"
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines", name="Random",
            line=dict(color="#dc3545", dash="dash")))
        fig_roc.update_layout(template="simple_white", title="ROC Curve",
                              xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, width='stretch', key="roc_curve")

    with col_d:
        st.markdown("#### Class Balance (Original Split)")
        cb = metrics["class_balance"]
        bal_df = pd.DataFrame({
            "Split": ["Train","Train","Test","Test"],
            "Class": ["Positive","Negative","Positive","Negative"],
            "Count": [cb["train_pos"],cb["train_neg"],cb["test_pos"],cb["test_neg"]]
        })
        fig_bal = px.bar(bal_df, x="Split", y="Count", color="Class",
                         color_discrete_map={"Positive":"#28a745","Negative":"#dc3545"},
                         barmode="group", title="Class Distribution: Train vs Test")
        fig_bal.update_layout(template="simple_white")
        st.plotly_chart(fig_bal, width='stretch', key="class_balance")

        # SMOTE before/after
        if "smote_balance" in metrics:
            st.markdown("#### SMOTE Effect on Training Data")
            sb = metrics["smote_balance"]
            smote_df = pd.DataFrame({
                "Stage":  ["Before SMOTE","Before SMOTE","After SMOTE","After SMOTE"],
                "Class":  ["Positive","Negative","Positive","Negative"],
                "Count":  [sb["before_pos"], sb["before_neg"], sb["after_pos"], sb["after_neg"]],
            })
            fig_smote = px.bar(smote_df, x="Stage", y="Count", color="Class",
                               color_discrete_map={"Positive":"#28a745","Negative":"#dc3545"},
                               barmode="group", title="SMOTE: Before vs After")
            fig_smote.update_layout(template="simple_white")
            st.plotly_chart(fig_smote, width='stretch', key="smote_balance")


# ══════════════════════════════════════════════
# TAB 4 · Predict
# ══════════════════════════════════════════════
with tab4:
    st.subheader("🔍 Real-Time Sentiment Prediction")
    st.markdown("The model is pre-loaded — predictions are **instant**.")

    user_rev = st.text_area(
        "Enter an Alexa review:",
        placeholder="e.g. I absolutely love my Alexa! It plays music perfectly and understands every command.",
        height=130
    )

    if st.button(" Predict Sentiment", type="primary"):
        if not user_rev.strip():
            st.warning("Please enter a review.")
        else:
            pred, prob = predict(user_rev)

            pos_conf = prob[1]
            neg_conf = prob[0]

            # Neutral condition (confidence in middle range)
            if abs(pos_conf - neg_conf) < 0.2:
                label = "Neutral 😐"
                pill = "neu-pill"
            elif pos_conf > 0.6:
                label = "Positive ✅"
                pill = "pos-pill"
            else:
                label = "Negative ❌"
                pill = "neg-pill"

            st.markdown("---")
            c1,c2,c3 = st.columns(3)
            with c1:
                st.markdown(f'<br><span class="{pill}">{label}</span>',
                            unsafe_allow_html=True)
            with c2:
                st.metric("Positive confidence", f"{prob[1]*100:.1f}%")
            with c3:
                st.metric("Negative confidence", f"{prob[0]*100:.1f}%")

            # Decide color based on confidence
            if abs(pos_conf - neg_conf) < 0.2:
                gauge_color = "#ffc107"   # Yellow (Neutral)
            elif pos_conf > 0.6:
                gauge_color = "#28a745"   # Green (Positive)
            else:
                gauge_color = "#dc3545"   # Red (Negative)
            # Gauge
            fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pos_conf * 100,
            delta={"reference":50},
            title={"text":"Positive Sentiment Score (%)"},
            gauge={
                "axis": {"range":[0,100]},
                "bar": {"color": gauge_color},   # ✅ UPDATED
                "steps":[
                    {"range":[0,40],  "color":"#fde8ea"},   # red zone
                    {"range":[40,60], "color":"#fff3cd"},   # yellow zone ✅ FIXED
                    {"range":[60,100],"color":"#e8f5e9"},   # green zone
                ],
                    "threshold":{"line":{"color":"black","width":3},"value":50}
            }
            ))
            fig_g.update_layout(height=300, margin=dict(t=60,b=20,l=20,r=20))
            st.plotly_chart(fig_g, width='stretch', key="gauge_chart")

            # Show preprocessed tokens
            with st.expander(" See how the review was preprocessed"):
                tokens = preprocess(user_rev)
                st.markdown(f"**Tokens after cleaning & stemming ({len(tokens)} terms):**")
                st.code(" · ".join(tokens))

    st.markdown("---")
    st.markdown("#### Try Sample Reviews")
    samples = {
        "😊 Strong Positive": "Absolutely love this! Best smart speaker ever. Plays music perfectly, controls all my devices, answers every question. 5 stars!",
        "😡 Strong Negative": "Terrible product. Broke after a week. Support was useless. Wasted my money. Horrible experience. Will never buy again.",
        "😐 Neutral / Mixed":  "It works great for basic use but nothing special, don't buy",
    }
    cols = st.columns(2)
    for i, (lbl, txt) in enumerate(samples.items()):
        with cols[i % 2]:
            with st.expander(lbl):
                st.write(txt)
                if st.button("Predict this", key=lbl):
                    p, pr = predict(txt)

                    pos_conf = pr[1]

                    if 0.4 <= pos_conf <= 0.6:
                        result = "😐 Neutral"
                    elif pos_conf > 0.6:
                        result = "✅ Positive"
                    else:
                        result = "❌ Negative"

                    if "Positive" in result:
                        st.success(f"{result} — Positive: {pr[1]*100:.1f}% | Negative: {pr[0]*100:.1f}%")

                    elif "Negative" in result:
                        st.error(f"{result} — Positive: {pr[1]*100:.1f}% | Negative: {pr[0]*100:.1f}%")

                    else:  # Neutral
                        st.warning(f"{result} — Positive: {pr[1]*100:.1f}% | Negative: {pr[0]*100:.1f}%")

# ══════════════════════════════════════════════
# TAB 5 · Metrics
# ══════════════════════════════════════════════
with tab5:
    st.subheader("📈 Full Metrics Dashboard")

    rpt = metrics["classification_report"]

    # Per-class metrics
    st.markdown("### Per-Class Metrics")
    class_data = []
    for cls in ["Negative","Positive"]:
        if cls in rpt:
            class_data.append({
                "Class":     cls,
                "Precision": round(rpt[cls]["precision"]*100,2),
                "Recall":    round(rpt[cls]["recall"]*100,2),
                "F1-Score":  round(rpt[cls]["f1-score"]*100,2),
                "Support":   int(rpt[cls]["support"]),
            })
    cls_df = pd.DataFrame(class_data)
    st.dataframe(cls_df, width='stretch', hide_index=True)

    
    
    st.markdown("---")
    st.markdown("### Macro & Weighted Averages")
    avg_data = []
    for avg_type in ["macro avg","weighted avg"]:
        if avg_type in rpt:
            avg_data.append({
                "Average":   avg_type.title(),
                "Precision": round(rpt[avg_type]["precision"]*100,2),
                "Recall":    round(rpt[avg_type]["recall"]*100,2),
                "F1-Score":  round(rpt[avg_type]["f1-score"]*100,2),
            })
    avg_df = pd.DataFrame(avg_data)
    st.dataframe(avg_df, width='stretch', hide_index=True)

    st.markdown("---")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Test Accuracy",  f"{metrics['test_accuracy']*100:.2f}%")
    c2.metric("Train Accuracy", f"{metrics['train_accuracy']*100:.2f}%")
    c3.metric("ROC-AUC",        f"{metrics['roc_auc']:.4f}")
    c4.metric("Avg Precision",  f"{metrics['avg_precision']:.4f}")

    st.markdown("---")
    st.markdown("### 🏁 Model Justification")
    bp = metrics["best_params"]
    st.markdown(f"""
| Decision | Choice | Why |
|---|---|---|
| **Algorithm** | LinearSVC + CalibratedCV | Max-margin classifier; excels in high-dim sparse spaces like TF-IDF; CalibratedCV adds `predict_proba` |
| **Kernel** | `{bp['kernel']}` (LinearSVC) | Linear kernel consistently best for NLP; selected by GridSearchCV |
| **C** | `{bp['C']}` | Controls regularisation strength; tuned via grid search |
| **Imbalance fix** | SMOTE | Creates synthetic Negative samples in TF-IDF space — more principled than `class_weight` alone |
| **class_weight** | balanced | Extra safety net on top of SMOTE |
| **Threshold** | `{metrics.get('best_threshold', 0.5)}` | Tuned post-training to maximise Macro F1 on the minority (Negative) class |
| **Vectoriser** | TF-IDF (bigrams) | Better than BoW: penalises common words, captures "not good" |
| **Scaling** | MaxAbsScaler | Scales by max absolute value → [0,1] for TF-IDF; **preserves sparsity** (no centering); dense conversion only for SMOTE & inference |
| **Eval metric** | Macro F1 | Accuracy misleading with imbalanced data; Macro F1 treats both classes equally |

**GridSearchCV searched:** 9 C values × 5-fold stratified CV = **45 model fits** on SMOTE-balanced data.
""")