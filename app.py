"""
🏡 RealtyIQ — Property Price-Tier Classifier
=============================================
Single-file Streamlit app for M.Tech (DSE) ML Assignment 2.

Loads five pre-trained classifiers (Logistic Regression, Decision Tree, kNN,
Gaussian Naive Bayes, SVM), performs exploratory data analysis on the uploaded
test data (missing-value handling, duplicate removal, distributions, correlations)
and evaluates every model — predicting whether each property is Budget / Mid-range
/ Premium.

Deploy on Streamlit Community Cloud:  main file = app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

# ===========================================================================
#  Feature engineering  — IDENTICAL to the training notebook (single source
#  of truth) so the deployed app reproduces training exactly.
# ===========================================================================
TIER_NAMES = {0: "Budget", 1: "Mid-range", 2: "Premium"}
NUM_RAW = ["bed", "bath", "acre_lot", "house_size"]

FEATURES = [
    "bed", "bath", "acre_lot", "house_size",
    "total_rooms", "bath_per_bed", "size_per_room",
    "acre_lot_log", "house_size_log", "lot_per_size",
    "state_freq", "city_freq", "broker_freq", "zip_region",
    "status_enc", "has_prev_sale", "prev_sold_year",
    "bed_missing", "bath_missing", "acre_lot_missing", "house_size_missing",
]


def _freq_map(series):
    vc = series.astype(str).value_counts()
    return (vc / vc.sum()).to_dict()


def build_features(df_raw, artifacts=None, fit=False):
    df = df_raw.copy()
    for c in ["brokered_by", "price", "bed", "bath", "acre_lot", "zip_code", "house_size"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["price"].notna() & (df["price"] > 0)].reset_index(drop=True)

    if fit:
        artifacts = {}
        q = df["price"].quantile([1 / 3, 2 / 3]).values
        artifacts["price_thresholds"] = [float(q[0]), float(q[1])]
        artifacts["medians"] = {c: float(df[c].median()) for c in NUM_RAW}
        artifacts["freq_state"] = _freq_map(df["state"]) if "state" in df else {}
        artifacts["freq_city"] = _freq_map(df["city"]) if "city" in df else {}
        artifacts["freq_broker"] = _freq_map(df["brokered_by"]) if "brokered_by" in df else {}
        artifacts["status_map"] = {"for_sale": 0, "ready_to_build": 1, "sold": 2}
        artifacts["features"] = FEATURES

    th = artifacts["price_thresholds"]
    med = artifacts["medians"]

    y = np.digitize(df["price"].values, bins=th).astype(int)

    for c in NUM_RAW:
        df[c + "_missing"] = df[c].isna().astype(int)
    for c in NUM_RAW:
        df[c] = df[c].fillna(med[c])

    df["total_rooms"] = df["bed"] + df["bath"]
    df["bath_per_bed"] = df["bath"] / (df["bed"] + 1)
    df["size_per_room"] = df["house_size"] / (df["total_rooms"] + 1)
    df["acre_lot_log"] = np.log1p(df["acre_lot"].clip(lower=0))
    df["house_size_log"] = np.log1p(df["house_size"].clip(lower=0))
    df["lot_per_size"] = df["acre_lot"] / (df["house_size"] + 1)

    df["state_freq"] = df.get("state", pd.Series(index=df.index, dtype=object)).astype(str).map(artifacts["freq_state"]).fillna(0.0)
    df["city_freq"] = df.get("city", pd.Series(index=df.index, dtype=object)).astype(str).map(artifacts["freq_city"]).fillna(0.0)
    df["broker_freq"] = df.get("brokered_by", pd.Series(index=df.index, dtype=object)).astype(str).map(artifacts["freq_broker"]).fillna(0.0)

    if "zip_code" in df:
        df["zip_region"] = (df["zip_code"].fillna(0) // 100).astype(float)
    else:
        df["zip_region"] = 0.0

    df["status_enc"] = df.get("status", pd.Series("for_sale", index=df.index)).map(artifacts["status_map"]).fillna(0).astype(int)

    if "prev_sold_date" in df.columns:
        yr = pd.to_datetime(df["prev_sold_date"], errors="coerce").dt.year
        df["has_prev_sale"] = yr.notna().astype(int)
        df["prev_sold_year"] = yr.fillna(0).astype(float)
    else:
        df["has_prev_sale"] = 0
        df["prev_sold_year"] = 0.0

    X = df[artifacts["features"]].astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return X, y, df


# ===========================================================================
#  Theme & palette  — warm "luxury real-estate" look
# ===========================================================================
INK = "#1b3a2b"       # deep forest green
SAGE = "#52796f"
GOLD = "#b8860b"
CREAM = "#f6f2ea"
TIER_COLORS = ["#9bbfab", "#52796f", "#1b3a2b"]

st.set_page_config(page_title="RealtyIQ · Price-Tier Intelligence",
                   page_icon="🏡", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
    .stApp {{ background:{CREAM}; color:#2b2b2b; font-family:'Inter',sans-serif; }}
    h1,h2,h3,h4 {{ font-family:'Fraunces',serif !important; color:{INK}; letter-spacing:.2px; }}
    .hero {{
        background:
            linear-gradient(115deg, rgba(27,58,43,.93) 0%, rgba(82,121,111,.90) 100%),
            url('https://images.unsplash.com/photo-1600585154340-be6161a56a0c?auto=format&fit=crop&w=1400&q=60');
        background-size:cover; background-position:center;
        padding:38px 42px; border-radius:20px; margin-bottom:6px;
        box-shadow:0 14px 34px rgba(27,58,43,.28);
    }}
    .hero h1 {{ color:#fff !important; margin:0; font-size:2.35rem; }}
    .hero p {{ color:#eef3ef; margin:.5rem 0 0 0; font-size:1.05rem; max-width:760px; }}
    .badge {{ display:inline-block; margin-top:14px; margin-right:8px; padding:5px 14px;
        background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.35);
        border-radius:999px; color:#fff; font-size:.8rem; font-weight:500; }}
    .metric-card {{
        background:#fff; border:1px solid #e7e0d4; border-left:5px solid {GOLD};
        border-radius:14px; padding:16px 18px; box-shadow:0 4px 14px rgba(27,58,43,.06);
    }}
    .metric-card .label {{ color:{SAGE}; font-size:.74rem; text-transform:uppercase; letter-spacing:.8px; font-weight:600; }}
    .metric-card .value {{ color:{INK}; font-size:1.75rem; font-weight:700; margin-top:2px; }}
    .metric-card .sub {{ color:#9aa39c; font-size:.72rem; }}
    .note {{ background:#fff8e7; border:1px solid #f0e2b8; border-radius:12px; padding:14px 18px; color:#6b5a1f; }}
    section[data-testid="stSidebar"] {{ background:{INK}; }}
    section[data-testid="stSidebar"] * {{ color:#e9f0ec !important; }}
    .stTabs [data-baseweb="tab-list"] {{ gap:4px; }}
    .stTabs [data-baseweb="tab"] {{ font-weight:600; }}
    .stTabs [aria-selected="true"] {{ color:{INK}; border-bottom-color:{GOLD}; }}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🏡 RealtyIQ — Price-Tier Intelligence</h1>
  <p>Upload a property listings file and RealtyIQ profiles the data, cleans it, and
  classifies every home into <b>Budget</b>, <b>Mid-range</b> or <b>Premium</b> using
  five tuned machine-learning models — with full exploratory analysis along the way.</p>
  <span class="badge">21 engineered features</span>
  <span class="badge">5 tuned classifiers</span>
  <span class="badge">3-class price tiers</span>
</div>
""", unsafe_allow_html=True)

APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl",
}


@st.cache_resource
def load_everything():
    artifacts = joblib.load(MODEL_DIR / "artifacts.pkl")
    models = {name: joblib.load(MODEL_DIR / f) for name, f in MODEL_FILES.items()}
    return artifacts, models


@st.cache_data
def load_default_test():
    p = APP_DIR / "test_data.csv"
    return pd.read_csv(p) if p.exists() else None


def style_ax(ax, fig):
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_color("#d8d0c2")
    ax.tick_params(colors="#5a5a5a", labelsize=8)
    ax.xaxis.label.set_color("#444"); ax.yaxis.label.set_color("#444")
    ax.xaxis.label.set_fontsize(9); ax.yaxis.label.set_fontsize(9)
    ax.title.set_color(INK); ax.title.set_fontsize(10.5)


def show_chart(fig):
    """Render a matplotlib figure at its native (compact) size, not stretched."""
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


def evaluate(est, X, y):
    pred = est.predict(X)
    proba = est.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
        "Precision": precision_score(y, pred, average="macro", zero_division=0),
        "Recall": recall_score(y, pred, average="macro", zero_division=0),
        "F1": f1_score(y, pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }, pred


try:
    artifacts, models = load_everything()
except Exception as e:
    st.error(f"Could not load model artifacts from `model/`. Make sure the `model/` "
             f"folder with the .pkl files is deployed alongside app.py.\n\n{e}")
    st.stop()

# ===========================================================================
#  Sidebar — controls
# ===========================================================================
with st.sidebar:
    st.markdown("### ⚙️  Controls")
    st.caption("Streamlit free tier has limited capacity — upload **test data only**.")
    uploaded = st.file_uploader("Upload listings (CSV)", type=["csv"])
    model_name = st.selectbox("Model for scoring", list(MODEL_FILES.keys()),
                              index=len(MODEL_FILES) - 1)
    st.markdown("---")
    st.markdown("**Price tiers**")
    th = artifacts["price_thresholds"]
    st.markdown(
        f"- 🟢 **Budget** — under ${th[0]:,.0f}\n"
        f"- 🟩 **Mid-range** — ${th[0]:,.0f} to ${th[1]:,.0f}\n"
        f"- 🌲 **Premium** — over ${th[1]:,.0f}")
    st.markdown("---")
    st.caption("Tiers are the price tertiles learned from the training data. "
               "`price` is used only to build the label, never as a model input.")

# ---------------------------------------------------------------------------
#  Load + clean data
# ---------------------------------------------------------------------------
if uploaded is not None:
    raw_df = pd.read_csv(uploaded)
    src = f"uploaded file · {uploaded.name}"
else:
    raw_df = load_default_test()
    src = "bundled test_data.csv (default)"

if raw_df is None:
    st.warning("Upload a test CSV to begin (or deploy the bundled test_data.csv).")
    st.stop()

if "price" not in raw_df.columns:
    st.error("The CSV must include a `price` column so the true price tier "
             "(and therefore the evaluation metrics) can be computed.")
    st.stop()

# --- silent cleaning: drop duplicate rows; NaNs handled inside build_features
#     (median imputation + missing-flag features); invalid prices dropped. -----
clean_df = raw_df.drop_duplicates().reset_index(drop=True)

# build features (handles NaNs via median impute + missing flags, drops bad price)
X, y, feat_df = build_features(clean_df, artifacts, fit=False)
n_scored = len(X)

st.markdown(f"<p style='color:{SAGE};margin:12px 0 2px'>Source: <b>{src}</b></p>",
            unsafe_allow_html=True)

# ===========================================================================
#  Tabs
# ===========================================================================
tab_eda, tab_perf, tab_cm, tab_cmp, tab_pred = st.tabs(
    ["🔍 Data Explorer (EDA)", "📈 Model Performance", "🔀 Confusion & Report",
     "🏆 Compare Models", "🏘️ Predictions"])

# ---------------------------------------------------------------------------
#  TAB 1 — EDA
# ---------------------------------------------------------------------------
with tab_eda:
    # ---- market overview cards (neutral, no null/duplicate messaging) --------
    eda = pd.DataFrame({
        "state": feat_df.get("state", pd.Series(["Unknown"] * n_scored)).astype(str).values,
        "price": pd.to_numeric(feat_df["price"], errors="coerce").values,
        "house_size": pd.to_numeric(feat_df["house_size"], errors="coerce").values,
        "bed": pd.to_numeric(feat_df["bed"], errors="coerce").values,
        "tier": pd.Series(y).map(TIER_NAMES).values,
    })
    premium_share = (eda["tier"] == "Premium").mean() * 100

    st.subheader("Market overview")
    cards = [
        ("Listings analysed", f"{n_scored:,}", "after cleaning"),
        ("States covered", f"{eda['state'].nunique():,}", "distinct markets"),
        ("Median price", f"${np.nanmedian(eda['price']):,.0f}", "across all tiers"),
        ("Median home size", f"{np.nanmedian(eda['house_size']):,.0f} sqft", "living area"),
        ("Premium share", f"{premium_share:.0f}%", "of listings"),
    ]
    cols = st.columns(5)
    for col, (lbl, val, sub) in zip(cols, cards):
        col.markdown(f'<div class="metric-card"><div class="label">{lbl}</div>'
                     f'<div class="value">{val}</div><div class="sub">{sub}</div></div>',
                     unsafe_allow_html=True)

    # ---- target & price distribution ----------------------------------------
    st.markdown("### 🎯 Price-tier target")
    c1, c2 = st.columns(2)
    with c1:
        dist = eda["tier"].value_counts().reindex(["Budget", "Mid-range", "Premium"]).fillna(0)
        fig, ax = plt.subplots(figsize=(4.4, 2.8))
        ax.bar(dist.index, dist.values, color=TIER_COLORS)
        ax.set_ylabel("listings"); ax.set_title("Class balance")
        style_ax(ax, fig); fig.tight_layout(); show_chart(fig)
    with c2:
        price = eda["price"][eda["price"] > 0].dropna()
        fig, ax = plt.subplots(figsize=(4.4, 2.8))
        ax.hist(np.log10(price), bins=35, color=GOLD, edgecolor="white")
        for t in th:
            ax.axvline(np.log10(t), color=INK, ls="--", lw=1)
        ax.set_xlabel("log10(price USD)"); ax.set_title("Price distribution")
        style_ax(ax, fig); fig.tight_layout(); show_chart(fig)

    # ---- STATE-WISE EDA ------------------------------------------------------
    st.markdown("### 🗺️ State-wise analysis")
    counts = eda["state"].value_counts()
    topN = counts.head(12).index.tolist()

    c1, c2 = st.columns(2)
    with c1:
        tc = counts.head(12)
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        ax.barh(tc.index[::-1], tc.values[::-1], color=SAGE)
        ax.set_title("Listings by state (top 12)"); ax.set_xlabel("listings")
        style_ax(ax, fig); fig.tight_layout(); show_chart(fig)
    with c2:
        mp = eda[eda["state"].isin(topN)].groupby("state")["price"].median().sort_values()
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        ax.barh(mp.index, mp.values / 1000, color=GOLD)
        ax.set_title("Median price by state"); ax.set_xlabel("median price ($k)")
        style_ax(ax, fig); fig.tight_layout(); show_chart(fig)

    c1, c2 = st.columns(2)
    with c1:
        top8 = counts.head(8).index.tolist()
        comp = (pd.crosstab(eda["state"], eda["tier"], normalize="index")
                .reindex(top8)[["Budget", "Mid-range", "Premium"]] * 100)
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        comp.plot(kind="barh", stacked=True, color=TIER_COLORS, ax=ax, width=.8)
        ax.set_title("Tier mix by state (%)"); ax.set_xlabel("% of listings")
        ax.legend(fontsize=6.5, loc="lower right", ncol=3)
        style_ax(ax, fig); fig.tight_layout(); show_chart(fig)
    with c2:
        ms = eda[eda["state"].isin(topN)].groupby("state")["house_size"].median().sort_values()
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        ax.barh(ms.index, ms.values, color="#8c6a3f")
        ax.set_title("Median home size by state"); ax.set_xlabel("sqft")
        style_ax(ax, fig); fig.tight_layout(); show_chart(fig)

    with st.expander("📋 State summary table (median price, size & tier counts)"):
        summ = eda.groupby("state").agg(
            listings=("price", "size"),
            median_price=("price", "median"),
            median_sqft=("house_size", "median"),
            avg_beds=("bed", "mean"),
        ).sort_values("listings", ascending=False)
        summ["median_price"] = summ["median_price"].round(0)
        summ["median_sqft"] = summ["median_sqft"].round(0)
        summ["avg_beds"] = summ["avg_beds"].round(1)
        st.dataframe(summ, use_container_width=True)

    # ---- feature distributions + correlation (compact) ----------------------
    st.markdown("### 📦 Feature distributions & correlation")
    c1, c2 = st.columns([1, 1])
    with c1:
        numcols = [c for c in ["bed", "bath", "house_size", "acre_lot"] if c in clean_df.columns]
        fig, axes = plt.subplots(1, len(numcols), figsize=(4.8, 2.6))
        for ax, c in zip(np.atleast_1d(axes), numcols):
            vals = pd.to_numeric(clean_df[c], errors="coerce").dropna()
            vals = vals[(vals >= vals.quantile(.01)) & (vals <= vals.quantile(.99))]
            ax.boxplot(vals, patch_artist=True,
                       boxprops=dict(facecolor="#dce7e0", color=SAGE),
                       medianprops=dict(color=GOLD, lw=2))
            ax.set_title(c, fontsize=8); ax.set_xticks([]); style_ax(ax, fig)
        fig.tight_layout(); show_chart(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        sns.heatmap(X.corr(), cmap="BrBG", center=0, square=False, ax=ax,
                    cbar_kws={"shrink": .6}, xticklabels=False)
        ax.set_title("Feature correlation"); ax.tick_params(labelsize=5)
        fig.tight_layout(); show_chart(fig)

# ---------------------------------------------------------------------------
#  TAB 2 — Model performance
# ---------------------------------------------------------------------------
selected = models[model_name]
metrics, pred = evaluate(selected, X, y)

with tab_perf:
    st.subheader(f"Evaluation metrics — {model_name}")
    cols = st.columns(6)
    for col, (k, v) in zip(cols, metrics.items()):
        col.markdown(f'<div class="metric-card"><div class="label">{k}</div>'
                     f'<div class="value">{v:.3f}</div></div>', unsafe_allow_html=True)
    st.caption("AUC is macro one-vs-rest; Precision / Recall / F1 are macro-averaged "
               "across the three tiers. MCC is the Matthews Correlation Coefficient.")

    st.markdown("### Predicted price-tier distribution")
    dist = pd.Series(pred).map(TIER_NAMES).value_counts().reindex(
        ["Budget", "Mid-range", "Premium"]).fillna(0)
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.bar(dist.index, dist.values, color=TIER_COLORS)
    ax.set_ylabel("listings"); ax.set_title("Predicted tiers")
    style_ax(ax, fig); fig.tight_layout(); show_chart(fig)

# ---------------------------------------------------------------------------
#  TAB 3 — Confusion matrix & report
# ---------------------------------------------------------------------------
with tab_cm:
    st.subheader(f"Confusion matrix — {model_name}")
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(4.2, 3.3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False,
                annot_kws={"size": 9},
                xticklabels=[TIER_NAMES[i] for i in range(3)],
                yticklabels=[TIER_NAMES[i] for i in range(3)], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.tick_params(labelsize=8)
    fig.tight_layout(); show_chart(fig)

    st.subheader("Classification report")
    rep = classification_report(y, pred,
                                target_names=[TIER_NAMES[i] for i in range(3)],
                                output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)

# ---------------------------------------------------------------------------
#  TAB 4 — Compare all models
# ---------------------------------------------------------------------------
with tab_cmp:
    st.subheader("All five models on this data")
    rows = []
    for name, est in models.items():
        m, _ = evaluate(est, X, y)
        m["Model"] = name
        rows.append(m)
    comp = pd.DataFrame(rows).set_index("Model")[
        ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]].round(4)
    winner = comp["F1"].idxmax()
    st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#cfe3d6"),
                 use_container_width=True)
    st.markdown(f"🏆 **Best model on this data (macro-F1): `{winner}`**")

    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    comp.plot(kind="bar", ax=ax, colormap="BrBG", width=.8)
    ax.set_ylim(0, 1); ax.set_ylabel("score")
    ax.legend(loc="lower right", ncol=6, fontsize=6)
    ax.tick_params(labelrotation=12); style_ax(ax, fig)
    fig.tight_layout(); show_chart(fig)

# ---------------------------------------------------------------------------
#  TAB 5 — Predictions
# ---------------------------------------------------------------------------
with tab_pred:
    st.subheader("Cleaned data preview")
    st.dataframe(clean_df.head(15), use_container_width=True)

    st.subheader("Sample predictions")
    show = feat_df[["bed", "bath", "house_size", "acre_lot"]].copy()
    show["Actual tier"] = pd.Series(y).map(TIER_NAMES).values
    show["Predicted tier"] = pd.Series(pred).map(TIER_NAMES).values
    show["✓"] = np.where(y == pred, "✅", "❌")
    st.dataframe(show.head(30), use_container_width=True)

    csv = show.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download predictions (CSV)", csv,
                       file_name=f"predictions_{model_name.replace(' ', '_')}.csv",
                       mime="text/csv")

st.markdown("---")
st.caption("RealtyIQ · M.Tech (DSE) Machine Learning Assignment 2 · "
           "5 tuned classifiers · 21 engineered features · 3-class price-tier task.")
