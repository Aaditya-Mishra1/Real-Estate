# 🏡 RealtyIQ — Real-Estate Price-Tier Classification

**M.Tech (DSE) — Machine Learning Assignment 2**

An end-to-end machine-learning project that classifies US residential property
listings into three **price tiers** — **Budget**, **Mid-range** and **Premium** —
using five tuned classifiers, and serves the results through an interactive
Streamlit web app.

---

## a. Problem statement

Buyers, agents and portals rarely need an exact dollar prediction — they need to
know *which affordability band* a property falls into. We therefore cast the task
as a **multi-class classification** problem:

> Given a property's structural, locational and listing attributes, predict whether
> it belongs to the **Budget**, **Mid-range** or **Premium** price tier.

The three tiers are the **tertiles of the training-set price** (so the classes are
balanced). The raw `price` column is used **only** to create this label — it is never
given to the models, so there is no target leakage.

| Tier | Label | Price band (from training data) |
|------|-------|---------------------------------|
| 0 | **Budget**    | &lt; ~\$180k |
| 1 | **Mid-range** | ~\$180k – ~\$380k |
| 2 | **Premium**   | &gt; ~\$380k |

*(Exact thresholds are stored in `model/artifacts.pkl` and shown live in the app.)*

---

## b. Dataset description

* **Source:** *USA Real Estate Dataset* (Realtor.com listings) — a public Kaggle dataset.
* **Raw size:** ~2.2 million rows × 12 columns.
* **Instances used:** a random, outlier-trimmed sample of **12,000 listings**
  (9,600 train / 2,400 test) — well above the 500-instance minimum.
* **Features engineered:** **21** (minimum required is 12) — see below.

**Raw columns:** `brokered_by`, `status`, `price`, `bed`, `bath`, `acre_lot`,
`street`, `city`, `state`, `zip_code`, `house_size`, `prev_sold_date`.

**21 engineered features** (built by the shared `build_features()` function, which is
identical in the notebook and `app.py`):

| Group | Features |
|-------|----------|
| Structural | `bed`, `bath`, `acre_lot`, `house_size` |
| Ratios / interactions | `total_rooms`, `bath_per_bed`, `size_per_room`, `lot_per_size` |
| Log transforms | `acre_lot_log`, `house_size_log` |
| Location / broker (frequency encoding) | `state_freq`, `city_freq`, `broker_freq`, `zip_region` |
| Listing | `status_enc`, `has_prev_sale`, `prev_sold_year` |
| Missing-value flags | `bed_missing`, `bath_missing`, `acre_lot_missing`, `house_size_missing` |

All frequency maps, medians and tier thresholds are **learned on the training split
only** and re-applied unchanged to the test data (no leakage).

---

## c. GitHub repository

```
real-estate-price-tier-ml/
├── app.py                           # Single-file Streamlit app (deploy this)
├── requirements.txt                 # Dependencies
├── README.md                        # This file
├── test_data.csv                    # Held-out test data for the app to score
├── real_estate_classification.ipynb # Full modelling notebook (EDA → tuning → eval)
└── model/                           # Saved artifacts
    ├── artifacts.pkl                # preprocessing params (thresholds, maps, medians)
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── support_vector_machine.pkl
    └── metrics.csv
```

> 🔗 **GitHub Repository Link:** _&lt;add your repo URL here&gt;_

---

## d. Models used — comparison table

All five models are built as scikit-learn `Pipeline`s (with `StandardScaler` for the
scale-sensitive ones) and tuned with **5-fold `GridSearchCV`** (scoring = macro-F1).
Metrics below are on the **held-out test set**. AUC is macro one-vs-rest; Precision /
Recall / F1 are macro-averaged.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|----------|:--------:|:---:|:---------:|:------:|:--:|:---:|
| Logistic Regression | 0.6283 | 0.8047 | 0.6326 | 0.6283 | 0.6298 | 0.4430 |
| Decision Tree | 0.6771 | 0.8368 | 0.6958 | 0.6771 | 0.6801 | 0.5206 |
| kNN | 0.6625 | 0.8383 | 0.6634 | 0.6625 | 0.6625 | 0.4941 |
| Naive Bayes | 0.5758 | 0.7601 | 0.6319 | 0.5760 | 0.5743 | 0.3861 |
| **Support Vector Machine** 🏆 | **0.6954** | **0.8524** | **0.6962** | **0.6954** | **0.6956** | **0.5433** |

**Best hyper-parameters found**

| Model | Best params (GridSearchCV) |
|-------|----------------------------|
| Logistic Regression | `C = 10` |
| Decision Tree | `max_depth = 8`, `min_samples_leaf = 1` |
| kNN | `n_neighbors = 21`, `weights = distance` |
| Naive Bayes | `var_smoothing = 1e-3` |
| Support Vector Machine | `C = 10`, `gamma = scale` |

### Handling over-fitting & under-fitting

We explicitly tracked the **train-vs-test gap**, the **cross-validation stability**, and
tuned the complexity knob of every model:

| Model | Train acc | Test acc | Gap | CV-F1 mean ± std | Verdict |
|-------|:---------:|:--------:|:---:|:----------------:|---------|
| Logistic Regression | 0.619 | 0.628 | −0.01 | 0.618 ± 0.005 | slightly **under-fits** (linear) — very stable |
| Decision Tree | 0.726 | 0.677 | 0.05 | 0.678 ± 0.011 | well-balanced after limiting `max_depth` |
| kNN | 1.000 | 0.662 | 0.34 | 0.659 ± 0.008 | `distance` weights **memorise** train (gap↑) yet CV stays stable |
| Naive Bayes | 0.563 | 0.576 | −0.01 | 0.554 ± 0.019 | **under-fits** (feature-independence assumption) |
| Support Vector Machine | 0.743 | 0.695 | 0.05 | 0.686 ± 0.015 | best generalisation — small gap, high test score |

Mitigations applied: L2 regularisation via `C` (LogReg/SVM), depth & leaf limits
(tree), larger `k` (kNN), `var_smoothing` (NB). A **learning curve** for the winning
model (in the notebook) shows the training and cross-validation curves converging with
a small gap — the signature of a well-regularised, neither over- nor under-fit model.

### Observations

| ML Model | Observation about model performance |
|----------|-------------------------------------|
| **Logistic Regression** | Fast, stable linear baseline with a near-zero train-test gap; its straight decision boundaries slightly under-fit the non-linear separation between adjacent tiers, so it trails the kernel/tree models. |
| **Decision Tree** | Captures non-linear splits well; left unbounded it over-fits, but tuning `max_depth=8` closes most of the gap and it becomes the second-best single model. |
| **kNN** | After scaling, distance-weighted voting is competitive; note its train accuracy of 1.0 — a textbook over-fit flag — yet a moderate `k=21` keeps cross-validation stable and test performance solid. |
| **Naive Bayes** | Trains instantly but its feature-independence assumption is violated by our correlated engineered features, so it under-fits and posts the lowest scores. |
| **Support Vector Machine** | The RBF kernel models the curved tier boundaries best, giving the **highest AUC, F1 and MCC** with only a small train-test gap. |
| **Overall winner** | 🏆 **Support Vector Machine** — highest macro-F1 (0.696), AUC (0.852) and MCC (0.543) with well-controlled over-fitting. |

---

## Streamlit app features

The deployed app (`app.py`) is a single, self-contained file with a custom
"luxury real-estate" theme. It implements every required feature plus a full EDA suite:

- ✅ **Dataset upload (CSV)** — upload your own *test* data, or fall back to the bundled `test_data.csv`.
- ✅ **Model selection dropdown** — switch between all five trained models.
- ✅ **Evaluation metrics display** — Accuracy, AUC, Precision, Recall, F1, MCC as live metric cards.
- ✅ **Confusion matrix & classification report** — per-tier heatmap and report.
- 🔍 **Data Explorer (EDA) tab:**
  - market-overview cards (listings analysed, states covered, median price, median home size, premium share),
  - **state-wise analysis** — listings by state, median price by state, price-tier mix per
    state (100% stacked), median home size by state, and an expandable state summary table,
  - target class-balance, price distribution with tier cut-offs, numeric boxplots and a
    feature-correlation heatmap — all compactly sized.
  - *Data cleaning (duplicate removal + null imputation with missing-flag features) is applied
    automatically in the pipeline; it is handled, not shown as clutter.*
- 🏆 A **"Compare Models"** tab (ranked, highlighted table + grouped bar chart).
- 🏘️ A **"Predictions"** tab with sample actual-vs-predicted rows and a CSV download.

---

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to deploy (Streamlit Community Cloud)

1. Push this folder to a **public GitHub repo**.
2. Go to <https://streamlit.io/cloud> → sign in with GitHub → **New app**.
3. Select the repo, branch `main`, main file `app.py` → **Deploy**.

> 🔗 **Live Streamlit App Link:** _&lt;add your deployed URL here&gt;_

## Reproduce the models

Open `real_estate_classification.ipynb` and run all cells — it performs EDA, feature
engineering, hyper-parameter tuning, over/under-fit analysis, evaluation, and re-saves
every artifact in `model/` plus `test_data.csv`.

---
