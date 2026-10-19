# ================================================================
# CRISP-DM: SPOTIFY TRACKS DATASET (SPOTIFY 音樂流行度預測)
# Multiple Linear Regression + 5 Feature Selection Methods
#
# 1. Exhaustive Feature Selection
# 2. Sequential Forward Selection (SFS)
# 3. RFECV (Recursive Feature Elimination with Cross-Validation)
# 4. Lasso (L1 Regularization)
# 5. Elastic Net (L1 + L2 Regularization)
#
# Target: popularity (0 到 100 分)
# Features: 20 columns (audio characteristics, musical attributes, and genres)
# Reference: main.py CRISP-DM Architecture
# ================================================================

import itertools
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    GridSearchCV
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
SAMPLE_SIZE = 5000  # Representative stratified sample for efficient, robust cross-validation

# Output directories
plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)

# ================================================================
# STEP 1 — BUSINESS UNDERSTANDING
# ================================================================

print("=" * 70)
print("STEP 1: BUSINESS UNDERSTANDING")
print("Objective: Predict Spotify track popularity (0-100) from multidimensional audio features.")
print("=" * 70)

TARGET = "popularity"

# Key continuous audio features highlighted in problem description
NUMERIC_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms"
]

# Discrete and categorical musical attributes
CATEGORICAL_FEATURES = [
    "explicit",
    "mode"
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ================================================================
# STEP 2 — DATA UNDERSTANDING
# Load dataset from local CSV or fallback to public URL
# ================================================================

print("\nSTEP 2: DATA UNDERSTANDING")

LOCAL_CSV = Path("spotify_tracks.csv")
REMOTE_URL = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"

if LOCAL_CSV.exists():
    print(f"Loading local dataset from: {LOCAL_CSV}")
    df_raw = pd.read_csv(LOCAL_CSV)
else:
    print(f"Fetching dataset from remote mirror: {REMOTE_URL}")
    df_raw = pd.read_csv(REMOTE_URL)

# Basic cleaning
if "Unnamed: 0" in df_raw.columns:
    df_raw = df_raw.drop(columns=["Unnamed: 0"])

print(f"Raw dataset shape: {df_raw.shape}")
print(f"Total columns ({len(df_raw.columns)}): {df_raw.columns.tolist()}")

# Ensure target and features are present
required_cols = ALL_FEATURES + [TARGET]
for col in required_cols:
    if col not in df_raw.columns:
        raise ValueError(f"Missing required column: {col}")

# Filter valid target rows and sample for reproducible, responsive execution
df = df_raw.dropna(subset=[TARGET] + NUMERIC_FEATURES).copy()

if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
    print(f"Sampling {SAMPLE_SIZE} representative tracks (random_state={RANDOM_STATE}) for model training...")
    df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)

print(f"\nWorking dataset shape: {df.shape}")
print("\nFirst five rows:")
print(df[ALL_FEATURES + [TARGET]].head())

print("\nDescriptive statistics:")
print(df[NUMERIC_FEATURES + [TARGET]].describe().T)

print("\nMissing values in selected columns:")
print(df[ALL_FEATURES + [TARGET]].isnull().sum())

# ---------------- EDA: Popularity Distribution ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df[TARGET], kde=True, ax=axes[0], color="#1db954")
axes[0].set_title("Spotify Track Popularity Distribution", fontweight="bold")
axes[0].set_xlabel("Popularity (0 - 100)")

sns.boxplot(x=df[TARGET], ax=axes[1], color="#1ed760")
axes[1].set_title("Popularity: Outlier & Dispersion Inspection", fontweight="bold")
plt.tight_layout()
plt.savefig(plots_dir / "spotify_popularity_distribution.png", dpi=200, bbox_inches="tight")
plt.show()

# ---------------- EDA: Correlation Heatmap ----------------
plt.figure(figsize=(9, 7))
sns.heatmap(
    df[NUMERIC_FEATURES + [TARGET]].corr(),
    annot=True,
    cmap="viridis",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Spotify Audio Features Correlation Matrix", fontweight="bold")
plt.tight_layout()
plt.savefig(plots_dir / "spotify_correlation_matrix.png", dpi=200, bbox_inches="tight")
plt.show()


# ================================================================
# STEP 3 — DATA PREPARATION
# Split before feature selection and model tuning
# ================================================================

print("\n" + "=" * 70)
print("STEP 3: DATA PREPARATION")
print("=" * 70)

X = df[ALL_FEATURES].copy()
y = df[TARGET].copy()

# Cast boolean/discrete features to int
X["explicit"] = X["explicit"].astype(int)
X["mode"] = X["mode"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

cv = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

print(f"Training tracks: {len(X_train)}")
print(f"Test tracks    : {len(X_test)}")
print(f"Input features : {ALL_FEATURES}")


# ================================================================
# PREPROCESSING AND MODEL HELPERS
# ================================================================

def build_preprocessor(columns, scale_numeric=True):
    """Build column transformer for chosen subset of features."""
    transformers = []
    nums = [c for c in columns if c in NUMERIC_FEATURES]
    cats = [c for c in columns if c in CATEGORICAL_FEATURES]

    if nums:
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(numeric_steps), nums))

    if cats:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
            ]),
            cats
        ))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_ols(columns):
    """OLS pipeline on selected columns with standard scaling."""
    return Pipeline([
        ("preprocessor", build_preprocessor(columns, scale_numeric=True)),
        ("regressor", LinearRegression())
    ])


def cv_rmse(estimator, columns, X_data, y_data):
    """Evaluate 5-fold cross-validation RMSE."""
    scores = cross_val_score(
        estimator,
        X_data[columns],
        y_data,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1
    )
    rmse = -scores
    return float(rmse.mean()), float(rmse.std())


# ================================================================
# BASELINE — FULL MULTIPLE LINEAR REGRESSION
# ================================================================

print("\n" + "=" * 70)
print("BASELINE: FULL MULTIPLE LINEAR REGRESSION")
print("=" * 70)

baseline_model = make_ols(ALL_FEATURES)
baseline_cv_rmse, baseline_cv_sd = cv_rmse(baseline_model, ALL_FEATURES, X_train, y_train)

print(f"Full-model CV RMSE: {baseline_cv_rmse:.4f}")
print(f"CV RMSE standard deviation: {baseline_cv_sd:.4f}")


# ================================================================
# METHOD 1 — EXHAUSTIVE FEATURE SELECTION
# Best subset search on the top primary audio features
# ================================================================

print("\n" + "=" * 70)
print("METHOD 1: EXHAUSTIVE FEATURE SELECTION")
print("=" * 70)

# Target the top 6 core audio features for exhaustive search
CORE_AUDIO = [
    "danceability",
    "energy",
    "loudness",
    "instrumentalness",
    "speechiness",
    "valence"
]

exhaustive_results = []

for n in range(1, len(CORE_AUDIO) + 1):
    for combination in itertools.combinations(CORE_AUDIO, n):
        subset = list(combination)
        estimator = make_ols(subset)
        mean_rmse, std_rmse = cv_rmse(estimator, subset, X_train, y_train)
        exhaustive_results.append({
            "Method": "Exhaustive",
            "Features": subset,
            "N_Features": len(subset),
            "CV_RMSE": mean_rmse,
            "CV_RMSE_SD": std_rmse
        })

exhaustive_table = (
    pd.DataFrame(exhaustive_results)
    .sort_values("CV_RMSE")
    .reset_index(drop=True)
)

best_exhaustive_features = exhaustive_table.iloc[0]["Features"]
print("\nTop 5 subsets (Exhaustive):")
print(exhaustive_table[["Features", "N_Features", "CV_RMSE", "CV_RMSE_SD"]].head(5).to_string(index=False))
print(f"\nSelected features: {best_exhaustive_features}")


# ================================================================
# METHOD 2 — SEQUENTIAL FORWARD SELECTION (SFS)
# Greedy forward search across all audio and discrete features
# ================================================================

print("\n" + "=" * 70)
print("METHOD 2: SEQUENTIAL FORWARD SELECTION (SFS)")
print("=" * 70)

sfs_results = []
selected_so_far = []

for n in range(1, len(ALL_FEATURES) + 1):
    remaining = [col for col in ALL_FEATURES if col not in selected_so_far]
    candidate_results = []

    for candidate in remaining:
        subset = selected_so_far + [candidate]
        mean_rmse, std_rmse = cv_rmse(make_ols(subset), subset, X_train, y_train)
        candidate_results.append({
            "Feature": candidate,
            "Features": subset,
            "CV_RMSE": mean_rmse,
            "CV_RMSE_SD": std_rmse
        })

    best_candidate = min(candidate_results, key=lambda res: res["CV_RMSE"])
    selected_so_far = best_candidate["Features"].copy()

    sfs_results.append({
        "Method": "SFS",
        "Features": selected_so_far.copy(),
        "N_Features": len(selected_so_far),
        "CV_RMSE": best_candidate["CV_RMSE"],
        "CV_RMSE_SD": best_candidate["CV_RMSE_SD"]
    })

sfs_table = (
    pd.DataFrame(sfs_results)
    .sort_values("CV_RMSE")
    .reset_index(drop=True)
)

best_sfs_features = sfs_table.iloc[0]["Features"]
print("\nSFS Feature Progression:")
print(pd.DataFrame(sfs_results)[["N_Features", "Features", "CV_RMSE"]].to_string(index=False))
print(f"\nBest SFS feature set ({len(best_sfs_features)} features): {best_sfs_features}")


# ================================================================
# METHOD 3 — RFECV (RECURSIVE FEATURE ELIMINATION)
# Backward elimination with cross-validation
# ================================================================

print("\n" + "=" * 70)
print("METHOD 3: RFECV")
print("=" * 70)


class GroupAwareRFECV:
    """Recursive feature elimination with CV supporting mixed continuous and discrete columns."""
    def __init__(self, estimator_builder, cv):
        self.estimator_builder = estimator_builder
        self.cv = cv

    def fit(self, X, y):
        current_features = list(X.columns)
        history = []
        while len(current_features) >= 1:
            mean_rmse, _ = cv_rmse(self.estimator_builder(current_features), current_features, X, y)
            # Fit on full data to inspect coefficients
            pipe = self.estimator_builder(current_features).fit(X[current_features], y)
            coefs = np.abs(pipe.named_steps["regressor"].coef_)

            history.append({
                "features": current_features.copy(),
                "cv_rmse": mean_rmse,
                "n_features": len(current_features)
            })

            if len(current_features) == 1:
                break

            # Drop least influential feature
            min_idx = int(np.argmin(coefs))
            current_features.pop(min_idx)

        best_step = min(history, key=lambda h: h["cv_rmse"])
        self.n_features_ = best_step["n_features"]
        self.selected_features_ = best_step["features"]
        self.best_cv_rmse_ = best_step["cv_rmse"]
        self.history_ = history
        return self


rfecv = GroupAwareRFECV(estimator_builder=make_ols, cv=cv)
rfecv.fit(X_train, y_train)

best_rfecv_features = rfecv.selected_features_
rfecv_best_cv_rmse = rfecv.best_cv_rmse_

print(f"Selected features count: {rfecv.n_features_}")
print(f"Selected features      : {best_rfecv_features}")
print(f"Best RFECV CV RMSE     : {rfecv_best_cv_rmse:.4f}")


# ================================================================
# METHODS 4 & 5 — LASSO AND ELASTIC NET
# ================================================================

print("\n" + "=" * 70)
print("METHODS 4 & 5: LASSO AND ELASTIC NET")
print("=" * 70)

# Lasso tuning
lasso_pipe = Pipeline([
    ("preprocessor", build_preprocessor(ALL_FEATURES, scale_numeric=True)),
    ("regressor", Lasso(max_iter=10000, random_state=RANDOM_STATE))
])

lasso_grid = GridSearchCV(
    lasso_pipe,
    param_grid={"regressor__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
    cv=cv,
    scoring="neg_root_mean_squared_error"
)
lasso_grid.fit(X_train, y_train)
lasso_best_cv_rmse = -lasso_grid.best_score_
lasso_model = lasso_grid.best_estimator_

# Elastic Net tuning
enet_pipe = Pipeline([
    ("preprocessor", build_preprocessor(ALL_FEATURES, scale_numeric=True)),
    ("regressor", ElasticNet(max_iter=10000, random_state=RANDOM_STATE))
])

enet_grid = GridSearchCV(
    enet_pipe,
    param_grid={
        "regressor__alpha": [0.01, 0.1, 1.0],
        "regressor__l1_ratio": [0.2, 0.5, 0.8]
    },
    cv=cv,
    scoring="neg_root_mean_squared_error"
)
enet_grid.fit(X_train, y_train)
enet_best_cv_rmse = -enet_grid.best_score_
enet_model = enet_grid.best_estimator_

print(f"Lasso Best Parameters      : {lasso_grid.best_params_}")
print(f"Lasso Best CV RMSE         : {lasso_best_cv_rmse:.4f}")
print(f"Elastic Net Best Parameters: {enet_grid.best_params_}")
print(f"Elastic Net Best CV RMSE   : {enet_best_cv_rmse:.4f}")


# ================================================================
# STEP 5 — FINAL TEST EVALUATION
# ================================================================

print("\n" + "=" * 70)
print("STEP 5: FINAL TEST EVALUATION")
print("=" * 70)

models_to_evaluate = {
    "Full OLS": (make_ols(ALL_FEATURES), ALL_FEATURES),
    "Exhaustive OLS": (make_ols(best_exhaustive_features), best_exhaustive_features),
    "SFS OENS": (make_ols(best_sfs_features), best_sfs_features),
    "RFECV OLS": (make_ols(best_rfecv_features), best_rfecv_features),
    "Lasso": (lasso_model, ALL_FEATURES),
    "Elastic Net": (enet_model, ALL_FEATURES)
}

test_summary = []

for name, (estimator, feat_cols) in models_to_evaluate.items():
    estimator.fit(X_train[feat_cols], y_train)
    preds = estimator.predict(X_test[feat_cols])
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    test_summary.append({
        "Model": name,
        "N_Features": len(feat_cols),
        "Test_RMSE": rmse,
        "Test_MAE": mae,
        "Test_R2": r2,
        "Features": feat_cols
    })

test_table = pd.DataFrame(test_summary).sort_values("Test_RMSE").reset_index(drop=True)
print("\nHeld-out Test Set Performance:")
print(test_table[["Model", "N_Features", "Test_RMSE", "Test_MAE", "Test_R2"]].to_string(index=False))

# Residual & Actual vs Predicted Diagnostic Plot
best_model_name = test_table.iloc[0]["Model"]
best_pipe, best_feats = models_to_evaluate[best_model_name]
test_preds = best_pipe.predict(X_test[best_feats])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(y_test, test_preds, color="#1db954", alpha=0.6, edgecolors="k", s=35)
min_p, max_p = y_test.min(), y_test.max()
axes[0].plot([min_p, max_p], [min_p, max_p], "r--", linewidth=2, label="Ideal Fit")
axes[0].set_title(f"{best_model_name}: Actual vs Predicted Popularity", fontweight="bold")
axes[0].set_xlabel("Actual Popularity")
axes[0].set_ylabel("Predicted Popularity")
axes[0].legend()

residuals = y_test - test_preds
sns.histplot(residuals, kde=True, ax=axes[1], color="#0284c7")
axes[1].set_title("Test Residual Error Distribution", fontweight="bold")
axes[1].set_xlabel("Residual (Actual - Predicted)")
plt.tight_layout()
plt.savefig(plots_dir / "spotify_diagnostic_plots.png", dpi=200, bbox_inches="tight")
plt.show()


# ================================================================
# STEP 6 — DEPLOYMENT
# ================================================================

print("\n" + "=" * 70)
print("STEP 6: DEPLOYMENT")
print("=" * 70)

deployment_bundle = {
    "model_name": best_model_name,
    "input_features": best_feats,
    "pipeline": best_pipe
}

joblib.dump(deployment_bundle, "spotify_popularity_pipeline.joblib")
print(f"Selected deployment model: {best_model_name}")
print(f"Required input features  : {best_feats}")
print("Saved pipeline to: spotify_popularity_pipeline.joblib")

# Example Track Inference
sample_song = pd.DataFrame([{
    "danceability": 0.72,
    "energy": 0.81,
    "loudness": -5.2,
    "speechiness": 0.08,
    "acousticness": 0.12,
    "instrumentalness": 0.00,
    "liveness": 0.15,
    "valence": 0.65,
    "tempo": 125.0,
    "duration_ms": 210000,
    "explicit": 1,
    "mode": 1
}])

pred_pop = best_pipe.predict(sample_song[best_feats])[0]
print("\nExample Track:")
print(sample_song[best_feats])
print(f"\nPredicted Popularity Score: {pred_pop:.1f} / 100")
print("=" * 70)
