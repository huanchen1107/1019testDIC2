
# ================================================================
# CRISP-DM: KAGGLE 50 STARTUPS
# Multiple Linear Regression + 5 Feature Selection Methods
#
# 1. Exhaustive Feature Selection
# 2. Sequential Forward Selection (SFS)
# 3. RFECV
# 4. Lasso (L1)
# 5. Elastic Net
#
# Data source: GitHub raw CSV
# Target: Profit
# ================================================================

# Install if necessary:
# pip install pandas numpy matplotlib seaborn scikit-learn joblib

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
from sklearn.feature_selection import RFECV
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

warnings.filterwarnings("default")

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

# ================================================================
# STEP 1 — BUSINESS UNDERSTANDING
# ================================================================

print("=" * 70)
print("STEP 1: BUSINESS UNDERSTANDING")
print("Objective: Predict startup Profit from spending and State.")
print("=" * 70)

TARGET = "Profit"

NUMERIC_FEATURES = [
    "R&D Spend",
    "Administration",
    "Marketing Spend"
]

CATEGORICAL_FEATURES = ["State"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ================================================================
# STEP 2 — DATA UNDERSTANDING
# Download the dataset directly from GitHub
# ================================================================

print("\nSTEP 2: DATA UNDERSTANDING")

URL = (
    "https://raw.githubusercontent.com/"
    "SantiagoMorenoV/50-Startups-Multiple-Reg/"
    "main/50_Startups.csv"
)

df = pd.read_csv(URL)

required_columns = ALL_FEATURES + [TARGET]
missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Dataset is missing expected columns: {missing_columns}"
    )

print("\nDataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nColumn types:")
print(df.dtypes)

print("\nDescriptive statistics:")
print(df.describe(include="all").T)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nUnique State values:")
print(df["State"].value_counts(dropna=False))

# Clean target and numeric data
for col in NUMERIC_FEATURES + [TARGET]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[TARGET]).copy()

if len(df) < N_SPLITS + 1:
    raise ValueError("Not enough labeled rows for cross-validation.")

print("\nRows available for modeling:", len(df))

# ---------------- EDA: Profit distribution ----------------

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(df[TARGET], kde=True, ax=axes[0])
axes[0].set_title("Profit Distribution")

sns.boxplot(x=df[TARGET], ax=axes[1])
axes[1].set_title("Profit: Outlier Inspection")

plt.tight_layout()
plt.show()

# ---------------- EDA: spending versus Profit ----------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, col in zip(axes, NUMERIC_FEATURES):
    sns.scatterplot(data=df, x=col, y=TARGET, ax=ax)
    ax.set_title(f"{col} vs Profit")

plt.tight_layout()
plt.show()

# ---------------- EDA: correlations ----------------

plt.figure(figsize=(8, 6))

sns.heatmap(
    df[NUMERIC_FEATURES + [TARGET]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Numerical Correlation Matrix")
plt.tight_layout()
plt.show()

# ---------------- EDA: State comparison ----------------

plt.figure(figsize=(7, 4))

sns.boxplot(data=df, x="State", y=TARGET)

plt.title("Profit by State")
plt.tight_layout()
plt.show()


# ================================================================
# STEP 3 — DATA PREPARATION
# Split before feature selection and model tuning
# ================================================================

print("\nSTEP 3: DATA PREPARATION")

X = df[ALL_FEATURES].copy()
y = df[TARGET].copy()

# Normalize categorical missing values to actual missing values
X["State"] = X["State"].replace(
    r"^\s*$", np.nan, regex=True
)

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

print("Training observations:", len(X_train))
print("Test observations:", len(X_test))
print("Original input features:", ALL_FEATURES)


# ================================================================
# PREPROCESSING AND MODEL HELPERS
# All transformations are fitted inside each CV training fold.
# ================================================================

def build_preprocessor(columns, scale_numeric=False):
    """Build a preprocessor for a chosen subset of original columns."""

    transformers = []

    nums = [
        c for c in columns
        if c in NUMERIC_FEATURES
    ]

    cats = [
        c for c in columns
        if c in CATEGORICAL_FEATURES
    ]

    if nums:
        numeric_steps = [
            ("imputer", SimpleImputer(strategy="median"))
        ]

        if scale_numeric:
            numeric_steps.append(
                ("scaler", StandardScaler())
            )

        transformers.append((
            "num",
            Pipeline(numeric_steps),
            nums
        ))

    if cats:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer",
                 SimpleImputer(strategy="most_frequent")),
                ("onehot",
                 OneHotEncoder(
                     drop="first",
                     handle_unknown="ignore"
                 ))
            ]),
            cats
        ))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


def make_ols(columns):
    """Ordinary least-squares regression on selected columns."""

    return Pipeline([
        ("preprocessor", build_preprocessor(columns)),
        ("regressor", LinearRegression())
    ])


def cv_rmse(estimator, columns, X_data, y_data):
    """Mean and SD of RMSE across the same CV splits."""

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

baseline_cv_rmse, baseline_cv_sd = cv_rmse(
    baseline_model, ALL_FEATURES, X_train, y_train
)

print("Full-model CV RMSE:", round(baseline_cv_rmse, 3))
print("CV RMSE standard deviation:", round(baseline_cv_sd, 3))


# ================================================================
# METHOD 1 — EXHAUSTIVE FEATURE SELECTION
# Four original features => 2^4 - 1 = 15 subsets
# State dummy variables are selected as a group.
# ================================================================

print("\n" + "=" * 70)
print("METHOD 1: EXHAUSTIVE FEATURE SELECTION")
print("=" * 70)

exhaustive_results = []

for n in range(1, len(ALL_FEATURES) + 1):
    for combination in itertools.combinations(ALL_FEATURES, n):

        subset = list(combination)

        estimator = make_ols(subset)

        mean_rmse, std_rmse = cv_rmse(
            estimator, subset, X_train, y_train
        )

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

print("\nTop 10 subsets:")
print(
    exhaustive_table[
        ["Features", "N_Features", "CV_RMSE", "CV_RMSE_SD"]
    ].head(10).to_string(index=False)
)

print("\nSelected features:", best_exhaustive_features)


# ================================================================
# METHOD 2 — SEQUENTIAL FORWARD SELECTION (SFS)
# Greedy search using CV RMSE.
# ================================================================

print("\n" + "=" * 70)
print("METHOD 2: SEQUENTIAL FORWARD SELECTION")
print("=" * 70)

sfs_results = []
selected_so_far = []

for n in range(1, len(ALL_FEATURES) + 1):

    remaining = [
        col for col in ALL_FEATURES
        if col not in selected_so_far
    ]

    candidate_results = []

    for candidate in remaining:

        subset = selected_so_far + [candidate]

        mean_rmse, std_rmse = cv_rmse(
            make_ols(subset),
            subset,
            X_train,
            y_train
        )

        candidate_results.append({
            "Feature": candidate,
            "Features": subset,
            "CV_RMSE": mean_rmse,
            "CV_RMSE_SD": std_rmse
        })

    best_candidate = min(
        candidate_results,
        key=lambda result: result["CV_RMSE"]
    )

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

print("\nSFS results:")
print(sfs_table.to_string(index=False))

print("\nSelected features:", best_sfs_features)


# ================================================================
# METHOD 3 — RFECV
# Custom estimator maps encoded coefficients back to original
# features. Numeric columns are scaled for comparable importance.
# ================================================================

print("\n" + "=" * 70)
print("METHOD 3: RFECV")
print("=" * 70)


class GroupAwareLinearRegressor(RegressorMixin, BaseEstimator):
    """
    Regression estimator exposing one importance per original
    feature, so RFECV can eliminate columns without splitting
    State into independently selected dummy variables.
    """

    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def fit(self, X, y):

        self.columns_ = list(X.columns)

        self.preprocessor_ = build_preprocessor(
            self.columns_,
            scale_numeric=True
        )

        transformed = self.preprocessor_.fit_transform(X, y)

        self.regressor_ = LinearRegression()
        self.regressor_.fit(transformed, y)

        names = self.preprocessor_.get_feature_names_out()
        coefficients = np.asarray(
            self.regressor_.coef_
        ).reshape(-1)

        grouped_importance = []

        for col in self.columns_:

            if col in self.numeric_features:
                indexes = [
                    i for i, name in enumerate(names)
                    if name == f"num__{col}"
                ]
            else:
                indexes = [
                    i for i, name in enumerate(names)
                    if name.startswith(f"cat__{col}_")
                ]

            importance = (
                float(np.linalg.norm(coefficients[indexes]))
                if indexes else 0.0
            )

            grouped_importance.append(importance)

        self.coef_ = np.asarray(grouped_importance)
        self.intercept_ = self.regressor_.intercept_

        return self

    def predict(self, X):
        transformed = self.preprocessor_.transform(X)
        return self.regressor_.predict(transformed)


class GroupAwareRFECV:
    """Recursive feature elimination with CV supporting grouped categorical columns."""
    def __init__(self, estimator, cv, scoring="neg_root_mean_squared_error"):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring

    def fit(self, X, y):
        current_features = list(X.columns)
        history = []
        while len(current_features) >= 1:
            mean_rmse, _ = cv_rmse(make_ols(current_features), current_features, X, y)
            est = clone(self.estimator).fit(X[current_features], y)
            history.append({
                "features": current_features.copy(),
                "cv_rmse": mean_rmse,
                "n_features": len(current_features)
            })
            if len(current_features) == 1:
                break
            min_idx = int(np.argmin(est.coef_))
            current_features.pop(min_idx)
        history_by_n = sorted(history, key=lambda h: h["n_features"])
        self.cv_results_ = {
            "mean_test_score": np.array([-h["cv_rmse"] for h in history_by_n])
        }
        best_step = min(history, key=lambda h: h["cv_rmse"])
        self.n_features_ = best_step["n_features"]
        self.support_ = np.array([col in best_step["features"] for col in X.columns])
        self.selected_features_ = best_step["features"]
        return self

rfecv = GroupAwareRFECV(
    estimator=GroupAwareLinearRegressor(
        numeric_features=tuple(NUMERIC_FEATURES),
        categorical_features=tuple(CATEGORICAL_FEATURES)
    ),
    cv=cv,
    scoring="neg_root_mean_squared_error"
)
rfecv.fit(X_train, y_train)

best_rfecv_features = list(
    X_train.columns[rfecv.support_]
)

rfecv_scores = -rfecv.cv_results_["mean_test_score"]
rfecv_best_cv_rmse = float(rfecv_scores.min())

print("Selected features:", best_rfecv_features)
print("Selected feature count:", rfecv.n_features_)
print("Best RFECV CV RMSE:", round(rfecv_best_cv_rmse, 3))


# ================================================================
# METHODS 4 & 5 — LASSO AND ELASTIC NET
# Numeric variables are scaled before regularization.
# State dummy coefficients are mapped to the original State group.
# ================================================================

print("\n" + "=" * 70)
print("METHODS 4 & 5: LASSO AND ELASTIC NET")
print("=" * 70)


def make_regularized_pipeline(kind):

    if kind == "Lasso":
        regressor = Lasso(max_iter=100000)

    elif kind == "ElasticNet":
        regressor = ElasticNet(max_iter=100000)

    else:
        raise ValueError(f"Unknown estimator: {kind}")

    return Pipeline([
        (
            "preprocessor",
            build_preprocessor(
                ALL_FEATURES,
                scale_numeric=True
            )
        ),
        ("regressor", regressor)
    ])


def tune_regularized_model(kind):

    pipeline = make_regularized_pipeline(kind)

    if kind == "Lasso":
        parameter_grid = {
            "regressor__alpha": [
                0.001, 0.01, 0.1, 1.0, 10.0, 100.0
            ]
        }

    else:
        parameter_grid = {
            "regressor__alpha": [
                0.001, 0.01, 0.1, 1.0, 10.0, 100.0
            ],
            "regressor__l1_ratio": [
                0.1, 0.5, 0.9, 0.95
            ]
        }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=1,
        refit=True
    )

    search.fit(X_train, y_train)

    return search


def get_grouped_coefficients(fitted_pipeline):

    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    regressor = fitted_pipeline.named_steps["regressor"]

    names = preprocessor.get_feature_names_out()
    coefficients = np.asarray(regressor.coef_).reshape(-1)

    grouped = {}

    for col in ALL_FEATURES:

        if col in NUMERIC_FEATURES:
            indexes = [
                i for i, name in enumerate(names)
                if name == f"num__{col}"
            ]
        else:
            indexes = [
                i for i, name in enumerate(names)
                if name.startswith(f"cat__{col}_")
            ]

        grouped[col] = (
            float(np.linalg.norm(coefficients[indexes]))
            if indexes else 0.0
        )

    return grouped


def selected_nonzero_features(importance, tolerance=1e-8):

    selected = [
        col for col in ALL_FEATURES
        if importance[col] > tolerance
    ]

    if not selected:
        # Report the strongest group if regularization sets all
        # coefficients to zero.
        selected = [
            max(importance, key=importance.get)
        ]

    return selected


# METHOD 4: Lasso
lasso_search = tune_regularized_model("Lasso")

lasso_importance = get_grouped_coefficients(
    lasso_search.best_estimator_
)

best_lasso_features = selected_nonzero_features(
    lasso_importance
)

lasso_cv_rmse = -float(lasso_search.best_score_)

print("\nLASSO")
print("Best parameters:", lasso_search.best_params_)
print("Best CV RMSE:", round(lasso_cv_rmse, 3))
print("Selected original features:", best_lasso_features)
print("Grouped coefficient magnitudes:", lasso_importance)


# METHOD 5: Elastic Net
elastic_search = tune_regularized_model("ElasticNet")

elastic_importance = get_grouped_coefficients(
    elastic_search.best_estimator_
)

best_elastic_features = selected_nonzero_features(
    elastic_importance
)

elastic_cv_rmse = -float(elastic_search.best_score_)

print("\nELASTIC NET")
print("Best parameters:", elastic_search.best_params_)
print("Best CV RMSE:", round(elastic_cv_rmse, 3))
print("Selected original features:", best_elastic_features)
print("Grouped coefficient magnitudes:", elastic_importance)


# ================================================================
# TRAINING-CV SUMMARY
# These CV scores are exploratory selection scores, not unbiased
# estimates after searching many subsets and hyperparameters.
# ================================================================

cv_summary = pd.DataFrame([
    {
        "Model": "Full OLS",
        "Features": ALL_FEATURES,
        "CV_RMSE": baseline_cv_rmse,
        "CV_RMSE_SD": baseline_cv_sd
    },
    {
        "Model": "Exhaustive OLS",
        "Features": best_exhaustive_features,
        "CV_RMSE": float(exhaustive_table.iloc[0]["CV_RMSE"]),
        "CV_RMSE_SD": float(exhaustive_table.iloc[0]["CV_RMSE_SD"])
    },
    {
        "Model": "SFS OLS",
        "Features": best_sfs_features,
        "CV_RMSE": float(sfs_table.iloc[0]["CV_RMSE"]),
        "CV_RMSE_SD": float(sfs_table.iloc[0]["CV_RMSE_SD"])
    },
    {
        "Model": "RFECV OLS",
        "Features": best_rfecv_features,
        "CV_RMSE": rfecv_best_cv_rmse,
        "CV_RMSE_SD": np.nan
    },
    {
        "Model": "Lasso",
        "Features": best_lasso_features,
        "CV_RMSE": lasso_cv_rmse,
        "CV_RMSE_SD": np.nan
    },
    {
        "Model": "Elastic Net",
        "Features": best_elastic_features,
        "CV_RMSE": elastic_cv_rmse,
        "CV_RMSE_SD": np.nan
    }
]).sort_values("CV_RMSE").reset_index(drop=True)

print("\n" + "=" * 70)
print("TRAINING CROSS-VALIDATION SUMMARY")
print("=" * 70)
print(cv_summary.to_string(index=False))


# ================================================================
# STEP 5 — EVALUATION
# Evaluate each candidate once on the held-out test set.
# Do not choose the deployment model based on test performance.
# ================================================================

print("\nSTEP 5: FINAL TEST EVALUATION")

candidate_models = {
    "Full OLS": {
        "estimator": make_ols(ALL_FEATURES),
        "input_features": ALL_FEATURES,
        "selected_features": ALL_FEATURES
    },
    "Exhaustive OLS": {
        "estimator": make_ols(best_exhaustive_features),
        "input_features": best_exhaustive_features,
        "selected_features": best_exhaustive_features
    },
    "SFS OLS": {
        "estimator": make_ols(best_sfs_features),
        "input_features": best_sfs_features,
        "selected_features": best_sfs_features
    },
    "RFECV OLS": {
        "estimator": make_ols(best_rfecv_features),
        "input_features": best_rfecv_features,
        "selected_features": best_rfecv_features
    },
    "Lasso": {
        "estimator": clone(lasso_search.best_estimator_),
        "input_features": ALL_FEATURES,
        "selected_features": best_lasso_features
    },
    "Elastic Net": {
        "estimator": clone(elastic_search.best_estimator_),
        "input_features": ALL_FEATURES,
        "selected_features": best_elastic_features
    }
}

test_results = []
fitted_models = {}

for name, info in candidate_models.items():

    estimator = info["estimator"]
    input_features = info["input_features"]

    estimator.fit(
        X_train[input_features],
        y_train
    )

    predictions = estimator.predict(
        X_test[input_features]
    )

    test_results.append({
        "Model": name,
        "Selected_Features": ", ".join(
            info["selected_features"]
        ),
        "N_Features": len(info["selected_features"]),
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(
            mean_squared_error(y_test, predictions)
        ),
        "R2": r2_score(y_test, predictions)
    })

    fitted_models[name] = estimator


evaluation = (
    pd.DataFrame(test_results)
    .sort_values("RMSE")
    .reset_index(drop=True)
)

print("\nHeld-out test results:")
print(evaluation.round(4).to_string(index=False))

# Actual vs predicted for the full baseline
baseline_predictions = fitted_models["Full OLS"].predict(
    X_test[ALL_FEATURES]
)

plt.figure(figsize=(6, 6))

plt.scatter(
    y_test,
    baseline_predictions,
    alpha=0.8
)

lower = min(y_test.min(), baseline_predictions.min())
upper = max(y_test.max(), baseline_predictions.max())

plt.plot(
    [lower, upper],
    [lower, upper],
    "k--",
    label="Perfect prediction"
)

plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Full OLS: Actual vs Predicted Test Profit")
plt.legend()
plt.tight_layout()
plt.show()

# Residual diagnostics
residuals = y_test - baseline_predictions

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.scatterplot(
    x=baseline_predictions,
    y=residuals,
    ax=axes[0]
)
axes[0].axhline(0, linestyle="--", color="black")
axes[0].set_xlabel("Predicted Profit")
axes[0].set_ylabel("Residual")
axes[0].set_title("Residuals vs Predicted Profit")

sns.histplot(residuals, kde=True, ax=axes[1])
axes[1].set_title("Residual Distribution")

plt.tight_layout()
plt.show()

# Coefficients for the full OLS baseline
full_model = fitted_models["Full OLS"]

full_preprocessor = full_model.named_steps["preprocessor"]
full_regressor = full_model.named_steps["regressor"]

print("\nFull OLS intercept:")
print(full_regressor.intercept_)

print("\nFull OLS coefficients:")
print(pd.DataFrame({
    "Feature": full_preprocessor.get_feature_names_out(),
    "Coefficient": full_regressor.coef_
}).to_string(index=False))


# ================================================================
# STEP 6 — DEPLOYMENT
#
# Choose the candidate using training CV, NOT test RMSE.
# The CV score is exploratory because feature selection and
# tuning were performed on the same training CV folds.
# Nested CV is preferable for a rigorous model comparison.
# ================================================================

print("\nSTEP 6: DEPLOYMENT")

deployment_name = cv_summary.iloc[0]["Model"]
deployment_info = candidate_models[deployment_name]

deployment_features = deployment_info["input_features"]
deployment_model = fitted_models[deployment_name]

# Save a bundle with the model and its required input columns.
# This preserves the feature contract for future predictions.
deployment_bundle = {
    "model_name": deployment_name,
    "input_features": deployment_features,
    "selected_features": deployment_info["selected_features"],
    "model": deployment_model
}

joblib.dump(
    deployment_bundle,
    "startup_profit_pipeline.joblib"
)

print("Selected deployment candidate:", deployment_name)
print("Required input columns:", deployment_features)
print("Saved: startup_profit_pipeline.joblib")


# Example prediction using all four original input fields.
# The selected model uses only the columns it needs.
new_startup = pd.DataFrame([{
    "R&D Spend": 100000,
    "Administration": 120000,
    "Marketing Spend": 200000,
    "State": "California"
}])

loaded_bundle = joblib.load(
    "startup_profit_pipeline.joblib"
)

prediction = loaded_bundle["model"].predict(
    new_startup[loaded_bundle["input_features"]]
)[0]

print("\nExample startup:")
print(new_startup)

print(
    f"\nPredicted Profit ({deployment_name}): "
    f"{prediction:,.2f}"
)

# Later, load the saved model:
#
# bundle = joblib.load("startup_profit_pipeline.joblib")
# new_prediction = bundle["model"].predict(
#     new_startup[bundle["input_features"]]
# )
# print(new_prediction)
