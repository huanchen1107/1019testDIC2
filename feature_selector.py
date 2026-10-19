"""
Automated Feature Selection and Performance Evaluation Framework
=================================================================
A modular, production-ready framework for systematic feature selection,
metric evaluation, and publication-ready diagnostic visualization.

Supports:
  - Tasks: 'regression' and 'classification'
  - Algorithms:
      1. 'sfs' / 'forward': Sequential Forward Selection
      2. 'rfe' / 'backward': Recursive Feature Elimination / Backward Elimination
      3. 'exhaustive': Best-subset search across all k-combinations
      4. 'importance': Model feature importance / coefficient ranking
  - Preprocessing: Automated handling of missing values, categorical encoding,
                   and feature scaling with scikit-learn Pipelines.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Filter unnecessary deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class EvaluationRecord:
    """Stores evaluation metrics and selected features for a specific subset size k."""
    k: int
    selected_features: List[str]
    metrics: Dict[str, float]


@dataclass
class SelectionResult:
    """Holds the complete result of a feature selection run."""
    method: str
    task: str
    history: List[EvaluationRecord]
    best_record: EvaluationRecord
    table: pd.DataFrame
    feature_names: List[str]


class TabularPreprocessor:
    """
    Standard preprocessing for arbitrary tabular data:
      - Median imputation for numerical columns (optional scaling)
      - Most-frequent imputation + One-Hot Encoding for categorical columns
    """

    def __init__(
        self,
        scale_numeric: bool = True,
        categorical_drop: Optional[Literal["first", "if_binary"]] = "first",
    ):
        self.scale_numeric = scale_numeric
        self.categorical_drop = categorical_drop
        self.pipeline: Optional[ColumnTransformer] = None
        self.feature_names_: List[str] = []
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []

    def fit_transform(
        self,
        X: pd.DataFrame,
        numeric_cols: Optional[Sequence[str]] = None,
        categorical_cols: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Fit preprocessor and transform X into a numerical NumPy array."""
        if numeric_cols is None or categorical_cols is None:
            # Auto-detect column types
            detected_num = X.select_dtypes(include=[np.number]).columns.tolist()
            detected_cat = [c for c in X.columns if c not in detected_num]
            self.numeric_cols_ = numeric_cols if numeric_cols is not None else detected_num
            self.categorical_cols_ = categorical_cols if categorical_cols is not None else detected_cat
        else:
            self.numeric_cols_ = list(numeric_cols)
            self.categorical_cols_ = list(categorical_cols)

        transformers = []
        if self.numeric_cols_:
            num_steps = [("imputer", SimpleImputer(strategy="median"))]
            if self.scale_numeric:
                num_steps.append(("scaler", StandardScaler()))
            transformers.append(("num", Pipeline(num_steps), self.numeric_cols_))

        if self.categorical_cols_:
            cat_steps = [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        drop=self.categorical_drop,
                        sparse_output=False,
                        handle_unknown="ignore",
                    ),
                ),
            ]
            transformers.append(("cat", Pipeline(cat_steps), self.categorical_cols_))

        self.pipeline = ColumnTransformer(transformers=transformers, remainder="drop")
        X_trans = self.pipeline.fit_transform(X)

        # Retrieve readable transformed feature names
        out_names = []
        if hasattr(self.pipeline, "get_feature_names_out"):
            raw_names = self.pipeline.get_feature_names_out()
            for name in raw_names:
                # Clean prefix for readability (e.g., 'num__R&D Spend' -> 'R&D Spend')
                if name.startswith("num__"):
                    out_names.append(name[5:])
                elif name.startswith("cat__"):
                    out_names.append(name[5:])
                else:
                    out_names.append(name)
        else:
            out_names = [f"feature_{i}" for i in range(X_trans.shape[1])]

        self.feature_names_ = out_names
        return X_trans, self.feature_names_

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform unseen data using fitted pipeline."""
        if self.pipeline is None:
            raise ValueError("Preprocessor has not been fitted yet.")
        return self.pipeline.transform(X)


class AutomatedFeatureSelector:
    """
    Automated Feature Selection & Evaluation System.

    Supports:
      - Systematic subset selection (SFS, RFE, Exhaustive, Importance-Ranked)
      - Regression and Classification metrics
      - Tabular output tables
      - Dual side-by-side diagnostic visualization (Error vs Fit)
    """

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        task: Literal["regression", "classification"] = "regression",
        method: Literal["sfs", "forward", "rfe", "backward", "exhaustive", "importance"] = "sfs",
        max_features: Optional[int] = None,
        scale_numeric: bool = False,
        categorical_drop: Optional[Literal["first", "if_binary"]] = "first",
        scoring_metric: Optional[str] = None,
        random_state: int = 42,
    ):
        self.task = task.lower()
        self.method = method.lower()
        self.max_features = max_features
        self.scale_numeric = scale_numeric
        self.categorical_drop = categorical_drop
        self.scoring_metric = scoring_metric
        self.random_state = random_state

        # Default estimator if none provided
        if estimator is None:
            if self.task == "regression":
                self.estimator = LinearRegression()
            else:
                self.estimator = LogisticRegression(max_iter=1000, random_state=random_state)
        else:
            self.estimator = estimator

        self.preprocessor = TabularPreprocessor(
            scale_numeric=scale_numeric,
            categorical_drop=categorical_drop,
        )

        self.results_: Optional[SelectionResult] = None

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute standard metrics based on task type."""
        metrics: Dict[str, float] = {}

        if self.task == "regression":
            mse = mean_squared_error(y_true, y_pred)
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(y_true, y_pred))
            r2 = float(r2_score(y_true, y_pred))
            metrics["RMSE"] = rmse
            metrics["MSE"] = float(mse)
            metrics["MAE"] = mae
            metrics["R-squared"] = r2

        elif self.task == "classification":
            acc = float(accuracy_score(y_true, y_pred))
            f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            metrics["Error Rate"] = 1.0 - acc
            metrics["Accuracy"] = acc
            metrics["F1-Score"] = f1

            if y_proba is not None:
                try:
                    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                        auc = float(roc_auc_score(y_true, y_proba[:, 1]))
                    else:
                        auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
                    metrics["ROC-AUC"] = auc
                except Exception:
                    metrics["ROC-AUC"] = np.nan
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        return metrics

    def _evaluate_subset(
        self,
        feature_indices: Sequence[int],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        """Train model on specified feature indices and evaluate on validation/test set."""
        idx = list(feature_indices)
        X_tr_sub = X_train[:, idx]
        X_val_sub = X_val[:, idx]

        model = clone(self.estimator)
        model.fit(X_tr_sub, y_train)

        preds = model.predict(X_val_sub)
        probas = None
        if self.task == "classification" and hasattr(model, "predict_proba"):
            try:
                probas = model.predict_proba(X_val_sub)
            except Exception:
                probas = None

        return self._compute_metrics(y_val, preds, probas)

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        val_size: float = 0.2,
        numeric_cols: Optional[Sequence[str]] = None,
        categorical_cols: Optional[Sequence[str]] = None,
        X_val_custom: Optional[pd.DataFrame] = None,
        y_val_custom: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> SelectionResult:
        """
        Run the complete feature selection workflow.
        """
        # Step 1: Preprocess Features
        X_proc, feature_names = self.preprocessor.fit_transform(
            X, numeric_cols=numeric_cols, categorical_cols=categorical_cols
        )
        y_arr = np.asarray(y)

        # Step 2: Split Train and Validation
        if X_val_custom is not None and y_val_custom is not None:
            X_train = X_proc
            y_train = y_arr
            X_val = self.preprocessor.transform(X_val_custom)
            y_val = np.asarray(y_val_custom)
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_proc, y_arr, test_size=val_size, random_state=self.random_state
            )

        n_total_features = X_train.shape[1]
        k_limit = min(n_total_features, self.max_features or n_total_features)

        history: List[EvaluationRecord] = []

        # Step 3: Run Feature Selection by Algorithm
        method_normalized = self.method
        if method_normalized in ["sfs", "forward"]:
            history = self._run_forward_selection(X_train, y_train, X_val, y_val, feature_names, k_limit)
        elif method_normalized in ["rfe", "backward"]:
            history = self._run_backward_elimination(X_train, y_train, X_val, y_val, feature_names, k_limit)
        elif method_normalized == "exhaustive":
            history = self._run_exhaustive_selection(X_train, y_train, X_val, y_val, feature_names, k_limit)
        elif method_normalized == "importance":
            history = self._run_importance_selection(X_train, y_train, X_val, y_val, feature_names, k_limit)
        else:
            raise ValueError(f"Unknown feature selection method: {self.method}")

        # Step 4: Construct Output Table & Determine Best
        primary_err = "RMSE" if self.task == "regression" else "Error Rate"
        primary_fit = "R-squared" if self.task == "regression" else "Accuracy"

        # Selection criterion (minimize error)
        best_record = min(history, key=lambda rec: rec.metrics[primary_err])

        table_rows = []
        for rec in history:
            row = {
                "Number of Features": rec.k,
                "Selected Features": rec.selected_features,
            }
            # Append all computed metrics
            for m_key, m_val in rec.metrics.items():
                row[m_key] = m_val
            table_rows.append(row)

        table_df = pd.DataFrame(table_rows).sort_values("Number of Features").reset_index(drop=True)

        self.results_ = SelectionResult(
            method=self.method,
            task=self.task,
            history=history,
            best_record=best_record,
            table=table_df,
            feature_names=feature_names,
        )

        return self.results_

    # ---------------- Search Algorithms ----------------

    def _run_forward_selection(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        k_limit: int,
    ) -> List[EvaluationRecord]:
        """Sequential Forward Selection (SFS)."""
        history = []
        selected_idx: List[int] = []
        primary_err = "RMSE" if self.task == "regression" else "Error Rate"

        for k in range(1, k_limit + 1):
            candidates = [i for i in range(len(feature_names)) if i not in selected_idx]
            best_cand = None
            best_metrics = None
            best_score = float("inf")

            for cand in candidates:
                current_subset = selected_idx + [cand]
                metrics = self._evaluate_subset(current_subset, X_train, y_train, X_val, y_val)
                err = metrics[primary_err]
                if err < best_score:
                    best_score = err
                    best_cand = cand
                    best_metrics = metrics

            selected_idx.append(best_cand)  # type: ignore
            history.append(
                EvaluationRecord(
                    k=k,
                    selected_features=[feature_names[i] for i in selected_idx],
                    metrics=best_metrics,  # type: ignore
                )
            )

        return history

    def _run_backward_elimination(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        k_limit: int,
    ) -> List[EvaluationRecord]:
        """Recursive Feature Elimination / Backward Elimination."""
        current_idx = list(range(len(feature_names)))
        history_reverse = []
        primary_err = "RMSE" if self.task == "regression" else "Error Rate"

        while len(current_idx) >= 1:
            metrics = self._evaluate_subset(current_idx, X_train, y_train, X_val, y_val)
            if len(current_idx) <= k_limit:
                history_reverse.append(
                    EvaluationRecord(
                        k=len(current_idx),
                        selected_features=[feature_names[i] for i in current_idx],
                        metrics=metrics,
                    )
                )

            if len(current_idx) == 1:
                break

            # Choose the feature whose removal results in lowest validation error
            best_drop_idx = None
            best_drop_err = float("inf")

            for cand in current_idx:
                trial_sub = [i for i in current_idx if i != cand]
                trial_metrics = self._evaluate_subset(trial_sub, X_train, y_train, X_val, y_val)
                trial_err = trial_metrics[primary_err]
                if trial_err < best_drop_err:
                    best_drop_err = trial_err
                    best_drop_idx = cand

            current_idx.remove(best_drop_idx)  # type: ignore

        # Return sorted by k ascending
        return sorted(history_reverse, key=lambda rec: rec.k)

    def _run_exhaustive_selection(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        k_limit: int,
    ) -> List[EvaluationRecord]:
        """Exhaustive Search: optimal subset evaluated across all combinations for each k."""
        history = []
        n_features = len(feature_names)
        primary_err = "RMSE" if self.task == "regression" else "Error Rate"

        if n_features > 16:
            warnings.warn(
                f"Exhaustive search on {n_features} features may be slow. "
                "Consider 'sfs' or 'rfe' for larger feature sets.",
                UserWarning,
            )

        for k in range(1, k_limit + 1):
            best_combo = None
            best_metrics = None
            best_err = float("inf")

            for combo in itertools.combinations(range(n_features), k):
                metrics = self._evaluate_subset(combo, X_train, y_train, X_val, y_val)
                err = metrics[primary_err]
                if err < best_err:
                    best_err = err
                    best_combo = combo
                    best_metrics = metrics

            history.append(
                EvaluationRecord(
                    k=k,
                    selected_features=[feature_names[i] for i in best_combo],  # type: ignore
                    metrics=best_metrics,  # type: ignore
                )
            )

        return history

    def _run_importance_selection(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        k_limit: int,
    ) -> List[EvaluationRecord]:
        """Rank features by estimator's coef_ or feature_importances_, then evaluate top-k."""
        full_model = clone(self.estimator)
        full_model.fit(X_train, y_train)

        if hasattr(full_model, "coef_"):
            coef = np.asarray(full_model.coef_)
            if coef.ndim > 1:
                coef = np.linalg.norm(coef, axis=0)
            importance = np.abs(coef)
        elif hasattr(full_model, "feature_importances_"):
            importance = np.asarray(full_model.feature_importances_)
        else:
            raise ValueError("Estimator has neither coef_ nor feature_importances_.")

        ranked_indices = list(np.argsort(importance)[::-1])
        history = []

        for k in range(1, k_limit + 1):
            subset = ranked_indices[:k]
            metrics = self._evaluate_subset(subset, X_train, y_train, X_val, y_val)
            history.append(
                EvaluationRecord(
                    k=k,
                    selected_features=[feature_names[i] for i in subset],
                    metrics=metrics,
                )
            )

        return history

    # ---------------- Output & Visualization ----------------

    def print_summary_table(self) -> None:
        """Print a formatted comparison table and highlight the optimal subset."""
        if self.results_ is None:
            raise ValueError("Run fit() before printing summary table.")

        table = self.results_.table.copy()
        primary_err = "RMSE" if self.task == "regression" else "Error Rate"
        primary_fit = "R-squared" if self.task == "regression" else "Accuracy"

        print("\n" + "=" * 90)
        print(f"FEATURE SELECTION RESULTS: {self.results_.method.upper()} ({self.task.upper()})")
        print("=" * 90)

        # Format column headers and values
        display_cols = ["Number of Features", "Selected Features", primary_err, primary_fit]
        if "MAE" in table.columns and self.task == "regression":
            display_cols.insert(3, "MAE")
        elif "F1-Score" in table.columns and self.task == "classification":
            display_cols.insert(3, "F1-Score")

        # Format float precision
        format_dict = {
            col: (lambda x: f"{x:.6f}")
            for col in display_cols
            if col not in ["Number of Features", "Selected Features"]
        }
        print(table[display_cols].to_string(index=False, formatters=format_dict))

        best = self.results_.best_record
        print("-" * 90)
        print(f"★ OPTIMAL SUBSET (k = {best.k}):")
        print(f"  Selected Features : {best.selected_features}")
        print(f"  {primary_err:<18}: {best.metrics[primary_err]:.6f}")
        print(f"  {primary_fit:<18}: {best.metrics[primary_fit]:.6f}")
        print("=" * 90 + "\n")

    def plot_evaluation(
        self,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (12, 5),
        show: bool = True,
    ) -> plt.Figure:
        """
        Generate two side-by-side plots:
          - Left: Error Metric vs. Number of Features (with marked minimum/elbow)
          - Right: Goodness-of-Fit / Performance Metric vs. Number of Features (with marked peak)
        """
        if self.results_ is None:
            raise ValueError("Run fit() before generating plots.")

        df = self.results_.table
        ks = df["Number of Features"].values

        primary_err = "RMSE" if self.task == "regression" else "Error Rate"
        primary_fit = "R-squared" if self.task == "regression" else "Accuracy"

        err_values = df[primary_err].values
        fit_values = df[primary_fit].values

        best_k = self.results_.best_record.k
        best_err = self.results_.best_record.metrics[primary_err]
        best_fit = self.results_.best_record.metrics[primary_fit]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Style & Palette
        line_color = "#1f77b4"
        highlight_color = "#d62728"

        # ---------------- Plot 1: Error Metric ----------------
        ax1.plot(ks, err_values, marker="o", color=line_color, linewidth=1.8, markersize=6)
        ax1.scatter([best_k], [best_err], color=highlight_color, s=90, zorder=5, label=f"Optimal (k={best_k})")
        ax1.set_title(f"{primary_err} by Number of Features", fontsize=12, fontweight="bold", pad=10)
        ax1.set_xlabel("Number of Features", fontsize=11)
        ax1.set_ylabel(primary_err, fontsize=11)
        ax1.set_xticks(ks)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="upper right", framealpha=0.9)

        # ---------------- Plot 2: Performance / Fit Metric ----------------
        ax2.plot(ks, fit_values, marker="o", color=line_color, linewidth=1.8, markersize=6)
        ax2.scatter([best_k], [best_fit], color=highlight_color, s=90, zorder=5, label=f"Optimal (k={best_k})")
        ax2.set_title(f"{primary_fit} by Number of Features", fontsize=12, fontweight="bold", pad=10)
        ax2.set_xlabel("Number of Features", fontsize=11)
        ax2.set_ylabel(primary_fit, fontsize=11)
        ax2.set_xticks(ks)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="lower right", framealpha=0.9)

        plt.tight_layout()

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=200, bbox_inches="tight")
            print(f"Plot saved to: {p.resolve()}")

        if show:
            plt.show()

        return fig


# =====================================================================
# DEMONSTRATION & EXAMPLE USE CASE
# =====================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING AUTOMATED FEATURE SELECTION DEMONSTRATION")
    print("Dataset: Kaggle 50 Startups (Predicting Profit)")
    print("====================================================================\n")

    # 1. Dataset Configuration (Swap in any dataset or URL here)
    DATASET_URL = (
        "https://raw.githubusercontent.com/SantiagoMorenoV/"
        "50-Startups-Multiple-Reg/main/50_Startups.csv"
    )
    df = pd.read_csv(DATASET_URL)
    TARGET_COLUMN = "Profit"
    TASK = "regression"

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Output directory for saving diagnostic plots
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # 2. Compare the 3 Primary Feature Selection Algorithms
    algorithms: List[Literal["sfs", "rfe", "exhaustive"]] = ["sfs", "rfe", "exhaustive"]

    for algo in algorithms:
        print(f"\n>>> Running Algorithm: {algo.upper()} ...")

        selector = AutomatedFeatureSelector(
            estimator=LinearRegression(),
            task=TASK,
            method=algo,
            scale_numeric=False,           # Raw scale matches baseline regression
            categorical_drop="first",      # Dummy encoding drops reference category
            random_state=0,                # Preserves deterministic comparison split
        )

        results = selector.fit(X, y, val_size=0.2)

        # Print Comparison Table
        selector.print_summary_table()

        # Generate & Save Side-by-Side Diagnostic Plots
        plot_filename = plots_dir / f"{algo}_feature_selection_results.png"
        selector.plot_evaluation(save_path=plot_filename, show=False)

    print("\n" + "=" * 80)
    print("ALL THREE ALGORITHMS COMPLETED SUCCESSFULLY!")
    print(f"Diagnostic plots saved in: {plots_dir.resolve()}")
    print("====================================================================\n")
