from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelRunResult:
    summary: pd.DataFrame
    folds: pd.DataFrame
    feature_importance: pd.DataFrame
    approved_signals: pd.DataFrame
    comparison: pd.DataFrame
    approved_breakdown: pd.DataFrame
    approved_breakdown_raw: pd.DataFrame
    warnings: list[str]


def prepare_model_matrix(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    model_frame = dataset.copy().sort_values("timestamp").reset_index(drop=True)
    y = model_frame["label_good_signal"].astype(int)
    feature_columns = [
        "close",
        "return_1d",
        "return_5d",
        "return_10d",
        "return_20d",
        "realized_vol_10d",
        "realized_vol_20d",
        "rsi_14",
        "sma_distance_20",
        "sma_distance_50",
        "sma_distance_200",
        "close_above_200_sma",
        "atr_14",
        "atr_pct",
        "atr_percentile",
        "drawdown_from_20d_high",
        "drawdown_from_50d_high",
        "day_of_week",
        "month",
        "entry_strategy_name",
        "exit_structure_name",
        "timeframe",
    ]
    available = [column for column in feature_columns if column in model_frame.columns]
    X = pd.get_dummies(model_frame[available], columns=[column for column in ["entry_strategy_name", "exit_structure_name", "timeframe"] if column in available], dummy_na=False)
    X = X.fillna(0.0)
    return X, y


def build_time_series_split(n_samples: int, requested_splits: int = 5, gap: int = 2) -> TimeSeriesSplit:
    usable_splits = max(2, min(requested_splits, max(2, n_samples // 30)))
    return TimeSeriesSplit(n_splits=usable_splits, gap=min(gap, max(0, n_samples // 20)))


def build_practical_approved_breakdown(approved_breakdown_raw: pd.DataFrame, min_signals: int = 10) -> pd.DataFrame:
    if approved_breakdown_raw.empty:
        return pd.DataFrame()
    filtered = approved_breakdown_raw[approved_breakdown_raw["approved_signal_count"] >= min_signals].copy()
    if filtered.empty:
        return pd.DataFrame()
    return filtered.sort_values(
        ["model_name", "approved_signal_count", "avg_forward_return_10d", "win_rate"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def train_time_series_models(dataset: pd.DataFrame) -> ModelRunResult:
    warnings: list[str] = []
    if dataset.empty or len(dataset) < 60:
        warnings.append("Too few signals are available for time-series model training.")
        return ModelRunResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), warnings)
    if dataset["label_good_signal"].sum() < 10:
        warnings.append("Too few positive labels are available for a meaningful model run.")
        return ModelRunResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), warnings)

    X, y = prepare_model_matrix(dataset)
    splitter = build_time_series_split(len(dataset))
    estimators = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, solver="lbfgs")),
            ]
        ),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, min_samples_leaf=5),
    }
    fold_rows: list[dict[str, Any]] = []
    approved_rows: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []
    for model_name, estimator in estimators.items():
        for fold_number, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                warnings.append(f"{model_name} fold {fold_number} had only one class and was skipped.")
                continue
            estimator.fit(X_train, y_train)
            probabilities = estimator.predict_proba(X_test)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
            fold_row = {
                "model_name": model_name,
                "fold": fold_number,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
                "roc_auc": roc_auc_score(y_test, probabilities),
                "average_precision": average_precision_score(y_test, probabilities),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
            test_rows = dataset.iloc[test_idx].copy()
            test_rows["model_name"] = model_name
            test_rows["fold"] = fold_number
            test_rows["predicted_probability"] = probabilities
            test_rows["predicted_label"] = predictions
            approved = test_rows[test_rows["predicted_label"] == 1].copy()
            fold_row["approved_signal_count"] = int(len(approved))
            fold_row["approved_avg_forward_return_10d"] = float(approved["forward_return_10d"].mean()) if not approved.empty else 0.0
            fold_row["approved_win_rate"] = float((approved["forward_return_10d"] > 0).mean()) if not approved.empty else 0.0
            fold_row["approved_avg_r_multiple"] = float((approved["forward_return_10d"] / approved["atr_pct"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).mean()) if not approved.empty else 0.0
            fold_row["baseline_avg_forward_return_10d"] = float(test_rows["forward_return_10d"].mean())
            fold_row["baseline_win_rate"] = float((test_rows["forward_return_10d"] > 0).mean())
            fold_rows.append(fold_row)
            approved_rows.append(approved)

            if hasattr(estimator, "coef_"):
                for feature_name, importance in zip(X.columns, estimator.coef_[0], strict=False):
                    importance_rows.append({"model_name": model_name, "feature_name": feature_name, "importance": abs(float(importance))})
            elif hasattr(estimator, "feature_importances_"):
                for feature_name, importance in zip(X.columns, estimator.feature_importances_, strict=False):
                    importance_rows.append({"model_name": model_name, "feature_name": feature_name, "importance": float(importance)})

    fold_df = pd.DataFrame(fold_rows)
    if fold_df.empty:
        warnings.append("No valid time-series model folds completed.")
        return ModelRunResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), warnings)

    summary_df = (
        fold_df.groupby("model_name", as_index=False)
        .agg(
            folds=("fold", "count"),
            accuracy=("accuracy", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            roc_auc=("roc_auc", "mean"),
            average_precision=("average_precision", "mean"),
            approved_signal_count=("approved_signal_count", "sum"),
            approved_avg_forward_return_10d=("approved_avg_forward_return_10d", "mean"),
            approved_win_rate=("approved_win_rate", "mean"),
            approved_avg_r_multiple=("approved_avg_r_multiple", "mean"),
            baseline_avg_forward_return_10d=("baseline_avg_forward_return_10d", "mean"),
            baseline_win_rate=("baseline_win_rate", "mean"),
        )
    )
    comparison_df = summary_df.copy()
    comparison_df["approval_rate"] = comparison_df["approved_signal_count"] / max(len(dataset), 1)
    comparison_df["forward_return_edge_10d"] = comparison_df["approved_avg_forward_return_10d"] - comparison_df["baseline_avg_forward_return_10d"]
    comparison_df["win_rate_edge"] = comparison_df["approved_win_rate"] - comparison_df["baseline_win_rate"]

    importance_df = pd.DataFrame(importance_rows)
    if not importance_df.empty:
        importance_df = (
            importance_df.groupby(["model_name", "feature_name"], as_index=False)["importance"]
            .mean()
            .sort_values(["model_name", "importance"], ascending=[True, False])
        )
        dominant = importance_df.groupby("model_name").head(1)
        if (dominant["importance"] > 0.6).any():
            warnings.append("One model appears to be dominated by a single feature. Review for suspicious leakage or regime overfit.")
    approved_signals = pd.concat(approved_rows, ignore_index=True) if approved_rows else pd.DataFrame()
    approved_breakdown = pd.DataFrame()
    approved_breakdown_raw = pd.DataFrame()
    if not approved_signals.empty:
        requested_breakdown_keys = [
            "model_name",
            "entry_strategy_name",
            "entry_parameters_json",
            "exit_structure_name",
            "exit_parameters_json",
            "timeframe",
        ]
        breakdown_keys = [column for column in requested_breakdown_keys if column in approved_signals.columns]
        base_breakdown = (
            approved_signals.groupby(
                breakdown_keys,
                as_index=False,
            )
            .agg(
                approved_signal_count=("predicted_label", "size"),
                avg_forward_return_10d=("forward_return_10d", "mean"),
                win_rate=("forward_return_10d", lambda series: float((series > 0).mean())),
                positive_label_rate=("label_good_signal", "mean"),
            )
        )
        r_multiple_breakdown = (
            approved_signals.groupby(
                breakdown_keys,
                as_index=False,
            )
            .apply(
                lambda group: pd.Series(
                    {
                        "avg_r_multiple": float(
                            (
                                group["forward_return_10d"] / group["atr_pct"].replace(0, np.nan)
                            )
                            .replace([np.inf, -np.inf], np.nan)
                            .mean()
                        )
                    }
                ),
                include_groups=False,
            )
        )
        approved_breakdown_raw = base_breakdown.merge(
            r_multiple_breakdown,
            on=breakdown_keys,
            how="left",
        ).sort_values(["model_name", "avg_forward_return_10d", "approved_signal_count"], ascending=[True, False, False])
        approved_breakdown = build_practical_approved_breakdown(approved_breakdown_raw)
    if not approved_signals.empty and len(approved_signals) < 10:
        warnings.append("The model approved too few signals to be practically useful.")
    if not approved_breakdown_raw.empty and approved_breakdown.empty:
        warnings.append("Approved-signal groups exist, but none met the minimum-size threshold for practical breakdown review.")
    fold_spread = fold_df.groupby("model_name")["f1"].agg(lambda series: float(series.max() - series.min()))
    if (fold_spread > 0.35).any():
        warnings.append("Model performance varies widely across folds. Treat the results cautiously.")
    if (summary_df["approved_avg_forward_return_10d"] <= summary_df["baseline_avg_forward_return_10d"]).any():
        warnings.append("At least one model did not improve forward-return quality over the unfiltered signal set.")
    if not comparison_df.empty and (comparison_df["forward_return_edge_10d"] <= 0).all():
        warnings.append("No trained model improved average 10-day forward return over the baseline signal set.")
    return ModelRunResult(summary_df, fold_df, importance_df, approved_signals, comparison_df, approved_breakdown, approved_breakdown_raw, warnings)
