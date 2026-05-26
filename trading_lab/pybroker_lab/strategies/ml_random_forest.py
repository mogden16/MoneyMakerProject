from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from trading_lab.pybroker_lab.config import PyBrokerLabConfig, PyBrokerStrategyDefinition
from trading_lab.pybroker_lab.strategies import (
    ensure_model,
    make_drawdown_indicator,
    make_return_indicator,
    make_rsi_indicator,
    make_sma_distance_indicator,
    make_vol_indicator,
    make_volume_change_indicator,
)


def build_strategy(config: PyBrokerLabConfig) -> PyBrokerStrategyDefinition:
    lookahead = int(config.strategy_params.get("label_lookahead", 5))
    label_threshold = float(config.strategy_params.get("label_threshold", 0.0))
    buy_threshold = float(config.strategy_params.get("buy_threshold", 0.55))
    sell_threshold = float(config.strategy_params.get("sell_threshold", 0.45))
    indicators = (
        make_return_indicator("pbl_rf_ret_1", 1),
        make_return_indicator("pbl_rf_ret_5", 5),
        make_return_indicator("pbl_rf_ret_10", 10),
        make_return_indicator("pbl_rf_ret_20", 20),
        make_sma_distance_indicator("pbl_rf_sma_dist_20", 20),
        make_vol_indicator("pbl_rf_realized_vol_20", 20),
        make_rsi_indicator("pbl_rf_rsi_14", 14),
        make_volume_change_indicator("pbl_rf_volume_change_5", 5),
        make_drawdown_indicator("pbl_rf_drawdown_20", 20),
    )
    feature_columns = tuple(ind.name for ind in indicators)
    model_name = f"pbl_random_forest_{uuid4().hex[:8]}"
    models_dir = Path(config.output_dir) / "models"

    def _prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[:, list(feature_columns)].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def train_fn(symbol: str, train_data: pd.DataFrame, test_data: pd.DataFrame, *, model_output_dir: str, threshold: float, lookahead_bars: int):
        train = train_data.sort_values("date").copy()
        forward_returns = train["close"].shift(-lookahead_bars) / train["close"] - 1.0
        labels = (forward_returns > threshold).astype(int)
        features = _prepare_features(train)
        valid = labels.notna()
        features = features.loc[valid]
        labels = labels.loc[valid]
        if features.empty:
            model_obj = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        else:
            model_obj = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42)
            model_obj.fit(features, labels)
        output_dir = Path(model_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = output_dir / f"{symbol}_{model_name}.joblib"
        joblib.dump({"model": model_obj, "features": feature_columns, "label_threshold": threshold}, artifact_path)
        return model_obj, feature_columns

    def predict_fn(model_obj, frame: pd.DataFrame) -> np.ndarray:
        features = _prepare_features(frame)
        if not hasattr(model_obj, "predict_proba") or features.empty:
            return np.zeros(len(frame), dtype=float)
        probabilities = model_obj.predict_proba(features)
        if probabilities.shape[1] == 1:
            return np.zeros(len(frame), dtype=float)
        return probabilities[:, 1]

    model_source = ensure_model(
        model_name,
        train_fn,
        indicators=indicators,
        predict_fn=predict_fn,
        model_output_dir=str(models_dir),
        threshold=label_threshold,
        lookahead_bars=lookahead,
    )

    def exec_fn(ctx) -> None:
        preds = ctx.preds(model_name)
        if len(preds) == 0:
            return
        probability = float(preds[-1])
        if ctx.long_pos() is None and probability > buy_threshold:
            ctx.buy_shares = ctx.calc_target_shares(1.0)
            ctx.hold_bars = lookahead
        elif ctx.long_pos() is not None and probability < sell_threshold:
            ctx.sell_all_shares()

    return PyBrokerStrategyDefinition(
        name="ml_random_forest",
        symbols=("SPY",),
        indicators=indicators,
        models=(model_source,),
        execution=exec_fn,
        lookahead=lookahead,
        description="Walk-forward RandomForestClassifier on simple SPY technical features.",
        assumptions=(
            f"Label = future {lookahead}-day return > {label_threshold:.2%}.",
            f"Buy/sell probability thresholds = {buy_threshold:.2f}/{sell_threshold:.2f}.",
        ),
    )
