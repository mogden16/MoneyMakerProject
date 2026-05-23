from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from trading_lab.backtest.engine import BacktestConfig
from trading_lab.backtest.robustness import parameter_stability_summary
from trading_lab.backtest.train_test import split_data_by_percentage
from trading_lab.strategies.base import StrategyBase


@dataclass
class SignalScanResult:
    ticker: str
    strategy: str
    signal_type: str
    signal_date: pd.Timestamp | None
    latest_close: float
    suggested_entry_reference: float
    suggested_stop: float
    suggested_target: float
    risk_per_share: float
    reward_per_share: float
    reward_risk_ratio: float
    explanation: str
    notes_warnings: list[str]
    latest_run_id: str | None = None
    qualification_id: str | None = None
    robustness_score: int | None = None
    qualification_status: str | None = None
    signal_quality_score: int | None = None
    signal_quality_label: str | None = None
    signal_quality_bullets: list[str] | None = None

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        if self.signal_date is not None:
            record["signal_date"] = pd.Timestamp(self.signal_date)
        record["notes_warnings"] = " | ".join(self.notes_warnings)
        record["signal_quality_bullets"] = " | ".join(self.signal_quality_bullets or [])
        return record


@dataclass
class SignalQualityAssessment:
    score: int
    label: str
    bullets: list[str]


def _append_position_state(frame: pd.DataFrame) -> pd.DataFrame:
    """Track whether a strategy would still be in a long state after each bar."""
    state = False
    states: list[bool] = []
    for _, row in frame.iterrows():
        if bool(row.get("exit_signal", False)):
            state = False
            states.append(state)
            continue
        if bool(row.get("entry_signal", False)):
            state = True
        states.append(state)
    result = frame.copy()
    result["position_active"] = states
    return result


def determine_signal_type(frame: pd.DataFrame) -> str:
    """Classify the latest strategy state into a scanner-friendly label."""
    if frame.empty:
        return "no_signal"
    latest = frame.iloc[-1]
    if bool(latest.get("entry_signal", False)):
        return "new_buy_signal"
    if bool(latest.get("exit_signal", False)):
        return "exit_signal"
    if bool(latest.get("position_active", False)):
        return "active_long_signal"
    return "no_signal"


def explain_signal(strategy_name: str, frame: pd.DataFrame, signal_type: str) -> tuple[str, list[str]]:
    """Generate a plain-English explanation tied to the actual latest bar conditions."""
    if frame.empty:
        return "No bars were available for this symbol.", ["No data was available for the selected symbol."]
    latest = frame.iloc[-1]
    warnings: list[str] = []
    symbol = str(latest.get("symbol", "ticker"))

    if strategy_name == "Moving Average Crossover":
        fast = latest.get("fast_ma")
        slow = latest.get("slow_ma")
        if signal_type == "new_buy_signal":
            return (
                f"{symbol} triggered a moving-average entry because the fast average crossed above the slow average.",
                warnings,
            )
        if signal_type == "exit_signal":
            return (
                f"{symbol} triggered an exit because the fast average crossed back below the slow average.",
                warnings,
            )
        if signal_type == "active_long_signal":
            return (
                f"{symbol} remains in an active moving-average long state because the fast average is still above the slow average.",
                warnings,
            )
        return (
            f"{symbol} has no moving-average signal because the fast average is not in a fresh bullish crossover state.",
            warnings,
        )

    if strategy_name == "SPY 200-Day Trend Filter":
        if signal_type == "new_buy_signal":
            return (
                f"{symbol} triggered a trend-filter entry because the close moved back above the long-term SMA.",
                warnings,
            )
        if signal_type == "exit_signal":
            return (
                f"{symbol} triggered a trend-filter exit because the close fell back below the long-term SMA.",
                warnings,
            )
        if signal_type == "active_long_signal":
            return (
                f"{symbol} remains in an active trend-filter long state because the close is still above the long-term SMA.",
                warnings,
            )
        return (
            f"{symbol} has no trend-filter signal because price is not in a fresh move above the long-term SMA.",
            warnings,
        )

    if strategy_name == "RSI Mean Reversion":
        rsi = latest.get("rsi")
        sma_200 = latest.get("sma_200")
        close = latest.get("close")
        if pd.notna(rsi) and pd.notna(sma_200) and float(rsi) < float(frame["rsi"].min(skipna=True) if frame["rsi"].notna().any() else rsi) - 1:
            warnings.append("RSI is unusually stretched. Expect bounce timing to remain uncertain.")
        if signal_type == "new_buy_signal":
            return (
                f"{symbol} triggered an RSI mean-reversion entry because RSI fell below the buy threshold while price stayed above the 200-day SMA.",
                warnings,
            )
        if signal_type == "exit_signal":
            return (
                f"{symbol} triggered an RSI mean-reversion exit because RSI moved above the sell threshold.",
                warnings,
            )
        if pd.notna(rsi) and pd.notna(sma_200) and pd.notna(close) and float(rsi) < 30 and float(close) <= float(sma_200):
            return (
                f"{symbol} generated an oversold RSI reading, but the stock is below its 200-day SMA, so the long signal was filtered out.",
                warnings,
            )
        if signal_type == "active_long_signal":
            return (
                f"{symbol} remains in an active RSI mean-reversion long after a prior oversold entry.",
                warnings,
            )
        return (
            f"{symbol} has no RSI mean-reversion signal because the oversold threshold was not met with the trend filter satisfied.",
            warnings,
        )

    if strategy_name == "Daily Breakout":
        if signal_type == "new_buy_signal":
            return (
                f"{symbol} triggered a breakout because the close exceeded the prior lookback high.",
                warnings,
            )
        if signal_type == "exit_signal":
            return (
                f"{symbol} triggered a breakout exit because the close fell below the prior lookback low.",
                warnings,
            )
        if signal_type == "active_long_signal":
            return (
                f"{symbol} remains in an active breakout long after a prior breakout entry.",
                warnings,
            )
        return (
            f"{symbol} has no breakout signal because price did not close beyond the prior breakout range.",
            warnings,
        )

    if strategy_name == "QQE/HMA Daily":
        trend = latest.get("trend")
        close = latest.get("close")
        hma = latest.get("hma")
        if signal_type == "new_buy_signal":
            return (
                f"{symbol} triggered a QQE/HMA entry because the QQE trend turned bullish while price was above the HMA.",
                warnings,
            )
        if signal_type == "exit_signal":
            return (
                f"{symbol} triggered a QQE/HMA exit because the QQE trend turned bearish or price broke below the HMA.",
                warnings,
            )
        if signal_type == "active_long_signal":
            return (
                f"{symbol} is above the HMA and the QQE trend remains bullish, so the strategy is still in an active long state.",
                warnings,
            )
        if pd.notna(trend) and int(trend) == 1 and pd.notna(close) and pd.notna(hma) and float(close) <= float(hma):
            return (
                f"{symbol} has a bullish QQE trend, but price is still below the HMA, so the daily long setup is filtered out.",
                warnings,
            )
        return (
            f"{symbol} has no QQE/HMA signal because the trend and HMA filter are not aligned for a fresh daily entry.",
            warnings,
        )

    return (f"{symbol} has no supported explanation for {strategy_name}.", warnings)


def _default_target(entry: float, stop: float, take_profit_pct: float | None) -> float:
    if take_profit_pct is not None and take_profit_pct > 0:
        return entry * (1 + take_profit_pct)
    risk = max(entry - stop, 0.01)
    return entry + 2 * risk


def _build_plan_levels(strategy_name: str, frame: pd.DataFrame, config: BacktestConfig) -> tuple[float, float, float]:
    latest = frame.iloc[-1]
    entry = float(latest.get("close", 0.0) or 0.0)
    stop_pct = config.stop_loss_pct if config.stop_loss_pct not in (None, 0) else 0.08
    if strategy_name == "Moving Average Crossover":
        stop = float(latest.get("slow_ma")) if pd.notna(latest.get("slow_ma")) else entry * (1 - stop_pct)
    elif strategy_name == "SPY 200-Day Trend Filter":
        stop = float(latest.get("trend_sma")) if pd.notna(latest.get("trend_sma")) else entry * (1 - stop_pct)
    elif strategy_name == "RSI Mean Reversion":
        stop = min(float(latest.get("low", entry)), float(latest.get("sma_200")) if pd.notna(latest.get("sma_200")) else entry)
    elif strategy_name == "Daily Breakout":
        stop = float(latest.get("prior_low")) if pd.notna(latest.get("prior_low")) else entry * (1 - stop_pct)
    else:
        stop = float(latest.get("hma")) if pd.notna(latest.get("hma")) else entry * (1 - stop_pct)
    if stop <= 0 or stop >= entry:
        stop = entry * (1 - stop_pct)
    target = _default_target(entry, stop, config.take_profit_pct)
    return entry, stop, target


def scan_symbol_strategy(
    *,
    ticker: str,
    bars: pd.DataFrame,
    strategy_name: str,
    strategy: StrategyBase,
    config: BacktestConfig,
    latest_run_id: str | None = None,
    qualification_id: str | None = None,
    robustness_score: int | None = None,
    qualification_status: str | None = None,
    quality_inputs: dict[str, object] | None = None,
) -> SignalScanResult:
    """Scan one symbol with one strategy and return a normalized signal record."""
    if bars.empty:
        result = SignalScanResult(
            ticker=ticker,
            strategy=strategy_name,
            signal_type="no_signal",
            signal_date=None,
            latest_close=0.0,
            suggested_entry_reference=0.0,
            suggested_stop=0.0,
            suggested_target=0.0,
            risk_per_share=0.0,
            reward_per_share=0.0,
            reward_risk_ratio=0.0,
            explanation="No bars were available for scanning.",
            notes_warnings=["No data available."],
            latest_run_id=latest_run_id,
            qualification_id=qualification_id,
            robustness_score=robustness_score,
            qualification_status=qualification_status,
        )
        quality = evaluate_signal_quality(result, quality_inputs or {})
        result.signal_quality_score = quality.score
        result.signal_quality_label = quality.label
        result.signal_quality_bullets = quality.bullets
        return result

    frame = strategy.generate_signals(bars.copy().sort_values("timestamp").reset_index(drop=True))
    frame["symbol"] = ticker
    frame = _append_position_state(frame)
    signal_type = determine_signal_type(frame)
    latest_row = frame.iloc[-1]
    signal_date = pd.Timestamp(latest_row["timestamp"]) if signal_type != "no_signal" else None
    latest_close = float(latest_row.get("close", 0.0) or 0.0)
    explanation, warnings = explain_signal(strategy_name, frame, signal_type)
    entry, stop, target = _build_plan_levels(strategy_name, frame, config)
    risk = max(entry - stop, 0.0)
    reward = max(target - entry, 0.0)
    ratio = reward / risk if risk > 0 else 0.0
    result = SignalScanResult(
        ticker=ticker,
        strategy=strategy_name,
        signal_type=signal_type,
        signal_date=signal_date,
        latest_close=latest_close,
        suggested_entry_reference=entry,
        suggested_stop=stop,
        suggested_target=target,
        risk_per_share=risk,
        reward_per_share=reward,
        reward_risk_ratio=ratio,
        explanation=explanation,
        notes_warnings=warnings,
        latest_run_id=latest_run_id,
        qualification_id=qualification_id,
        robustness_score=robustness_score,
        qualification_status=qualification_status,
    )
    quality = evaluate_signal_quality(result, quality_inputs or {})
    result.signal_quality_score = quality.score
    result.signal_quality_label = quality.label
    result.signal_quality_bullets = quality.bullets
    return result


def evaluate_signal_quality(signal: SignalScanResult, context: dict[str, object]) -> SignalQualityAssessment:
    """Score a scanner result as a research triage heuristic."""
    score = 50
    bullets = ["Signal quality score is a research heuristic, not a recommendation."]

    robustness_score = int(signal.robustness_score or 0)
    if robustness_score >= 70:
        score += 12
        bullets.append("The underlying strategy has a healthy robustness score.")
    elif robustness_score < 40:
        score -= 15
        bullets.append("The underlying strategy has a weak robustness score.")

    qualification_status = str(signal.qualification_status or "")
    if "strong" in qualification_status.lower():
        score += 12
        bullets.append("This strategy passed a stronger qualification result.")
    elif "possible" in qualification_status.lower():
        score += 6
        bullets.append("This strategy has some qualification support.")
    elif qualification_status:
        score -= 8
        bullets.append("Qualification status is weak or not ready.")

    if signal.signal_type == "new_buy_signal":
        score += 10
        bullets.append("The signal is fresh rather than stale.")
    elif signal.signal_type == "exit_signal":
        score -= 15
        bullets.append("This is an exit signal, not a fresh long setup.")
    elif signal.signal_type == "no_signal":
        score -= 20
        bullets.append("There is no actionable signal on the latest bar.")

    if signal.reward_risk_ratio >= 2:
        score += 10
        bullets.append("The proposed reward/risk ratio is favorable.")
    elif signal.reward_risk_ratio < 1:
        score -= 10
        bullets.append("The proposed reward/risk ratio is weak.")

    if context.get("data_quality_warnings"):
        score -= 8
        bullets.append("Data-quality warnings exist for this symbol.")
    if context.get("corporate_action_warnings"):
        score -= 6
        bullets.append("Corporate-action warnings could distort the signal context.")

    trade_count = int(context.get("trade_count", 0) or 0)
    if trade_count >= 30:
        score += 6
        bullets.append("Trade count for the underlying strategy is adequate.")
    elif trade_count < 10:
        score -= 10
        bullets.append("Trade count for the underlying strategy is thin.")

    if context.get("parameter_stability_poor"):
        score -= 8
        bullets.append("Parameter stability looked fragile in saved sweeps.")
    if context.get("slippage_sensitive"):
        score -= 8
        bullets.append("Saved slippage sensitivity suggests the edge may be fragile.")
    if context.get("bear_regime"):
        score -= 4
        bullets.append("The benchmark regime is currently less favorable.")

    score = max(0, min(100, int(round(score))))
    if score >= 75:
        label = "High quality"
    elif score >= 55:
        label = "Watch"
    elif score >= 35:
        label = "Low quality"
    else:
        label = "Ignore"
    return SignalQualityAssessment(score=score, label=label, bullets=bullets)


def plan_trade_from_signal(
    signal: SignalScanResult,
    *,
    portfolio_value: float,
    sizing_method: str,
    sizing_value: float,
    notes: str = "",
    tags: str = "",
) -> dict[str, object]:
    """Convert a scan result into a manual trade plan."""
    entry = float(signal.suggested_entry_reference)
    stop = float(signal.suggested_stop)
    target = float(signal.suggested_target)
    risk_per_share = max(entry - stop, 0.0)
    if entry <= 0:
        shares = 0
    elif sizing_method == "fixed_dollar_allocation":
        shares = int(max(sizing_value, 0) / entry)
    elif sizing_method == "percent_of_portfolio":
        shares = int(max(portfolio_value * sizing_value, 0) / entry)
    elif sizing_method == "fixed_dollar_risk":
        shares = int(max(sizing_value, 0) / risk_per_share) if risk_per_share > 0 else 0
    else:
        raise ValueError(f"Unsupported sizing method: {sizing_method}")
    capital_required = shares * entry
    max_dollar_risk = shares * risk_per_share
    return {
        "ticker": signal.ticker,
        "strategy": signal.strategy,
        "setup_date": signal.signal_date,
        "planned_entry": entry,
        "stop_loss": stop,
        "take_profit": target,
        "risk_per_share": risk_per_share,
        "position_size": shares,
        "estimated_capital_required": capital_required,
        "max_dollar_risk": max_dollar_risk,
        "reward_risk_ratio": signal.reward_risk_ratio,
        "quality_label": signal.signal_quality_label,
        "quality_score": signal.signal_quality_score,
        "notes": notes,
        "tags": tags,
        "linked_backtest_run_id": signal.latest_run_id,
        "linked_qualification_id": signal.qualification_id,
    }


def summarize_parameter_stability_for_signal(results: pd.DataFrame) -> dict[str, object]:
    if results.empty:
        return {}
    return parameter_stability_summary(results.rename(columns={"cagr": "CAGR", "max_drawdown": "Max Drawdown", "total_return": "Total Return"}))


def infer_train_test_adequacy(data_by_symbol: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Expose a simple train/test split for scanners if a future daily workflow needs it."""
    return split_data_by_percentage(data_by_symbol, 0.7)
