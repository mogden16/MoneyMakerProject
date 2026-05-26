from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from trading_lab.backtest.engine import BacktestEngine
from trading_lab.backtest.metrics import calculate_sharpe_ratio
from trading_lab.backtest.robustness import compute_robustness_score, profit_concentration_analysis
from trading_lab.data.database import TradingLabDatabase
from trading_lab.data.providers.yfinance_provider import YFinanceDataProvider
from trading_lab.research_runner.config import ResearchRunnerConfig
from trading_lab.research_runner.dataset import SignalDatasetResult, build_signal_dataset
from trading_lab.research_runner.labeling import LabelingConfig
from trading_lab.research_runner.models import ModelRunResult, train_time_series_models
from trading_lab.research_runner.reporting import write_csv, write_recommendation_markdown, write_summary_markdown, write_warnings_markdown
from trading_lab.research_runner.review import build_manual_review_shortlist, recommend_candidate_for_implementation
from trading_lab.spy_lab import (
    average_r_multiple,
    apply_spy_exit_structure,
    build_spy_backtest_config,
    build_spy_search_summary_comment,
    build_spy_strategy,
    build_spy_workbench_config,
    describe_spy_exit_archetype,
    describe_spy_search_archetype,
    generate_approved_spy_entry_presets,
    generate_approved_spy_exit_presets,
    grade_spy_search_candidate,
    prepare_spy_timeframe_bars,
    rank_spy_search_results,
    spy_strategy_summary,
)


@dataclass(frozen=True)
class ResearchRunnerResult:
    output_path: Path
    candidates: pd.DataFrame
    rejected_candidates: pd.DataFrame
    top_candidates: pd.DataFrame
    signal_dataset: pd.DataFrame
    model_summary: pd.DataFrame
    model_folds: pd.DataFrame
    model_comparison: pd.DataFrame
    approved_breakdown: pd.DataFrame
    approved_breakdown_raw: pd.DataFrame
    manual_review_shortlist: pd.DataFrame
    recommendation: dict[str, Any]
    warnings: list[str]


def run_research_pipeline(config: ResearchRunnerConfig) -> ResearchRunnerResult:
    timestamp_slug = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = Path(config.output_dir) / timestamp_slug
    output_path.mkdir(parents=True, exist_ok=True)

    db = TradingLabDatabase()
    provider = YFinanceDataProvider(database=db)
    bars = provider.get_stock_bars(config.symbol, config.start, config.end, timeframe=config.timeframe, force_refresh=False)
    daily_context_start = config.start
    if config.timeframe != "1d":
        daily_context_start = str((pd.Timestamp(config.start) - pd.Timedelta(days=450)).date())
    daily_bars = provider.get_stock_bars(config.symbol, daily_context_start, config.end, timeframe="1d", force_refresh=False) if config.timeframe != "1d" else bars

    run_warnings: list[str] = []
    last_status = provider.get_last_fetch_status(config.symbol)
    if last_status is not None:
        run_warnings.extend(last_status.validation_warnings)
    if bars.empty:
        raise ValueError("No SPY bars were returned for the requested range.")

    entry_presets = generate_approved_spy_entry_presets(config.timeframe)
    exit_presets = generate_approved_spy_exit_presets()
    if config.max_combinations is not None:
        total_limit = max(config.max_combinations, 0)
    else:
        total_limit = len(entry_presets) * len(exit_presets)

    engine = BacktestEngine(database=None)
    candidate_rows: list[dict[str, Any]] = []
    signal_frames: list[pd.DataFrame] = []
    combinations_tested = 0
    labeling_config = LabelingConfig(
        label_horizon_days=config.label_horizon_days,
        target_r_multiple=config.target_r_multiple,
        stop_r_multiple=config.stop_r_multiple,
    )

    for entry_preset in entry_presets:
        for exit_preset in exit_presets:
            if combinations_tested >= total_limit:
                break
            combinations_tested += 1
            workbench = build_spy_workbench_config(
                preset_key=entry_preset.preset_key,
                entry_parameters=entry_preset.parameters,
                timeframe=config.timeframe,
                exit_structure_key=exit_preset.exit_structure_key,
                exit_parameters=exit_preset.parameters,
                start_date=config.start,
                end_date=config.end,
                price_mode="adjusted_price_mode",
                initial_capital=100000.0,
                position_sizing_method="percent_of_portfolio",
                position_size_value=1.0,
                max_positions=1,
                slippage_pct=0.0005,
                commission_per_trade=1.0,
            )
            strategy = apply_spy_exit_structure(build_spy_strategy(entry_preset.preset_key, entry_preset.parameters), workbench)
            config_obj = build_spy_backtest_config(workbench)
            prepared_bars = prepare_spy_timeframe_bars(primary_bars=bars, timeframe=config.timeframe, daily_bars=daily_bars)
            result = engine.run(data_by_symbol={config.symbol: prepared_bars}, strategy=strategy, config=config_obj, benchmark_symbol=config.symbol)
            concentration = profit_concentration_analysis(result.trade_log)
            robustness = compute_robustness_score(result.metrics, concentration=concentration)
            benchmark_sharpe = 0.0
            if not result.benchmark_curve.empty:
                benchmark_sharpe = calculate_sharpe_ratio(result.benchmark_curve.rename(columns={"benchmark_equity": "equity"}))
            summary = spy_strategy_summary(result.metrics, benchmark_sharpe=benchmark_sharpe)
            row = {
                "timeframe": config.timeframe,
                "entry_strategy_name": entry_preset.entry_strategy_name,
                "entry_preset_label": entry_preset.label,
                "entry_parameters_json": entry_preset.parameters,
                "strategy_archetype": describe_spy_search_archetype(
                    {
                        "entry_strategy_name": entry_preset.entry_strategy_name,
                        "entry_preset_label": entry_preset.label,
                    }
                ),
                "exit_structure_name": exit_preset.exit_structure_name,
                "exit_preset_label": exit_preset.label,
                "exit_parameters_json": exit_preset.parameters,
                "exit_archetype": describe_spy_exit_archetype(
                    {
                        "exit_structure_name": exit_preset.exit_structure_name,
                        "exit_preset_label": exit_preset.label,
                    }
                ),
                "total_return": float(result.metrics.get("Total Return", 0.0) or 0.0),
                "cagr": float(result.metrics.get("CAGR", 0.0) or 0.0),
                "spy_cagr": float(summary["Buy-and-Hold SPY CAGR"]),
                "excess_cagr": float(result.metrics.get("Excess CAGR", 0.0) or 0.0),
                "max_drawdown": float(result.metrics.get("Max Drawdown", 0.0) or 0.0),
                "spy_max_drawdown": float(result.metrics.get("Benchmark Max Drawdown", 0.0) or 0.0),
                "drawdown_improvement": float(summary["Drawdown Improvement vs SPY"]),
                "sharpe": float(result.metrics.get("Sharpe Ratio", 0.0) or 0.0),
                "sortino": float(result.metrics.get("Sortino Ratio", 0.0) or 0.0),
                "calmar": float(result.metrics.get("Calmar Ratio", 0.0) or 0.0),
                "number_of_trades": int(result.metrics.get("Number of Trades", 0) or 0),
                "win_rate": float(result.metrics.get("Win Rate", 0.0) or 0.0),
                "profit_factor": float(result.metrics.get("Profit Factor", 0.0) or 0.0),
                "avg_trade_return": float(result.metrics.get("Average Trade Return", 0.0) or 0.0),
                "avg_r_multiple": float(average_r_multiple(result.trade_log, exit_preset.parameters)),
                "exposure_pct": float(result.metrics.get("Exposure %", 0.0) or 0.0),
                "robustness_score": int(robustness.score),
                "experimental": bool(entry_preset.experimental),
                "complexity_score": int(entry_preset.complexity_score),
            }
            candidate_label, red_flag_count = grade_spy_search_candidate(row)
            row["candidate_label"] = candidate_label
            row["red_flag_count"] = red_flag_count
            row["summary_comment"] = build_spy_search_summary_comment(row)
            candidate_rows.append(row)
            signal_result: SignalDatasetResult = build_signal_dataset(
                bars=prepared_bars,
                signal_frame=strategy.generate_signals(prepared_bars.copy()),
                strategy_name=entry_preset.entry_strategy_name,
                entry_parameters=entry_preset.parameters,
                exit_structure_name=exit_preset.exit_structure_name,
                exit_parameters=exit_preset.parameters,
                timeframe=config.timeframe,
                labeling_config=labeling_config,
            )
            if signal_result.warnings:
                run_warnings.extend(signal_result.warnings)
            if not signal_result.frame.empty:
                signal_frames.append(signal_result.frame)
        if combinations_tested >= total_limit:
            break

    all_candidates = pd.DataFrame(candidate_rows)
    if all_candidates.empty:
        raise ValueError("No SPY strategy combinations were completed.")
    top_candidates = all_candidates[
        (all_candidates["candidate_label"].isin(["Strong candidate", "Possible candidate"]))
        & (all_candidates["number_of_trades"] >= config.min_trades)
    ].sort_values(["candidate_label", "calmar", "excess_cagr"], ascending=[True, False, False])
    rejected_candidates = all_candidates[~all_candidates.index.isin(top_candidates.index)].copy()
    highlights = rank_spy_search_results(all_candidates)

    signal_dataset = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    if not signal_dataset.empty:
        signal_dataset = signal_dataset.sort_values("timestamp").reset_index(drop=True)
    if len(signal_dataset) < 100:
        run_warnings.append("The signal dataset is small. Treat any model result cautiously.")
    if config.timeframe != "1d":
        run_warnings.append("Intraday yfinance history is short. Treat this run as forward-validation infrastructure, not deep historical proof.")

    model_result = ModelRunResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [])
    if config.include_models:
        model_result = train_time_series_models(signal_dataset)

    manual_review_shortlist = build_manual_review_shortlist(
        candidates=all_candidates,
        model_comparison=model_result.comparison,
        approved_breakdown=model_result.approved_breakdown,
    )
    recommendation = recommend_candidate_for_implementation(
        candidates=all_candidates,
        model_comparison=model_result.comparison,
        manual_review_shortlist=manual_review_shortlist,
    )

    _write_artifacts(
        output_path=output_path,
        candidates=all_candidates,
        rejected_candidates=rejected_candidates,
        top_candidates=top_candidates,
        signal_dataset=signal_dataset,
        model_result=model_result,
        manual_review_shortlist=manual_review_shortlist,
        recommendation=recommendation,
        config=config,
        highlights=highlights,
        run_warnings=run_warnings,
    )
    return ResearchRunnerResult(
        output_path=output_path,
        candidates=all_candidates,
        rejected_candidates=rejected_candidates,
        top_candidates=top_candidates,
        signal_dataset=signal_dataset,
        model_summary=model_result.summary,
        model_folds=model_result.folds,
        model_comparison=model_result.comparison,
        approved_breakdown=model_result.approved_breakdown,
        approved_breakdown_raw=model_result.approved_breakdown_raw,
        manual_review_shortlist=manual_review_shortlist,
        recommendation=recommendation,
        warnings=[*run_warnings, *model_result.warnings],
    )


def _write_artifacts(
    *,
    output_path: Path,
    candidates: pd.DataFrame,
    rejected_candidates: pd.DataFrame,
    top_candidates: pd.DataFrame,
    signal_dataset: pd.DataFrame,
    model_result: ModelRunResult,
    manual_review_shortlist: pd.DataFrame,
    recommendation: dict[str, Any],
    config: ResearchRunnerConfig,
    highlights: dict[str, dict[str, Any]],
    run_warnings: list[str],
) -> None:
    combined_warnings = [*run_warnings, *model_result.warnings]
    write_csv(candidates, output_path / "candidates.csv")
    write_csv(rejected_candidates, output_path / "rejected_candidates.csv")
    write_csv(top_candidates, output_path / "top_candidates.csv")
    write_csv(signal_dataset, output_path / "signal_dataset.csv")
    write_csv(model_result.summary, output_path / "model_results.csv")
    write_csv(model_result.folds, output_path / "model_fold_results.csv")
    write_csv(model_result.feature_importance, output_path / "feature_importance.csv")
    write_csv(model_result.approved_signals, output_path / "model_approved_signals.csv")
    write_csv(model_result.comparison, output_path / "model_comparison.csv")
    write_csv(model_result.approved_breakdown, output_path / "approved_signal_breakdown.csv")
    write_csv(model_result.approved_breakdown_raw, output_path / "approved_signal_breakdown_raw.csv")
    write_csv(manual_review_shortlist, output_path / "manual_review_shortlist.csv")
    write_recommendation_markdown(output_path / "recommendation.md", recommendation)
    write_summary_markdown(
        output_path=output_path / "summary.md",
        config=asdict(config),
        candidates=candidates,
        rejected=rejected_candidates,
        top_candidates=top_candidates,
        highlights=highlights,
        signal_dataset=signal_dataset,
        dataset_warnings=[],
        model_summary=model_result.summary,
        model_comparison=model_result.comparison,
        approved_breakdown=model_result.approved_breakdown,
        manual_review_shortlist=manual_review_shortlist,
        recommendation=recommendation,
        model_warnings=model_result.warnings,
        run_warnings=run_warnings,
    )
    write_warnings_markdown(output_path / "warnings.md", combined_warnings)
