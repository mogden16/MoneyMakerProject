from __future__ import annotations

import runpy
from pathlib import Path

import pandas as pd

from trading_lab.research_runner.cli import build_parser, normalize_end_date
from trading_lab.research_runner.config import ResearchRunnerConfig
from trading_lab.research_runner.dataset import build_signal_dataset
from trading_lab.research_runner.features import build_feature_frame
from trading_lab.research_runner.labeling import LabelingConfig, apply_labels
from trading_lab.research_runner.models import build_practical_approved_breakdown, build_time_series_split, train_time_series_models
from trading_lab.research_runner.reporting import write_recommendation_markdown, write_summary_markdown, write_warnings_markdown
from trading_lab.research_runner.review import build_manual_review_shortlist, recommend_candidate_for_implementation


def make_bars(length: int = 260) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=length, freq="B")
    closes = pd.Series([100 + idx * 0.2 for idx in range(length)], index=index)
    return pd.DataFrame(
        {
            "source_vendor": ["test"] * length,
            "symbol": ["SPY"] * length,
            "timeframe": ["1d"] * length,
            "timestamp": index,
            "session_date": index.date,
            "open": closes.values,
            "high": (closes + 1.0).values,
            "low": (closes - 1.0).values,
            "close": closes.values,
            "adj_close": closes.values,
            "volume": [1000.0] * length,
            "dividends": [0.0] * length,
            "stock_splits": [0.0] * length,
            "adjusted_flag": [True] * length,
            "retrieved_at": [pd.Timestamp("2024-01-01")] * length,
        }
    )


def test_cli_argument_parsing():
    parser = build_parser()
    args = parser.parse_args(["--symbol", "SPY", "--timeframe", "1d", "--start", "2000-01-01", "--end", "today", "--include-models"])
    assert args.symbol == "SPY"
    assert args.include_models is True
    assert normalize_end_date("today") != "today"


def test_feature_generation_has_expected_columns():
    frame = build_feature_frame(make_bars())
    assert {"return_1d", "return_5d", "realized_vol_10d", "rsi_14", "atr_14", "close_above_200_sma"} <= set(frame.columns)


def test_no_feature_leakage_from_future_data():
    bars = make_bars()
    original = build_feature_frame(bars).iloc[220].copy()
    modified = bars.copy()
    modified.loc[221:, "close"] = modified.loc[221:, "close"] * 5
    updated = build_feature_frame(modified).iloc[220].copy()
    assert original["return_20d"] == updated["return_20d"]
    assert original["rsi_14"] == updated["rsi_14"]


def test_label_generation_uses_future_bars_only():
    bars = make_bars(40)
    signal_row = pd.Series(
        {
            "signal_price": float(bars.iloc[20]["close"]),
            "atr_pct": 0.02,
            "exit_parameters_json": {"stop_loss_pct": 0.03},
        }
    )
    labels = apply_labels(signal_row, bars, 20, LabelingConfig())
    assert {"forward_return_10d", "target_before_stop_10d", "label_good_signal"} <= set(labels.keys())


def test_signal_dataset_construction_and_no_data_edge_case():
    bars = make_bars()
    signal_frame = bars.copy()
    signal_frame["entry_signal"] = False
    signal_frame.loc[[210, 220, 230], "entry_signal"] = True
    signal_frame["exit_signal"] = False
    result = build_signal_dataset(
        bars=bars,
        signal_frame=signal_frame,
        strategy_name="Test",
        entry_parameters={"alpha": 1},
        exit_structure_name="Signal exit only",
        exit_parameters={},
        timeframe="1d",
        labeling_config=LabelingConfig(),
    )
    assert not result.frame.empty
    empty = build_signal_dataset(
        bars=bars,
        signal_frame=bars.assign(entry_signal=False, exit_signal=False),
        strategy_name="Test",
        entry_parameters={},
        exit_structure_name="Signal exit only",
        exit_parameters={},
        timeframe="1d",
        labeling_config=LabelingConfig(),
    )
    assert empty.frame.empty


def test_time_series_split_usage():
    split = build_time_series_split(150)
    assert split.__class__.__name__ == "TimeSeriesSplit"
    assert split.n_splits >= 2


def test_model_training_runs_on_small_synthetic_data_and_too_few_signals_warning():
    bars = make_bars(220)
    signal_frame = build_feature_frame(bars)
    dataset = signal_frame.iloc[180:220].copy()
    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"])
    dataset["entry_strategy_name"] = ["trend"] * len(dataset)
    dataset["exit_structure_name"] = ["signal"] * len(dataset)
    dataset["timeframe"] = ["1d"] * len(dataset)
    dataset["label_good_signal"] = ([0, 1] * 20)[: len(dataset)]
    dataset["forward_return_10d"] = [0.02 if label else -0.01 for label in dataset["label_good_signal"]]
    dataset["atr_pct"] = 0.02
    model_result = train_time_series_models(dataset)
    assert model_result.summary.empty
    assert model_result.warnings

    larger = pd.concat([dataset] * 3, ignore_index=True)
    larger["timestamp"] = pd.date_range("2021-01-01", periods=len(larger), freq="B")
    larger_result = train_time_series_models(larger)
    assert not larger_result.summary.empty
    assert not larger_result.folds.empty
    assert not larger_result.comparison.empty
    assert not larger_result.approved_breakdown.empty
    assert not larger_result.approved_breakdown_raw.empty
    assert {"forward_return_edge_10d", "win_rate_edge", "approval_rate"} <= set(larger_result.comparison.columns)


def test_report_file_creation(tmp_path: Path):
    output = tmp_path / "summary.md"
    write_summary_markdown(
        output_path=output,
        config={"symbol": "SPY"},
        candidates=pd.DataFrame([{"entry_preset_label": "Trend", "exit_preset_label": "Signal", "cagr": 0.1, "excess_cagr": 0.02, "max_drawdown": -0.1, "number_of_trades": 30, "candidate_label": "Strong candidate"}]),
        rejected=pd.DataFrame(),
        top_candidates=pd.DataFrame([{"entry_preset_label": "Trend", "exit_preset_label": "Signal", "cagr": 0.1, "excess_cagr": 0.02, "max_drawdown": -0.1, "number_of_trades": 30, "candidate_label": "Strong candidate"}]),
        highlights={"Best Overall": {"entry_preset_label": "Trend", "exit_preset_label": "Signal", "summary_comment": "Good"}} ,
        signal_dataset=pd.DataFrame([{"timestamp": pd.Timestamp("2024-01-01"), "label_good_signal": 1}]),
        dataset_warnings=[],
        model_summary=pd.DataFrame(),
        model_comparison=pd.DataFrame(),
        approved_breakdown=pd.DataFrame(),
        manual_review_shortlist=pd.DataFrame(),
        recommendation={},
        model_warnings=[],
        run_warnings=[],
    )
    assert output.exists()
    assert "SPY Research Runner Summary" in output.read_text(encoding="utf-8")


def test_manual_review_shortlist_builds_from_positive_model_edge():
    candidates = pd.DataFrame(
        [
            {
                "timeframe": "1d",
                "entry_strategy_name": "RSI Mean Reversion",
                "entry_parameters_json": "{\"a\":1}",
                "exit_structure_name": "OCO bracket",
                "exit_parameters_json": "{\"b\":2}",
                "candidate_label": "Possible candidate",
                "profit_factor": 1.3,
                "calmar": 0.4,
                "red_flag_count": 1,
                "complexity_score": 2,
                "experimental": False,
            }
        ]
    )
    comparison = pd.DataFrame(
        [
            {
                "model_name": "random_forest",
                "forward_return_edge_10d": 0.002,
                "win_rate_edge": 0.01,
                "approval_rate": 0.4,
            }
        ]
    )
    approved_breakdown = pd.DataFrame(
        [
            {
                "model_name": "random_forest",
                "entry_strategy_name": "RSI Mean Reversion",
                "entry_parameters_json": "{\"a\":1}",
                "exit_structure_name": "OCO bracket",
                "exit_parameters_json": "{\"b\":2}",
                "timeframe": "1d",
                "approved_signal_count": 50,
                "avg_forward_return_10d": 0.01,
                "win_rate": 0.62,
                "positive_label_rate": 0.58,
                "avg_r_multiple": 0.7,
            }
        ]
    )
    shortlist = build_manual_review_shortlist(
        candidates=candidates,
        model_comparison=comparison,
        approved_breakdown=approved_breakdown,
    )
    assert not shortlist.empty
    assert shortlist.iloc[0]["model_name"] == "random_forest"
    assert "manual_review_comment" in shortlist.columns


def test_recommend_candidate_for_implementation_prefers_model_backed_shortlist():
    shortlist = pd.DataFrame(
        [
            {
                "timeframe": "15m",
                "entry_strategy_name": "Opening Range Breakout",
                "strategy_archetype": "opening-range breakout with volume-pressure confirmation",
                "exit_structure_name": "OCO bracket",
                "exit_archetype": "OCO bracket",
                "candidate_label": "Possible candidate",
                "approved_signal_count": 42,
                "avg_forward_return_10d": 0.012,
                "model_name": "random_forest",
                "manual_review_comment": "model improved forward return",
            }
        ]
    )
    recommendation = recommend_candidate_for_implementation(
        candidates=pd.DataFrame(),
        model_comparison=pd.DataFrame(),
        manual_review_shortlist=shortlist,
    )
    assert recommendation["source"] == "model_backed_shortlist"
    assert recommendation["entry_strategy_name"] == "Opening Range Breakout"
    assert "survived both the candidate screen" in recommendation["recommendation_reason"]


def test_practical_approved_breakdown_filters_tiny_groups():
    raw = pd.DataFrame(
        [
            {"model_name": "rf", "approved_signal_count": 1, "avg_forward_return_10d": 0.05, "win_rate": 1.0},
            {"model_name": "rf", "approved_signal_count": 12, "avg_forward_return_10d": 0.01, "win_rate": 0.6},
            {"model_name": "rf", "approved_signal_count": 25, "avg_forward_return_10d": 0.008, "win_rate": 0.64},
        ]
    )
    practical = build_practical_approved_breakdown(raw, min_signals=10)
    assert len(practical) == 2
    assert practical.iloc[0]["approved_signal_count"] == 25


def test_warnings_file_creation(tmp_path: Path):
    output = tmp_path / "warnings.md"
    write_warnings_markdown(output, ["warning one", "warning two"])
    text = output.read_text(encoding="utf-8")
    assert output.exists()
    assert "Research Runner Warnings" in text
    assert "warning one" in text


def test_recommendation_file_creation(tmp_path: Path):
    output = tmp_path / "recommendation.md"
    write_recommendation_markdown(
        output,
        {
            "source": "model_backed_shortlist",
            "timeframe": "15m",
            "entry_strategy_name": "Opening Range Breakout",
            "strategy_archetype": "opening-range breakout",
            "exit_structure_name": "OCO bracket",
            "exit_archetype": "OCO bracket",
            "candidate_label": "Possible candidate",
            "model_name": "random_forest",
            "recommendation_reason": "Start with opening-range breakout.",
        },
    )
    text = output.read_text(encoding="utf-8")
    assert output.exists()
    assert "Research Recommendation" in text
    assert "Opening Range Breakout" in text


def test_config_dataclass_round_trip():
    config = ResearchRunnerConfig(
        symbol="SPY",
        timeframe="1d",
        start="2000-01-01",
        end="2026-05-23",
        include_models=True,
        max_combinations=5,
        output_dir="reports/research_runs",
        min_trades=20,
        label_horizon_days=10,
        target_r_multiple=1.5,
        stop_r_multiple=1.0,
    )
    assert config.symbol == "SPY"
    assert config.max_combinations == 5


def test_script_files_exist_and_parse():
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "scripts" / "run_spy_research.py").exists()
    assert (repo_root / "scripts" / "build_signal_dataset.py").exists()
    assert (repo_root / "scripts" / "train_signal_filter.py").exists()
    assert (repo_root / "scripts" / "nightly_research_suite.py").exists()


def test_runner_module_exports():
    module_globals = runpy.run_path(str(Path(__file__).resolve().parents[1] / "trading_lab" / "research_runner" / "__init__.py"))
    assert "ResearchRunnerConfig" in module_globals
    assert "run_research_pipeline" in module_globals
