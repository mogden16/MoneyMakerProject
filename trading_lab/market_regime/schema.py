from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DataSourceInfo:
    label: str
    detail: str
    is_demo: bool = False


@dataclass(frozen=True)
class ReportSection:
    title: str
    rows: list[dict[str, Any]]
    source: DataSourceInfo
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScoreComponent:
    name: str
    score: int
    active: bool
    rationale: str


@dataclass(frozen=True)
class RegimeSummary:
    regime_label: str
    trend_direction: str
    recommended_posture: str
    short_summary: str
    total_score: int
    components: list[ScoreComponent]


@dataclass(frozen=True)
class MarketRegimeReport:
    generated_at: str
    summary: RegimeSummary
    analyst_summary: str
    index_momentum: ReportSection
    breadth: ReportSection
    options_positioning: ReportSection
    sector_leadership: ReportSection
    seasonality: ReportSection
    macro_calendar: ReportSection
    earnings_watch: ReportSection
    warnings: list[str] = field(default_factory=list)
