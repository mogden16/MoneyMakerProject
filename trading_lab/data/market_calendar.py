from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

try:
    import pandas_market_calendars as mcal
except Exception:  # pragma: no cover - fallback exercised in runtime if import fails
    mcal = None


EASTERN_TZ = ZoneInfo("America/New_York")


@dataclass
class CalendarStatus:
    calendar_name: str
    using_fallback: bool


class MarketCalendar:
    """NYSE calendar helper with weekday fallback when calendar loading fails."""

    def __init__(self, calendar_name: str = "NYSE") -> None:
        self.calendar_name = calendar_name
        self._calendar = None
        self.status = CalendarStatus(calendar_name=calendar_name, using_fallback=True)
        if mcal is not None:
            try:
                self._calendar = mcal.get_calendar(calendar_name)
                self.status = CalendarStatus(calendar_name=calendar_name, using_fallback=False)
            except Exception:
                self._calendar = None
        self._fallback_business_day = CustomBusinessDay(calendar=USFederalHolidayCalendar())

    def expected_sessions(self, start_date: str | date, end_date: str | date) -> list[date]:
        start = pd.Timestamp(start_date).date()
        end = pd.Timestamp(end_date).date()
        if start > end:
            return []
        if self._calendar is None:
            return [ts.date() for ts in pd.date_range(start, end, freq=self._fallback_business_day)]
        schedule = self._calendar.schedule(start_date=start, end_date=end)
        return [ts.date() for ts in schedule.index]

    def latest_completed_session(self, requested_end: str | date, now: datetime | None = None) -> date:
        """Return the latest completed trading session on or before requested_end."""
        now_est = now.astimezone(EASTERN_TZ) if now is not None else datetime.now(EASTERN_TZ)
        capped_end = min(pd.Timestamp(requested_end).date(), now_est.date())
        start_window = (pd.Timestamp(capped_end) - pd.Timedelta(days=14)).date()
        if self._calendar is None:
            candidate = capped_end
            if candidate.weekday() >= 5:
                candidate = (pd.Timestamp(candidate) - self._fallback_business_day).date()
            elif candidate == now_est.date() and now_est.hour < 16:
                candidate = (pd.Timestamp(candidate) - self._fallback_business_day).date()
            while candidate not in self.expected_sessions(start_window, capped_end):
                candidate = (pd.Timestamp(candidate) - self._fallback_business_day).date()
            return candidate

        schedule = self._calendar.schedule(start_date=start_window, end_date=capped_end)
        if schedule.empty:
            return capped_end
        latest = schedule.iloc[-1]
        latest_date = schedule.index[-1].date()
        latest_close = latest["market_close"].tz_convert(EASTERN_TZ)
        if latest_date == now_est.date() and now_est < latest_close:
            if len(schedule) >= 2:
                return schedule.index[-2].date()
        return latest_date

    def session_bounds(self, session_date: str | date) -> tuple[datetime, datetime] | None:
        session = pd.Timestamp(session_date).date()
        if self._calendar is None:
            open_ts = datetime.combine(session, datetime.min.time(), tzinfo=EASTERN_TZ).replace(hour=9, minute=30)
            close_ts = datetime.combine(session, datetime.min.time(), tzinfo=EASTERN_TZ).replace(hour=16, minute=0)
            return open_ts, close_ts
        schedule = self._calendar.schedule(start_date=session, end_date=session)
        if schedule.empty:
            return None
        row = schedule.iloc[0]
        return row["market_open"].tz_convert(EASTERN_TZ).to_pydatetime(), row["market_close"].tz_convert(EASTERN_TZ).to_pydatetime()

    def latest_completed_bar(self, requested_end: str | date, timeframe: str, now: datetime | None = None) -> pd.Timestamp:
        now_est = now.astimezone(EASTERN_TZ) if now is not None else datetime.now(EASTERN_TZ)
        latest_session = self.latest_completed_session(requested_end, now=now_est)
        bounds = self.session_bounds(latest_session)
        if bounds is None:
            return pd.Timestamp(latest_session)
        open_ts, close_ts = bounds
        current = min(now_est, close_ts) if latest_session == now_est.date() else close_ts
        interval_minutes = 15 if timeframe == "15m" else 5
        if current <= open_ts:
            previous_session = self.expected_sessions((pd.Timestamp(latest_session) - pd.Timedelta(days=7)).date(), latest_session)
            if len(previous_session) >= 2:
                latest_session = previous_session[-2]
                bounds = self.session_bounds(latest_session)
                if bounds is None:
                    return pd.Timestamp(latest_session)
                open_ts, close_ts = bounds
                current = close_ts
        elapsed_minutes = max(int((current - open_ts).total_seconds() // 60), 0)
        completed_blocks = elapsed_minutes // interval_minutes
        if completed_blocks <= 0:
            return pd.Timestamp(open_ts.replace(tzinfo=None))
        bar_ts = open_ts + pd.Timedelta(minutes=(completed_blocks - 1) * interval_minutes)
        return pd.Timestamp(bar_ts.replace(tzinfo=None))

    def missing_sessions(self, observed_sessions: list[date], start_date: str | date, end_date: str | date) -> list[date]:
        expected = set(self.expected_sessions(start_date, end_date))
        observed = set(pd.to_datetime(observed_sessions).date)
        return sorted(expected - observed)


def get_default_calendar() -> MarketCalendar:
    return MarketCalendar(calendar_name="NYSE")
