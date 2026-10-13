"""5-minute bars from Alpaca's full-market (SIP) feed, regular trading hours only."""
from pathlib import Path

import numpy as np
import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.requests import GetCalendarRequest

from broker import data_client, trading_client

SYMBOLS = ["SPY", "QQQ"]
TZ = "America/New_York"
SLOTS = 78  # 5-minute bars from 9:30 to 16:00
CACHE = Path(__file__).parent / "data" / "bars_5min.csv.gz"


def load_bars(refresh=False, start="2016-01-01", symbols=None, name=None):
    """Return 5-minute bars (symbol, timestamp in New York time, open/high/low/close/volume).
    Downloads complete days only (through yesterday) and caches them. By default this is
    SPY and QQQ; pass `symbols` and a cache `name` for others."""
    symbols = symbols or SYMBOLS
    cache = CACHE if name is None else CACHE.with_name(f"bars_5min_{name}.csv.gz")
    if refresh or not cache.exists():
        today = pd.Timestamp.now(tz=TZ).normalize()
        req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                               start=pd.Timestamp(start, tz=TZ).to_pydatetime(),
                               end=today.to_pydatetime(), feed=DataFeed.SIP,
                               adjustment=Adjustment.SPLIT)
        df = data_client().get_stock_bars(req).df.reset_index()
        df["timestamp"] = df["timestamp"].dt.tz_convert(TZ)
        minutes = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
        df = df[(minutes >= 9 * 60 + 30) & (minutes < 16 * 60)]
        cache.parent.mkdir(exist_ok=True)
        df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]].to_csv(cache, index=False)
    df = pd.read_csv(cache)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(TZ)
    return session_only(df, refresh=refresh)


def session_only(df, refresh=False):
    """Keep bars inside official trading hours. Early-close days (e.g. the day after
    Thanksgiving) end at 13:00, and later bars are thin after-hours trading."""
    cal = market_calendar(refresh=refresh)
    local = df["timestamp"].dt.tz_localize(None)
    close = local.dt.normalize().map(cal)
    minutes = local.dt.hour * 60 + local.dt.minute
    return df[(minutes >= 9 * 60 + 30) & (local < close)].reset_index(drop=True)


CAL_CACHE = Path(__file__).parent / "data" / "calendar.csv"


def market_calendar(refresh=False, start="2016-01-01"):
    """Official closing time (New York, naive) for every trading day, from Alpaca, two
    weeks ahead. Re-downloads whenever the cache doesn't reach today."""
    today = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
    stale = not CAL_CACHE.exists() or pd.read_csv(CAL_CACHE, parse_dates=["date"])["date"].max() < today
    if refresh or stale:
        end = (today + pd.Timedelta(days=14)).date()
        days = trading_client().get_calendar(GetCalendarRequest(start=pd.Timestamp(start).date(), end=end))
        CAL_CACHE.parent.mkdir(exist_ok=True)
        pd.DataFrame({"date": [d.date for d in days], "close": [d.close for d in days]}).to_csv(CAL_CACHE, index=False)
    cal = pd.read_csv(CAL_CACHE, parse_dates=["date", "close"])
    return pd.Series(cal["close"].to_numpy(), index=cal["date"])


def panels(bars):
    """Per symbol, a dict of (days x 78 slots) DataFrames: open, close, volume.
    Slot 0 is the 9:30 bar; half-days leave the later slots empty (NaN)."""
    ts = bars["timestamp"]
    bars = bars.assign(date=ts.dt.date,
                       slot=((ts.dt.hour * 60 + ts.dt.minute) - (9 * 60 + 30)) // 5)
    out = {}
    for sym, g in bars.groupby("symbol"):
        out[sym] = {f: g.pivot(index="date", columns="slot", values=f)
                    .reindex(columns=range(SLOTS)) for f in ("open", "high", "low", "close", "volume")}
        for f in out[sym]:
            out[sym][f].index = pd.DatetimeIndex(out[sym][f].index)
    return out


def missing_report(pan):
    """Count missing bars inside each trading day (after the day's first bar, before its last)."""
    for sym, p in pan.items():
        c = p["close"].to_numpy()
        valid = ~np.isnan(c)
        last = valid.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)
        inside = np.arange(c.shape[1]) <= last[:, None]
        print(f"{sym}: {len(c)} days, {int((inside & ~valid).sum())} missing bars inside days, "
              f"{int((last < SLOTS - 1).sum())} short days (half-days)")
