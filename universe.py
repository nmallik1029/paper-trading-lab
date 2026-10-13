"""Daily OHLC data for the wider v2 universe: leveraged and inverse ETFs, volatility
products, Treasuries, gold and crypto. Cached per named universe.

Leveraged funds only exist since 2009-2011, so `synthetic_leveraged` rebuilds their daily
returns from the underlying index for longer backtests (leverage x the index's daily
return, minus the extra borrowing cost and fees). Where the real fund exists, the real
prices are used.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DATA = Path(__file__).parent / "data"

DAILY = ["SPY", "QQQ", "IWM", "TQQQ", "SQQQ", "UPRO", "SPXU", "TLT", "TMF", "IEF", "GLD",
         "SVXY", "UVXY", "^VIX", "^VIX3M", "^VIX9D", "^IRX", "BTC-USD", "ETH-USD"]


def load_daily(tickers=DAILY, name="v2", refresh=False, start="1999-01-01", keep_today=False):
    """Return {"open", "high", "low", "close", "volume"} -> DataFrame (dates x tickers),
    dividend- and split-adjusted. Crypto trades every day; everything is kept on its own
    calendar and aligned by the caller.

    keep_today: keep today's unfinished bar, so the latest price stands in for today's
    close. The live bot uses this at 15:40; backtests never do."""
    path = DATA / f"daily_{name}{'_live' if keep_today else ''}.csv"
    if refresh or not path.exists():
        try:
            raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
            if raw.empty:
                raise RuntimeError("Yahoo Finance returned no data")
            now = pd.Timestamp.now(tz="America/New_York")
            if not keep_today and now.hour * 60 + now.minute < 16 * 60 + 15:  # drop today's unfinished bar
                raw = raw[raw.index < now.normalize().tz_localize(None)]
            DATA.mkdir(exist_ok=True)
            raw.stack(level=1, future_stack=True).rename_axis(["date", "ticker"]).to_csv(path)
        except Exception as e:  # Yahoo sometimes rate-limits cloud servers
            if not path.exists():
                raise
            print(f"warning: price download failed ({e}); using cached prices")
    df = pd.read_csv(path, parse_dates=["date"]).set_index(["date", "ticker"])
    return {f.lower(): df[f].unstack("ticker") for f in ("Open", "High", "Low", "Close", "Volume")}


def synthetic_leveraged(index_close, leverage, rf_daily, fee=0.0095):
    """Daily returns of a fund that delivers `leverage` x the index's daily return,
    paying the borrowing cost on the leveraged part and a yearly fee."""
    r = index_close / index_close.shift(1) - 1
    return leverage * r - (leverage - 1) * rf_daily - fee / 252


def spliced(real_close, index_close, leverage, rf_daily, fee=0.0095):
    """Daily returns: the real fund where it traded, the synthetic version before that."""
    synth = synthetic_leveraged(index_close, leverage, rf_daily, fee)
    real = real_close / real_close.shift(1) - 1
    return real.where(real.notna(), synth)
