"""Strategies. Each returns the target weights it wants on its signal dates.

Rows are decision dates (using closes up to and including that date), columns are
tickers, values are fractions of the portfolio. Whatever a row leaves unallocated
(1 - row sum) sits in T-bills. Execution timing and costs live in backtest.py, so a
strategy only has to decide *what* it wants to hold.
"""
import pandas as pd

from data import BENCHMARK, SECTORS


def month_ends(prices):
    """Last trading day of each month, dropping the current month if it isn't over yet."""
    dates = prices.index.to_series().groupby(prices.index.to_period("M")).last()
    dates = pd.DatetimeIndex(dates.to_numpy())
    if (dates[-1] + pd.offsets.BDay(1)).month == dates[-1].month:
        dates = dates[:-1]
    return dates


def buy_and_hold(prices, start, ticker=BENCHMARK):
    """Buy once at `start` and never trade again."""
    return pd.DataFrame({ticker: [1.0]}, index=[pd.Timestamp(start)])


def equal_weight(prices, tickers=SECTORS):
    """Hold every sector equally, resetting to equal weights at each month-end."""
    return pd.DataFrame(1.0 / len(tickers), index=month_ends(prices), columns=tickers)


def sector_momentum(prices, lookback=12, skip=1, top_n=3, tickers=SECTORS):
    """Each month-end, hold the `top_n` sectors with the best return from `lookback`
    months ago to `skip` months ago, equally weighted. Skipping the latest month is the
    standard construction: very recent winners tend to give some of it back."""
    monthly = prices.loc[month_ends(prices), tickers]
    score = (monthly.shift(skip) / monthly.shift(lookback) - 1).dropna()
    rank = score.rank(axis=1, ascending=False, method="first")
    return (rank <= top_n).astype(float) / top_n


def trend_following(prices, window=200, ticker=BENCHMARK):
    """At each month-end, hold `ticker` if it closed above its `window`-day average,
    otherwise sit in T-bills."""
    px = prices[ticker]
    avg = px.rolling(window).mean()
    dates = month_ends(prices)
    dates = dates[avg.loc[dates].notna().to_numpy()]
    above = px.loc[dates] > avg.loc[dates]
    return above.astype(float).to_frame(ticker)
