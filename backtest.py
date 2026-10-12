"""Turn target weights into daily portfolio returns: execution lag, drift and costs."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Result:
    returns: pd.Series     # daily portfolio return, after costs
    weights: pd.DataFrame  # weights at each close, after any rebalance; the rest is cash
    turnover: pd.Series    # fraction of the portfolio traded each day
    trades: pd.Series      # number of tickers bought or sold each day


def run_backtest(targets, prices, rf, start, cost_bps=5.0, lag=1):
    """Simulate holding a strategy's target weights day by day.

    targets:  weights decided at the close of each signal date (rows), by ticker.
              Weights may sum to less than 1; the remainder is cash earning `rf`.
    lag:      trading days between a signal and its fill. Fills happen at that day's
              close, so lag=1 means a signal never trades on the price it came from.
    cost_bps: cost per dollar traded (bid-ask spread + slippage; commissions assumed $0).

    Between rebalances, weights drift with prices instead of being reset every day.
    """
    targets = targets.loc[start:].reindex(columns=prices.columns, fill_value=0.0).fillna(0.0)
    days = prices.index
    sig_pos = days.get_indexer(targets.index)
    if (sig_pos < 0).any():
        raise ValueError("every signal date must be a trading day in `prices`")
    fill_pos = sig_pos + lag
    keep = fill_pos < len(days)  # the newest signal may not have filled yet
    fills = dict(zip(fill_pos[keep].tolist(), targets.to_numpy()[keep]))

    first = int(fill_pos[0])
    window = prices.iloc[first - 1:]
    if window.isna().any().any():
        raise ValueError("missing prices inside the backtest window")
    asset_ret = (window / window.shift(1) - 1).iloc[1:].to_numpy()
    cash_ret = rf.reindex(days[first:]).fillna(0.0).to_numpy()

    n_days, n_assets = len(days) - first, prices.shape[1]
    w = np.zeros(n_assets)  # start fully in cash
    rets, turnover = np.zeros(n_days), np.zeros(n_days)
    trades = np.zeros(n_days, dtype=int)
    weights = np.zeros((n_days, n_assets))
    for k in range(n_days):
        gross = w @ asset_ret[k] + (1 - w.sum()) * cash_ret[k]
        w = w * (1 + asset_ret[k]) / (1 + gross)  # yesterday's weights drift with today's prices
        target = fills.get(first + k)
        if target is not None:
            traded = np.abs(target - w)
            turnover[k] = traded.sum()
            trades[k] = (traded > 1e-9).sum()
            w = target.copy()
        rets[k] = (1 + gross) * (1 - turnover[k] * cost_bps / 1e4) - 1
        weights[k] = w

    idx = days[first:]
    return Result(
        returns=pd.Series(rets, idx),
        weights=pd.DataFrame(weights, idx, prices.columns),
        turnover=pd.Series(turnover, idx),
        trades=pd.Series(trades, idx),
    )
