"""Pass/fail checks, with a bar that rises with every variant tried."""
from datetime import date
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from metrics import TRADING_DAYS, return_stats

LOG = Path(__file__).parent / "experiments.csv"


def t_needed(n_variants, confidence=0.975):
    """The t-stat that the luckiest of n zero-edge variants would beat only 2.5% of the
    time (Sidak correction). One variant gives the familiar 1.96; 100 give about 3.5."""
    return NormalDist().inv_cdf(confidence ** (1 / max(n_variants, 1)))


def vs_risk_matched(returns, bench, rf):
    """Compare with a SPY + cash mix that has the same volatility as the strategy.

    A low-risk strategy shouldn't fail just for being low-risk, and a high-risk one
    shouldn't pass just for being high-risk. Returns (excess per year, t-stat).
    """
    rf = rf.reindex(returns.index)
    bench = bench.reindex(returns.index)
    k = returns.std() / bench.std()
    active = returns - (rf + k * (bench - rf))
    if active.std() < 1e-12:  # the benchmark compared with itself
        return np.nan, np.nan
    return active.mean() * TRADING_DAYS, active.mean() / active.std() * np.sqrt(len(active))


def score(result, bench, rf, dd_limit, n_variants):
    r = result.returns
    stats = return_stats(r, rf)
    excess, t = vs_risk_matched(r, bench, rf)
    years = len(r) / TRADING_DAYS
    row = {
        "CAGR": stats["CAGR"],
        "Sharpe": stats["Sharpe"],
        "Max drawdown": stats["Max drawdown"],
        "Turnover / yr": result.turnover.sum() / years,
        "Excess vs risk-matched SPY / yr": excess,
        "t-stat vs risk-matched SPY": t,
    }
    if dd_limit is None:  # a reference row, not a candidate
        return {**row, "Needed t": np.nan, "Drawdown limit": np.nan, "Result": "reference"}
    bar = t_needed(n_variants)
    passed = t >= bar and stats["Max drawdown"] >= dd_limit
    return {**row, "Needed t": bar, "Drawdown limit": dd_limit, "Result": "PASS" if passed else "FAIL"}


def log_variants(configs):
    """Record each configuration the first time it is evaluated; return the total count.
    Delete nothing from this file: forgotten failures are how backtests lie."""
    log = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame(columns=["first_tried", "config"])
    new = [c for c in configs if c not in set(log["config"])]
    if new:
        log = pd.concat([log, pd.DataFrame({"first_tried": str(date.today()), "config": new})])
        log.to_csv(LOG, index=False)
    return log
