"""Ways to split one account across several strategy sleeves.

Each allocator takes the sleeves' daily returns (after their own costs) and returns the
budget for each sleeve each day, decided at the close and used for the next day. A
sleeve that hasn't started yet (e.g. crypto before 2015) gets nothing until it has
enough history.
"""
import numpy as np
import pandas as pd

WARMUP = 60  # days of a sleeve's history needed before it gets money


def available(R):
    """True where a sleeve has at least WARMUP days of returns behind it."""
    return R.notna().cumsum() > WARMUP


def equal_weight(R):
    ok = available(R).astype(float)
    return ok.div(ok.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def risk_parity(R, window=120):
    """Budgets proportional to 1 / (sleeve's volatility over the last `window` days), so
    each sleeve contributes about the same risk."""
    vol = R.rolling(window, min_periods=WARMUP).std()
    inv = (1 / vol).where(available(R))
    return inv.div(inv.sum(axis=1), axis=0).fillna(0.0)


def hedge(R, rf, eta=0.05, forget=0.995, floor=0.05, window=60):
    """The 'Hedge' online-learning algorithm (multiplicative weights), starting from
    scratch with equal trust in every sleeve.

    Each day it scores every sleeve by that day's excess return divided by the sleeve's
    recent volatility, adds eta x score to the sleeve's running log-weight, and slowly
    forgets old evidence (x0.995 a day, a half-life of about 140 trading days). Budgets
    are the softmax of the log-weights, mixed with a 5% floor per sleeve so it keeps
    exploring sleeves that have fallen out of favour. Returns (budgets, log-weights).
    """
    X = R.sub(rf.reindex(R.index), axis=0)
    vol = R.rolling(window, min_periods=20).std()
    ok = available(R)
    theta = np.zeros(R.shape[1])
    budgets, thetas = [], []
    for t in range(len(R)):
        live = ok.iloc[t].to_numpy()
        if live.any():
            z = np.where(live, theta, -np.inf)
            w = np.exp(z - z[live].max())
            w = w / w.sum()
            n = live.sum()
            w = np.where(live, (1 - floor * n) * w + floor, 0.0) if floor * n < 1 else live / n
        else:
            w = np.zeros(R.shape[1])
        budgets.append(w)
        thetas.append(theta.copy())
        # Learn from today's results (known at today's close) for tomorrow's budgets.
        score = (X.iloc[t] / vol.iloc[t]).to_numpy()
        theta = forget * theta + eta * np.where(live & np.isfinite(score), score, 0.0)
    return (pd.DataFrame(budgets, index=R.index, columns=R.columns),
            pd.DataFrame(thetas, index=R.index, columns=R.columns))


def combine(R, budgets, rf, cost_per_switch=None):
    """Portfolio daily returns: yesterday's budgets times today's sleeve returns. Moving
    money between sleeves costs roughly the sleeves' own trading cost on the amount moved
    (`cost_per_switch`: sleeve -> cost fraction)."""
    b = budgets.shift(1).fillna(0.0)
    cash = (1 - b.sum(axis=1)).clip(lower=0)
    ret = (b * R.fillna(0.0)).sum(axis=1) + cash * rf.reindex(R.index).fillna(0.0)
    if cost_per_switch is not None:
        c = pd.Series(cost_per_switch).reindex(R.columns).fillna(3e-4)
        ret -= (budgets.diff().abs().fillna(0.0) * c).sum(axis=1)
    return ret
