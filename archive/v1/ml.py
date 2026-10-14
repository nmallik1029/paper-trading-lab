"""Machine-learning sector strategies, trained walk-forward.

At each month-end the model only sees data up to that date: it trains on every earlier
month whose next-month outcome is already known, then ranks the sectors for the month
ahead. Nothing from the future can leak into a decision.
"""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import BENCHMARK, SECTORS
from strategies import month_ends

# Bump this whenever the features change: a new feature set is a new variant to count.
FEATURE_VERSION = "v1"
MIN_TRAIN_MONTHS = 36

FEATURE_NAMES = {
    "ret_1m": "Last month's return",
    "ret_3m": "Last 3 months' return",
    "ret_6m": "Last 6 months' return",
    "ret_12m_skip1": "12-month return, skipping last month",
    "vol_3m": "Volatility, last 3 months",
    "vs_200d": "Distance above its 200-day average",
    "mkt_vs_200d": "Market: SPY vs its 200-day average",
    "mkt_vol_3m": "Market: SPY volatility, last 3 months",
    "tbill": "Market: T-bill rate",
    "tbill_chg_3m": "Market: change in T-bill rate, 3 months",
}


def features(prices, rf):
    """Return (X, y): one row per (month-end, sector).

    Sector features are ranked across the nine sectors each month (-0.5 = worst,
    +0.5 = best), so the model learns relative strength, not the market's level.
    Market features are the same for every sector in a month; only the tree model can
    use them, by interacting them with the sector features.
    y is the rank of each sector's *next*-month return: the label, never a feature.
    """
    dates = month_ends(prices)
    px = prices[SECTORS]
    daily = px / px.shift(1) - 1
    m = px.loc[dates]
    sector = {
        "ret_1m": m / m.shift(1) - 1,
        "ret_3m": m / m.shift(3) - 1,
        "ret_6m": m / m.shift(6) - 1,
        "ret_12m_skip1": m.shift(1) / m.shift(12) - 1,
        "vol_3m": daily.rolling(63).std().loc[dates],
        "vs_200d": (px / px.rolling(200).mean() - 1).loc[dates],
    }
    X = pd.DataFrame({k: (v.rank(axis=1, pct=True) - 0.5).stack() for k, v in sector.items()})
    X.index.names = ["date", "ticker"]

    spy = prices[BENCHMARK]
    tbill = (rf * 252).loc[dates]
    market = pd.DataFrame({
        "mkt_vs_200d": (spy / spy.rolling(200).mean() - 1).loc[dates],
        "mkt_vol_3m": (spy / spy.shift(1) - 1).rolling(63).std().loc[dates],
        "tbill": tbill,
        "tbill_chg_3m": tbill.diff(3),
    })
    X = X.join(market, on="date").dropna()

    nxt = m.shift(-1) / m - 1
    y = (nxt.rank(axis=1, pct=True) - 0.5).stack()
    y.index.names = ["date", "ticker"]
    return X, y.reindex(X.index)


def make_model(kind):
    if kind == "ridge":  # linear: a weighted sum of the features
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if kind == "gbm":    # gradient-boosted trees: can learn interactions and thresholds
        return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                             min_samples_leaf=50, l2_regularization=1.0,
                                             early_stopping=False, random_state=0)
    raise ValueError(f"unknown model kind: {kind}")


def walk_forward(prices, rf, kind="ridge"):
    """Retrain every month-end on all earlier months, then score the sectors.

    Returns (scores, coefs): predicted next-month rank per sector (rows = month-ends),
    and for ridge the fitted weights at each retrain (None for gbm).
    """
    X, y = features(prices, rf)
    when = X.index.get_level_values("date")
    dates = when.unique().sort_values()
    scores, coefs = {}, {}
    for d in dates[MIN_TRAIN_MONTHS:]:
        # Every month before d has a label that was known by d's close.
        train = (when < d) & y.notna().to_numpy()
        model = make_model(kind).fit(X[train].to_numpy(), y[train].to_numpy())
        today = X[when == d]
        scores[d] = pd.Series(model.predict(today.to_numpy()),
                              index=today.index.get_level_values("ticker"))
        if kind == "ridge":
            coefs[d] = pd.Series(model[-1].coef_, index=X.columns)
    return pd.DataFrame(scores).T, (pd.DataFrame(coefs).T if coefs else None)


def ml_sectors(scores, prices, top_n=3, crash_filter=False, window=200):
    """Hold the top_n sectors by model score, equally weighted. With crash_filter,
    move to T-bills whenever SPY closes the month below its `window`-day average."""
    rank = scores.rank(axis=1, ascending=False, method="first")
    weights = (rank <= top_n).astype(float) / top_n
    if crash_filter:
        spy = prices[BENCHMARK]
        uptrend = (spy > spy.rolling(window).mean()).loc[weights.index]
        weights = weights.mul(uptrend.astype(float), axis=0)
    return weights
