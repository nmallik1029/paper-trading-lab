"""ML day trader: SPY and QQQ in 30-minute blocks, always out by the close.

Every half hour from 10:00 to 15:30 the model predicts each ETF's return over the next
30 minutes from what has happened so far today and in recent days. It buys the better
one when the predicted gain beats the round-trip trading cost, keeps holding while the
prediction stays positive, and otherwise sits in cash. Everything is sold at the close,
so nothing is held overnight.

Timing: a decision at 10:00 uses only bars that finished by 10:00, and fills at the open
of the 10:00 bar (the first trades after 10:00). Each block runs to the next decision's
fill, and the last block runs to the closing price.
"""
import numpy as np
import pandas as pd

from intraday_data import SLOTS
from ml import make_model

FEATURE_VERSION = "intraday-v1"
DECISION_SLOTS = list(range(6, SLOTS, 6))  # 10:00, 10:30, ..., 15:30
MIN_TRAIN_MONTHS = 12
SEED_DAYS = 5  # a from-scratch model starts out knowing only this many trading days
CLIP = 0.015  # cap training targets at +/-1.5% so a few crash days don't dominate the fit

FEATURE_NAMES = {
    "ret_5m": "Return, last 5 minutes",
    "ret_15m": "Return, last 15 minutes",
    "ret_30m": "Return, last 30 minutes",
    "ret_since_open": "Return since today's open",
    "gap": "Overnight gap (today's open vs yesterday's close)",
    "ret_prev_day": "Yesterday's return",
    "ret_prev_5d": "Return over the previous 5 days",
    "vol_30m": "Volatility, last 30 minutes",
    "vol_prev_day": "Volatility, yesterday",
    "rel_volume_30m": "Volume, last 30 min vs usual for this time",
    "time_of_day": "Time of day (block 1-12)",
    "is_qqq": "QQQ rather than SPY",
}


def _symbol_rows(sym, p):
    O, C, V = (p[f].to_numpy() for f in ("open", "close", "volume"))
    dates = p["close"].index
    n = len(dates)
    R = np.full_like(C, np.nan)  # 5-minute returns; the first bar is measured from the open
    R[:, 0] = C[:, 0] / O[:, 0] - 1
    R[:, 1:] = C[:, 1:] / C[:, :-1] - 1

    has = ~np.isnan(C)
    last = SLOTS - 1 - np.argmax(has[:, ::-1], axis=1)  # each day's final bar (earlier on half-days)
    day_close = C[np.arange(n), last]
    closes = pd.Series(day_close)
    prev_close = closes.shift(1).to_numpy()
    with np.errstate(invalid="ignore"):
        vol_prev_day = pd.Series(np.nanstd(R, axis=1)).shift(1).to_numpy()

    rows = []
    for k in DECISION_SLOTS:
        vol30 = V[:, k - 6:k].sum(axis=1)
        usual = pd.Series(vol30).rolling(20, min_periods=15).mean().shift(1).to_numpy()
        next_open = O[:, k + 6] if k + 6 < SLOTS else np.full(n, np.nan)
        exit_px = np.where(k + 6 <= last, next_open, np.where(last == k + 5, day_close, np.nan))
        rows.append(pd.DataFrame({
            "date": dates, "slot": k, "symbol": sym,
            "ret_5m": R[:, k - 1],
            "ret_15m": C[:, k - 1] / C[:, k - 4] - 1,
            "ret_30m": C[:, k - 1] / O[:, k - 6] - 1,
            "ret_since_open": C[:, k - 1] / O[:, 0] - 1,
            "gap": O[:, 0] / prev_close - 1,
            "ret_prev_day": prev_close / closes.shift(2).to_numpy() - 1,
            "ret_prev_5d": prev_close / closes.shift(6).to_numpy() - 1,
            "vol_30m": R[:, k - 6:k].std(axis=1),
            "vol_prev_day": vol_prev_day,
            "rel_volume_30m": vol30 / usual,
            "time_of_day": k // 6,
            "is_qqq": float(sym == "QQQ"),
            "target": exit_px / O[:, k] - 1,  # the next 30 minutes: the label, never a feature
        }))
    return rows


def features(pan, keep_unlabeled=False):
    """Return (X, y), one row per (date, decision slot, symbol). Rows for blocks that
    don't exist (half-days) or have missing bars are dropped. Live trading passes
    keep_unlabeled=True to keep today's rows, whose outcome isn't known yet."""
    df = pd.concat([r for sym, p in pan.items() for r in _symbol_rows(sym, p)])
    df = df.set_index(["date", "slot", "symbol"]).sort_index()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=df.columns.drop("target") if keep_unlabeled else None)
    return df.drop(columns="target"), df["target"]


def train(X, y, kind="ridge"):
    """Fit one model on every row given (used live; walk_forward does the same monthly)."""
    return make_model(kind).fit(X.to_numpy(), y.clip(-CLIP, CLIP).to_numpy())


def choose(preds, pos, cost_bps):
    """The trading rule for one block. preds: {symbol: predicted return}; pos: the symbol
    held now, or ''. Enter only when the predicted gain beats the round-trip cost; keep
    holding while it's still positive; otherwise cash."""
    preds = {s: p for s, p in preds.items() if not np.isnan(p)}
    if not preds:
        return ""
    best = max(preds, key=preds.get)
    if preds[best] > 2 * cost_bps / 1e4:
        return best
    if pos and preds.get(pos, np.nan) > 0:
        return pos
    return ""


def walk_forward(X, y, kind="ridge"):
    """Retrain at the start of every month on all earlier months, then predict that month.

    Returns (predictions, coefs): predicted next-30-minute return per row, and for ridge
    the fitted weights at each retrain (None for gbm).
    """
    month = X.index.get_level_values("date").to_period("M")
    preds = pd.Series(np.nan, index=X.index)
    coefs = {}
    for m in month.unique().sort_values()[MIN_TRAIN_MONTHS:]:
        train, test = np.asarray(month < m), np.asarray(month == m)
        model = make_model(kind).fit(X[train].to_numpy(), y[train].clip(-CLIP, CLIP).to_numpy())
        preds[test] = model.predict(X[test].to_numpy())
        if kind == "ridge":
            coefs[m.to_timestamp()] = pd.Series(model[-1].coef_, index=X.columns)
    return preds.dropna(), (pd.DataFrame(coefs).T if coefs else None)


def seed_start(days, first_day, seed_days=SEED_DAYS):
    """The first day a from-scratch model learns from: `seed_days` trading days before
    `first_day`. `days` is the sorted list of trading days available."""
    i = days.searchsorted(pd.Timestamp(first_day))
    return days[i - seed_days]


def cold_start(X, y, first_day, last_day=None, seed_days=SEED_DAYS, kind="ridge"):
    """A model that starts out knowing only the `seed_days` trading days before
    `first_day`, then retrains every morning on everything it has seen since and never
    forgets. Early on it is mostly noise; it steadies as examples pile up.

    Returns (predictions, weights): predicted returns for each trading day from
    first_day to last_day, and the model's weights each morning (bps of predicted
    return per standard deviation of each feature).
    """
    dates = X.index.get_level_values("date")
    days = dates.unique().sort_values()
    start = seed_start(days, first_day, seed_days)
    run = days[(days >= pd.Timestamp(first_day)) & (days <= pd.Timestamp(last_day or days[-1]))]
    date_arr = np.asarray(dates)
    lo = date_arr.searchsorted(np.datetime64(start))  # rows are sorted by date
    Xv, yv = X.to_numpy(), y.clip(-CLIP, CLIP).to_numpy()
    preds, weights = [], {}
    for d in run:
        hi, end = date_arr.searchsorted(np.datetime64(d)), date_arr.searchsorted(np.datetime64(d), "right")
        model = make_model(kind).fit(Xv[lo:hi], yv[lo:hi])  # everything before this morning
        preds.append(pd.Series(model.predict(Xv[hi:end]), index=X.index[hi:end]))
        if kind == "ridge":
            weights[d] = model[-1].coef_ * 1e4
    return pd.concat(preds), (pd.DataFrame(weights, index=X.columns).T if weights else None)


def decide(preds, cost_bps):
    """For each (date, slot), the symbol to hold through the next block, or '' for cash.
    Enter only when the predicted gain beats the round-trip cost; stay while it's positive."""
    p = preds.unstack("symbol")
    cols, vals = list(p.columns), p.to_numpy()
    holds, pos, today = [], "", None
    for (d, _), row in zip(p.index, vals):
        if d != today:
            pos, today = "", d  # every day starts in cash
        pos = choose(dict(zip(cols, row)), pos, cost_bps)
        holds.append(pos)
    return pd.Series(holds, index=p.index)


def simulate(holds, block_ret, cost_bps, rf):
    """Daily results from per-block holdings.

    Each entry, exit and switch costs cost_bps per dollar traded, including the sale at
    the close. The account is always in cash overnight, so it earns that day's T-bill
    interest. Returns a DataFrame by date: ret, trades, invested (share of blocks held),
    plus the list of gross block returns while invested.
    """
    c = cost_bps / 1e4
    r = block_ret.unstack("symbol").reindex(holds.index)
    cols = {s: i for i, s in enumerate(r.columns)}
    rv = r.to_numpy()
    out, held_returns = {}, []
    dates = holds.index.get_level_values("date")
    start = 0
    for d, n in pd.Series(dates).value_counts(sort=False).sort_index().items():
        growth, trades, invested, pos = 1.0, 0, 0, ""
        for i in range(start, start + n):
            h = holds.iat[i]
            if h != pos:
                trades += (pos != "") + (h != "")
                pos = h
            if pos:
                x = rv[i, cols[pos]]
                growth *= 1 + x
                invested += 1
                held_returns.append(x)
        trades += pos != ""
        out[d] = ((growth * (1 - c) ** trades) * (1 + rf.get(d, 0.0)) - 1, trades, invested / n)
        start += n
    daily = pd.DataFrame.from_dict(out, orient="index", columns=["ret", "trades", "invested"])
    return daily, np.array(held_returns)


def always_hold(block_ret, sym="SPY"):
    """Reference: hold one ETF through every block (10:00 to the close), never overnight."""
    idx = block_ret.xs(sym, level="symbol").index
    return pd.Series(sym, index=idx)


def last_half_hour_momentum(X, block_ret, sym="SPY"):
    """Reference from published research (Gao, Han, Li & Zhou 2018): if the ETF is up from
    yesterday's close to 10:00, hold it for the last half hour; otherwise stay in cash."""
    idx = block_ret.xs(sym, level="symbol").index
    first = X.xs((6, sym), level=("slot", "symbol"))
    up = (1 + first["gap"]) * (1 + first["ret_since_open"]) - 1 > 0
    slot = idx.get_level_values("slot")
    date = idx.get_level_values("date")
    hold = (slot == DECISION_SLOTS[-1]) & up.reindex(date).fillna(False).astype(bool).to_numpy()
    return pd.Series(np.where(hold, sym, ""), index=idx)
