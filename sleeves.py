"""v2 strategy sleeves on daily data. Each function returns daily strategy returns (after
costs) with its own exposure series, so sleeves can be scored alone or combined.

Timing: a signal uses closes up to day t and trades at day t's close (the live bot runs at
15:40, about 20 minutes before the close, using the latest price as the close). The
position then earns day t+1's return. Overnight strategies instead hold from one close
to the next open.

Parameters are fixed from the published or practitioner defaults cited in each docstring,
not tuned here.
"""
import numpy as np
import pandas as pd

from universe import spliced

TRADING_DAYS = 252


def sma(s, n):
    return s.rolling(n).mean()


def realized_vol(r, n=20):
    return r.rolling(n).std() * np.sqrt(TRADING_DAYS)


def rsi(close, n=2):
    """Wilder's RSI."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down)


def run_weights(w, asset_ret, rf, cost_bps):
    """Daily returns for holding weight w (decided at day t's close) in an asset, the rest
    in T-bills. Pays cost_bps per dollar traded when w changes."""
    w = w.reindex(asset_ret.index).ffill().fillna(0.0)
    held = w.shift(1).fillna(0.0)
    turnover = w.diff().abs().fillna(w.abs())
    ret = held * asset_ret.fillna(0.0) + (1 - held) * rf.reindex(asset_ret.index).fillna(0.0) \
        - turnover * cost_bps / 1e4
    # target = the weight decided at each day's close (what the live bot trades today).
    return pd.DataFrame({"ret": ret, "exposure": held, "turnover": turnover, "target": w})


def lev_returns(d, rf, fund, index, leverage):
    """The real leveraged fund's daily returns, rebuilt from its index before it existed.
    Computed on the index's own trading days: the data also has crypto's weekend rows,
    and a stock's return must be measured from its previous trading day, not from Sunday."""
    idx = d["close"][index].dropna()
    real = d["close"][fund].dropna().reindex(idx.index)
    return spliced(real, idx, leverage, rf.reindex(idx.index).fillna(0.0))


# --- sleeves ---------------------------------------------------------------------------

def trend_voltarget(d, rf, target_vol=0.40, window=200, cost_bps=3.0):
    """Leveraged trend with volatility targeting (Moreira & Muir 2017; the popular TQQQ
    200-day rule). Hold TQQQ while QQQ is above its 200-day average, sized so the
    position's expected volatility is ~40%/yr (capped at 100%). Rebalance only when the
    target weight moves by more than 10 points."""
    qqq = d["close"]["QQQ"].dropna()
    tq = lev_returns(d, rf, "TQQQ", "QQQ", 3).reindex(qqq.index)
    vol = realized_vol(qqq / qqq.shift(1) - 1) * 3
    raw = (qqq > sma(qqq, window)).astype(float) * (target_vol / vol).clip(upper=1.0)
    w, last = [], 0.0
    for x in raw.fillna(0.0):  # trade only on big changes, to save costs
        if abs(x - last) > 0.10 or (x == 0) != (last == 0):
            last = x
        w.append(last)
    return run_weights(pd.Series(w, raw.index), tq, rf, cost_bps)


def short_vol(d, rf, enter=0.95, leave=1.0, cost_bps=3.0):
    """Short-volatility carry with a term-structure switch (practitioner rule: hold the
    inverse VIX fund while the curve is in contango). Hold SVXY while VIX/VIX3M < 0.95;
    go to cash once it reaches 1.0 (backwardation, i.e. panic)."""
    ratio = (d["close"]["^VIX"] / d["close"]["^VIX3M"]).dropna()
    sv = d["close"]["SVXY"].dropna()
    ratio = ratio.reindex(sv.index).ffill()
    w, on = [], False
    for x in ratio:
        on = (x < enter) or (on and x < leave)
        w.append(float(on))
    return run_weights(pd.Series(w, sv.index), sv / sv.shift(1) - 1, rf, cost_bps)


def overnight(d, rf, trend_filter=False, cost_bps=3.0):
    """Overnight drift, leveraged (Lou, Polk & Skouras 2019; the 'night effect'): buy TQQQ
    near the close, sell at the next open, every day. Optionally only while QQQ is above
    its 200-day average. Two trades a day."""
    o, c = d["open"], d["close"]
    qo, qc = o["QQQ"].dropna(), c["QQQ"].dropna()  # stock trading days only (no crypto weekends)
    qqq_night = (qo / qc.shift(1) - 1).dropna()
    to, tc = o["TQQQ"].dropna(), c["TQQQ"].dropna()
    real = (to / tc.shift(1) - 1).reindex(qqq_night.index)
    night = real.where(real.notna(), 3 * qqq_night)
    tonight = pd.Series(1.0, index=night.index)  # decided at each day's close
    if trend_filter:
        q = qc.reindex(night.index)
        tonight = (q > sma(q, 200)).astype(float)
    on = tonight.shift(1).fillna(0.0)  # the night ending at this day's open
    ret = on * night - on * 2 * cost_bps / 1e4  # buy at the close, sell at the open
    # T-bill interest isn't earned overnight-only (the cash sits in the fund), so skip it.
    return pd.DataFrame({"ret": ret, "exposure": on, "turnover": 2 * on, "target": tonight})


def turn_of_month(d, rf, cost_bps=3.0):
    """Turn-of-the-month effect (Ariel 1987; Xu & McConnell 2008), leveraged: hold UPRO
    from the close of the 2nd-to-last trading day of each month through the close of the
    3rd trading day of the next month (4 trading days); T-bills otherwise."""
    spy = d["close"]["SPY"].dropna()
    up = lev_returns(d, rf, "UPRO", "SPY", 3).reindex(spy.index)
    month = spy.index.to_period("M")
    pos = pd.Series(np.arange(len(spy)), index=spy.index)
    from_end = pos.groupby(month).transform(lambda x: x.max() - x)   # 0 = last day
    from_start = pos.groupby(month).transform(lambda x: x - x.min())  # 0 = first day
    # Weight decided at day t's close earns day t+1: hold through last day + first 3 days.
    w = ((from_end == 1) | (from_end == 0) | (from_start <= 1)).astype(float)
    return run_weights(w, up, rf, cost_bps)


def rsi2_dip(d, rf, cost_bps=3.0):
    """Connors RSI(2) mean reversion, leveraged: buy TQQQ at the close when QQQ's 2-day
    RSI drops below 10 while QQQ is above its 200-day average; sell when RSI(2) > 70."""
    qqq = d["close"]["QQQ"].dropna()
    tq = lev_returns(d, rf, "TQQQ", "QQQ", 3).reindex(qqq.index)
    r2, up = rsi(qqq, 2), qqq > sma(qqq, 200)
    w, on = [], False
    for x, ok in zip(r2, up):
        if not on and ok and x < 10:
            on = True
        elif on and x > 70:
            on = False
        w.append(float(on))
    return run_weights(pd.Series(w, qqq.index), tq, rf, cost_bps)


def crypto_trend(d, rf, target_vol=0.60, lookback=28, cost_bps=30.0):
    """Crypto time-series momentum (trend-following in Bitcoin is among the strongest
    documented crypto effects; e.g. Quantpedia's MAX/MIN studies). Hold BTC when its
    28-day return is positive, sized to ~60%/yr volatility (capped at 100%). Rebalance only
    on 10-point changes. Costs include Alpaca's 0.25% crypto fee."""
    btc = d["close"]["BTC-USD"].dropna()
    r = btc / btc.shift(1) - 1
    vol = r.rolling(20).std() * np.sqrt(365)
    raw = ((btc / btc.shift(lookback) - 1) > 0).astype(float) * (target_vol / vol).clip(upper=1.0)
    w, last = [], 0.0
    for x in raw.fillna(0.0):
        if abs(x - last) > 0.10 or (x == 0) != (last == 0):
            last = x
        w.append(last)
    rf_c = rf.reindex(btc.index).ffill().fillna(0.0) * 252 / 365  # crypto trades every day
    return run_weights(pd.Series(w, btc.index), r, rf_c, cost_bps)
