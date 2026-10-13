"""v2 intraday strategies on QQQ, executed with 3x funds: TQQQ to bet on a rise, SQQQ to
bet on a fall (so a $1,000 account can trade both directions without short selling).

Both are published day-trading strategies, with their published parameters:

* Noise-area momentum (Zarattini, Aziz & Barbon 2024, "Beat the Market"): around each
  day's open, draw bands as wide as the average move from the open to this time of day
  over the last 14 days (widened for overnight gaps). Inside the bands is noise; a break
  above (below) at a half-hour mark starts a long (short). Trail a stop at the band or
  VWAP, whichever is tighter; everything closes at the end of the day.
* Opening-range breakout (Zarattini & Aziz 2023): trade in the direction of the first
  5-minute bar, stop at the other side of that bar, target 10x the risk or the close.

Signals use QQQ prices, and a 3x fund moves about 3x QQQ within a day, so positions are
valued in QQQ terms times the leverage used. Costs come from the fund actually traded:
half its bid-ask spread (one cent at its price) plus 0.5 bp of slippage per trade.
"""
import numpy as np
import pandas as pd

from intraday_data import SLOTS

DECISIONS = list(range(6, SLOTS, 6))  # bar indexes at 10:00, 10:30, ..., 15:30
LOOKBACK = 14


def fund_cost(price):
    """Cost per side, as a fraction: half a one-cent spread plus 0.5 bp of slippage."""
    return 0.5 * 0.01 / price + 0.5e-4


def _day_arrays(q):
    O, H, L, C, V = (q[f].to_numpy() for f in ("open", "high", "low", "close", "volume"))
    has = ~np.isnan(C)
    last = SLOTS - 1 - np.argmax(has[:, ::-1], axis=1)
    day_close = C[np.arange(len(C)), last]
    return O, H, L, C, V, last, day_close


def daily_vol(day_close, n=LOOKBACK):
    r = pd.Series(day_close).pct_change()
    return r.rolling(n).std().shift(1).to_numpy()


def noise_bands(q):
    """The noise area for every day and 5-minute bar of a price panel (days x slots).

    upper/lower at bar m: today's open (or yesterday's close, if the gap went that way)
    plus/minus the average absolute move from the open to bar m over the previous 14 days.
    vwap at bar m: the volume-weighted average price from the open through bar m.
    """
    O, H, L, C, V, last, day_close = _day_arrays(q)
    o0 = O[:, 0]
    prev_close = np.r_[np.nan, day_close[:-1]]
    move = np.abs(C / o0[:, None] - 1)
    # At least 10 of the 14 days: an early-close day has no afternoon bars, and that
    # shouldn't blank out the afternoon bands for the next two weeks.
    sigma = pd.DataFrame(move).rolling(LOOKBACK, min_periods=10).mean().shift(1).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.nancumsum(C * V, axis=1) / np.nancumsum(V, axis=1)
    top, bottom = np.fmax(o0, prev_close), np.fmin(o0, prev_close)
    return {"O": O, "C": C, "last": last, "day_close": day_close, "sigma": sigma, "vwap": vwap,
            "upper": top[:, None] * (1 + sigma), "lower": bottom[:, None] * (1 - sigma)}


def nam_decide(pos, price, upper, lower, vwap):
    """The rule at one half-hour mark. pos/return value: +1 long, -1 short, 0 flat.
    Flat: break above the upper band -> long, below the lower band -> short.
    Long: exit if the price falls below the band or VWAP, whichever is higher.
    Short: exit if the price rises above the band or VWAP, whichever is lower."""
    if pos == 0:
        return 1 if price > upper else (-1 if price < lower else 0)
    if pos == 1:
        return 0 if price < np.fmax(upper, vwap) else 1
    return 0 if price > np.fmin(lower, vwap) else -1


def noise_area(pan, rf, sizing="paper", target_daily_vol=0.02, max_lev=3.0, trace=None):
    """Daily returns of the noise-area momentum strategy. sizing='paper' scales leverage
    to target 2% daily volatility (capped at 3x); sizing='full' always uses 3x. Pass a
    list as `trace` to collect every decision (for checking the live bot)."""
    q, tq, sq = pan["QQQ"], pan["TQQQ"], pan["SQQQ"]
    dates = q["close"].index.intersection(tq["close"].index).intersection(sq["close"].index)
    q = {k: v.loc[dates] for k, v in q.items()}
    b = noise_bands(q)
    O, C, last, day_close = b["O"], b["C"], b["last"], b["day_close"]
    sigma, vwap, upper, lower = b["sigma"], b["vwap"], b["upper"], b["lower"]
    tq_px, sq_px = (f["open"].loc[dates].ffill(axis=1).bfill(axis=1).to_numpy() for f in (tq, sq))
    n = len(dates)
    lev = (np.minimum(max_lev, target_daily_vol / daily_vol(day_close)) if sizing == "paper"
           else np.full(n, max_lev))

    rows = []
    for d in range(n):
        if np.isnan(sigma[d, 5]) or np.isnan(lev[d]):
            rows.append((0.0, 0, 0.0))
            continue
        growth, trades, pos, entry, held = 1.0, 0, 0, np.nan, 0
        for k in DECISIONS:
            if k + 1 > last[d] + 1:
                break
            fill = O[d, k]
            want = nam_decide(pos, C[d, k - 1], upper[d, k - 1], lower[d, k - 1], vwap[d, k - 1])
            if trace is not None:
                trace.append((dates[d], k, C[d, k - 1], upper[d, k - 1], lower[d, k - 1],
                              vwap[d, k - 1], pos, want))
            if want != pos:
                if pos != 0:  # close the open trade
                    growth *= 1 + pos * lev[d] * (fill / entry - 1)
                    growth *= 1 - fund_cost((tq_px if pos > 0 else sq_px)[d, k])
                    trades += 1
                if want != 0:
                    growth *= 1 - fund_cost((tq_px if want > 0 else sq_px)[d, k])
                    entry, trades = fill, trades + 1
                pos = want
            held += pos != 0
        if pos != 0:  # out by the close
            growth *= 1 + pos * lev[d] * (day_close[d] / entry - 1)
            growth *= 1 - fund_cost((tq_px if pos > 0 else sq_px)[d, last[d]])
            trades += 1
        rows.append((growth - 1, trades, held / len(DECISIONS)))
    out = pd.DataFrame(rows, index=dates, columns=["ret", "trades", "exposure"])
    out["ret"] = (1 + out["ret"]) * (1 + rf.reindex(dates).fillna(0.0)) - 1  # cash overnight
    out["turnover"] = out["trades"].astype(float)
    return out


def opening_range(pan, rf, sizing="paper", risk=0.01, max_lev=3.0, target_r=10):
    """Daily returns of the 5-minute opening-range breakout. sizing='paper' risks 1% of
    the account between entry and stop (capped at 3x); sizing='full' always uses 3x."""
    q, tq, sq = pan["QQQ"], pan["TQQQ"], pan["SQQQ"]
    dates = q["close"].index.intersection(tq["close"].index).intersection(sq["close"].index)
    q = {k: v.loc[dates] for k, v in q.items()}
    O, H, L, C, V, last, day_close = _day_arrays(q)
    tq_px, sq_px = (f["open"].loc[dates].ffill(axis=1).bfill(axis=1).to_numpy() for f in (tq, sq))
    rows = []
    for d in range(len(dates)):
        side = np.sign(C[d, 0] - O[d, 0])
        if side == 0 or np.isnan(O[d, 1]):
            rows.append((0.0, 0, 0.0))
            continue
        entry = O[d, 1]
        stop = L[d, 0] if side > 0 else H[d, 0]
        r = abs(entry - stop)
        if r <= 0:
            rows.append((0.0, 0, 0.0))
            continue
        lev = min(max_lev, risk / (r / entry)) if sizing == "paper" else max_lev
        target = entry + side * target_r * r
        exit_px, k_exit = day_close[d], last[d]
        for k in range(1, last[d] + 1):
            if side > 0 and L[d, k] <= stop:
                exit_px, k_exit = min(O[d, k], stop), k
                break
            if side < 0 and H[d, k] >= stop:
                exit_px, k_exit = max(O[d, k], stop), k
                break
            if side > 0 and H[d, k] >= target:
                exit_px, k_exit = max(O[d, k], target), k
                break
            if side < 0 and L[d, k] <= target:
                exit_px, k_exit = min(O[d, k], target), k
                break
        px = tq_px if side > 0 else sq_px
        growth = (1 - fund_cost(px[d, 1])) * (1 + side * lev * (exit_px / entry - 1)) \
            * (1 - fund_cost(px[d, min(k_exit, last[d])]))
        rows.append((growth - 1, 2, (k_exit - 1) / max(last[d], 1)))
    out = pd.DataFrame(rows, index=dates, columns=["ret", "trades", "exposure"])
    out["ret"] = (1 + out["ret"]) * (1 + rf.reindex(dates).fillna(0.0)) - 1
    out["turnover"] = out["trades"].astype(float)
    return out
