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


def vwap_scaler(pan, rf, mode="momentum", fund="3x", step=0.10, first=3, last_slot=75):
    """An always-adjusting intraday strategy, checked every 5 minutes (9:45-15:50).

    Its signal is how far QQQ is from today's VWAP, measured in units of the usual move
    from the open by this time of day (the noise-area width):  s = (price - VWAP) / (open x sigma).
    mode='momentum': lean with the move, exposure = clip(s, -1, 1) (above VWAP: long; below: short).
    mode='reversion': lean against it, exposure = clip(-s, -1, 1).
    fund='3x' trades TQQQ/SQQQ; fund='1x-long' trades QQQ itself and only goes long.
    It trades whenever the target exposure moves by `step` (10 points) or more, so the
    position changes many times a day in small steps. Out by the close.
    """
    q, tq, sq = pan["QQQ"], pan["TQQQ"], pan["SQQQ"]
    dates = q["close"].index.intersection(tq["close"].index).intersection(sq["close"].index)
    q = {k: v.loc[dates] for k, v in q.items()}
    b = noise_bands(q)
    O, C, last, day_close = b["O"], b["C"], b["last"], b["day_close"]
    sigma, vwap = b["sigma"], b["vwap"]
    o0 = O[:, 0]
    tq_px, sq_px, q_px = (f["open"].loc[dates].ffill(axis=1).bfill(axis=1).to_numpy() for f in (tq, sq, q))
    lev = 3.0 if fund == "3x" else 1.0
    rows = []
    for d in range(len(dates)):
        if np.isnan(sigma[d, 5]):
            rows.append((0.0, 0, 0.0))
            continue
        growth, trades, w, held = 1.0, 0, 0.0, 0
        end = min(last_slot, last[d])
        for k in range(first, end + 1):
            s = (C[d, k - 1] - vwap[d, k - 1]) / (o0[d] * sigma[d, k - 1]) if sigma[d, k - 1] > 0 else np.nan
            if np.isnan(s):
                target = w
            else:
                target = float(np.clip(s if mode == "momentum" else -s, -1, 1))
                if fund == "1x-long":
                    target = max(target, 0.0)
                target = round(target / step) * step
            if k == end:
                target = 0.0  # out by the close
            if abs(target - w) >= step - 1e-9 or (target == 0 and w != 0):
                px = lambda x: (tq_px if x > 0 else sq_px)[d, k] if fund == "3x" else q_px[d, k]  # noqa: E731
                cost = (0.5 * 0.01 / q_px[d, k] + 0.5e-4) if fund != "3x" else None
                if fund == "3x" and np.sign(target) != np.sign(w) and w != 0:
                    growth *= 1 - abs(w) * fund_cost(px(w))
                    growth *= 1 - abs(target) * fund_cost(px(target)) if target != 0 else 1
                    trades += 1 + (target != 0)
                else:
                    side = target if target != 0 else w
                    growth *= 1 - abs(target - w) * (fund_cost(px(side)) if fund == "3x" else cost)
                    trades += 1
                w = target
            # Hold w through the next 5 minutes (fill at this bar's open, to the next bar's open).
            nxt = O[d, k + 1] if k + 1 <= last[d] else day_close[d]
            if w != 0 and not np.isnan(nxt) and not np.isnan(O[d, k]):
                growth *= 1 + w * lev * (nxt / O[d, k] - 1)
                held += 1
        rows.append((growth - 1, trades, held / max(end - first + 1, 1)))
    out = pd.DataFrame(rows, index=dates, columns=["ret", "trades", "exposure"])
    out["ret"] = (1 + out["ret"]) * (1 + rf.reindex(dates).fillna(0.0)) - 1
    out["turnover"] = out["trades"].astype(float)
    return out


def breakout_scaler(pan, rf, every=1, base=0.5, step=0.25, first=6, last_slot=75, scale=True):
    """The noise-area breakout, made more active: checked every `every` 5-minute bars
    instead of every 30 minutes, and (if `scale`) sized in steps instead of all-in.

    Outside the band, exposure starts at `base` (50%) and grows with how far QQQ has run
    past the band, in units of the band's width: base + distance, capped at 100%. It
    trims as QQQ falls back, and exits on the usual trailing stop (back past the band or
    VWAP). Trades happen whenever the target moves by `step` (25 points) or more.
    Executed in TQQQ (long) / SQQQ (short).
    """
    q, tq, sq = pan["QQQ"], pan["TQQQ"], pan["SQQQ"]
    dates = q["close"].index.intersection(tq["close"].index).intersection(sq["close"].index)
    q = {k: v.loc[dates] for k, v in q.items()}
    b = noise_bands(q)
    O, C, last, day_close = b["O"], b["C"], b["last"], b["day_close"]
    sigma, vwap, upper, lower = b["sigma"], b["vwap"], b["upper"], b["lower"]
    o0 = O[:, 0]
    tq_px, sq_px = (f["open"].loc[dates].ffill(axis=1).bfill(axis=1).to_numpy() for f in (tq, sq))
    rows = []
    for d in range(len(dates)):
        if np.isnan(sigma[d, 5]):
            rows.append((0.0, 0, 0.0))
            continue
        growth, trades, w, side, held = 1.0, 0, 0.0, 0, 0
        end = min(last_slot, last[d])
        for k in range(first, end + 1):
            if (k - first) % every == 0 or k == end:
                p, ub, lb, vw = C[d, k - 1], upper[d, k - 1], lower[d, k - 1], vwap[d, k - 1]
                new_side = nam_decide(side, p, ub, lb, vw)
                width = o0[d] * sigma[d, k - 1]
                if new_side == 0 or not width > 0:
                    target = 0.0
                elif not scale:
                    target = float(new_side)
                else:
                    dist = (p - ub) / width if new_side > 0 else (lb - p) / width
                    target = new_side * float(np.clip(base + max(dist, 0.0), base, 1.0))
                    target = np.sign(target) * round(abs(target) / step) * step
                if k == end:
                    target, new_side = 0.0, 0
                if np.sign(target) != np.sign(w) and w != 0:  # exit (and maybe reverse)
                    growth *= 1 - abs(w) * fund_cost((tq_px if w > 0 else sq_px)[d, k])
                    trades += 1
                    w = 0.0
                if abs(target - w) >= step - 1e-9:
                    growth *= 1 - abs(target - w) * fund_cost((tq_px if target > 0 else sq_px)[d, k])
                    trades += 1
                    w = target
                side = new_side if w != 0 else (new_side if new_side == 0 else new_side)
            nxt = O[d, k + 1] if k + 1 <= last[d] else day_close[d]
            if w != 0 and not np.isnan(nxt) and not np.isnan(O[d, k]):
                growth *= 1 + w * 3.0 * (nxt / O[d, k] - 1)
                held += 1
        rows.append((growth - 1, trades, held / max(end - first + 1, 1)))
    out = pd.DataFrame(rows, index=dates, columns=["ret", "trades", "exposure"])
    out["ret"] = (1 + out["ret"]) * (1 + rf.reindex(dates).fillna(0.0)) - 1
    out["turnover"] = out["trades"].astype(float)
    return out


# --- the multi-market breakout trader -------------------------------------------------
# Market -> (long fund, short fund, leverage). Leveraged funds move about leverage x the
# market within a day, so a position's value is modeled as leverage x the market's move;
# trading costs come from the fund's own price (its one-cent spread is wider in bps when
# the fund is cheap).
MARKETS = {
    "SPY": ("UPRO", "SPXU", 3.0),   # S&P 500
    "IWM": ("TNA", "TZA", 3.0),     # small caps
    "SMH": ("SOXL", "SOXS", 3.0),   # semiconductors
    "TLT": ("TMF", "TMV", 3.0),     # long-term Treasuries
    "XLF": ("FAS", "FAZ", 3.0),     # banks and financials
    "XLE": ("ERX", "ERY", 2.0),     # energy (2x funds)
}


def constant_cost(fund_price_now, floor=1.5e-4):
    """Cost per trade from the fund's current price (half a one-cent spread + 0.5 bp),
    at least 1.5 bp. Used for every date: split-adjusted history would distort it, since
    inverse funds reverse-split often (old prices look huge, costs tiny) and leveraged long
    funds split forward (old prices look tiny, costs huge)."""
    return max(fund_cost(fund_price_now), floor)


def breakout_sleeve(q, long_px, short_px, lev, check_every=6, offset=0, trace=None):
    """One market's day-by-day intraday return under the noise-area rules (no cash
    interest), checked every `check_every` 5-minute bars from 10:00. long_px/short_px:
    the funds' daily prices; only the latest one is used, for the cost per trade."""
    dates = q["close"].index
    b = noise_bands(q)
    O, C, last, day_close = b["O"], b["C"], b["last"], b["day_close"]
    sigma, vwap, upper, lower = b["sigma"], b["vwap"], b["upper"], b["lower"]
    lp = np.full(len(dates), long_px.dropna().iloc[-1])
    sp = np.full(len(dates), short_px.dropna().iloc[-1])
    rows = []
    for d in range(len(dates)):
        if np.isnan(sigma[d, 5]):
            rows.append((0.0, 0, 0.0))
            continue
        growth, trades, pos, entry, held = 1.0, 0, 0, np.nan, 0
        checks = [k for k in range(6 + offset, SLOTS, check_every) if k + 1 <= last[d] + 1]
        for k in checks:
            want = nam_decide(pos, C[d, k - 1], upper[d, k - 1], lower[d, k - 1], vwap[d, k - 1])
            if trace is not None:
                trace.append((dates[d], k, pos, want))
            if want != pos and not np.isnan(O[d, k]):
                if pos != 0:
                    growth *= (1 + pos * lev * (O[d, k] / entry - 1)) * (1 - constant_cost((lp if pos > 0 else sp)[d]))
                    trades += 1
                if want != 0:
                    growth *= 1 - constant_cost((lp if want > 0 else sp)[d])
                    entry, trades = O[d, k], trades + 1
                pos = want
            held += pos != 0
        if pos != 0:
            growth *= (1 + pos * lev * (day_close[d] / entry - 1)) * (1 - constant_cost((lp if pos > 0 else sp)[d]))
            trades += 1
        rows.append((growth - 1, trades, held / max(len(checks), 1)))
    return pd.DataFrame(rows, index=dates, columns=["ret", "trades", "exposure"])


def multi_breakout(pans, fund_close, rf, markets=MARKETS, check_every=6):
    """Each market gets an equal slice of the account, reset every morning; cash earns
    T-bill interest overnight. Returns the portfolio's daily results plus each market's."""
    sleeves = {m: breakout_sleeve(pans[m], fund_close[lf], fund_close[sf], lev, check_every)
               for m, (lf, sf, lev) in markets.items()}
    rets = pd.DataFrame({m: s["ret"] for m, s in sleeves.items()}).dropna()
    out = pd.DataFrame({"ret": rets.mean(axis=1),
                        "trades": pd.DataFrame({m: s["trades"] for m, s in sleeves.items()}).reindex(rets.index).sum(axis=1),
                        "exposure": pd.DataFrame({m: s["exposure"] for m, s in sleeves.items()}).reindex(rets.index).mean(axis=1)})
    out["ret"] = (1 + out["ret"]) * (1 + rf.reindex(out.index).fillna(0.0)) - 1
    out["turnover"] = out["trades"].astype(float)
    return out, sleeves
