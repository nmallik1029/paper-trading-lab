"""Live paper trading on Alpaca (lab v2), using exactly the backtested rules.

    python live.py daily [--dry-run]           # 15:40: multi-strategy fund + learning allocator
    python live.py unwind [--dry-run]          # 9:30: sell the overnight TQQQ the daily accounts hold
    python live.py daytrader --session morning|afternoon [--dry-run]   # breakout day trader
    python live.py daytrader --replay 2026-10-09     # rerun a past day with live data, no orders
    python live.py flatten daytrader|moderate|safe   # emergency: sell everything
    python live.py safety                      # sell the day trader's position if its session died

cron-job.org starts these through GitHub Actions (.github/workflows). Every decision is
appended to logs/<account>.jsonl, which the dashboard reads.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import GetCalendarRequest, MarketOrderRequest

import allocators as al
import intraday_v2 as iv
import research_v2 as rv
from broker import data_client, trading_client
from intraday_data import TZ, load_bars, panels, session_only
from universe import load_daily

LOGS = Path(__file__).parent / "logs"
LAUNCH_V2 = pd.Timestamp("2026-10-13")  # v2 started; the learning allocator starts from scratch here
CASH_BUFFER = 0.005   # leave 0.5% in cash so a small price move can't make a buy fail
MIN_ORDER = 1.0       # Alpaca's smallest dollar order
SMALL_TRADE = 0.02    # skip rebalancing trades under 2% of the account (entries and exits still happen)
LATE = pd.Timedelta(minutes=5)       # skip a day-trader decision if we reach it later than this
AFTERNOON = pd.Timedelta(hours=13, minutes=15)  # decisions from 13:15 belong to the afternoon run
CLOSE_OUT = pd.Timedelta(minutes=3)  # the day trader sells everything this long before the close

# The five daily sleeves both daily accounts hold, and what each one trades.
TREND, SHORT_VOL, OVERNIGHT, DIP, CRYPTO = (
    "Leveraged trend + vol target", "Short volatility (VIX curve)", "Overnight TQQQ, uptrend only",
    "RSI(2) dip buying, 3x", "Crypto trend (BTC)")
SLEEVES = [CRYPTO, SHORT_VOL, OVERNIGHT, DIP, TREND]
LONG_FUND, SHORT_FUND = "TQQQ", "SQQQ"  # the day trader's 3x long and 3x short QQQ funds


def now_et():
    return pd.Timestamp.now(tz=TZ)


def sleep_until(t):
    while (left := (t - now_et()).total_seconds()) > 0:
        time.sleep(min(left, 30))


def log(account, **row):
    LOGS.mkdir(exist_ok=True)
    row = {"time": now_et().isoformat(timespec="seconds"), **row}
    with open(LOGS / f"{account}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
    print(f"[{account}] " + ", ".join(f"{k}={v}" for k, v in row.items()))


def read_log(account):
    path = LOGS / f"{account}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def logged_today(account, actions, **match):
    """Has an entry like this been logged today? Lets a duplicate run skip work another
    run already did."""
    today = str(now_et().date())
    return any(r.get("action") in actions and r["time"].startswith(today)
               and all(r.get(k) == v for k, v in match.items()) for r in read_log(account))


def persist(message):
    """On GitHub Actions, commit the logs so the next run and the dashboard see them."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    subprocess.run(["git", "add", "logs"], check=False)
    if subprocess.run(["git", "commit", "-q", "-m", message], check=False).returncode == 0:
        for _ in range(3):  # another workflow may have pushed in the meantime
            subprocess.run(["git", "pull", "--rebase", "-q"], check=False)
            if subprocess.run(["git", "push", "-q"], check=False).returncode == 0:
                break


def schedule(day):
    """The day's (open, decision times, close), or None if the market is closed."""
    cal = trading_client().get_calendar(GetCalendarRequest(start=day, end=day))
    if not cal or cal[0].date != day:
        return None
    opens = pd.Timestamp(cal[0].open).tz_localize(TZ)
    close = pd.Timestamp(cal[0].close).tz_localize(TZ)
    times, t = [], opens + pd.Timedelta(minutes=30)
    while t + pd.Timedelta(minutes=30) <= close:  # same half-hour marks as the backtest
        times.append(t)
        t += pd.Timedelta(minutes=30)
    return opens, times, close


# --- orders ---------------------------------------------------------------------------

def is_crypto(symbol):
    return "/" in symbol


def holdings(client):
    """Market value by symbol. Crypto positions come back as e.g. BTCUSD; orders use BTC/USD."""
    out = {}
    for p in client.get_all_positions():
        sym = p.symbol
        if p.asset_class.value == "crypto" and "/" not in sym:
            sym = sym[:-3] + "/" + sym[-3:]
        out[sym] = float(p.market_value)
    return out


def wait_filled(client, order, timeout=90):
    for _ in range(timeout):
        o = client.get_order_by_id(order.id)
        if o.status.value == "filled":
            return o
        if o.status.value in ("canceled", "expired", "rejected"):
            raise RuntimeError(f"order for {o.symbol} was {o.status.value}")
        time.sleep(1)
    raise TimeoutError(f"order for {order.symbol} not filled after {timeout}s")


def trade(client, account, symbol, side, dollars=None, everything=False, dry_run=False, why=""):
    """Buy or sell by dollar amount, or sell the whole position. Returns the fill."""
    if dry_run:
        log(account, action=f"would {side}", symbol=symbol,
            dollars="all" if everything else round(dollars, 2), why=why)
        return None
    if everything:
        order = client.close_position(symbol.replace("/", ""))
    else:
        order = client.submit_order(MarketOrderRequest(
            symbol=symbol, notional=round(dollars, 2),
            time_in_force=TimeInForce.GTC if is_crypto(symbol) else TimeInForce.DAY,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL))
    fill = wait_filled(client, order)
    qty, price = float(fill.filled_qty), float(fill.filled_avg_price)
    log(account, action=side, symbol=symbol, qty=qty, price=price,
        dollars=round(qty * price, 2), why=why)
    return fill


def rebalance(client, account, target, dry_run, why):
    """Move the account to `target` weights (symbol -> fraction of the account). Sells
    first, to free cash. Skips small top-ups and trims, so tiny drifts don't cost fees."""
    equity = float(client.get_account().equity)
    want = {s: w * equity * (1 - CASH_BUFFER) for s, w in target.items() if w > 1e-4}
    small = SMALL_TRADE * equity
    held = holdings(client)
    for s, value in held.items():
        if s not in want:
            trade(client, account, s, "sell", everything=True, dry_run=dry_run, why=why)
        elif value - want[s] > small:
            trade(client, account, s, "sell", value - want[s], dry_run=dry_run, why=why)
    if not dry_run:
        held = holdings(client)
    for s, value in want.items():
        gap = value - held.get(s, 0.0)
        if gap > small or (s not in held and gap >= MIN_ORDER):
            trade(client, account, s, "buy", gap, dry_run=dry_run, why=why)


# --- the two daily accounts: multi-strategy fund and learning allocator ---------------

def sleeve_inputs(keep_today=True):
    """Every sleeve's daily returns and its decision at the latest close (today's latest
    price at 15:40), on the stock-trading calendar."""
    d = load_daily(refresh=True, keep_today=keep_today)
    rf = (d["close"]["^IRX"].ffill().shift(1) / 100 / 252).fillna(0.0)
    built = rv.build(d, rf)
    R = pd.DataFrame({k: built[k][0]["ret"] for k in SLEEVES})
    T = pd.DataFrame({k: built[k][0]["target"] for k in SLEEVES}).ffill().fillna(0.0)
    return R, T, rf


def learner_budgets(R, rf):
    """The Hedge allocator, learning from scratch since LAUNCH_V2: it sees just enough
    earlier history to estimate each sleeve's volatility, then starts with equal trust.
    Returns (budget history, tomorrow's budgets)."""
    i = R.index.searchsorted(min(LAUNCH_V2, R.index[-1]))  # (a practice run before launch day)
    window = R.iloc[max(i - al.WARMUP, 0):]
    tomorrow = pd.DataFrame(np.nan, index=[window.index[-1] + pd.Timedelta(days=1)], columns=R.columns)
    budgets, _ = al.hedge(pd.concat([window, tomorrow]), rf)
    return budgets.iloc[:-1], budgets.iloc[-1]


def instrument_targets(budgets, signals):
    """Turn sleeve budgets and today's sleeve signals into account weights per symbol.
    The overnight sleeve's TQQQ is bought now and sold at tomorrow's open."""
    day = budgets[TREND] * signals[TREND] + budgets[DIP] * signals[DIP]
    night = day + budgets[OVERNIGHT] * signals[OVERNIGHT]
    return {"TQQQ": night, "SVXY": budgets[SHORT_VOL] * signals[SHORT_VOL],
            "BTC/USD": budgets[CRYPTO] * signals[CRYPTO]}, day


def daily(dry_run):
    if not dry_run and not trading_client("moderate").get_clock().is_open:
        print("market is closed; nothing to do")
        return
    today = now_et().normalize().tz_localize(None)
    R, T, rf = sleeve_inputs()
    if R.index[-1] != today and not dry_run:
        print(f"no price for today yet (latest {R.index[-1].date()}); trying again on the next run")
        return
    signals = T.iloc[-1]
    _, learned = learner_budgets(R, rf)
    plans = {"moderate": (al.equal_weight(R).iloc[-1], "equal weight"),
             "safe": (learned, "learned (Hedge)")}
    for account, (budgets, how) in plans.items():
        if logged_today(account, ("rebalanced",)):
            print(f"[{account}] already rebalanced today")
            continue
        targets, day_tqqq = instrument_targets(budgets, signals)
        client = trading_client(account)
        rebalance(client, account, targets, dry_run, why=f"daily rebalance ({how})")
        log(account, action="rebalanced", date=str(R.index[-1].date()), how=how,
            budgets={k: round(float(v), 4) for k, v in budgets.items()},
            signals={k: round(float(v), 4) for k, v in signals.items()},
            targets={k: round(float(v), 4) for k, v in targets.items()},
            day_tqqq=round(float(day_tqqq), 4), equity=float(client.get_account().equity))
        persist(f"{account}: daily rebalance")


def unwind(dry_run):
    """At the open, sell the TQQQ each daily account bought for the overnight sleeve,
    keeping whatever the daytime sleeves want."""
    plan = schedule(now_et().date())
    if plan is None:
        print("market is closed today")
        return
    if not dry_run:
        sleep_until(plan[0] + pd.Timedelta(seconds=30))
    for account in ("moderate", "safe"):
        if logged_today(account, ("unwound",)):
            continue
        rebs = [r for r in read_log(account) if r.get("action") == "rebalanced"]
        if not rebs:
            continue
        keep = rebs[-1]["day_tqqq"]
        client = trading_client(account)
        equity, tq = float(client.get_account().equity), holdings(client).get("TQQQ", 0.0)
        excess = tq - keep * equity * (1 - CASH_BUFFER)
        if excess >= MIN_ORDER:
            trade(client, account, "TQQQ", "sell", everything=keep < 1e-4,
                  dollars=None if keep < 1e-4 else excess, dry_run=dry_run,
                  why="overnight sleeve: sell at the open")
        log(account, action="unwound", sold=round(max(excess, 0.0), 2) if excess >= MIN_ORDER else 0.0)
        persist(f"{account}: overnight unwind")


# --- the day trader: noise-area breakout ------------------------------------------------

def _ny(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize(TZ) if ts.tz is None else ts.tz_convert(TZ)


def recent_bars(start, end, symbols=("QQQ",), feed=DataFeed.SIP):
    """5-minute bars from `start` up to (not including) `end`, official hours only."""
    req = StockBarsRequest(symbol_or_symbols=list(symbols), timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                           start=_ny(start).to_pydatetime(), end=_ny(end).to_pydatetime(), feed=feed)
    bars = data_client().get_stock_bars(req).df.reset_index()
    if bars.empty:
        return bars
    bars["timestamp"] = bars["timestamp"].dt.tz_convert(TZ)
    return session_only(bars)


def nam_state(history, at):
    """(price, upper band, lower band, VWAP) for the decision at `at`: the bands come from
    full-market history through yesterday plus today's live (IEX) bars finished by `at`.
    All NaN if today's bars haven't arrived, which makes the rule hold its position."""
    today = recent_bars(at.normalize(), at, feed=DataFeed.IEX)
    if today.empty:
        return (np.nan,) * 4
    today = today[today["timestamp"] < at]
    pan = panels(pd.concat([history, today], ignore_index=True))["QQQ"]
    if pan["close"].index[-1] != pd.Timestamp(at.date()):
        return (np.nan,) * 4
    b = iv.noise_bands(pan)
    k = (at.hour * 60 + at.minute - (9 * 60 + 30)) // 5  # the decision's bar; use the one before
    return tuple(float(b[f][-1, k - 1]) for f in ("C", "upper", "lower", "vwap"))


def fund_position(client):
    """+1 holding TQQQ, -1 holding SQQQ, 0 in cash. Anything else gets sold first."""
    held = holdings(client)
    for sym in list(held):
        if sym not in (LONG_FUND, SHORT_FUND):
            trade(client, "daytrader", sym, "sell", everything=True, why="not part of this strategy")
            held.pop(sym)
    if not held:
        return 0
    return 1 if max(held, key=held.get) == LONG_FUND else -1


def daytrader(session, dry_run):
    client = trading_client("daytrader")
    today = now_et().date()
    plan = schedule(today)
    if plan is None:
        print("market is closed today")
        return
    opens, times, close = plan
    noon = pd.Timestamp(today, tz=TZ) + AFTERNOON
    mine = [t for t in times if (t < noon) == (session == "morning")
            and not logged_today("daytrader", ("decision", "skipped"), decision=t.strftime("%H:%M"))]
    close_out = close - CLOSE_OUT
    i_close = (close_out < noon) == (session == "morning") and not logged_today("daytrader", ("day closed",))
    if not mine and not i_close:
        print(f"nothing left to do in the {session} session today")
        return
    history = recent_bars(pd.Timestamp(today) - pd.Timedelta(days=45), pd.Timestamp(today))

    if session == "morning" and now_et() < times[0]:
        sleep_until(opens + pd.Timedelta(minutes=1))
        if holdings(client):  # every day starts flat, as in the backtest
            for sym in holdings(client):
                trade(client, "daytrader", sym, "sell", everything=True, dry_run=dry_run,
                      why="leftover position from a previous day")

    names = {1: LONG_FUND, -1: SHORT_FUND, 0: "cash"}
    for t in mine:
        if now_et() > t + LATE:
            log("daytrader", action="skipped", decision=t.strftime("%H:%M"), why="run started late")
            continue
        sleep_until(t + pd.Timedelta(seconds=15))  # let the bar that ended at t publish
        price, upper, lower, vwap = nam_state(history, t)
        pos = fund_position(client) if not dry_run else 0
        want = iv.nam_decide(pos, price, upper, lower, vwap)
        why = f"{t:%H:%M} decision"
        if want != pos:
            if pos:
                trade(client, "daytrader", names[pos], "sell", everything=True, dry_run=dry_run, why=why)
            if want:
                cash = float(client.get_account().buying_power) * (1 - CASH_BUFFER)
                trade(client, "daytrader", names[want], "buy", cash, dry_run=dry_run, why=why)
        log("daytrader", action="decision", decision=t.strftime("%H:%M"), strategy="noise-area",
            qqq=round(price, 2), upper=round(upper, 2), lower=round(lower, 2), vwap=round(vwap, 2),
            before=names[pos], after=names[want], dry_run=dry_run)
        persist(f"daytrader: {t:%H:%M} decision")

    if i_close:
        sleep_until(close_out)
        for sym in holdings(client):
            trade(client, "daytrader", sym, "sell", everything=True, dry_run=dry_run,
                  why="end of day: never hold overnight")
        log("daytrader", action="day closed", equity=float(client.get_account().equity))
        persist("daytrader: end of day")


def replay(day):
    """Rerun a past day's decisions with the live data path (IEX bars) and compare with
    the backtest's decisions on full-market data for the same day. Places no orders."""
    day = pd.Timestamp(day)
    plan = schedule(day.date())
    if plan is None:
        print("market was closed that day")
        return
    history = recent_bars(day - pd.Timedelta(days=45), day)
    trace = []
    pan = {**panels(load_bars()), **panels(load_bars(symbols=["TQQQ", "SQQQ"], name="lev"))}
    iv.noise_area(pan, pd.Series(dtype=float), sizing="full", trace=trace)
    test = {k: r for r in trace if r[0] == day for k in [r[1]]}
    pos = 0
    print(f"{'time':>5}  {'live QQQ':>9} {'upper':>8} {'lower':>8} {'vwap':>8}  {'live':>5}  "
          f"{'test QQQ':>9} {'upper':>8} {'lower':>8}  {'test':>5}")
    for t in plan[1]:
        price, upper, lower, vwap = nam_state(history, t)
        want = iv.nam_decide(pos, price, upper, lower, vwap)
        k = (t.hour * 60 + t.minute - 570) // 5
        r = test.get(k)
        print(f"{t:%H:%M}  {price:9.2f} {upper:8.2f} {lower:8.2f} {vwap:8.2f}  {want:+5d}  "
              + (f"{r[2]:9.2f} {r[3]:8.2f} {r[4]:8.2f}  {r[7]:+5d}" if r else "   (no backtest row)"))
        pos = want
    print("(+1 = hold TQQQ, -1 = hold SQQQ, 0 = cash)")


def safety():
    """Run by the dashboard job: in the last 10 minutes of the day, if the day trader
    still holds something and its final decision never got logged (so the afternoon run
    probably died), sell rather than hold overnight."""
    plan = schedule(now_et().date())
    if plan is None:
        return
    _, times, close = plan
    if not close - pd.Timedelta(minutes=10) <= now_et() < close:
        return
    if logged_today("daytrader", ("decision", "skipped"), decision=times[-1].strftime("%H:%M")):
        return  # the session is alive and will close out itself
    client = trading_client("daytrader")
    for sym in holdings(client):
        trade(client, "daytrader", sym, "sell", everything=True, why="safety net: session missing")
    log("daytrader", action="day closed", equity=float(client.get_account().equity))
    persist("daytrader: safety close-out")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("daily", "unwind"):
        sub.add_parser(name).add_argument("--dry-run", action="store_true",
                                          help="log intended orders without placing them")
    d = sub.add_parser("daytrader")
    d.add_argument("--session", choices=["morning", "afternoon"])
    d.add_argument("--replay", metavar="YYYY-MM-DD", help="rerun a past day without trading")
    d.add_argument("--dry-run", action="store_true")
    f = sub.add_parser("flatten")
    f.add_argument("account", choices=["daytrader", "moderate", "safe"])
    sub.add_parser("safety", help="sell the day trader's position if its session died")
    args = parser.parse_args()
    if getattr(args, "dry_run", False):
        global LOGS  # keep practice runs out of the real logs
        LOGS = LOGS.parent / "logs_dryrun"

    if args.cmd == "daily":
        daily(args.dry_run)
    elif args.cmd == "unwind":
        unwind(args.dry_run)
    elif args.cmd == "daytrader" and args.replay:
        replay(args.replay)
    elif args.cmd == "daytrader":
        if not args.session:
            parser.error("daytrader needs --session or --replay")
        daytrader(args.session, args.dry_run)
    elif args.cmd == "safety":
        safety()
    else:
        trading_client(args.account).close_all_positions(cancel_orders=True)
        log(args.account, action="flattened", why="manual or safety close-out")
        persist(f"{args.account}: flattened")


if __name__ == "__main__":
    main()
