"""Live paper trading on Alpaca, using exactly the backtested rules.

    python live.py monthly [--dry-run]                          # ML moderate + ML safe
    python live.py daytrader --session morning|afternoon [--dry-run]
    python live.py daytrader --replay 2026-10-09                # rerun a past day, no orders
    python live.py flatten daytrader                            # emergency: sell everything
    python live.py safety                                       # sell if the afternoon run died

GitHub Actions runs these on a schedule (.github/workflows). Every decision is appended
to logs/<account>.jsonl, which the dashboard reads.
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

import daytrader as dt
from broker import data_client, trading_client
from data import load_prices
from intraday_data import SYMBOLS, TZ, load_bars, market_calendar, panels, session_only
from ml import ml_sectors, walk_forward

LOGS = Path(__file__).parent / "logs"
CASH_BUFFER = 0.005   # leave 0.5% in cash so a small price move can't make a buy fail
MIN_ORDER = 1.0       # Alpaca's smallest dollar order
DT_COST_BPS = 1.0     # must match run_daytrader.COST_BPS: it sets the entry threshold
LATE = pd.Timedelta(minutes=5)       # skip a decision if we reach it later than this
AFTERNOON = pd.Timedelta(hours=13, minutes=15)  # decisions from 13:15 belong to the afternoon
CLOSE_OUT = pd.Timedelta(minutes=3)  # sell everything this long before the close


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
    """Has an entry like this been logged today? Lets a duplicate scheduled run skip
    work another run already did."""
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


# --- orders ---------------------------------------------------------------------------

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
        order = client.close_position(symbol)
    else:
        order = client.submit_order(MarketOrderRequest(
            symbol=symbol, notional=round(dollars, 2), time_in_force=TimeInForce.DAY,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL))
    fill = wait_filled(client, order)
    qty, price = float(fill.filled_qty), float(fill.filled_avg_price)
    log(account, action=side, symbol=symbol, qty=qty, price=price,
        dollars=round(qty * price, 2), why=why)
    return fill


def holdings(client):
    return {p.symbol: float(p.market_value) for p in client.get_all_positions()}


# --- monthly strategies -----------------------------------------------------------------

def rebalance(client, account, target, dry_run, why):
    """Move the account to `target` weights (ticker -> fraction). Sells first, to free cash."""
    equity = float(client.get_account().equity)
    want = {s: w * equity * (1 - CASH_BUFFER) for s, w in target.items() if w > 0}
    for s, value in holdings(client).items():
        if s not in want:
            trade(client, account, s, "sell", everything=True, dry_run=dry_run, why=why)
        elif value - want[s] >= MIN_ORDER:
            trade(client, account, s, "sell", value - want[s], dry_run=dry_run, why=why)
    held = holdings(client)
    for s, value in want.items():
        if value - held.get(s, 0.0) >= MIN_ORDER:
            trade(client, account, s, "buy", value - held.get(s, 0.0), dry_run=dry_run, why=why)


def monthly(dry_run):
    if not dry_run and not trading_client("moderate").get_clock().is_open:
        print("market is closed; nothing to do")
        return
    prices, rf = load_prices(refresh=True)
    scores, _ = walk_forward(prices, rf, "ridge")
    signal = scores.index[-1]  # the latest finished month-end
    for account, crash_filter in (("moderate", False), ("safe", True)):
        if any(r.get("action") == "rebalanced" and r.get("signal_date") == str(signal.date())
               for r in read_log(account)):
            print(f"[{account}] already rebalanced for the {signal.date()} signal")
            continue
        target = ml_sectors(scores, prices, crash_filter=crash_filter).loc[signal]
        why = f"monthly rebalance, {signal.date()} signal"
        client = trading_client(account)
        rebalance(client, account, target, dry_run, why)
        if not dry_run:
            log(account, action="rebalanced", signal_date=str(signal.date()),
                target={s: round(w, 4) for s, w in target.items() if w > 0},
                equity=float(client.get_account().equity))
            persist(f"{account}: rebalance for {signal.date()} signal")


# --- day trader -------------------------------------------------------------------------

def schedule(day):
    """The day's (open, decision times, close), or None if the market is closed."""
    cal = trading_client().get_calendar(GetCalendarRequest(start=day, end=day))
    if not cal or cal[0].date != day:
        return None
    opens = pd.Timestamp(cal[0].open).tz_localize(TZ)
    close = pd.Timestamp(cal[0].close).tz_localize(TZ)
    times, t = [], opens + pd.Timedelta(minutes=30)
    while t + pd.Timedelta(minutes=30) <= close:  # same blocks as the backtest
        times.append(t)
        t += pd.Timedelta(minutes=30)
    return opens, times, close


MODEL_LOG = "daytrader_model"  # logs/daytrader_model.jsonl: what the model believes each morning


def recent_bars(start, end):
    """Full-market 5-minute bars from `start` up to (not including) `end`'s date."""
    req = StockBarsRequest(symbol_or_symbols=SYMBOLS, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                           start=pd.Timestamp(start, tz=TZ).to_pydatetime(),
                           end=pd.Timestamp(end, tz=TZ).to_pydatetime(), feed=DataFeed.SIP)
    bars = data_client().get_stock_bars(req).df.reset_index()
    bars["timestamp"] = bars["timestamp"].dt.tz_convert(TZ)
    return session_only(bars)


def daytrader_model(day, seed=None, record=True):
    """The learn-from-scratch model for `day`: trained on every trading day from its seed
    start (the 5 trading days before its first live day) through yesterday, and never
    forgetting anything. Each morning's weights are logged so the dashboard can show
    how its beliefs change."""
    day = pd.Timestamp(day)
    history = read_log(MODEL_LOG)
    if seed is None:
        seed = (pd.Timestamp(history[0]["seed_start"]) if history
                else dt.seed_start(market_calendar().index, day))
    # Fetch extra days before the seed: some features look back up to 20 trading days.
    bars = recent_bars(seed - pd.Timedelta(days=45), day)
    X, y = dt.features(panels(bars))
    when = X.index.get_level_values("date")
    keep = np.asarray((when >= seed) & (when < day))
    model = dt.train(X[keep], y[keep], "ridge")
    if record and not logged_today(MODEL_LOG, ("retrained",)):
        weights = pd.Series(model[-1].coef_ * 1e4, index=X.columns)
        log(MODEL_LOG, action="retrained", seed_start=str(seed.date()), examples=int(keep.sum()),
            days_learned=int(when[keep].nunique()),
            weights={k: round(v, 4) for k, v in weights.items()},
            intercept_bps=round(float(model[-1].intercept_) * 1e4, 4))
        persist("daytrader: morning retrain")
    return model


def live_features(at):
    """Feature rows (one per symbol) for decision time `at`, from bars finished before it.
    Live prices come from IEX, the real-time feed on Alpaca's free plan."""
    req = StockBarsRequest(symbol_or_symbols=SYMBOLS, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                           start=(at - pd.Timedelta(days=50)).to_pydatetime(),
                           end=at.to_pydatetime(), feed=DataFeed.IEX)
    bars = data_client().get_stock_bars(req).df.reset_index()
    bars["timestamp"] = bars["timestamp"].dt.tz_convert(TZ)
    bars = session_only(bars[bars["timestamp"] < at])
    X, _ = dt.features(panels(bars), keep_unlabeled=True)
    slot = (at.hour * 60 + at.minute - (9 * 60 + 30)) // 5
    key = (pd.Timestamp(at.date()), slot)
    return X.xs(key, level=("date", "slot")) if key in X.droplevel("symbol").index else X.iloc[:0]


def predict(model, at):
    rows = live_features(at)
    return {s: float(model.predict(rows.loc[[s]].to_numpy())[0]) if s in rows.index else np.nan
            for s in SYMBOLS}


def position(client):
    held = holdings(client)
    return max(held, key=held.get) if held else ""


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
    model = daytrader_model(today)

    if session == "morning" and now_et() < times[0]:
        sleep_until(opens + pd.Timedelta(minutes=1))
        if pos := position(client):  # every day starts in cash, as in the backtest
            trade(client, "daytrader", pos, "sell", everything=True, dry_run=dry_run,
                  why="leftover position from a previous day")

    for t in mine:
        if now_et() > t + LATE:
            log("daytrader", action="skipped", decision=t.strftime("%H:%M"), why="run started late")
            continue
        sleep_until(t + pd.Timedelta(seconds=15))  # let the bar that ended at t publish
        preds = predict(model, t)
        pos = position(client)
        new = dt.choose(preds, pos, DT_COST_BPS)
        why = f"{t:%H:%M} decision"
        if new != pos:
            if pos:
                trade(client, "daytrader", pos, "sell", everything=True, dry_run=dry_run, why=why)
            if new:
                cash = float(client.get_account().buying_power) * (1 - CASH_BUFFER)
                trade(client, "daytrader", new, "buy", cash, dry_run=dry_run, why=why)
        log("daytrader", action="decision", decision=t.strftime("%H:%M"),
            predicted_bps={s: round(p * 1e4, 2) for s, p in preds.items()},
            before=pos or "cash", after=new or "cash", dry_run=dry_run)
        persist(f"daytrader: {t:%H:%M} decision")

    if i_close:
        sleep_until(close_out)
        pos = position(client)
        if pos:
            trade(client, "daytrader", pos, "sell", everything=True, dry_run=dry_run,
                  why="end of day: never hold overnight")
        log("daytrader", action="day closed", equity=float(client.get_account().equity))
        persist("daytrader: end of day")


def safety():
    """Safety net, run by the dashboard job: in the last 10 minutes of the day, if the
    day trader still holds a position and its final decision never got logged (so the
    afternoon session probably died), sell rather than hold overnight."""
    plan = schedule(now_et().date())
    if plan is None:
        return
    _, times, close = plan
    if not close - pd.Timedelta(minutes=10) <= now_et() < close:
        return
    if logged_today("daytrader", ("decision", "skipped"), decision=times[-1].strftime("%H:%M")):
        return  # the session is alive and will close out itself
    client = trading_client("daytrader")
    if pos := position(client):
        trade(client, "daytrader", pos, "sell", everything=True, why="safety net: session missing")
        log("daytrader", action="day closed", equity=float(client.get_account().equity))
        persist("daytrader: safety close-out")


def replay(day):
    """Rerun a past day's decisions with the live pipeline (IEX data), placing no orders,
    and compare with the backtest's predictions (full-market data) for the same blocks."""
    day = pd.Timestamp(day).date()
    plan = schedule(day)
    if plan is None:
        print("market was closed that day")
        return
    # As if that day were its first: seeded with the 5 trading days before it.
    model = daytrader_model(day, seed=dt.seed_start(market_calendar().index, day), record=False)
    X, _ = dt.features(panels(load_bars()))  # the backtest's full-market features
    pos = ""
    print(f"{'time':>5}  {'live SPY':>9} {'live QQQ':>9}  {'test SPY':>9} {'test QQQ':>9}  action")
    for t in plan[1]:
        live = predict(model, t)
        slot = (t.hour * 60 + t.minute - 570) // 5
        rows = X.xs((pd.Timestamp(day), slot), level=("date", "slot"))
        test = {s: float(model.predict(rows.loc[[s]].to_numpy())[0]) for s in rows.index}
        new = dt.choose(live, pos, DT_COST_BPS)
        print(f"{t:%H:%M}  " + " ".join(f"{live[s] * 1e4:+9.2f}" for s in SYMBOLS) + "  "
              + " ".join(f"{test.get(s, np.nan) * 1e4:+9.2f}" for s in SYMBOLS)
              + f"  {pos or 'cash'} -> {new or 'cash'}")
        pos = new
    print("(predictions in bps of expected 30-minute return)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("monthly")
    m.add_argument("--dry-run", action="store_true", help="log intended orders without placing them")
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

    if args.cmd == "monthly":
        monthly(args.dry_run)
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
