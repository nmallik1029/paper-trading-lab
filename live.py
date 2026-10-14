"""Live paper trading on Alpaca (lab v2), using exactly the backtested rules.

    python live.py daily [--dry-run]           # 15:40: the learning allocator's rebalance
    python live.py unwind [--dry-run]          # 9:30: sell its overnight TQQQ at the open
    python live.py daytrader --session morning|afternoon [--dry-run]   # both breakout day traders
    python live.py daytrader --replay 2026-10-09     # rerun a past day with live data, no orders
    python live.py flatten daytrader|moderate|safe   # emergency: sell everything
    python live.py safety                      # sell the day trader's position if its session died
    python live.py watchdog                    # restart the day traders if their session died

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
LATE = pd.Timedelta(minutes=5)       # skip a decision that is reached later than this
AFTERNOON = pd.Timedelta(hours=13, minutes=15)  # decisions from 13:15 belong to the afternoon run
CLOSE_OUT = pd.Timedelta(minutes=3)  # the day trader sells everything this long before the close
REBALANCE_BEFORE_CLOSE = pd.Timedelta(minutes=20)  # the daily accounts rebalance at 15:40

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


def retry(fn, *args, tries=3, wait=15, **kwargs):
    """Call fn, retrying a couple of times on errors (network blips, API hiccups)."""
    for attempt in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == tries - 1:
                raise
            print(f"retrying {getattr(fn, '__name__', 'call')} after error: {e}")
            time.sleep(wait)


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


def trade(client, account, symbol, side, dollars=None, everything=False, dry_run=False, why="", timeout=90):
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
    fill = wait_filled(client, order, timeout)
    qty, price = float(fill.filled_qty), float(fill.filled_avg_price)
    log(account, action=side, symbol=symbol, qty=qty, price=price,
        dollars=round(qty * price, 2), why=why)
    return fill


def sell_all(client, account, dry_run, why, timeout=90):
    """Sell every position, logging (not raising) any order that fails or is slow to fill,
    so one stuck order can't stop the rest of a run."""
    for sym in holdings(client):
        try:
            trade(client, account, sym, "sell", everything=True, dry_run=dry_run, why=why, timeout=timeout)
        except Exception as e:
            log(account, action="error", symbol=sym, error=str(e)[:300], why=why)


def rebalance(client, account, target, dry_run, why):
    """Move the account to `target` weights (symbol -> fraction of the account). Sells
    first, to free cash. Skips small top-ups and trims, so tiny drifts don't cost fees."""
    equity = float(client.get_account().equity)
    want = {s: w * equity * (1 - CASH_BUFFER) for s, w in target.items() if w > 1e-4}
    small = SMALL_TRADE * equity
    errors = []

    def attempt(*args, **kwargs):  # one failed order shouldn't stop the rest
        try:
            trade(client, account, *args, dry_run=dry_run, why=why, **kwargs)
        except Exception as e:
            errors.append(f"{args[0]}: {e}")
            log(account, action="error", symbol=args[0], error=str(e)[:300], why=why)

    held = holdings(client)
    for s, value in held.items():
        if s not in want:
            attempt(s, "sell", everything=True)
        elif value - want[s] > small:
            attempt(s, "sell", value - want[s])
    if not dry_run:
        held = holdings(client)
    for s, value in want.items():
        gap = value - held.get(s, 0.0)
        if gap > small or (s not in held and gap >= MIN_ORDER):
            if is_crypto(s) and not dry_run:
                # Alpaca only buys crypto with settled cash; sale proceeds settle the next day.
                settled = float(client.get_account().non_marginable_buying_power) * (1 - CASH_BUFFER)
                if settled < MIN_ORDER:
                    log(account, action="note", symbol=s, why="waiting for settled cash to buy crypto")
                    continue
                gap = min(gap, settled)
            attempt(s, "buy", gap)
    return errors


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
    if not dry_run:
        clock = trading_client("moderate").get_clock()
        if not clock.is_open:
            print("market is closed; nothing to do")
            return
        if pd.Timestamp(clock.next_close).tz_convert(TZ) - now_et() < pd.Timedelta(minutes=5):
            print("too close to the close to rebalance safely; skipping")
            return
    today = now_et().normalize().tz_localize(None)
    R, T, rf = sleeve_inputs()
    if R.index[-1] != today and not dry_run:
        print(f"no price for today yet (latest {R.index[-1].date()}); trying again on the next run")
        return
    signals = T.iloc[-1]
    _, learned = learner_budgets(R, rf)
    plans = {"moderate": (learned, "learned (Hedge)")}  # the learning allocator's account
    for account, (budgets, how) in plans.items():
        if logged_today(account, ("rebalanced",)):
            print(f"[{account}] already rebalanced today")
            continue
        targets, day_tqqq = instrument_targets(budgets, signals)
        client = trading_client(account)
        errors = rebalance(client, account, targets, dry_run, why=f"daily rebalance ({how})")
        log(account, action="rebalanced", date=str(R.index[-1].date()), how=how, errors=errors,
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
    for account in ("moderate",):
        if logged_today(account, ("unwound",)):
            continue
        rebs = [r for r in read_log(account) if r.get("action") == "rebalanced" and "day_tqqq" in r]
        if not rebs:  # (v1 rebalances had no overnight sleeve)
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


# The QQQ breakout trader (day-trader account): all-in on one market, half-hour clock.
QQQ_MARKET = {"QQQ": (LONG_FUND, SHORT_FUND, 3.0)}
# The staggered breakout portfolio ("safe" account): three markets, each on three half-hour
# clocks offset by 10 minutes, nine equal slices. Offsets are in 5-minute bars.
STAGGER_MARKETS = {"QQQ": ("TQQQ", "SQQQ", 3.0), "SPY": ("UPRO", "SPXU", 3.0), "SMH": ("SOXL", "SOXS", 3.0)}
STAGGER_OFFSETS = (0, 2, 4)
DAY_ACCOUNTS = ("daytrader", "safe")
DAY_SYMBOLS = sorted(STAGGER_MARKETS)


def nam_states(history, at, symbols=DAY_SYMBOLS):
    """{market: (price, upper band, lower band, VWAP)} for the decision at `at`. Bands come
    from full-market history through yesterday plus today's live (IEX) bars finished by
    `at`. All NaN for a market whose bars haven't arrived, which makes the rule hold."""
    blank = {m: (np.nan,) * 4 for m in symbols}
    today = recent_bars(at.normalize(), at, symbols=symbols, feed=DataFeed.IEX)
    if today.empty:
        return blank
    today = today[today["timestamp"] < at]
    pans = panels(pd.concat([history, today], ignore_index=True))
    k = (at.hour * 60 + at.minute - (9 * 60 + 30)) // 5  # the decision's bar; use the one before
    out = dict(blank)
    for m in symbols:
        if m in pans and pans[m]["close"].index[-1] == pd.Timestamp(at.date()):
            b = iv.noise_bands(pans[m])
            out[m] = tuple(float(b[f][-1, k - 1]) for f in ("C", "upper", "lower", "vwap"))
    return out


def nam_state(history, at):
    """QQQ's (price, upper, lower, VWAP) at `at` (used by the replay)."""
    return nam_states(history, at, symbols=["QQQ"])["QQQ"]


def day_marks(opens, close):
    """Every 10-minute decision mark of the day, with its clock offset: (time, offset).
    Matches the backtest: bar k = 6, 8, 10, ... up to the day's last bar."""
    last = int((close - opens).total_seconds() // 300) - 1
    return [(opens + pd.Timedelta(minutes=5 * k), (k - 6) % 6) for k in range(6, last + 1, 2)]


def clock_name(offset):
    return f":{offset * 5:02d}/:{30 + offset * 5:02d}"


def slices_file():
    """Each staggered slice's open position, so its exits sell only its own shares."""
    return LOGS / "safe_slices.json"


def load_slices(today):
    if slices_file().exists():
        st = json.loads(slices_file().read_text(encoding="utf-8"))
        if st.get("date") == str(today):
            return st
    return {"date": str(today), "budget": None, "slices": {}}


def save_slices(st):
    LOGS.mkdir(exist_ok=True)
    slices_file().write_text(json.dumps(st, indent=1), encoding="utf-8")


def sell_shares(client, account, symbol, qty, dry_run, why):
    """Sell `qty` shares of `symbol` (all of them if that's about what's held)."""
    if dry_run:
        return trade(client, account, symbol, "sell", dollars=0.0, dry_run=True, why=why)
    try:
        held = float(client.get_open_position(symbol).qty)
    except Exception:  # already sold (a safety close-out, or a manual flatten)
        log(account, action="note", symbol=symbol, why="position already closed")
        return None
    if qty >= held * 0.999:
        return trade(client, account, symbol, "sell", everything=True, why=why)
    order = client.submit_order(MarketOrderRequest(symbol=symbol, qty=round(qty, 9), side=OrderSide.SELL,
                                                   time_in_force=TimeInForce.DAY))
    fill = wait_filled(client, order)
    q, price = float(fill.filled_qty), float(fill.filled_avg_price)
    log(account, action="sell", symbol=symbol, qty=q, price=price, dollars=round(q * price, 2), why=why)
    return fill


def reconcile_slices(client, st, dry_run):
    """Make the slices' records agree with what the account actually holds, in case a
    previous run stopped before saving: shrink claims to the shares that exist, and sell
    any fund no slice claims."""
    if dry_run:
        return
    held = {p.symbol: float(p.qty) for p in client.get_all_positions()}
    claims = {}
    for key, sl in st["slices"].items():
        if sl.get("side"):
            claims.setdefault(sl["fund"], []).append(key)
    for fund, keys in claims.items():
        claimed = sum(st["slices"][k].get("qty", 0.0) for k in keys)
        actual = held.get(fund, 0.0)
        if actual <= 0:
            for k in keys:
                st["slices"][k] = {"side": 0}
        elif claimed > 0 and actual < claimed * 0.999:
            for k in keys:
                st["slices"][k]["qty"] *= actual / claimed
    for sym in held:
        if sym not in claims:
            try:
                trade(client, "safe", sym, "sell", everything=True, why="not held by any slice")
            except Exception as e:
                log("safe", action="error", symbol=sym, error=str(e)[:300])
    save_slices(st)


def qqq_decision(client, states, t, dry_run):
    """The QQQ breakout trader at a half-hour mark: all-in TQQQ, SQQQ or cash."""
    why = f"{t:%H:%M} decision"
    held = holdings(client)
    for sym in list(held):
        if sym not in (LONG_FUND, SHORT_FUND):
            trade(client, "daytrader", sym, "sell", everything=True, dry_run=dry_run, why="not part of this strategy")
            held.pop(sym)
    pos = (1 if LONG_FUND in held else -1 if SHORT_FUND in held else 0) if not dry_run else 0
    names = {1: LONG_FUND, -1: SHORT_FUND, 0: "cash"}
    price, upper, lower, vwap = states["QQQ"]
    want = iv.nam_decide(pos, price, upper, lower, vwap)
    if want != pos:
        if pos:
            trade(client, "daytrader", names[pos], "sell", everything=True, dry_run=dry_run, why=why)
        if want:
            cash = float(client.get_account().buying_power) * (1 - CASH_BUFFER)
            trade(client, "daytrader", names[want], "buy", cash, dry_run=dry_run, why=why)
    log("daytrader", action="decision", decision=t.strftime("%H:%M"), strategy="noise-area",
        qqq=round(price, 2), upper=round(upper, 2), lower=round(lower, 2), vwap=round(vwap, 2),
        before=names[pos], after=names[want], dry_run=dry_run)


def stagger_decision(client, st, states, t, offset, dry_run):
    """The staggered portfolio at one 10-minute mark: the three slices on this mark's clock
    each apply the breakout rule with their own budget and their own shares."""
    why = f"{t:%H:%M} decision ({clock_name(offset)} clock)"
    if st["budget"] is None:
        st["budget"] = float(client.get_account().equity) * (1 - CASH_BUFFER) / (len(STAGGER_MARKETS) * len(STAGGER_OFFSETS))
    detail = {}
    for m, (lf, sf, _) in STAGGER_MARKETS.items():
        key = f"{m} {clock_name(offset)}"
        cur = st["slices"].get(key, {"side": 0})
        want = iv.nam_decide(cur["side"], *states[m])
        names = {1: lf, -1: sf, 0: "cash"}
        if want != cur["side"]:
            if cur["side"]:
                sell_shares(client, "safe", cur["fund"], cur["qty"], dry_run, why)
                st["slices"][key] = {"side": 0}
            if want:
                cash = float(client.get_account().buying_power) * (1 - CASH_BUFFER)
                dollars = min(st["budget"], cash)
                if dollars >= MIN_ORDER:
                    fill = trade(client, "safe", names[want], "buy", dollars, dry_run=dry_run, why=why)
                    qty = float(fill.filled_qty) if fill is not None else 0.0
                    st["slices"][key] = {"side": want, "fund": names[want], "qty": qty}
        detail[m] = {"price": round(states[m][0], 2), "upper": round(states[m][1], 2),
                     "lower": round(states[m][2], 2), "vwap": round(states[m][3], 2),
                     "before": names[cur["side"]], "after": names[st["slices"].get(key, {"side": 0})["side"]]}
    save_slices(st)
    log("safe", action="decision", decision=t.strftime("%H:%M"), strategy="staggered-breakout",
        clock=clock_name(offset), markets=detail, dry_run=dry_run)


def refresh_dashboard():
    """On GitHub Actions, ask for a dashboard rebuild so the site shows trades within minutes."""
    if os.environ.get("GITHUB_ACTIONS") == "true" and os.environ.get("GH_TOKEN"):
        subprocess.run(["gh", "workflow", "run", "dashboard.yml"], check=False)


def daytrader(session, dry_run):
    """Run both day-trading accounts through one half of the day, as a timeline of events:
    10-minute decision marks, the learning allocator's 15:40 rebalance, and the close-out."""
    clients = {a: trading_client(a) for a in DAY_ACCOUNTS}
    today = now_et().date()
    plan = retry(schedule, today)
    if plan is None:
        print("market is closed today")
        return
    opens, _, close = plan
    noon = pd.Timestamp(today, tz=TZ) + AFTERNOON
    close_out = close - CLOSE_OUT
    rebalance_at = close - REBALANCE_BEFORE_CLOSE
    mine = lambda t: (t < noon) == (session == "morning")  # noqa: E731
    events = [(t, "mark", o) for t, o in day_marks(opens, close) if mine(t)
              and not logged_today("safe", ("decision", "skipped"), decision=t.strftime("%H:%M"))]
    if mine(close_out) and not logged_today("daytrader", ("day closed",)):
        events += [(rebalance_at, "rebalance", None), (close_out, "close", None)]
    events.sort(key=lambda e: e[0])
    if not events:
        print(f"nothing left to do in the {session} session today")
        return
    history = retry(recent_bars, pd.Timestamp(today) - pd.Timedelta(days=45), pd.Timestamp(today),
                    symbols=DAY_SYMBOLS)
    st = load_slices(today)

    if session == "morning" and now_et() < opens + pd.Timedelta(minutes=30):
        sleep_until(opens + pd.Timedelta(minutes=1))
        for account, client in clients.items():  # every day starts flat, as in the backtest
            sell_all(client, account, dry_run, why="leftover position from a previous day", timeout=300)
        st = {"date": str(today), "budget": None, "slices": {}}
        save_slices(st)
    else:  # a later start (the afternoon, or a restart): pick up where the records say
        reconcile_slices(clients["safe"], st, dry_run)

    for t, kind, offset in events:
        if kind == "mark":
            if now_et() > t + LATE:
                log("safe", action="skipped", decision=t.strftime("%H:%M"), why="run started late")
                if offset == 0:
                    log("daytrader", action="skipped", decision=t.strftime("%H:%M"), why="run started late")
                continue
            sleep_until(t + pd.Timedelta(seconds=15))  # let the bars that ended at t publish
            try:
                states = nam_states(history, t)
            except Exception as e:
                log("safe", action="error", decision=t.strftime("%H:%M"), error=f"prices: {str(e)[:300]}")
                continue
            if offset == 0:
                try:
                    qqq_decision(clients["daytrader"], states, t, dry_run)
                except Exception as e:  # keep going: the next decision re-reads the real position
                    log("daytrader", action="error", decision=t.strftime("%H:%M"), error=str(e)[:300])
            try:
                stagger_decision(clients["safe"], st, states, t, offset, dry_run)
            except Exception as e:
                log("safe", action="error", decision=t.strftime("%H:%M"), error=str(e)[:300])
                save_slices(st)
            persist(f"day traders: {t:%H:%M} decision")
            if offset == 0:  # every 30 minutes (GitHub Pages limits rebuilds per hour)
                refresh_dashboard()
        elif kind == "rebalance":
            # The learning allocator rebalances here, so it doesn't depend on a separate alarm.
            sleep_until(t)
            try:
                daily(dry_run)
            except Exception as e:  # never let it get in the way of the close-out
                log("moderate", action="error", error=f"daily rebalance failed: {str(e)[:300]}")
                persist("daily rebalance failed")
        else:
            sleep_until(t)
            for account, client in clients.items():
                sell_all(client, account, dry_run, why="end of day: never hold overnight")
                log(account, action="day closed", equity=float(client.get_account().equity))
            save_slices({"date": str(today), "budget": None, "slices": {}})
            persist("day traders: end of day")
            refresh_dashboard()


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


def watchdog():
    """Run by the hourly dashboard job: if the market is open but the day traders haven't
    logged a decision in 25 minutes and no session is running, start one. Duplicate starts
    are harmless: a session skips every decision that's already logged."""
    now = now_et()
    plan = schedule(now.date())
    if plan is None:
        return
    opens, _, close = plan
    if not opens + pd.Timedelta(minutes=45) <= now < close - pd.Timedelta(minutes=15):
        return
    today = str(now.date())
    recent = [pd.Timestamp(r["time"]) for r in read_log("safe")
              if r["time"].startswith(today) and r.get("action") in ("decision", "skipped", "error")]
    if recent and now - max(recent) < pd.Timedelta(minutes=25):
        return
    for status in ("in_progress", "queued"):
        out = subprocess.run(["gh", "run", "list", "--workflow", "daytrader.yml", "--status", status,
                              "--json", "databaseId"], capture_output=True, text=True)
        if out.returncode == 0 and json.loads(out.stdout or "[]"):
            print(f"a day-trader run is {status}; leaving it alone")
            return
    session = "morning" if now < pd.Timestamp(now.date(), tz=TZ) + AFTERNOON else "afternoon"
    started = subprocess.run(["gh", "workflow", "run", "daytrader.yml", "-f", f"session={session}"]).returncode == 0
    gap = f"{(now - max(recent)).total_seconds() / 60:.0f} minutes" if recent else "since the open"
    log("daytrader", action="restarted" if started else "error",
        why=f"no decisions for {gap}; {'started' if started else 'could not start'} the {session} session")
    persist("watchdog: restarted the day traders")


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
    for account in DAY_ACCOUNTS:
        client = trading_client(account)
        sell_all(client, account, dry_run=False, why="safety net: session missing")
        log(account, action="day closed", equity=float(client.get_account().equity))
    persist("day traders: safety close-out")


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
    sub.add_parser("watchdog", help="restart the day traders if their session died")
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
    elif args.cmd == "watchdog":
        watchdog()
    else:
        trading_client(args.account).close_all_positions(cancel_orders=True)
        log(args.account, action="flattened", why="manual or safety close-out")
        persist(f"{args.account}: flattened")


if __name__ == "__main__":
    main()
