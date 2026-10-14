"""Build the dashboard website (site/index.html) from the Alpaca paper accounts and logs.

    python dashboard.py
"""
import html
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.trading.requests import GetPortfolioHistoryRequest

from broker import data_client, trading_client
from intraday_data import TZ, market_calendar
from report import REPORT_CSS
from live import LAUNCH_V2, SLEEVES, STAGGER_MARKETS, STAGGER_OFFSETS, clock_name, read_log
from universe import load_daily

ROOT = Path(__file__).parent
SITE = ROOT / "site"
REPO = "https://github.com/nmallik1029/paper-trading-lab"
START_VALUE = 1000.0

# Display order. Colors are the categorical palette's first three slots, in this order.
STRATEGIES = {
    "daytrader": {
        "name": "QQQ breakout trader", "tag": "Intraday · 3x leveraged", "slot": 1, "band": "daytrader",
        "about": (
            "QQQ, the Nasdaq-100 fund, usually stays within a predictable distance of its opening "
            "price, and how far it typically strays depends on the time of day. Every 30 minutes the "
            "bot compares QQQ's price with the average move for that time over the past two weeks. A "
            "break above that range buys TQQQ, a fund that moves about three times as much as QQQ; a "
            "break below buys SQQQ, which moves three times in the opposite direction. The position is "
            "closed when the price falls back through the range or the day's volume-weighted average "
            "price, and everything is sold before the market closes."),
        "facts": [("Trades", "TQQQ and SQQQ (3x long and short QQQ)"),
                  ("Decisions", "Every 30 minutes, 10:00 AM to 3:30 PM"),
                  ("Based on", "Zarattini, Aziz and Barbon (2024)")],
    },
    "safe": {
        "name": "Staggered breakout portfolio", "tag": "Intraday · most active", "slot": 2, "band": "staggered",
        "about": (
            "This account applies the same breakout rule to three markets: the Nasdaq-100, the S&P 500 "
            "and the semiconductor index. Each market runs on three separate half-hour clocks offset by "
            "ten minutes, which splits the account into nine independent slices. Checking a single "
            "market more often than every 30 minutes made the strategy worse in testing, so staggering "
            "the clocks is how this account trades throughout the hour without giving up the edge."),
        "facts": [("Trades", "TQQQ/SQQQ, UPRO/SPXU and SOXL/SOXS"),
                  ("Decisions", "Every 10 minutes, 10:00 AM to 3:50 PM"),
                  ("Typical activity", "About 16 trades a day")],
    },
    "moderate": {
        "name": "Learning allocator", "tag": "Daily · adaptive", "slot": 3, "band": "learner",
        "about": (
            "Five daily strategies share this account: Bitcoin trend-following, short volatility timed "
            "by the VIX futures curve, holding TQQQ overnight during uptrends, buying sharp two-day "
            "dips, and a 3x trend position sized to its recent volatility. An online-learning "
            "algorithm called Hedge decides how much money each one gets. It began with an equal split "
            f"on {LAUNCH_V2:%B} {LAUNCH_V2.day} and moves money toward the strategies that have been "
            "performing well, while keeping at least 5% in each so it can notice when one recovers."),
        "facts": [("Trades", "TQQQ, SVXY and Bitcoin"),
                  ("Decisions", "Daily at 3:40 PM; the overnight position is sold at the open")],
    },
}
# When each account's current strategy started. Earlier trades in the same accounts belong to
# retired strategies, so the trade lists and counts leave them out.
CURRENT_SINCE = {"daytrader": pd.Timestamp("2026-10-13"), "safe": pd.Timestamp("2026-10-14"),
                 "moderate": pd.Timestamp("2026-10-13")}
MARKET_NAMES = {"QQQ": "Nasdaq-100", "SPY": "S&P 500", "SMH": "Semiconductors"}
SLEEVE_INFO = {
    "Crypto trend (BTC)": ("Bitcoin trend", "Bitcoin", "Holds Bitcoin while its 28-day return is positive."),
    "Short volatility (VIX curve)": ("Short volatility", "SVXY", "Holds SVXY while short-term volatility is below its three-month level."),
    "Overnight TQQQ, uptrend only": ("Overnight TQQQ", "TQQQ", "Holds TQQQ from 3:40 PM to the next open while QQQ is in an uptrend."),
    "RSI(2) dip buying, 3x": ("Dip buying", "TQQQ", "Buys TQQQ after a sharp two-day drop in an uptrend and sells on the rebound."),
    "Leveraged trend + vol target": ("3x trend", "TQQQ", "Holds TQQQ while QQQ is above its 200-day average, sized to its volatility."),
}


def money(v):
    return f"${v:,.2f}"


def pct(v, signed=True):
    return f"{v:+.2%}" if signed else f"{v:.2%}"


def esc(s):
    return html.escape(str(s))


def table(headers, rows, align=None, empty="Nothing yet."):
    """An HTML table. `align` is a string like 'lrr', one letter per column."""
    if not rows:
        return f'<p class="empty">{esc(empty)}</p>'
    align = align or "l" * len(headers)
    cls = lambda i: ' class="num"' if align[i] == "r" else ""  # noqa: E731
    head = "".join(f"<th{cls(i)}>{esc(h)}</th>" for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f"<td{cls(i)}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def when(ts):
    return pd.Timestamp(ts).strftime("%b %d, %I:%M %p").replace(" 0", " ")


# --- market and account data ---------------------------------------------------------------

def market_status(now):
    cal = market_calendar()
    today = now.normalize().tz_localize(None)
    if today in cal.index:
        opens, close = today + pd.Timedelta(hours=9, minutes=30), pd.Timestamp(cal.loc[today])
        if opens <= now.tz_localize(None) < close:
            return True
    return False


def trading_days(launch, now):
    """Trading days from the launch through today (today only once the market has opened)."""
    cal = market_calendar()
    today = now.normalize().tz_localize(None)
    days = [d for d in cal.index if launch <= d <= today]
    opened = now.hour * 60 + now.minute >= 9 * 60 + 30
    return [d for d in days if d < today or opened]


def spy_values(days, now):
    """$1,000 put in SPY at the close before the launch."""
    spy = load_daily(["SPY"], name="spy", refresh=True)["close"]["SPY"].dropna()
    today = now.normalize().tz_localize(None)
    if days and days[-1] == today and today not in spy.index:
        latest = data_client().get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols="SPY", feed=DataFeed.IEX))["SPY"].price
        spy.loc[today] = float(latest)
    base = spy[spy.index < days[0]].iloc[-1] if days else np.nan
    return [START_VALUE * spy.get(d, np.nan) / base for d in days]


def account_values(client, days):
    hist = client.get_portfolio_history(GetPortfolioHistoryRequest(period="1A", timeframe="1D"))
    s = pd.Series(hist.equity, index=pd.to_datetime(hist.timestamp, unit="s", utc=True)
                  .tz_convert(TZ).normalize().tz_localize(None), dtype=float)
    out = s[s > 0].reindex(days).ffill().fillna(START_VALUE)
    if days:
        out.iloc[-1] = float(client.get_account().equity)
    return out.tolist()


def session_series(clients):
    """Each account's value through the most recent session, every 5 minutes, as a change
    from the open. Returns (date, chart data) or (None, None)."""
    raw = {}
    for account, client in clients.items():
        try:
            h = client.get_portfolio_history(GetPortfolioHistoryRequest(period="1D", timeframe="5Min"))
        except Exception:
            continue
        s = pd.Series(h.equity, index=pd.to_datetime(h.timestamp, unit="s", utc=True).tz_convert(TZ), dtype=float)
        minutes = s.index.hour * 60 + s.index.minute
        raw[account] = s[(s > 0) & (minutes >= 570) & (minutes <= 960)]
    dates = [s.index[-1].date() for s in raw.values() if len(s)]
    if not dates:
        return None, None
    day = max(dates)
    series, labels = [], set()
    for account, s in raw.items():
        s = s[s.index.date == day]
        if len(s) < 2:
            continue
        values = {t.strftime("%H:%M"): float(v / s.iloc[0] - 1) for t, v in s.items()}
        labels |= set(values)
        series.append((account, values))
    if not series:
        return None, None
    labels = sorted(labels)
    return day, {"labels": labels, "series": [
        {"name": STRATEGIES[a]["name"], "slot": STRATEGIES[a]["slot"], "values": [v.get(t) for t in labels]}
        for a, v in series]}


def normal_range(band, horizon):
    """10th to 90th percentile of $1,000 after h trading days, for h = 0..horizon, from every
    h-day stretch of the strategy's full backtest."""
    r = pd.read_csv(ROOT / "report" / f"v2_{band}_daily_returns.csv", index_col=0)["ret"].dropna()
    growth = np.r_[1.0, (1 + r).cumprod().to_numpy()]
    lo, hi = [START_VALUE], [START_VALUE]
    for h in range(1, horizon + 1):
        ratio = growth[h:] / growth[:-h]
        lo.append(START_VALUE * np.percentile(ratio, 10))
        hi.append(START_VALUE * np.percentile(ratio, 90))
    return lo, hi


def backtest_facts(band, summary):
    f = summary["finalists"][band]
    dev, hold = f["dev"], f["holdout"]
    return [("Backtest, unseen years",
             f"{hold['cagr']:.0%} a year from {hold['start'][:4]} to {hold['end'][:4]}, "
             f"worst drop {abs(hold['max_drawdown']):.0%}"),
            ("Backtest, design years",
             f"{dev['cagr']:.0%} a year from {dev['start'][:4]} to {dev['end'][:4]}, "
             f"worst drop {abs(dev['max_drawdown']):.0%}")]


# --- panels --------------------------------------------------------------------------------

def strategy_trades(account):
    """The account's buys and sells since its current strategy started, oldest first."""
    since = CURRENT_SINCE[account]
    return [r for r in read_log(account) if r.get("action") in ("buy", "sell")
            and pd.Timestamp(r["time"]).tz_localize(None) >= since]


def all_trades():
    rows = [(pd.Timestamp(r["time"]), a, r) for a in STRATEGIES for r in strategy_trades(a)]
    return sorted(rows, key=lambda x: x[0], reverse=True)


def position(name):
    return "Cash" if name == "cash" else name


def moved(before, after):
    return f"<b>{position(before)} → {position(after)}</b>" if before != after else position(after)


def side(r):
    return f'<span class="{r["action"]}">{"Buy" if r["action"] == "buy" else "Sell"}</span>'


def trade_feed(trades, limit=8):
    rows = [[esc(when(t)), f"{side(r)} <b>{esc(r['symbol'])}</b><span class='sub'><i class='dot s{STRATEGIES[a]['slot']}'>"
             f"</i>{esc(STRATEGIES[a]['name'])}</span>", money(float(r["dollars"]))] for t, a, r in trades[:limit]]
    return table(["Time (ET)", "Trade", "Amount"], rows, "llr",
                 "No trades yet. The intraday strategies start at 10:00 AM on trading days.")


def holdings_table(client):
    rows = [[esc(p.symbol), f"{float(p.qty):,.4f}", money(float(p.market_value)),
             f"{money(float(p.unrealized_pl))} ({float(p.unrealized_plpc):+.2%})"]
            for p in client.get_all_positions()]
    return table(["Holding", "Shares", "Value", "Gain"], rows, "lrrr", "No positions. The account is in cash.")


def recent_trades(account, limit=12):
    trades = strategy_trades(account)
    rows = [[esc(when(r["time"])), side(r), esc(r["symbol"]), money(float(r["dollars"])),
             money(float(r["price"])), esc(r.get("why", ""))] for r in trades[-limit:][::-1]]
    if not rows:
        return ""
    return (f"<details><summary>Recent trades ({len(trades)} in total)</summary>"
            + table(["Time (ET)", "Side", "Symbol", "Amount", "Price", "Reason"], rows, "lllrrl") + "</details>")


def qqq_panel(now):
    rows = [r for r in read_log("daytrader") if r.get("action") == "decision" and r.get("strategy") == "noise-area"]
    if not rows:
        return panel("Today's decisions", "<p class='empty'>The first decision is made at 10:00 AM on a trading day.</p>")
    day = pd.Timestamp(rows[-1]["time"]).date()
    out = []
    for r in rows:
        if pd.Timestamp(r["time"]).date() != day:
            continue
        out.append([esc(r["decision"]), f"{r['qqq']:.2f}", f"{r['lower']:.2f} – {r['upper']:.2f}",
                    f"{r['vwap']:.2f}", moved(r["before"], r["after"])])
    title = "Today's decisions" if day == now.date() else f"Decisions on {day:%B} {day.day}"
    return panel(title, table(["Time", "QQQ", "Normal range", "VWAP", "Position"], out, "lrrrl"))


def staggered_panel(now):
    rows = [r for r in read_log("safe") if r.get("action") == "decision" and r.get("strategy") == "staggered-breakout"]
    if not rows:
        return panel("Positions by clock", "<p class='empty'>The first decision is made at 10:00 AM on a trading day.</p>")
    day = pd.Timestamp(rows[-1]["time"]).date()
    day_rows = [r for r in rows if pd.Timestamp(r["time"]).date() == day]
    latest = {r["clock"]: r for r in day_rows}
    clocks = [clock_name(o) for o in STAGGER_OFFSETS]
    grid = []
    for m in STAGGER_MARKETS:
        cells = []
        for c in clocks:
            v = latest.get(c, {}).get("markets", {}).get(m)
            cells.append("—" if v is None else (f"<b>{v['after']}</b>" if v["after"] != "cash" else "Cash"))
        grid.append([esc(MARKET_NAMES[m]), *cells])
    changes = sum(v["before"] != v["after"] for r in day_rows for v in r["markets"].values())
    label = "today" if day == now.date() else f"on {day:%B} {day.day}"
    note = (f"<p class='caption'>{len(day_rows)} decisions and {changes} position changes {label}. "
            "Each cell is one slice of the account: one market on one clock.</p>")
    last = day_rows[-1]
    detail = [[esc(MARKET_NAMES[m]), f"{v['price']:.2f}", f"{v['lower']:.2f} – {v['upper']:.2f}", f"{v['vwap']:.2f}",
               moved(v["before"], v["after"])] for m, v in last["markets"].items()]
    return (panel("Positions by clock", table(["Market", *[f"{c} clock" for c in clocks]], grid, "llll") + note)
            + panel(f"Latest decision ({last['decision']}, {last['clock']} clock)",
                    table(["Market", "Price", "Normal range", "VWAP", "Position"], detail, "lrrrl")))


def rebalances():
    return [r for r in read_log("moderate") if r.get("action") == "rebalanced" and "budgets" in r]


def learner_panel():
    rebs = rebalances()
    if not rebs:
        return panel("Current allocation", "<p class='empty'>The first rebalance happens at 3:40 PM on a trading day.</p>"), None
    last = rebs[-1]
    rows = []
    for s in SLEEVES:
        name, holds, rule = SLEEVE_INFO[s]
        sig, share = last["signals"].get(s, 0.0), last["budgets"].get(s, 0.0)
        status = "Out" if sig <= 0 else ("In" if sig >= 0.999 else f"In ({sig:.0%})")
        rows.append([f"<b>{esc(name)}</b><span class='sub'>{esc(rule)}</span>", esc(holds), status, f"{share:.1%}"])
    alloc = panel(f"Current allocation (as of {when(last['time'])})",
                  table(["Strategy", "Holds", "Signal", "Share"], rows, "lllr"))
    labels = [pd.Timestamp(r["time"]).strftime("%b %d") for r in rebs]
    minis = [{"title": SLEEVE_INFO[s][0], "labels": labels, "values": [r["budgets"].get(s, 0.0) for r in rebs]}
             for s in SLEEVES]
    grid = "".join(f'<figure class="mini"><figcaption>{esc(m["title"])}</figcaption>'
                   f'<div class="mini-chart"><canvas id="m-{i}" aria-label="{esc(m["title"])} share over time"></canvas></div></figure>'
                   for i, m in enumerate(minis))
    learned = panel("How the allocation has changed",
                    f"<p class='caption'>Each strategy's share of the account after each daily update. "
                    f"The gray line marks an equal 20% split.</p><div class='minis'>{grid}</div>")
    return alloc + learned, minis


def panel(title, body):
    return f'<div class="panel"><h3>{esc(title)}</h3>{body}</div>'


# --- page ----------------------------------------------------------------------------------

def build():
    now = pd.Timestamp.now(tz=TZ)
    logs = [r for a in STRATEGIES for r in read_log(a)]
    launch = (min(pd.Timestamp(r["time"]) for r in logs).tz_localize(None).normalize()
              if logs else now.normalize().tz_localize(None))
    days = trading_days(launch, now)
    labels = ["Start"] + [d.strftime("%b %d") for d in days]
    spy = [START_VALUE] + spy_values(days, now)
    summary = json.loads((ROOT / "report" / "v2_summary.json").read_text())
    trades = all_trades()
    n_today = sum(t.date() == now.date() for t, _, _ in trades)

    clients = {a: trading_client(a) for a in STRATEGIES}
    series, tiles, cards, minis = [], [], [], None
    for account, info in STRATEGIES.items():
        client = clients[account]
        acct = client.get_account()
        values = [START_VALUE] + account_values(client, days)
        lo, hi = normal_range(info["band"], len(days))
        series.append({"name": info["name"], "slot": info["slot"], "values": values, "lo": lo, "hi": hi})
        day_change = float(acct.equity) / float(acct.last_equity) - 1 if float(acct.last_equity) > 0 else 0.0
        tiles.append(tile(info["name"], info["slot"], values[-1], values[-1] / START_VALUE - 1, day_change))
        if account == "daytrader":
            extra = qqq_panel(now)
        elif account == "safe":
            extra = staggered_panel(now)
        else:
            extra, minis = learner_panel()
        facts = info["facts"] + backtest_facts(info["band"], summary)
        cards.append(f"""<article class="card strategy" id="{info['band']}">
<header class="card-head"><h3><i class="dot s{info['slot']}"></i>{esc(info['name'])}</h3>
<span class="tag">{esc(info['tag'])}</span></header>
<div class="strategy-top">
  <div><p>{esc(info['about'])}</p>
  <dl class="facts">{''.join(f'<dt>{esc(k)}</dt><dd>{esc(v)}</dd>' for k, v in facts)}</dl></div>
  <div class="chart"><canvas id="chart-{account}" aria-label="{esc(info['name'])} account value"></canvas></div>
</div>
<div class="panels">{panel('Holdings', holdings_table(client))}{extra}</div>
{recent_trades(account)}
</article>""")
    spy_day = (spy[-1] / spy[-2] - 1) if len(spy) > 2 else 0.0
    tiles.append(tile("S&P 500 (SPY), for comparison", 0, spy[-1], spy[-1] / START_VALUE - 1, spy_day))

    session_day, session = session_series(clients)
    data = {"labels": labels, "spy": spy, "series": series, "accounts": list(STRATEGIES), "minis": minis,
            "session": session}
    SITE.mkdir(exist_ok=True)
    reports = SITE / "reports"
    reports.mkdir(exist_ok=True)
    shutil.copy(ROOT / "report" / "v2_research.html", reports / "research.html")
    archive = ROOT / "archive" / "v1" / "report"
    if archive.exists():  # the retired v1 reports, restyled to match, with a link back
        (reports / "v1").mkdir(exist_ok=True)
        for f in archive.glob("*.html"):
            text = re.sub(r"<style>.*?</style>", f"<style>{REPORT_CSS}</style>", f.read_text(encoding="utf-8"), flags=re.S)
            back = '<a class="home" href="../research.html">← Research and backtests</a>'
            text = text.replace("<body><main>", f"<body><main>\n{back}", 1)
            (reports / "v1" / f.name).write_text(text, encoding="utf-8")
    page_html = page(now, launch, tiles, cards, data, summary, n_today, len(trades), trade_feed(trades),
                     session_day, market_status(now))
    (SITE / "index.html").write_text(page_html, encoding="utf-8")
    print(f"dashboard written: {SITE / 'index.html'}")


def tile(name, slot, value, total, today):
    return f"""<div class="tile"><div class="tile-label"><i class="dot s{slot}"></i>{esc(name)}</div>
<div class="tile-value">{money(value)}</div>
<div class="tile-sub"><span>{pct(total)} since start</span><span>{pct(today)} today</span></div></div>"""


CSS = """
:root { color-scheme: light; --page:#f7f7f5; --surface:#ffffff; --ink:#111110; --ink-2:#4f4e4a;
  --muted:#85837d; --grid:#e6e5df; --line:#e2e1db; --s0:#6b6a66; --s1:#2a78d6; --s2:#eb6834;
  --s3:#1baf7a; --good:#0a7a0a; --bad:#c23434; --band:rgba(17,17,16,.07); --open:#0ca30c; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark;
  --page:#0f0f0e; --surface:#1a1a19; --ink:#f4f4f2; --ink-2:#c3c2b7; --muted:#8f8d86; --grid:#2c2c2a;
  --line:#2f2f2c; --s0:#b5b4ab; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --good:#3dbb3d; --bad:#e66767;
  --band:rgba(255,255,255,.08); } }
:root[data-theme="dark"] { color-scheme: dark; --page:#0f0f0e; --surface:#1a1a19; --ink:#f4f4f2;
  --ink-2:#c3c2b7; --muted:#8f8d86; --grid:#2c2c2a; --line:#2f2f2c; --s0:#b5b4ab; --s1:#3987e5;
  --s2:#d95926; --s3:#199e70; --good:#3dbb3d; --bad:#e66767; --band:rgba(255,255,255,.08); }
* { box-sizing:border-box; }
html { -webkit-text-size-adjust:100%; }
body { margin:0; background:var(--page); color:var(--ink);
  font:15px/1.6 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width:1120px; margin:0 auto; padding:40px 20px 72px; }
a { color:var(--s1); text-decoration:none; } a:hover { text-decoration:underline; }
h1 { font-size:30px; line-height:1.2; margin:0 0 12px; letter-spacing:-.01em; }
h2 { font-size:20px; line-height:1.3; margin:48px 0 6px; letter-spacing:-.005em; }
h3 { font-size:15px; margin:0 0 10px; }
p { margin:0 0 12px; }
.lede { font-size:16.5px; color:var(--ink-2); max-width:760px; }
.links { display:flex; gap:20px; flex-wrap:wrap; margin:16px 0 0; font-weight:600; }
.status { display:flex; gap:8px 24px; flex-wrap:wrap; margin:24px 0 0; color:var(--ink-2); font-size:14px; }
.status b { color:var(--ink); font-weight:650; }
.live { display:inline-flex; align-items:center; gap:7px; }
.live i { width:8px; height:8px; border-radius:50%; background:var(--muted); }
.live.open i { background:var(--open); }
.section-intro { color:var(--ink-2); max-width:760px; margin-bottom:18px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:14px; margin:32px 0 0; }
.tile, .card { background:var(--surface); border:1px solid var(--line); border-radius:12px; }
.tile { padding:16px 18px; }
.tile-label { display:flex; align-items:center; gap:8px; font-size:13.5px; color:var(--ink-2); }
.tile-value { font-size:26px; font-weight:650; margin:4px 0 2px; letter-spacing:-.01em; }
.tile-sub { display:flex; gap:14px; flex-wrap:wrap; font-size:13px; color:var(--ink-2);
  font-variant-numeric:tabular-nums; }
.dot { width:10px; height:10px; border-radius:50%; display:inline-block; flex:none; margin-right:8px; vertical-align:-1px; }
.tile-label .dot { margin-right:0; }
.s0 { background:var(--s0); } .s1 { background:var(--s1); } .s2 { background:var(--s2); } .s3 { background:var(--s3); }
.card { padding:24px; margin:0 0 16px; }
.row { display:grid; grid-template-columns:minmax(0, 3fr) minmax(0, 2fr); gap:16px; margin-top:16px; }
.row .card { margin:0; }
.card-head { display:flex; justify-content:space-between; align-items:baseline; gap:12px; flex-wrap:wrap; margin-bottom:6px; }
.card-head h3 { font-size:18px; margin:0; display:flex; align-items:center; }
.tag { font-size:12.5px; color:var(--ink-2); border:1px solid var(--line); border-radius:999px; padding:2px 10px; white-space:nowrap; }
.caption { color:var(--ink-2); font-size:13.5px; margin:6px 0 12px; }
.chart { position:relative; height:260px; }
.chart.tall { height:300px; }
.strategy { margin-bottom:20px; }
.strategy-top { display:grid; grid-template-columns:minmax(0, 1fr) minmax(0, 1.15fr); gap:28px; align-items:start; margin-top:8px; }
.strategy-top p { color:var(--ink-2); }
.facts { display:grid; grid-template-columns:max-content 1fr; gap:6px 16px; margin:16px 0 0; font-size:14px; }
.facts dt { color:var(--muted); }
.facts dd { margin:0; }
.panels { display:grid; grid-template-columns:repeat(auto-fit, minmax(400px, 1fr)); gap:22px 32px; margin-top:24px; }
.panel { min-width:0; }
.empty { color:var(--muted); font-size:14px; margin:4px 0; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:13.5px; font-variant-numeric:tabular-nums; }
th, td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--grid); vertical-align:top; }
th { color:var(--muted); font-weight:600; font-size:12.5px; white-space:nowrap; }
td { white-space:nowrap; }
th:first-child, td:first-child { padding-left:0; }
th:last-child, td:last-child { padding-right:0; }
.num { text-align:right; }
td .sub { display:block; color:var(--muted); font-size:12.5px; white-space:normal; max-width:340px; margin-top:1px; }
td .sub .dot { width:8px; height:8px; margin-right:6px; }
.buy { color:var(--good); font-weight:600; } .sell { color:var(--bad); font-weight:600; }
details { margin-top:20px; border-top:1px solid var(--grid); padding-top:14px; }
summary { cursor:pointer; font-weight:600; font-size:14px; color:var(--ink-2); }
details[open] summary { margin-bottom:10px; }
.minis { display:grid; grid-template-columns:repeat(auto-fill, minmax(150px, 1fr)); gap:12px; }
.mini { margin:0; border:1px solid var(--grid); border-radius:8px; padding:10px 12px 6px; }
.mini figcaption { font-size:12.5px; color:var(--ink-2); }
.mini-chart { position:relative; height:84px; }
.method { display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:16px; }
.method .card { margin:0; }
.method p { color:var(--ink-2); margin:0; }
footer { margin-top:48px; padding-top:20px; border-top:1px solid var(--line); color:var(--muted); font-size:13.5px; }
footer p { max-width:760px; }
@media (max-width: 860px) {
  main { padding:28px 16px 56px; }
  .row, .strategy-top, .panels { grid-template-columns:minmax(0, 1fr); }
  .card { padding:18px; }
  h1 { font-size:26px; }
}
"""


def page(now, launch, tiles, cards, data, summary, n_today, n_total, feed, session_day, is_open):
    session_title = ("Today" if session_day == now.date() else
                     f"Most recent session: {session_day:%B} {session_day.day}" if session_day else "Today")
    session_body = ('<div class="chart"><canvas id="chart-session" aria-label="Change in value through the session"></canvas></div>'
                    if data["session"] else "<p class='empty'>The chart fills in once the market opens.</p>")
    launch_text = f"{launch:%B} {launch.day}"
    first = min(CURRENT_SINCE.values())
    since_text = f"{first:%B} {first.day}"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paper Trading Lab</title>
<meta name="description" content="An automated trading system running three strategies on paper-trading accounts, with live results, every trade, and the research behind it.">
<style>{CSS}</style></head>
<body><main>
<header>
<h1>Paper Trading Lab</h1>
<p class="lede">An automated trading system that runs three strategies on Alpaca paper-trading accounts,
each funded with $1,000 of simulated money. The strategies trade on their own every market day, and
this page rebuilds itself from the live accounts and the bots' decision logs.</p>
<nav class="links"><a href="reports/research.html">Research and backtests</a><a href="{REPO}">Source code</a></nav>
<div class="status"><span class="live{' open' if is_open else ''}"><i></i><b>{'Market open' if is_open else 'Market closed'}</b></span>
<span><b>{n_today}</b> trades today</span><span><b>{n_total}</b> trades since {since_text}</span>
<span>Updated {now:%B} {now.day} at {now.strftime("%I:%M %p").lstrip("0")} ET</span></div>
</header>

<section class="tiles">{''.join(tiles)}</section>

<div class="row">
<section class="card"><h3>{session_title}</h3>
<p class="caption">Change in each account's value since the open, in five-minute steps.</p>{session_body}</section>
<section class="card"><h3>Latest trades</h3>{feed}</section>
</div>

<section class="card" style="margin-top:16px"><h3>Since launch</h3>
<p class="caption">The value of each $1,000 account since {launch_text}, compared with putting $1,000 into
the S&amp;P 500 through SPY on the same day.</p>
<div class="chart tall"><canvas id="chart-all" aria-label="Account values since launch"></canvas></div></section>

<h2>The strategies</h2>
<p class="section-intro">Each strategy's chart shades the range that 80% of its backtested outcomes fell
within after the same number of trading days. A result inside that range is ordinary luck; a result well
outside it would suggest the strategy is no longer behaving the way it was tested.</p>
{''.join(cards)}

<h2>How it works</h2>
<div class="method">
<section class="card"><h3>Research and testing</h3><p>Each strategy began as published research or a
well-known trading rule and was rebuilt and tested on this project's own data, with trading costs
included. Design choices were made on one stretch of history and then checked once against later years
the strategy had never seen. Every variant tested is logged, including the failures, and the bar for
calling a result statistically significant rises with each attempt. {summary['variants']} variants have
been tried so far and none clears that bar, so the backtest figures here are evidence rather than proof.</p></section>
<section class="card"><h3>Trading</h3><p>The bots place market orders through Alpaca's paper-trading
API, using fractional shares so a $1,000 account can hold leveraged funds and Bitcoin. Before going live,
the intraday strategies were run through past trading days on a simulated clock to confirm that the live
code makes the same decisions as the backtest, and it matched on more than 99% of them.</p></section>
<section class="card"><h3>Infrastructure</h3><p>The bots run on GitHub Actions and are started on
schedule by cron-job.org. Every decision is written to a log that is committed to the repository. This
page is rebuilt from those logs and the account data about every 30 minutes while the market is open,
and the charts are drawn with Chart.js.</p></section>
</div>

<footer><p>Simulated money on paper-trading accounts. Nothing here is investment advice, and backtest
results are not a forecast of future returns. The project's earlier experiments, including machine-learning
strategies that were tested and retired, are described in the <a href="reports/research.html">research
write-up</a>.</p></footer>
</main>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.js"></script>
<script>
const DATA = {json.dumps(data)};
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const usd = v => '$' + v.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
const pct = v => (v >= 0 ? '+' : '') + (v * 100).toFixed(2) + '%';
const charts = [];
function options(fmt, tick) {{
  return {{
    responsive: true, maintainAspectRatio: false, animation: false, spanGaps: true,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ color: css('--ink-2'), usePointStyle: true, pointStyle: 'circle',
        boxWidth: 8, boxHeight: 8, padding: 16, filter: item => !item.text.startsWith('_') }} }},
      tooltip: {{ filter: item => !item.dataset.label.startsWith('_'),
        callbacks: {{ label: c => ` ${{c.dataset.label}}: ${{fmt(c.parsed.y)}}` }} }}
    }},
    scales: {{
      x: {{ grid: {{ display: false }}, border: {{ color: css('--grid') }},
           ticks: {{ color: css('--muted'), maxTicksLimit: 7, maxRotation: 0 }} }},
      y: {{ grid: {{ color: css('--grid') }}, border: {{ display: false }},
           ticks: {{ color: css('--muted'), maxTicksLimit: 6, callback: tick }} }}
    }}
  }};
}}
const dollars = () => options(usd, v => '$' + v.toLocaleString());
const percent = () => options(pct, v => (v * 100).toFixed(1) + '%');
const line = (label, data, color, n, width = 2) => ({{ label, data, borderColor: color, backgroundColor: color,
  borderWidth: width, pointRadius: n < 3 ? 4 : 0, pointHoverRadius: 4, tension: 0 }});
function draw() {{
  charts.forEach(c => c.destroy()); charts.length = 0;
  const n = DATA.labels.length;
  if (DATA.session) charts.push(new Chart(document.getElementById('chart-session'), {{ type: 'line', options: percent(),
    data: {{ labels: DATA.session.labels, datasets: DATA.session.series.map(s =>
      line(s.name, s.values, css('--s' + s.slot), DATA.session.labels.length)) }} }}));
  const spy = () => line('S&P 500 (SPY)', DATA.spy, css('--s0'), n, 1.5);
  charts.push(new Chart(document.getElementById('chart-all'), {{ type: 'line', options: dollars(),
    data: {{ labels: DATA.labels, datasets: [...DATA.series.map(s => line(s.name, s.values, css('--s' + s.slot), n)), spy()] }} }}));
  DATA.series.forEach((s, i) => {{
    const lo = {{ label: '_lo', data: s.lo, borderWidth: 0, pointRadius: 0, pointHoverRadius: 0, fill: false }};
    const hi = {{ label: 'Typical range', data: s.hi, borderWidth: 0, pointRadius: 0, pointHoverRadius: 0, fill: '-1',
      backgroundColor: css('--band'), borderColor: css('--band') }};
    const o = dollars();
    o.plugins.tooltip.filter = item => !item.dataset.label.startsWith('_') && item.dataset.label !== 'Typical range';
    charts.push(new Chart(document.getElementById('chart-' + DATA.accounts[i]), {{ type: 'line', options: o,
      data: {{ labels: DATA.labels, datasets: [lo, hi, line(s.name, s.values, css('--s' + s.slot), n), spy()] }} }}));
  }});
  if (DATA.minis) DATA.minis.forEach((m, i) => {{
    const o = options(v => (v * 100).toFixed(1) + '%', v => Math.round(v * 100) + '%');
    o.plugins.legend.display = false;
    o.scales.x.ticks.display = false;
    o.scales.y.min = 0;
    o.scales.y.ticks.maxTicksLimit = 3;
    const k = m.labels.length;
    charts.push(new Chart(document.getElementById('m-' + i), {{ type: 'line', options: o,
      data: {{ labels: m.labels, datasets: [line('Share', m.values, css('--s3'), k),
        {{ label: 'Equal split', data: Array(k).fill(0.2), borderColor: css('--s0'), borderWidth: 1,
           pointRadius: 0, pointHoverRadius: 0 }}] }} }}));
  }});
}}
draw();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
</script>
</body></html>
"""


if __name__ == "__main__":
    build()
