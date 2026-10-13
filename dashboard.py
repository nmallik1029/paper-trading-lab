"""Build the dashboard website (site/index.html) from the Alpaca paper accounts and logs.

    python dashboard.py
"""
import html
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.trading.requests import GetPortfolioHistoryRequest

from broker import data_client, trading_client
from data import load_prices
from intraday_data import TZ, market_calendar
from live import LAUNCH_V2, SLEEVES, read_log

ROOT = Path(__file__).parent
SITE = ROOT / "site"
START_VALUE = 1000.0

# Colors are categorical slots 1-3, in this order.
STRATEGIES = {
    "moderate": dict(name="Multi-strategy fund", risk="Aggressive", slot=1, band="fund",
                     about="Five unrelated strategies with 20% of the money each, rebalanced every "
                           "day at 3:40 PM: Bitcoin trend, short volatility, overnight TQQQ, RSI(2) "
                           "dip buying, and volatility-targeted 3x trend."),
    "safe": dict(name="Learning allocator", risk="Learns live", slot=2, band="learner",
                 about="The same five strategies, but an online-learning algorithm (Hedge) decides "
                       "how much money each one gets. It started with equal trust on "
                       f"{LAUNCH_V2:%b %d} and shifts money toward whatever has been working."),
    "daytrader": dict(name="Breakout day trader", risk="Ultra risky", slot=3, band="daytrader",
                      about="Every half hour, checks whether QQQ has broken out of its normal range "
                            "for that time of day. Upside break: buys TQQQ (3x long). Downside "
                            "break: buys SQQQ (3x short). Trails a stop at the band or VWAP and "
                            "sells everything by the close."),
}
SLEEVE_INFO = {
    "Crypto trend (BTC)": ("Bitcoin", "Holds Bitcoin while its 28-day return is positive, sized to ~60%/yr volatility"),
    "Short volatility (VIX curve)": ("SVXY", "Bets volatility falls while VIX is below its 3-month level"),
    "Overnight TQQQ, uptrend only": ("TQQQ overnight", "Holds TQQQ from 3:40 PM to the next open, while QQQ is in an uptrend"),
    "RSI(2) dip buying, 3x": ("TQQQ", "Buys TQQQ after a sharp 2-day dip in an uptrend, sells on the bounce"),
    "Leveraged trend + vol target": ("TQQQ", "Holds TQQQ while QQQ is above its 200-day average, sized to ~40%/yr volatility"),
}
SHORT_NAMES = {"Crypto trend (BTC)": "Crypto trend", "Short volatility (VIX curve)": "Short volatility",
               "Overnight TQQQ, uptrend only": "Overnight TQQQ", "RSI(2) dip buying, 3x": "Dip buying",
               "Leveraged trend + vol target": "3x trend"}


def money(v):
    return f"${v:,.2f}"


def signed_pct(v):
    return f"{v:+.2%}"


def trading_days(launch, now):
    """Trading days from the launch through today (today only once the market has opened)."""
    cal = market_calendar()
    today = now.normalize().tz_localize(None)
    days = [d for d in cal.index if launch <= d <= today]
    opened = now.hour * 60 + now.minute >= 9 * 60 + 30
    return [d for d in days if d < today or opened]


def spy_values(days, now):
    """$1,000 put in SPY at the close before the launch."""
    prices, _ = load_prices(refresh=True)
    spy = prices["SPY"]
    today = now.normalize().tz_localize(None)
    if days and days[-1] == today and today not in spy.index:
        latest = data_client().get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols="SPY", feed=DataFeed.IEX))["SPY"].price
        spy.loc[today] = float(latest)  # dividend adjustments don't matter within one day
    base = spy[spy.index < days[0]].iloc[-1] if days else np.nan
    return [START_VALUE * spy.get(d, np.nan) / base for d in days]


def account_values(client, days):
    hist = client.get_portfolio_history(GetPortfolioHistoryRequest(period="1A", timeframe="1D"))
    s = pd.Series(hist.equity, index=pd.to_datetime(hist.timestamp, unit="s", utc=True)
                  .tz_convert(TZ).normalize().tz_localize(None), dtype=float)
    s = s[s > 0]
    out = s.reindex(days).ffill().fillna(START_VALUE)
    if days:
        out.iloc[-1] = float(client.get_account().equity)  # the latest value, not last close
    return out.tolist()


def normal_range(band, horizon):
    """10th-90th percentile of $1,000 after h trading days, for h = 0..horizon, from every
    h-day stretch of the strategy's full backtest."""
    r = pd.read_csv(ROOT / "report" / f"v2_{band}_daily_returns.csv", index_col=0)["ret"].dropna()
    growth = np.r_[1.0, (1 + r).cumprod().to_numpy()]
    lo, hi = [START_VALUE], [START_VALUE]
    for h in range(1, horizon + 1):
        ratio = growth[h:] / growth[:-h]
        lo.append(START_VALUE * np.percentile(ratio, 10))
        hi.append(START_VALUE * np.percentile(ratio, 90))
    return lo, hi


def verdict(band, summary):
    f = summary["finalists"][band]
    dev, hold = f["dev"], f["holdout"]
    return (f"Backtest: {dev['cagr']:.0%}/yr in {dev['start'][:4]}–{dev['end'][:4]} (design years), "
            f"{hold['cagr']:.0%}/yr in {hold['start'][:4]}–{hold['end'][:4]} (never used for design), "
            f"worst drop {min(dev['max_drawdown'], hold['max_drawdown']):.0%}.")


def table(headers, rows, empty):
    if not rows:
        return f'<p class="empty">{html.escape(empty)}</p>'
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def positions_table(client):
    rows = [[html.escape(p.symbol), f"{float(p.qty):.4f}", money(float(p.market_value)),
             f"{money(float(p.unrealized_pl))} ({float(p.unrealized_plpc):+.2%})"]
            for p in client.get_all_positions()]
    return table(["Holding", "Shares", "Value", "Gain"], rows, "All cash right now.")


def trades_table(account, limit=10):
    trades = [r for r in read_log(account) if r.get("action") in ("buy", "sell")][-limit:][::-1]
    rows = [[html.escape(pd.Timestamp(r["time"]).strftime("%b %d, %I:%M %p")),
             f'<span class="{r["action"]}">{r["action"].upper()}</span>', html.escape(r["symbol"]),
             money(float(r["dollars"])), money(float(r["price"])), html.escape(r.get("why", ""))]
            for r in trades]
    return table(["When (ET)", "Side", "Symbol", "Amount", "Price", "Why"], rows, "No trades yet.")


def v2_rebalances(account):
    return [r for r in read_log(account) if r.get("action") == "rebalanced" and "budgets" in r]


def sleeves_table(account):
    """Each sleeve's signal, budget and resulting share of the account, as of the latest
    daily rebalance."""
    rebs = v2_rebalances(account)
    if not rebs:
        return "<p class='empty'>The first daily rebalance happens at 3:40 PM ET on a trading day.</p>"
    last = rebs[-1]
    rows = []
    for s in SLEEVES:
        sig, bud = last["signals"].get(s, 0.0), last["budgets"].get(s, 0.0)
        status = ("out" if sig <= 0 else "in" if sig >= 0.999 else f"in, {sig:.0%} of its share")
        rows.append([f"<b>{html.escape(SHORT_NAMES[s])}</b><br><span class='note'>"
                     f"{html.escape(SLEEVE_INFO[s][1])}</span>", html.escape(SLEEVE_INFO[s][0]),
                     status, f"{bud:.0%}", f"{bud * max(sig, 0):.0%}"])
    when = pd.Timestamp(last["time"]).strftime("%b %d, %I:%M %p")
    return (f"<p class='note'>As of the {when} rebalance. Share = the sleeve's budget; in the account = "
            "budget × how invested the sleeve is.</p>"
            + table(["Sleeve", "Trades", "Signal", "Share", "In the account"], rows, ""))


def learning_panel():
    """How the learning allocator's trust in each sleeve has moved since launch."""
    rebs = v2_rebalances("safe")
    if not rebs:
        return "", None
    labels = [pd.Timestamp(r["time"]).strftime("%b %d") for r in rebs]
    minis = [{"title": SHORT_NAMES[s], "labels": labels, "ref": 0.2,
              "values": [r["budgets"].get(s, 0.0) for r in rebs]} for s in SLEEVES]
    last, first = rebs[-1]["budgets"], rebs[0]["budgets"]
    top = max(SLEEVES, key=lambda s: last.get(s, 0))
    low = min(SLEEVES, key=lambda s: last.get(s, 0))
    note = (f"After {len(rebs)} day{'s' if len(rebs) != 1 else ''} of learning it trusts "
            f"<b>{html.escape(SHORT_NAMES[top])}</b> most ({last[top]:.0%}) and "
            f"<b>{html.escape(SHORT_NAMES[low])}</b> least ({last[low]:.0%}). "
            "It never goes below 5% on any sleeve, so it keeps checking ones that are out of favor.")
    grid = "".join(f'<div class="mini"><div class="mini-title">{html.escape(m["title"])}</div>'
                   f'<div class="mini-chart"><canvas id="m-{i}"></canvas></div></div>'
                   for i, m in enumerate(minis))
    return (f"<h4>What it has learned</h4><p class='note'>{note}</p><div class='minis'>{grid}</div>"
            "<p class='note'>Each panel is one sleeve's share of the money after each day's update. "
            "The gray line is 20%, an equal split (what the fund uses).</p>", minis)


def decisions_table(now):
    """The day trader's most recent day of breakout checks."""
    rows = [r for r in read_log("daytrader")
            if (r.get("action") == "decision" and r.get("strategy") == "noise-area")
            or (r.get("action") == "skipped" and pd.Timestamp(r["time"]).tz_localize(None) >= LAUNCH_V2)]
    if not rows:
        return "<p class='empty'>No breakout checks yet. The first one happens at 10:00 AM ET on a trading day.</p>"
    last_day = pd.Timestamp(rows[-1]["time"]).date()
    out = []
    for r in rows:
        if pd.Timestamp(r["time"]).date() != last_day:
            continue
        if r["action"] == "skipped":
            out.append([html.escape(r.get("decision", "")), "—", "—", "—", "—", "skipped: run started late"])
            continue
        moved = r["before"] != r["after"]
        out.append([html.escape(r["decision"]), f"{r['qqq']:.2f}", f"{r['upper']:.2f}", f"{r['lower']:.2f}",
                    f"{r['vwap']:.2f}", (f"<b>{r['before']} → {r['after']}</b>" if moved else r["after"])])
    label = "Today" if last_day == now.date() else last_day.strftime("%b %d")
    return (f"<h4>{label}'s breakout checks</h4><p class='note'>Above the upper band → buy TQQQ. Below "
            "the lower band → buy SQQQ. Once in, it exits when QQQ crosses back past the band or VWAP.</p>"
            + table(["Time", "QQQ", "Upper band", "Lower band", "VWAP", "Position"], out, ""))


def build():
    now = pd.Timestamp.now(tz=TZ)
    logs = [r for a in STRATEGIES for r in read_log(a)]
    launch = (min(pd.Timestamp(r["time"]) for r in logs).tz_localize(None).normalize()
              if logs else now.normalize().tz_localize(None))
    days = trading_days(launch, now)
    labels = ["Start"] + [d.strftime("%b %d") for d in days]
    spy = [START_VALUE] + spy_values(days, now)
    summary = json.loads((ROOT / "report" / "v2_summary.json").read_text())

    series, cards, tiles, minis = [], [], [], None
    for account, info in STRATEGIES.items():
        client = trading_client(account)
        values = [START_VALUE] + account_values(client, days)
        lo, hi = normal_range(info["band"], len(days))
        series.append({"name": info["name"], "slot": info["slot"], "values": values, "lo": lo, "hi": hi})
        tiles.append(f"""<div class="tile"><div class="tile-label"><i class="dot s{info['slot']}"></i>
{html.escape(info['name'])}</div><div class="tile-value">{money(values[-1])}</div>
<div class="tile-sub">{signed_pct(values[-1] / START_VALUE - 1)} since start</div></div>""")
        if account == "daytrader":
            extra = decisions_table(now)
        else:
            extra = "<h4>The five sleeves</h4>" + sleeves_table(account)
            if account == "safe":
                panel, minis = learning_panel()
                extra += panel
        cards.append(f"""<section class="card">
<div class="card-head"><h2><i class="dot s{info['slot']}"></i>{html.escape(info['name'])}</h2>
<span class="badge">{html.escape(info['risk'])}</span></div>
<p class="about">{html.escape(info['about'])}</p>
<p class="note">{html.escape(verdict(info['band'], summary))}</p>
<div class="chart"><canvas id="chart-{account}" aria-label="{html.escape(info['name'])} value over time"></canvas></div>
<p class="note">Shaded: the range its backtest says is normal (80% of outcomes) after this many days.</p>
<h4>Holdings</h4>{positions_table(client)}
{extra}
<h4>Recent trades</h4>{trades_table(account)}
</section>""")
    tiles.append(f"""<div class="tile"><div class="tile-label"><i class="dot s0"></i>Buy &amp; hold SPY</div>
<div class="tile-value">{money(spy[-1])}</div>
<div class="tile-sub">{signed_pct(spy[-1] / START_VALUE - 1)} since start</div></div>""")

    data = {"labels": labels, "spy": spy, "series": series, "accounts": list(STRATEGIES), "minis": minis}
    SITE.mkdir(exist_ok=True)
    reports = SITE / "reports"
    reports.mkdir(exist_ok=True)
    for name in ("v2_research", "report", "ml_dev", "daytrader_dev", "coldstart_dev"):
        src = ROOT / "report" / f"{name}.html"
        if src.exists():
            shutil.copy(src, reports / f"{name}.html")
    (SITE / "index.html").write_text(page(now, launch, tiles, cards, data, summary), encoding="utf-8")
    print(f"dashboard written: {SITE / 'index.html'}")


def page(now, launch, tiles, cards, data, summary):
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paper Trading Lab</title>
<style>
  :root {{ color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e;
    --muted:#898781; --grid:#e1e0d9; --line:rgba(11,11,11,.10); --s0:#52514e; --s1:#2a78d6;
    --s2:#eb6834; --s3:#1baf7a; --good:#006300; --bad:#d03b3b; --band:rgba(11,11,11,.07); }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark;
    --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --grid:#2c2c2a;
    --line:rgba(255,255,255,.10); --s0:#c3c2b7; --s1:#3987e5; --s2:#d95926; --s3:#199e70;
    --good:#0ca30c; --bad:#e66767; --band:rgba(255,255,255,.08); }} }}
  :root[data-theme="dark"] {{ color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff;
    --ink-2:#c3c2b7; --grid:#2c2c2a; --line:rgba(255,255,255,.10); --s0:#c3c2b7; --s1:#3987e5;
    --s2:#d95926; --s3:#199e70; --good:#0ca30c; --bad:#e66767; --band:rgba(255,255,255,.08); }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--page); color:var(--ink);
    font:14px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }}
  main {{ max-width:1040px; margin:0 auto; padding:28px 16px 64px; }}
  h1 {{ font-size:26px; margin:0 0 4px; }}
  h2 {{ font-size:17px; margin:0; display:flex; align-items:center; gap:8px; }}
  h4 {{ font-size:13px; margin:18px 0 6px; color:var(--ink-2); text-transform:uppercase; letter-spacing:.04em; }}
  .sub, .note, .about, .empty {{ color:var(--ink-2); }}
  .note {{ font-size:12.5px; margin:4px 0 8px; }}
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(160px, 1fr)); gap:12px; margin:20px 0; }}
  .tile, .card {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; }}
  .tile {{ padding:14px 16px; }}
  .tile-label {{ color:var(--ink-2); font-size:13px; display:flex; align-items:center; gap:6px; }}
  .tile-value {{ font-size:24px; font-weight:650; margin-top:2px; }}
  .tile-sub {{ color:var(--ink-2); font-size:13px; }}
  .card {{ padding:20px; margin:16px 0; }}
  .card-head {{ display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; }}
  .badge {{ font-size:12px; border:1px solid var(--line); border-radius:999px; padding:2px 10px; color:var(--ink-2); }}
  .dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; flex:none; }}
  .s0 {{ background:var(--s0); }} .s1 {{ background:var(--s1); }} .s2 {{ background:var(--s2); }} .s3 {{ background:var(--s3); }}
  .chart {{ position:relative; height:260px; margin:8px 0 4px; }}
  .minis {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(180px, 1fr)); gap:10px; margin:8px 0; }}
  .mini {{ border:1px solid var(--grid); border-radius:8px; padding:8px 10px 4px; }}
  .mini-title {{ font-size:12px; color:var(--ink-2); }}
  .mini-chart {{ position:relative; height:90px; }}
  .scroll {{ overflow-x:auto; }}
  table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; font-size:13px; }}
  th, td {{ text-align:left; padding:6px 10px; border-bottom:1px solid var(--grid); vertical-align:top; }}
  td {{ white-space:nowrap; }} td:first-child {{ white-space:normal; min-width:200px; }}
  th {{ color:var(--ink-2); font-weight:600; font-size:12px; }}
  .buy {{ color:var(--good); font-weight:650; }} .sell {{ color:var(--bad); font-weight:650; }}
  footer {{ color:var(--ink-2); font-size:12.5px; margin-top:28px; }}
  a {{ color:var(--s1); }}
</style></head>
<body><main>
<h1>Paper Trading Lab</h1>
<p class="sub">Three strategies, each with $1,000 of fake money in its own Alpaca paper account,
trading automatically. Started {launch:%b %d, %Y}; the v2 strategies took over on
{LAUNCH_V2:%b %d}. Updated {now:%b %d, %Y %I:%M %p} ET.</p>
<div class="tiles">{''.join(tiles)}</div>
<section class="card"><h2>Growth of $1,000</h2>
<p class="note">All three strategies against simply holding SPY. Hover for values.</p>
<div class="chart"><canvas id="chart-all" aria-label="All strategies and SPY over time"></canvas></div></section>
{''.join(cards)}
<footer>Fake money on paper accounts; not investment advice. {summary['variants']} strategy variants
have been tested in this lab and none clears the strict significance bar
(t ≥ {summary['t_needed']:.2f}), so treat this as an experiment.
<a href="reports/v2_research.html">How these strategies were chosen</a> · earlier reports:
<a href="reports/report.html">baselines</a> · <a href="reports/ml_dev.html">monthly ML</a> ·
<a href="reports/daytrader_dev.html">ML day trader</a> ·
<a href="reports/coldstart_dev.html">learning from scratch</a>.</footer>
</main>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.js"></script>
<script>
const DATA = {json.dumps(data)};
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const usd = v => '$' + v.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
const charts = [];
function base() {{
  return {{
    responsive: true, maintainAspectRatio: false, animation: false,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ color: css('--ink-2'), usePointStyle: true, boxWidth: 8,
        filter: item => !item.text.startsWith('_') }} }},
      tooltip: {{ filter: item => !item.dataset.label.startsWith('_'),
        callbacks: {{ label: c => ` ${{c.dataset.label}}: ${{usd(c.parsed.y)}}` }} }}
    }},
    scales: {{
      x: {{ grid: {{ display: false }}, ticks: {{ color: css('--muted'), maxTicksLimit: 8 }},
           border: {{ color: css('--grid') }} }},
      y: {{ grid: {{ color: css('--grid') }}, border: {{ display: false }},
           ticks: {{ color: css('--muted'), callback: v => '$' + v.toLocaleString() }} }}
    }}
  }};
}}
const line = (label, data, color, width = 2, n = DATA.labels.length) => ({{ label, data, borderColor: color,
  backgroundColor: color, borderWidth: width, pointRadius: n < 3 ? 4 : 0, pointHoverRadius: 4, tension: 0 }});
function draw() {{
  charts.forEach(c => c.destroy()); charts.length = 0;
  const spy = line('Buy & hold SPY', DATA.spy, css('--s0'), 1.5);
  charts.push(new Chart(document.getElementById('chart-all'), {{ type: 'line', options: base(),
    data: {{ labels: DATA.labels, datasets: [...DATA.series.map(s => line(s.name, s.values, css('--s' + s.slot))), spy] }} }}));
  DATA.series.forEach((s, i) => {{
    const lo = {{ label: '_lo', data: s.lo, borderWidth: 0, pointRadius: 0, fill: false }};
    const hi = {{ label: '_hi', data: s.hi, borderWidth: 0, pointRadius: 0, fill: '-1', backgroundColor: css('--band') }};
    charts.push(new Chart(document.getElementById('chart-' + DATA.accounts[i]), {{ type: 'line', options: base(),
      data: {{ labels: DATA.labels, datasets: [lo, hi, line(s.name, s.values, css('--s' + s.slot)),
        line('Buy & hold SPY', DATA.spy, css('--s0'), 1.5)] }} }}));
  }});
  if (DATA.minis) DATA.minis.forEach((m, i) => {{
    const o = base();
    o.plugins.legend.display = false;
    o.plugins.tooltip.callbacks.label = c => ` ${{c.dataset.label}}: ${{(c.parsed.y * 100).toFixed(1)}}%`;
    o.scales.x.ticks.display = false;
    o.scales.y.min = 0;
    o.scales.y.ticks.maxTicksLimit = 3;
    o.scales.y.ticks.callback = v => Math.round(v * 100) + '%';
    const n = m.labels.length;
    charts.push(new Chart(document.getElementById('m-' + i), {{ type: 'line', options: o,
      data: {{ labels: m.labels, datasets: [
        line('Learned share', m.values, css('--s2'), 2, n),
        {{ label: 'Equal split', data: Array(n).fill(m.ref), borderColor: css('--s0'), borderWidth: 1,
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
