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
from live import read_log

ROOT = Path(__file__).parent
SITE = ROOT / "site"
START_VALUE = 1000.0

# Display order runs from least to most risky; colors are categorical slots 1-3 in order.
STRATEGIES = {
    "safe": dict(name="ML safe", risk="Safe", slot=1, report=("ml_dev", "ML safe"),
                 about="Holds the 3 sectors the model ranks best; moves to cash when the market "
                       "is in a downtrend. Rebalances monthly."),
    "moderate": dict(name="ML moderate", risk="Moderate", slot=2, report=("ml_dev", "ML moderate"),
                     about="Holds the 3 sectors the model ranks best, always invested. "
                           "Rebalances monthly."),
    "daytrader": dict(name="ML day trader", risk="Ultra risky", slot=3,
                      report=("coldstart", "Day trader, from scratch"),
                      about="Learns from scratch: it started out knowing only the 5 trading days "
                            "before launch and retrains every morning on everything since. Every "
                            "30 minutes it buys SPY or QQQ when its predicted gain beats trading "
                            "costs, and it sells everything by the close."),
}
PERIODS = {"ml_dev": "Backtest 2003–2017", "coldstart": "Backtest 2017–2022, learning from scratch"}


def money(v):
    return f"${v:,.2f}"


def signed_pct(v):
    return f"{v:+.2%}"


def trading_days(launch, now):
    """Trading days from the launch through today (today only once the market has opened)."""
    cal = market_calendar()
    days = [d for d in cal.index if launch <= d <= now.normalize().tz_localize(None)]
    opened = now.hour * 60 + now.minute >= 9 * 60 + 30
    return [d for d in days if d < now.normalize().tz_localize(None) or opened]


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


def normal_range(name, column, horizon):
    """10th-90th percentile of $1,000 after h trading days, for h = 0..horizon.

    For the from-scratch day trader this comes from its 60 simulated fresh starts. For the
    others it comes from every h-day stretch of the development-period backtest."""
    if name == "coldstart":
        band = pd.read_csv(ROOT / "report" / "coldstart_band.csv", index_col=0)
        band = band.reindex(range(horizon + 1)).ffill()  # past a year, hold the last range
        return band["p10"].tolist(), band["p90"].tolist()
    r = pd.read_csv(ROOT / "report" / f"{name}_daily_returns.csv", index_col=0)[column].dropna()
    growth = np.r_[1.0, (1 + r).cumprod().to_numpy()]
    lo, hi = [START_VALUE], [START_VALUE]
    for h in range(1, horizon + 1):
        ratio = growth[h:] / growth[:-h]
        lo.append(START_VALUE * np.percentile(ratio, 10))
        hi.append(START_VALUE * np.percentile(ratio, 90))
    return lo, hi


def verdict(name, column):
    s = pd.read_csv(ROOT / "report" / f"{name}_scorecard.csv", index_col=0).loc[column]
    return (f"{PERIODS[name]}: {float(s['CAGR']):.1%}/yr, worst drop {float(s['Max drawdown']):.0%}. "
            f"{'Passed' if s['Result'] == 'PASS' else 'Failed'} the scorecard "
            f"(t = {float(s['t-stat vs risk-matched SPY']):+.1f}, needed {float(s['Needed t']):.1f}).")


def beliefs():
    """The day trader's weights after each morning's retrain, plus the long-history model's
    weights for comparison. Returns (html, chart data)."""
    ref = json.loads((ROOT / "report" / "daytrader_reference_weights.json").read_text())
    history = [r for r in read_log("daytrader_model") if r.get("action") == "retrained"]
    order = sorted(ref["weights"], key=lambda f: -abs(ref["weights"][f]))
    if not history:
        return ("<p class='empty'>Not trained yet. It trains for the first time at about 9:25 AM ET "
                "on its first trading day.</p>", None)
    last = history[-1]
    note = (f"Learned from <b>{last['examples']:,}</b> examples ({last['days_learned']} trading days "
            f"since {last['seed_start']}), retrained {pd.Timestamp(last['time']):%b %d}.")
    if len(history) > 1:
        prev = history[-2]["weights"]
        moved = max(order, key=lambda f: abs(last["weights"][f] - prev[f]))
        note += (f" Biggest change since the previous retrain: <b>{html.escape(ref['names'][moved])}</b>, "
                 f"{prev[moved]:+.2f} → {last['weights'][moved]:+.2f} bps.")
        flipped = [f for f in order if np.sign(last["weights"][f]) != np.sign(prev[f])]
        if flipped:
            note += f" Flipped direction: {html.escape(', '.join(ref['names'][f] for f in flipped))}."
    agree = sum(np.sign(last["weights"][f]) == np.sign(ref["weights"][f]) for f in order)
    note += (f" It agrees in direction with the long-history model on <b>{agree} of {len(order)}</b> "
             "features.")
    data = {"labels": [pd.Timestamp(r["time"]).strftime("%b %d") for r in history],
            "features": [{"name": ref["names"][f], "ref": ref["weights"][f],
                          "values": [r["weights"][f] for r in history]} for f in order]}
    minis = "".join(f'<div class="mini"><div class="mini-title">{html.escape(ref["names"][f])}</div>'
                    f'<div class="mini-chart"><canvas id="w-{i}"></canvas></div></div>'
                    for i, f in enumerate(order))
    return (f"<p class='note'>{note}</p><div class='minis'>{minis}</div>"
            "<p class='note'>Each panel is one weight (bps of predicted 30-minute return per typical "
            "move in that feature) after each morning's retrain. The gray line is where a model "
            "trained on 10 years of data settled. If learning works, the colored lines should "
            "drift toward the gray ones.</p>", data)


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


def trades_table(account, limit=8):
    trades = [r for r in read_log(account) if r.get("action") in ("buy", "sell")][-limit:][::-1]
    rows = [[html.escape(pd.Timestamp(r["time"]).strftime("%b %d, %I:%M %p")),
             f'<span class="{r["action"]}">{r["action"].upper()}</span>', html.escape(r["symbol"]),
             money(float(r["dollars"])), money(float(r["price"])), html.escape(r.get("why", ""))]
            for r in trades]
    return table(["When (ET)", "Side", "Symbol", "Amount", "Price", "Why"], rows, "No trades yet.")


def decisions_table(now):
    """The day trader's most recent day of decisions."""
    rows = [r for r in read_log("daytrader") if r.get("action") in ("decision", "skipped")]
    if not rows:
        return table([], [], "No decisions yet. The first one happens at 10:00 AM ET on a trading day.")
    last_day = pd.Timestamp(rows[-1]["time"]).date()
    rows = [r for r in rows if pd.Timestamp(r["time"]).date() == last_day]
    out = []
    for r in rows:
        p = r.get("predicted_bps", {})
        out.append([html.escape(r.get("decision", "")),
                    f"{p.get('SPY', float('nan')):+.2f}" if p else "—",
                    f"{p.get('QQQ', float('nan')):+.2f}" if p else "—",
                    html.escape(f"{r.get('before', '')} → {r.get('after', '')}" if r["action"] == "decision"
                                else f"skipped: {r.get('why', '')}")])
    label = "Today" if last_day == now.date() else last_day.strftime("%b %d")
    return (f"<h4>{label}'s decisions</h4><p class='note'>Predicted next-30-minute return, in bps "
            f"(1 bp = 0.01%). It buys only when a prediction beats 2 bps, the round-trip cost.</p>"
            + table(["Time", "SPY", "QQQ", "Position"], out, ""))


def build():
    now = pd.Timestamp.now(tz=TZ)
    logs = [r for a in STRATEGIES for r in read_log(a)]
    launch = (min(pd.Timestamp(r["time"]) for r in logs).tz_localize(None).normalize()
              if logs else now.normalize().tz_localize(None))
    days = trading_days(launch, now)
    labels = ["Start"] + [d.strftime("%b %d") for d in days]
    spy = [START_VALUE] + spy_values(days, now)

    series, cards, tiles = [], [], []
    for account, info in STRATEGIES.items():
        client = trading_client(account)
        values = [START_VALUE] + account_values(client, days)
        lo, hi = normal_range(*info["report"], len(days))
        series.append({"name": info["name"], "slot": info["slot"], "values": values,
                       "lo": lo, "hi": hi})
        change = values[-1] / START_VALUE - 1
        tiles.append(f"""<div class="tile"><div class="tile-label"><i class="dot s{info['slot']}"></i>
{html.escape(info['name'])}</div><div class="tile-value">{money(values[-1])}</div>
<div class="tile-sub">{signed_pct(change)} since start</div></div>""")
        extra = ""
        if account == "daytrader":
            belief_html, belief_data = beliefs()
            extra = f"<h4>What the model believes</h4>{belief_html}" + decisions_table(now)
        cards.append(f"""<section class="card">
<div class="card-head"><h2><i class="dot s{info['slot']}"></i>{html.escape(info['name'])}</h2>
<span class="badge">{html.escape(info['risk'])}</span></div>
<p class="about">{html.escape(info['about'])}</p>
<p class="note">{html.escape(verdict(*info['report']))}</p>
<div class="chart"><canvas id="chart-{account}" aria-label="{html.escape(info['name'])} value over time"></canvas></div>
<p class="note">Shaded: the range the backtest says is normal (80% of outcomes) after this many days.</p>
<h4>Holdings</h4>{positions_table(client)}
{extra}
<h4>Recent trades</h4>{trades_table(account)}
</section>""")
    tiles.append(f"""<div class="tile"><div class="tile-label"><i class="dot s0"></i>Buy &amp; hold SPY</div>
<div class="tile-value">{money(spy[-1])}</div>
<div class="tile-sub">{signed_pct(spy[-1] / START_VALUE - 1)} since start</div></div>""")

    data = {"labels": labels, "spy": spy, "series": series,
            "accounts": list(STRATEGIES), "beliefs": belief_data}
    SITE.mkdir(exist_ok=True)
    reports = SITE / "reports"
    reports.mkdir(exist_ok=True)
    for name in ("report", "ml_dev", "daytrader_dev", "coldstart_dev"):
        src = ROOT / "report" / f"{name}.html"
        if src.exists():
            shutil.copy(src, reports / f"{name}.html")
    (SITE / "index.html").write_text(page(now, launch, tiles, cards, data), encoding="utf-8")
    print(f"dashboard written: {SITE / 'index.html'}")


def page(now, launch, tiles, cards, data):
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
  .minis {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(220px, 1fr)); gap:10px; margin:8px 0; }}
  .mini {{ border:1px solid var(--grid); border-radius:8px; padding:8px 10px 4px; }}
  .mini-title {{ font-size:12px; color:var(--ink-2); line-height:1.3; min-height:2.6em; }}
  .mini-chart {{ position:relative; height:90px; }}
  .scroll {{ overflow-x:auto; }}
  table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; font-size:13px; }}
  th, td {{ text-align:left; padding:6px 10px; border-bottom:1px solid var(--grid); white-space:nowrap; }}
  th {{ color:var(--ink-2); font-weight:600; font-size:12px; }}
  .buy {{ color:var(--good); font-weight:650; }} .sell {{ color:var(--bad); font-weight:650; }}
  footer {{ color:var(--ink-2); font-size:12.5px; margin-top:28px; }}
  a {{ color:var(--s1); }}
</style></head>
<body><main>
<h1>Paper Trading Lab</h1>
<p class="sub">Three strategies, each with $1,000 of fake money in its own Alpaca paper account,
trading automatically. Started {launch:%b %d, %Y} · updated {now:%b %d, %Y %I:%M %p} ET.</p>
<div class="tiles">{''.join(tiles)}</div>
<section class="card"><h2>Growth of $1,000</h2>
<p class="note">All three strategies against simply holding SPY. Hover for values.</p>
<div class="chart"><canvas id="chart-all" aria-label="All strategies and SPY over time"></canvas></div></section>
{''.join(cards)}
<footer>Fake money on paper accounts; not investment advice. None of these strategies passed its
backtest scorecard, so this is an experiment, not an expectation of profit.
Backtest reports: <a href="reports/report.html">baselines</a> ·
<a href="reports/ml_dev.html">monthly ML</a> · <a href="reports/daytrader_dev.html">day trader</a> ·
<a href="reports/coldstart_dev.html">day trader learning from scratch</a>.</footer>
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
const line = (label, data, color, width = 2) => ({{ label, data, borderColor: color, backgroundColor: color,
  borderWidth: width, pointRadius: DATA.labels.length < 3 ? 4 : 0, pointHoverRadius: 4, tension: 0 }});
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
  if (DATA.beliefs) DATA.beliefs.features.forEach((f, i) => {{
    const o = base();
    o.plugins.legend.display = false;
    o.plugins.tooltip.callbacks.label = c => ` ${{c.dataset.label}}: ${{c.parsed.y >= 0 ? '+' : ''}}${{c.parsed.y.toFixed(2)}} bps`;
    o.scales.x.ticks.display = false;
    o.scales.y.ticks.maxTicksLimit = 3;
    o.scales.y.ticks.callback = v => (v > 0 ? '+' : '') + v;
    const n = DATA.beliefs.labels.length;
    charts.push(new Chart(document.getElementById('w-' + i), {{ type: 'line', options: o,
      data: {{ labels: DATA.beliefs.labels, datasets: [
        {{ label: 'Live model', data: f.values, borderColor: css('--s3'), backgroundColor: css('--s3'),
           borderWidth: 2, pointRadius: n < 3 ? 3 : 0, pointHoverRadius: 3 }},
        {{ label: '10-year model', data: Array(n).fill(f.ref), borderColor: css('--s0'),
           borderWidth: 1, pointRadius: 0, pointHoverRadius: 0 }}] }} }}));
  }});
}}
draw();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
</script>
</body></html>
"""


if __name__ == "__main__":
    build()
