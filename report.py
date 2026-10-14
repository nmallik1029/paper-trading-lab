"""Charts and self-contained HTML reports."""
import base64
import html

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from metrics import drawdown

SURFACE, INK, INK_2, MUTED, GRID, BASELINE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, assigned in fixed order
POS, NEG = (42, 120, 214), (227, 73, 72)    # diverging poles for return tints: blue up, red down
SHORT = {"Buy & hold SPY": "SPY", "Equal-weight sectors": "Equal-weight",
         "Sector momentum": "Momentum", "Trend following": "Trend",
         "Day trader (ridge)": "DT ridge", "Day trader (gbm)": "DT gbm",
         "SPY, daytime only": "SPY daytime", "Day trader, from scratch": "From scratch",
         "Day trader, full history": "Full history", "QQQ breakout trader": "QQQ breakout",
         "Staggered breakout portfolio": "Staggered", "Learning allocator": "Learner"}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titlecolor": INK,
    "figure.facecolor": SURFACE,
})


def _colors(names, bench):
    """The benchmark is neutral ink; strategies take categorical slots in order."""
    slots = iter(SERIES)
    return {n: INK_2 if n == bench else next(slots) for n in names}


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, length=0)


def _spread(values, min_gap):
    """Nudge label positions apart so no two sit closer than min_gap, keeping the
    group centered on the original positions."""
    order = np.argsort(values)
    orig = np.asarray(values, dtype=float)[order]
    pos = orig.copy()
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    pos -= pos.mean() - orig.mean()
    out = np.empty_like(pos)
    out[order] = pos
    return out


def growth_chart(results, colors, cost_bps, title=None):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    _style(ax)
    ends = {}
    for name, res in results.items():
        wealth = 100 * (1 + res.returns).cumprod()
        ax.plot(wealth.index, wealth.to_numpy(), color=colors[name], linewidth=1.5, label=name)
        ax.plot(wealth.index[-1], wealth.iloc[-1], "o", color=colors[name], markersize=4)
        ends[name] = (wealth.index[-1], wealth.iloc[-1])

    ax.set_xlim(min(r.returns.index[0] for r in results.values()), max(x for x, _ in ends.values()))
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    lo, hi = np.log10(ax.get_ylim())
    label_y = _spread([np.log10(v) for _, v in ends.values()], (hi - lo) * 0.05)
    for (name, (x, v)), y in zip(ends.items(), label_y):
        ax.text(x + pd.Timedelta(days=90), 10 ** y, f"{SHORT.get(name, name)}  ${v:,.0f}",
                va="center", color=INK_2)

    ax.legend(frameon=False, loc="upper left", labelcolor=INK_2)
    unit = "bp" if cost_bps == 1 else "bps"
    ax.set_title(title or f"Growth of $100 (log scale, after {cost_bps:g} {unit} trading costs)", loc="left")
    return fig


def drawdown_chart(results, colors, bench):
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.2), sharex=True, sharey=True)
    bench_dd = drawdown(results[bench].returns)
    for ax, (name, res) in zip(axes.flat, results.items()):
        _style(ax)
        dd = drawdown(res.returns)
        if name != bench:
            ax.plot(bench_dd.index, bench_dd.to_numpy(), color=MUTED, linewidth=0.8)
        ax.fill_between(dd.index, dd.to_numpy(), 0, color=colors[name], alpha=0.18, linewidth=0)
        ax.plot(dd.index, dd.to_numpy(), color=colors[name], linewidth=1.2)
        ax.set_title(f"{name}  ·  worst {dd.min():.0%}", loc="left")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    axes.flat[0].set_xlim(bench_dd.index[0], bench_dd.index[-1])
    fig.suptitle("Drawdowns: how far below its previous peak each strategy was",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.005, "Thin gray line in each panel = Buy & hold SPY, for reference.",
             color=INK_2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def learning_chart(curve, band, color):
    """Three panels sharing 'trading days since a fresh start': how wrong its beliefs
    are, how much it trades, and what $1,000 became, each across 60 fresh starts."""
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.2), sharex=True)
    x = curve.index.to_numpy()
    ax = axes[0]
    _style(ax)
    ax.fill_between(x, curve["error_p25"], curve["error_p75"], color=color, alpha=0.18, linewidth=0)
    ax.plot(x, curve["error_p50"], color=color, linewidth=1.6)
    ax.set_ylim(bottom=0)
    ax.set_title("How far its beliefs are from the six-year model's (avg gap per weight, bps)", loc="left")
    ax = axes[1]
    _style(ax)
    ax.plot(x, curve["trades"], color=color, linewidth=1.6)
    ax.set_ylim(bottom=0)
    ax.set_title("Trades per day (5-day average)", loc="left")
    ax = axes[2]
    _style(ax)
    bx = band.index.to_numpy()
    ax.fill_between(bx, band["p10"], band["p90"], color=color, alpha=0.18, linewidth=0)
    ax.plot(bx, band["p50"], color=color, linewidth=1.6)
    ax.axhline(1000, color=BASELINE, linewidth=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_title("What $1,000 became", loc="left")
    ax.set_xlabel("Trading days since starting from scratch", color=INK_2)
    ax.set_xlim(0, x[-1])
    fig.text(0.01, 0.005, "Line = median of 60 fresh starts (2017–2021). Shading = middle 50% "
             "(top panel) or middle 80% (bottom panel).", color=INK_2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    return fig


def _save(fig, path):
    """Write the chart to disk and return it base64-encoded for embedding."""
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return base64.b64encode(path.read_bytes()).decode()


def _badge(result):
    if result == "PASS":
        return '<span class="pass">✓ PASS</span>'
    if result == "FAIL":
        return '<span class="fail">✗ FAIL</span>'
    return html.escape(result)


def _fmt(col, v):
    if col == "Result":
        return _badge(v)
    if isinstance(v, str):
        return html.escape(v)
    if pd.isna(v):
        return "—"
    if col.startswith(("CAGR", "Volatility")):
        return f"{v:.1%}"
    if col.startswith(("Max drawdown", "Drawdown limit", "Turnover", "Time invested",
                       "Time in market")):
        return f"{v:.0%}"
    if col.startswith(("Win rate", "Direction right")):
        return f"{v:.1%}"
    if col.startswith(("Trades / day", "Cost / day")):
        return f"{v:.1f}"
    if col.startswith("Avg gain"):
        return f"{v:+.2f}"
    if col.startswith("Avg monthly IC"):
        return f"{v:+.3f}"
    if col.startswith("IC t-stat"):
        return f"{v:+.1f}"
    if col.startswith("Excess"):
        return f"{v:+.1%}"
    if col.startswith("Sharpe t-stat"):
        return f"{v:.1f}"
    if col.startswith("Sharpe"):
        return f"{v:.2f}"
    if col.startswith("t-stat"):
        return f"{v:+.1f}"
    if col.startswith("Trades"):
        return f"{v:.0f}"
    if col.startswith("Needed"):
        return f"{v:.1f}"
    if col.startswith("Average weight"):
        return f"{v:+.3f}"
    if col.startswith(("After", "1 ", "3 ", "6 ", "Six-year")) or col.endswith("years"):
        return f"{v:+.2f}"
    return html.escape(str(v))


def _table(df, show_index=True, fmt=_fmt, cell_bg=None, labels=None):
    """Render a DataFrame. Columns starting with '_' are hidden; a truthy '_default'
    column bolds that row. `labels` renames column headers for display."""
    cols = [c for c in df.columns if not str(c).startswith("_")]
    labels = labels or {}
    head = ("<th></th>" if show_index else "") + "".join(
        f"<th>{html.escape(str(labels.get(c, c)))}</th>" for c in cols)
    rows = []
    for idx, row in df.iterrows():
        cls = ' class="em"' if row.get("_default", False) else ""
        cells = f"<th>{html.escape(str(idx))}</th>" if show_index else ""
        for c in cols:
            style = f' style="background:{cell_bg(row[c])}"' if cell_bg else ""
            cells += f"<td{style}>{fmt(c, row[c])}</td>"
        rows.append(f"<tr{cls}>{cells}</tr>")
    return (f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def _tint(v):
    if pd.isna(v):
        return "transparent"
    r, g, b = POS if v >= 0 else NEG
    return f"rgba({r},{g},{b},{min(abs(v) / 0.4, 1) * 0.45:.2f})"


def _pct(col, v):
    return "—" if pd.isna(v) else f"{v:.1%}"


REPORT_CSS = """
:root { color-scheme: light; --page:#f7f7f5; --surface:#ffffff; --ink:#111110; --ink-2:#4f4e4a;
  --muted:#85837d; --grid:#e6e5df; --line:#e2e1db; --link:#2a78d6; --good:#0a7a0a; --bad:#c23434; }
@media (prefers-color-scheme: dark) { :root { color-scheme: dark; --page:#0f0f0e; --surface:#1a1a19;
  --ink:#f4f4f2; --ink-2:#c3c2b7; --muted:#8f8d86; --grid:#2c2c2a; --line:#2f2f2c; --link:#5598e7;
  --good:#3dbb3d; --bad:#e66767; } img { background:#fcfcfb; border-radius:8px; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--page); color:var(--ink);
  font:15px/1.65 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width:960px; margin:0 auto; padding:40px 20px 72px; }
a { color:var(--link); text-decoration:none; } a:hover { text-decoration:underline; }
.home { display:inline-block; margin-bottom:20px; font-size:14px; font-weight:600; }
h1 { font-size:28px; line-height:1.25; margin:0 0 12px; letter-spacing:-.01em; }
h2 { font-size:18px; line-height:1.35; margin:0 0 10px; }
p { margin:0 0 12px; }
.meta { color:var(--ink-2); font-size:16px; max-width:760px; }
.note { color:var(--ink-2); font-size:14px; margin:10px 0 14px; }
section { background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:24px; margin:18px 0; }
section > p, section li { color:var(--ink-2); }
ul { margin:0; padding-left:20px; } li { margin:6px 0; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:13.5px; font-variant-numeric:tabular-nums; margin:4px 0; }
th, td { padding:7px 10px; text-align:right; border-bottom:1px solid var(--grid); white-space:nowrap; }
th:first-child, td:first-child { text-align:left; padding-left:0; }
th:last-child, td:last-child { padding-right:0; }
thead th { color:var(--muted); font-weight:600; font-size:12.5px; }
tbody th { font-weight:500; }
tr.em td { font-weight:650; }
.pass { color:var(--good); font-weight:650; } .fail { color:var(--bad); font-weight:650; }
img { width:100%; height:auto; display:block; }
@media (max-width: 760px) { main { padding:28px 16px 56px; } section { padding:18px; } h1 { font-size:24px; } }
"""


def _page(title, body, home=None):
    """A standalone report page. `home`: link back to the dashboard, relative to the page."""
    back = f'<a class="home" href="{home}">← Paper Trading Lab</a>\n' if home else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{REPORT_CSS}</style></head>
<body><main>
{back}{body}
</main></body></html>
"""


def _bottom_line(summary, bench):
    b = summary.loc[bench]
    items = []
    for name, s in summary.drop(index=bench).iterrows():
        t = s["t-stat vs SPY"]
        verdict = "statistically significant" if abs(t) >= 2 else "not statistically significant"
        items.append(
            f"<li><b>{html.escape(name)}</b>: {s['CAGR']:.1%}/yr vs {b['CAGR']:.1%} for SPY · "
            f"Sharpe {s['Sharpe']:.2f} vs {b['Sharpe']:.2f} · worst drop {s['Max drawdown']:.0%} "
            f"vs {b['Max drawdown']:.0%} · difference from SPY: t = {t:+.1f} ({verdict})</li>")
    return "<ul>" + "".join(items) + "</ul>"


def write_report(out_dir, *, meta, summary, results, costs, momentum_grid, trend_grid,
                 periods, yearly):
    """The baseline-strategy report (run.py)."""
    out_dir.mkdir(exist_ok=True)
    bench = meta["benchmark"]
    colors = _colors(list(results), bench)
    growth = _save(growth_chart(results, colors, meta["cost_bps"]), out_dir / "growth.png")
    dds = _save(drawdown_chart(results, colors, bench), out_dir / "drawdowns.png")

    body = f"""
<h1>Baseline strategy backtest</h1>
<p class="meta">Daily data {meta['start']} → {meta['end']} ({meta['years']:.1f} years) ·
signals at the month-end close, filled at the next day's close · {meta['cost_bps']:g} bps cost
per dollar traded · uninvested cash earns the 13-week T-bill rate</p>

<section><h2>Bottom line</h2>{_bottom_line(summary, bench)}
<p class="note">A |t| of 2 or more is the usual bar for "probably not luck". Below that, the gap
from SPY is within what chance alone produces over this many years.</p></section>

<section><h2>Summary</h2>{_table(summary)}
<p class="note"><b>Sharpe t-stat</b>: is the return above T-bills distinguishable from zero?
<b>t-stat vs SPY</b>: is the return different from just holding SPY?
<b>Turnover / yr</b>: how much of the portfolio is traded per year (100% = the whole portfolio once).
<b>Trades / yr</b>: individual buy or sell orders, including small rebalancing trades.</p></section>

<section><img alt="Growth of $100 for each strategy, log scale" src="data:image/png;base64,{growth}"></section>
<section><img alt="Drawdowns for each strategy" src="data:image/png;base64,{dds}"></section>

<section><h2>How much do trading costs matter?</h2>
<p class="note">The same strategies at 0, 5 and 20 bps per dollar traded. Strategies that trade more
lose more as costs rise.</p>{_table(costs)}</section>

<section><h2>Parameter sensitivity</h2>
<p class="note">The bold rows are the settings fixed in advance. The others are a robustness check,
not a menu: if only one setting works, that's probably luck, and picking the best cell after
looking would be overfitting.</p>
<h2>Sector momentum</h2>{_table(momentum_grid, show_index=False)}
<h2 style="margin-top:20px">Trend following</h2>{_table(trend_grid, show_index=False)}</section>

<section><h2>By decade</h2>
<p class="note">Does any edge hold up across very different markets?</p>
{_table(periods, show_index=False)}</section>

<section><h2>Year by year</h2>
<p class="note">Blue = up, red = down; deeper shade = bigger move.</p>
{_table(yearly, fmt=_pct, cell_bg=_tint)}</section>

<section><h2>Method and caveats</h2><ul>
<li><b>Prices</b>: Yahoo Finance dividend-adjusted closes, so returns include dividends and are
after ETF fees.</li>
<li><b>Universe</b>: SPY plus the nine original Select Sector SPDR ETFs. All existed for the whole
period, which avoids survivorship bias (testing only on companies that survived to today).</li>
<li><b>No look-ahead</b>: each signal uses closes up to the month-end and trades at the next
trading day's close.</li>
<li><b>Costs</b>: {meta['cost_bps']:g} bps per dollar traded for spread and slippage;
commissions assumed $0.</li>
<li><b>Parameters</b> were fixed at standard values before running (12-month momentum skipping the
latest month, top 3 sectors, 200-day average), not tuned on this data.</li>
<li><b>Not modeled</b>: taxes (the trading strategies realize gains every year; buy-and-hold
defers them), fractional-share rounding, and intraday execution.</li>
<li><b>t-stats</b> assume independent daily returns. Treat them as rough.</li>
<li>Past results are not a forecast.</li>
</ul></section>"""
    path = out_dir / "report.html"
    path.write_text(_page("Baseline Strategy Backtest", body), encoding="utf-8")
    return path


def write_ml_report(out_dir, name, *, meta, card, results, yearly, coefs, log):
    """The ML scorecard report (run_ml.py)."""
    out_dir.mkdir(exist_ok=True)
    bench = meta["benchmark"]
    colors = _colors(list(results), bench)
    growth = _save(growth_chart(results, colors, meta["cost_bps"]), out_dir / f"{name}_growth.png")
    dds = _save(drawdown_chart(results, colors, bench), out_dir / f"{name}_drawdowns.png")

    b = card.loc[bench]
    items = []
    for n, s in card[card["Result"] != "reference"].iterrows():
        items.append(
            f"<li>{_badge(s['Result'])} <b>{html.escape(n)}</b>: {s['CAGR']:.1%}/yr vs "
            f"{b['CAGR']:.1%} for SPY · Sharpe {s['Sharpe']:.2f} vs {b['Sharpe']:.2f} · worst drop "
            f"{s['Max drawdown']:.0%} (limit {s['Drawdown limit']:.0%}) · "
            f"t = {s['t-stat vs risk-matched SPY']:+.1f}, needed {s['Needed t']:.1f}</li>")

    body = f"""
<h1>ML sector strategies</h1>
<p class="meta">{meta['label']}: {meta['start']} → {meta['end']} ({meta['years']:.1f} years) ·
trained walk-forward and retrained every month · {meta['cost_bps']:g} bps cost per dollar traded ·
{html.escape(meta['holdout_note'])}</p>

<section><h2>Bottom line</h2>
<p>{meta['n_variants']} variants tried so far, so a strategy needs <b>t ≥ {meta['t_needed']:.1f}</b>
against a risk-matched SPY, plus a worst drop within its limit, to pass.</p>
<ul>{''.join(items)}</ul></section>

<section><h2>Scorecard</h2>{_table(card)}
<p class="note"><b>Risk-matched SPY</b>: a mix of SPY and cash with the same volatility as the
strategy. The safe strategy isn't failed for being safe, and a risky one can't pass just by taking
more risk. <b>Needed t</b>: the score the luckiest of all variants tried would beat only 2.5% of the
time by chance; it rises with every variant. <b>Sector momentum</b> is the simple rule the ML has to
beat to be worth its complexity.</p></section>

<section><img alt="Growth of $100 for each strategy, log scale" src="data:image/png;base64,{growth}"></section>
<section><img alt="Drawdowns for each strategy" src="data:image/png;base64,{dds}"></section>

<section><h2>What the main (ridge) model learned</h2>
<p class="note">Average weight on each feature across the monthly retrains. Positive means the
model favors sectors ranked high on that feature. Market-wide features are the same for every
sector, so a linear model can't use them to rank sectors and their weights sit near zero; only the
gbm variant can use them, through interactions.</p>{_table(coefs)}</section>

<section><h2>Year by year</h2>
<p class="note">Blue = up, red = down; deeper shade = bigger move.</p>
{_table(yearly, fmt=_pct, cell_bg=_tint)}</section>

<section><h2>Every variant tried</h2>
<p class="note">Never delete rows from experiments.csv: forgotten failures are how backtests lie.</p>
{_table(log, show_index=False)}</section>

<section><h2>How it works</h2><ul>
<li><b>Features</b> (from past data only), for each sector each month-end: returns over 1, 3, 6 and
12 months (skipping the latest month), volatility, and distance from the 200-day average, each
ranked against the other sectors; plus market trend, market volatility and the T-bill rate.</li>
<li><b>Target</b>: how the sector ranks against the others next month.</li>
<li><b>Walk-forward training</b>: at each month-end the model trains on every earlier month (at
least 36), then predicts. It never sees a month before deciding on it.</li>
<li><b>Portfolios</b>: <i>ML moderate</i> holds the top 3 sectors, always invested. <i>ML safe</i>
holds the same 3 but moves to T-bills when SPY ends the month below its 200-day average.</li>
<li><b>Holdout</b>: 2018–today stays locked until the design is final, then gets tested once.
Caveat: the baseline report already showed how simple momentum did after 2018, so the holdout
isn't perfectly blind.</li>
<li>Same engine as the baselines: next-day fills, drift between rebalances,
{meta['cost_bps']:g} bps costs, cash earns T-bills. Past results are not a forecast.</li>
</ul></section>"""
    path = out_dir / f"{name}.html"
    path.write_text(_page("ML Sector Strategies", body), encoding="utf-8")
    return path


def write_daytrader_report(out_dir, name, *, meta, card, results, activity, costs, quality,
                           coefs, yearly, log):
    """The ML day-trader report (run_daytrader.py)."""
    out_dir.mkdir(exist_ok=True)
    bench = meta["benchmark"]
    colors = _colors(list(results), bench)
    growth = _save(growth_chart(results, colors, meta["cost_bps"]), out_dir / f"{name}_growth.png")
    dds = _save(drawdown_chart(results, colors, bench), out_dir / f"{name}_drawdowns.png")

    b = card.loc[bench]
    items = []
    for n, s in card[card["Result"] != "reference"].iterrows():
        items.append(
            f"<li>{_badge(s['Result'])} <b>{html.escape(n)}</b>: {s['CAGR']:.1%}/yr vs "
            f"{b['CAGR']:.1%} for SPY · Sharpe {s['Sharpe']:.2f} vs {b['Sharpe']:.2f} · worst drop "
            f"{s['Max drawdown']:.0%} · {activity.loc[n, 'Trades / day']:.1f} trades a day · "
            f"t = {s['t-stat vs risk-matched SPY']:+.1f}, needed {s['Needed t']:.1f}</li>")

    body = f"""
<h1>ML day trader</h1>
<p class="meta">{meta['label']}: {meta['start']} → {meta['end']} ({meta['years']:.1f} years) ·
SPY and QQQ in 30-minute blocks from 10:00, out by the close · retrained every month ·
{meta['cost_bps']:g} bp cost per dollar traded · {html.escape(meta['holdout_note'])}</p>

<section><h2>Bottom line</h2>
<p>{meta['n_variants']} variants tried so far (monthly and intraday together), so a strategy needs
<b>t ≥ {meta['t_needed']:.1f}</b> against a risk-matched SPY to pass. The ultra-risky slot has no
drawdown limit.</p>
<ul>{''.join(items)}</ul></section>

<section><h2>Scorecard</h2>{_table(card)}
<p class="note"><b>SPY, daytime only</b> holds SPY from 10:00 to the close every day and nothing
overnight, the simplest possible day trade. <b>Last-half-hour momentum</b> is a published intraday
pattern: if SPY is up from yesterday's close to 10:00, hold it for the final 30 minutes.</p></section>

<section><img alt="Growth of $100 for each strategy, log scale" src="data:image/png;base64,{growth}"></section>
<section><img alt="Drawdowns for each strategy" src="data:image/png;base64,{dds}"></section>

<section><h2>Trading activity</h2>
<p class="note">Day trading lives or dies on the average gain per trade versus the cost per trade.
A block's gain is before costs; each trade (buy or sell) costs {meta['cost_bps']:g} bp.</p>
{_table(activity)}</section>

<section><h2>How much do trading costs matter?</h2>
<p class="note">The same trades, charged 0, 1 and 3 bps per dollar traded.</p>{_table(costs)}</section>

<section><h2>Can the model predict anything?</h2>
<p class="note"><b>IC</b> (information coefficient): the rank correlation between the predicted and
actual next-30-minute returns, averaged by month. 0 means no skill; for intraday returns, even
+0.02 to +0.05 that holds up would be a real, useful signal. <b>Direction right</b>: how often the
predicted sign (up or down) was correct; 50% is a coin flip.</p>{_table(quality)}</section>

<section><h2>What the main (ridge) model learned</h2>
<p class="note">Average weight on each feature across the monthly retrains, in bps of predicted
30-minute return per standard deviation of the feature.</p>{_table(coefs)}</section>

<section><h2>Year by year</h2>
<p class="note">Blue = up, red = down; deeper shade = bigger move.</p>
{_table(yearly, fmt=_pct, cell_bg=_tint)}</section>

<section><h2>Every variant tried</h2>
<p class="note">Shared with the monthly ML strategies. Never delete rows.</p>
{_table(log, show_index=False)}</section>

<section><h2>How it works</h2><ul>
<li><b>Data</b>: 5-minute bars from Alpaca's full-market feed since 2016, official trading
hours only (early-close days end at 13:00).</li>
<li><b>Decisions</b> at 10:00, 10:30, …, 15:30. Each uses only bars finished by then and fills at
the next bar's open; the last block ends at the closing price. Nothing is held overnight.</li>
<li><b>Features</b>: recent returns (5, 15, 30 minutes, since the open, the overnight gap,
yesterday, the last 5 days), recent volatility, volume versus normal for that time of day, the
time of day, and which ETF.</li>
<li><b>Target</b>: the return over the next 30 minutes (capped at ±1.5% during training).</li>
<li><b>Rule</b>: buy the ETF with the higher prediction when it beats the round-trip cost
({2 * meta['cost_bps']:g} bps); keep holding while the prediction stays positive; otherwise cash.</li>
<li><b>Walk-forward</b>: retrained at the start of each month on all earlier months (at least 12),
so it never sees the month it trades.</li>
<li><b>Cash</b> earns the T-bill rate: the account is always in cash at the close, when brokers
calculate interest.</li>
<li><b>Not modeled</b>: delays in getting real-time data, partial fills, and taxes (every gain is
short-term). Past results are not a forecast.</li>
</ul></section>"""
    path = out_dir / f"{name}.html"
    path.write_text(_page("ML Day Trader", body), encoding="utf-8")
    return path


def write_coldstart_report(out_dir, *, meta, card, results, band, curve, evolution):
    """The learn-from-scratch day-trader report (run_coldstart.py)."""
    out_dir.mkdir(exist_ok=True)
    bench = meta["benchmark"]
    colors = _colors(list(results), bench)
    me = "Day trader, from scratch"
    learn = _save(learning_chart(curve, band, colors[me]), out_dir / "coldstart_learning.png")
    growth = _save(growth_chart(results, colors, meta["cost_bps"]), out_dir / "coldstart_growth.png")

    s, b = card.loc[me], card.loc[bench]
    first, year = curve.iloc[0], curve.loc[min(250, curve.index[-1])]
    body = f"""
<h1>Day trader that learns from scratch</h1>
<p class="meta">Development period {meta['start']} → {meta['end']} ({meta['years']:.1f} years) ·
starts knowing only {meta['seed_days']} trading days, retrains every morning on everything seen
since · {meta['cost_bps']:g} bp cost per dollar traded</p>

<section><h2>Bottom line</h2><ul>
<li>{_badge(s['Result'])} <b>Started from scratch in January 2017</b>: {s['CAGR']:.1%}/yr vs
{b['CAGR']:.1%} for SPY · worst drop {s['Max drawdown']:.0%} · {s['Trades / day']:.1f} trades a day ·
t = {s['t-stat vs risk-matched SPY']:+.1f}, needed {s['Needed t']:.1f}
({meta['n_variants']} variants tried).</li>
<li><b>It learns, slowly.</b> On day 1 its beliefs are off by {first['error_p50']:.1f} bps per weight
on average; after a year, by {year['error_p50']:.2f}. Trades per day fall from {first['trades']:.1f}
to {year['trades']:.1f} as it stops acting on noise.</li>
</ul></section>

<section><h2>Scorecard</h2>{_table(card)}
<p class="note"><b>Full history</b> is the earlier day trader, which starts with a year of data
and retrains monthly. Same features, same trading rule.</p></section>

<section><h2>The learning curve</h2>
<p class="note">Sixty fresh starts, one at the beginning of every month from 2017 to 2021, each
followed for {meta['horizon']} trading days. The bottom panel is the normal range the dashboard
shades for the live account.</p>
<img alt="Learning curve: belief error, trades per day and value of $1,000 versus days since start" src="data:image/png;base64,{learn}"></section>

<section><h2>Watching its beliefs settle</h2>
<p class="note">The continuous run's weights at each checkpoint (bps of predicted 30-minute return
per typical move in the feature), next to what a model trained on all six years settles on.
Early weights are big and often the wrong sign; they shrink toward the real, small effects.</p>
{_table(evolution)}</section>

<section><img alt="Growth of $100: from scratch, full history, and SPY" src="data:image/png;base64,{growth}"></section>

<section><h2>How it works</h2><ul>
<li>On its first morning the model trains on the {meta['seed_days']} trading days before it (about
{meta['seed_days'] * 24} examples: 12 half-hours × 2 ETFs × {meta['seed_days']} days).</li>
<li>Every morning after that it retrains on everything since, about 24 more examples a day,
and never forgets anything.</li>
<li>Same features, trading rule, costs and timing as the full-history day trader; see the day
trader report.</li>
<li>The fresh starts overlap in time, so they show the range of experiences, not independent
results. Past results are not a forecast.</li>
</ul></section>"""
    path = out_dir / "coldstart_dev.html"
    path.write_text(_page("From-Scratch Day Trader", body), encoding="utf-8")
    return path
