"""Rebuild the lab v2 research report and the files the dashboard reads.

    .venv\\Scripts\\python run_v2_report.py

Writes report/v2_research.html, report/v2_summary.json and each finalist's full-history
daily returns (report/v2_<name>_daily_returns.csv), used for the dashboard's normal ranges.
Every number here was already scored in research_v2.py / the intraday and allocator runs,
and the holdouts were looked at once (HOLDOUT_USED_V2.txt).
"""
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

import allocators as al
import intraday_v2 as iv
import research_v2 as rv
from research_multi import load as load_multi
from research_staggered import staggered
from backtest import Result
from intraday_data import load_bars, panels
from metrics import return_stats
from report import _colors, _page, _save, _table, growth_chart
from scorecard import t_needed
from universe import load_daily

OUT = Path(__file__).parent / "report"
FUND_SLEEVES = ["Crypto trend (BTC)", "Short volatility (VIX curve)", "Overnight TQQQ, uptrend only",
                "RSI(2) dip buying, 3x", "Leveraged trend + vol target"]
SOURCES = [
    ("Zarattini, Aziz & Barbon (2024), Beat the Market: An Effective Intraday Momentum Strategy for SPY",
     "https://www.sfi.ch/fr/publications/n-24-97-beat-the-market-an-effective-intraday-momentum-strategy-for-s-p500-etf-spy"),
    ("Zarattini & Aziz (2023), opening-range breakout, summary and replication notes",
     "https://www.cxoadvisory.com/technical-trading/day-trading-with-an-opening-range-breakout-strategy"),
    ("Independent ORB replication: gross reproduced, net zero",
     "https://www.mql5.com/en/blogs/post/776235"),
    ("Lou, Polk & Skouras (2019), A Tug of War: Overnight Versus Intraday Expected Returns",
     "https://www.fmg.ac.uk/cm/publications/discussion-papers/tug-war-overnight-versus-intraday-expected-returns"),
    ("Night Moves: is the overnight drift the grandmother of all anomalies?",
     "https://www.advisorperspectives.com/articles/2022/06/24/night-moves-is-the-overnight-drift-the-grandmother-of-all-market-anomalies"),
    ("Moreira & Muir (2017), Volatility-Managed Portfolios", "https://www.nber.org/papers/22208"),
    ("Three strategies for trading volatility ETPs (VIX term structure)",
     "https://tradingvolatility.substack.com/p/three-strategies-for-trading-volatility"),
    ("Trend-following and mean-reversion in Bitcoin (Quantpedia)",
     "https://quantpedia.com/trend-following-and-mean-reversion-in-bitcoin/"),
    ("The pre-FOMC announcement drift: more recent evidence (New York Fed)",
     "https://libertystreeteconomics.newyorkfed.org/2018/11/the-pre-fomc-announcement-drift-more-recent-evidence.html"),
    ("The turn-of-the-month effect (CXO Advisory)",
     "https://www.cxoadvisory.com/calendar-effects/the-turn-of-the-month-effect/"),
    ("Connors RSI(2) on the S&P 500, backtest notes", "https://backtrex.com/en/backtests/connors-rsi-2-sp-500"),
]


def stats(ret, rf):
    s = return_stats(ret.dropna(), rf)
    return {"start": str(ret.dropna().index[0].date()), "end": str(ret.dropna().index[-1].date()),
            "cagr": s["CAGR"], "sharpe": s["Sharpe"], "max_drawdown": s["Max drawdown"]}


def main():
    d = load_daily()
    rf = (d["close"]["^IRX"].ffill().shift(1) / 100 / 252).fillna(0.0)
    spy = d["close"]["SPY"].dropna()
    spy_r = spy / spy.shift(1) - 1
    built = rv.build(d, rf)

    # Every daily candidate, development window and holdout.
    rows = []
    for name, (df, (a, b)) in built.items():
        dev, hold = df["ret"].loc[a:b], df["ret"].loc[pd.Timestamp(b) + pd.Timedelta(days=1):]
        sd, sh = return_stats(dev, rf), return_stats(hold, rf)
        rows.append({"Strategy": name, "Dev years": f"{dev.index[0]:%Y}–{dev.index[-1]:%Y}",
                     "CAGR (dev)": sd["CAGR"], "Sharpe (dev)": sd["Sharpe"], "Max drawdown (dev)": sd["Max drawdown"],
                     "CAGR (holdout)": sh["CAGR"], "Sharpe (holdout)": sh["Sharpe"],
                     "Max drawdown (holdout)": sh["Max drawdown"], "_default": name in FUND_SLEEVES})
    daily_table = pd.DataFrame(rows)

    pan = {**panels(load_bars()), **panels(load_bars(symbols=["TQQQ", "SQQQ"], name="lev"))}
    intraday = {"Noise-area breakout, 3x": iv.noise_area(pan, rf, sizing="full"),
                "Noise-area breakout, paper sizing": iv.noise_area(pan, rf, sizing="paper"),
                "Opening-range breakout, 3x": iv.opening_range(pan, rf, sizing="full"),
                "Opening-range breakout, paper sizing": iv.opening_range(pan, rf, sizing="paper")}
    rows = []
    for name, df in intraday.items():
        sd, sh = return_stats(df["ret"].loc["2017":"2022"], rf), return_stats(df["ret"].loc["2023":], rf)
        rows.append({"Strategy": name, "Trades / day": df["trades"].mean(),
                     "CAGR (dev)": sd["CAGR"], "Sharpe (dev)": sd["Sharpe"], "Max drawdown (dev)": sd["Max drawdown"],
                     "CAGR (holdout)": sh["CAGR"], "Sharpe (holdout)": sh["Sharpe"],
                     "Max drawdown (holdout)": sh["Max drawdown"], "_default": name == "Noise-area breakout, 3x"})
    intraday_table = pd.DataFrame(rows)

    # v2.1: the live QQQ trader and the staggered portfolio, with costs from current fund
    # prices (split-adjusted history distorts them), plus the activity experiments.
    _, _, funds, pans = load_multi()
    funds = pd.concat([funds, d["close"][["TQQQ", "SQQQ"]]], axis=1)
    pans = {**pans, **pan}
    q = iv.breakout_sleeve(pans["QQQ"], funds["TQQQ"], funds["SQQQ"], 3.0)
    qqq_trader = (1 + q["ret"]) * (1 + rf.reindex(q.index).fillna(0.0)) - 1
    stag, _ = staggered(pans, funds, rf)
    activity = {
        "QQQ breakout, 30-min checks (live)": (qqq_trader, q["trades"]),
        "Staggered breakout portfolio (live)": (stag["ret"], stag["trades"]),
        "QQQ breakout, 5-min checks": "v21_breakout_every1_allin", "QQQ breakout, 5-min checks, scaled": "v21_breakout_every1_scaled",
        "VWAP follow, 3x, 5-min": "v21_momentum_3x", "VWAP fade, 3x, 5-min": "v21_reversion_3x",
        "VWAP follow, QQQ 1x, 5-min": "v21_momentum_1x-long", "VWAP fade, QQQ 1x, 5-min": "v21_reversion_1x-long",
        "Breakout on 6 other markets, 30-min": "v21_multi_every6", "Breakout on 6 other markets, 15-min": "v21_multi_every3",
    }
    rows = []
    for name, src in activity.items():
        if isinstance(src, str):
            df = pd.read_csv(OUT / f"{src}_daily.csv", index_col=0, parse_dates=True)
            ret, trades = df["ret"], df["trades"]
        else:
            ret, trades = src
        sd, sh = return_stats(ret.loc["2017":"2022"], rf), return_stats(ret.loc["2023":], rf)
        rows.append({"Strategy": name, "Trades / day": trades.loc["2017":].mean(), "CAGR (dev)": sd["CAGR"],
                     "Sharpe (dev)": sd["Sharpe"], "CAGR (holdout)": sh["CAGR"] if "live" in name or "markets" in name else np.nan,
                     "Sharpe (holdout)": sh["Sharpe"] if "live" in name or "markets" in name else np.nan,
                     "_default": "live" in name})
    activity_table = pd.DataFrame(rows)

    R = pd.DataFrame({k: built[k][0]["ret"] for k in FUND_SLEEVES}).loc["2011-10-01":]
    costs = {"Crypto trend (BTC)": 30e-4}
    fund = al.combine(R, al.equal_weight(R), rf, costs)
    learner = al.combine(R, al.hedge(R, rf)[0], rf, costs)
    risk_parity = al.combine(R, al.risk_parity(R), rf, costs)
    rows = []
    for name, ret in {"Equal weight (the fund)": fund, "Learning allocator, Hedge": learner,
                      "Risk parity": risk_parity}.items():
        sd, sh = return_stats(ret.loc["2012-07":"2017"], rf), return_stats(ret.loc["2018":], rf)
        rows.append({"Allocator": name, "CAGR (dev)": sd["CAGR"], "Sharpe (dev)": sd["Sharpe"],
                     "Max drawdown (dev)": sd["Max drawdown"], "CAGR (holdout)": sh["CAGR"],
                     "Sharpe (holdout)": sh["Sharpe"], "Max drawdown (holdout)": sh["Max drawdown"],
                     "_default": name != "Risk parity"})
    alloc_table = pd.DataFrame(rows)

    finalists = {"fund": fund.loc["2012-07":], "learner": learner.loc["2012-07":],
                 "daytrader": qqq_trader.loc["2016-02":], "staggered": stag["ret"].loc["2016-02":]}
    for k, ret in finalists.items():
        ret.rename("ret").to_csv(OUT / f"v2_{k}_daily_returns.csv")
    n = len(pd.read_csv(Path(__file__).parent / "experiments.csv"))
    summary = {"variants": n, "t_needed": t_needed(n), "finalists": {
        "fund": {"dev": stats(fund.loc["2012-07":"2017"], rf), "holdout": stats(fund.loc["2018":], rf)},
        "learner": {"dev": stats(learner.loc["2012-07":"2017"], rf), "holdout": stats(learner.loc["2018":], rf)},
        "daytrader": {"dev": stats(finalists["daytrader"].loc["2017":"2022"], rf),
                      "holdout": stats(finalists["daytrader"].loc["2023":], rf)},
        "staggered": {"dev": stats(finalists["staggered"].loc["2017":"2022"], rf),
                      "holdout": stats(finalists["staggered"].loc["2023":], rf)},
        "spy": {"dev": stats(spy_r.loc["2012-07":"2017"], rf), "holdout": stats(spy_r.loc["2018":], rf)}}}
    (OUT / "v2_summary.json").write_text(json.dumps(summary, indent=1, default=float))

    def res(r):
        r = r.dropna()
        return Result(r, pd.DataFrame({"x": 1.0}, r.index), pd.Series(0.0, r.index), pd.Series(0, r.index))
    start = "2017-01-03"
    charted = {"Buy & hold SPY": res(spy_r.loc[start:]), "QQQ breakout trader": res(finalists["daytrader"].loc[start:]),
               "Staggered breakout": res(finalists["staggered"].loc[start:]),
               "Learning allocator": res(learner.loc[start:])}
    growth = _save(growth_chart(charted, _colors(list(charted), "Buy & hold SPY"), 0,
                               title="Growth of $100 since 2017 (log scale, after each strategy's costs)"),
                   OUT / "v2_growth.png")

    sources = "".join(f'<li><a href="{u}">{html.escape(t)}</a></li>' for t, u in SOURCES)
    body = f"""
<h1>Lab v2: the research</h1>
<p class="meta">What I read, what I tested, and why these three strategies are live. Development
periods were used to pick; each holdout (daily strategies 2018–today, intraday 2023–today) was
looked at once, after the picks were made. {n} variants have been tried in this lab, so passing
the strict scorecard now needs t ≥ {t_needed(n):.2f}. Nothing does; that is normal.</p>

<section><h2>The live lineup (v2.1, from Oct 14, 2026)</h2><ul>
<li><b>QQQ breakout trader</b> (day-trader account): QQQ's "noise area" is the range it usually
moves from the open by each time of day. A break out of it at a half-hour mark means real buying
or selling pressure: buy TQQQ (3x long) on an upside break, SQQQ (3x short) on a downside break.
Trail a stop at the band or VWAP, sell everything by the close.</li>
<li><b>Staggered breakout portfolio</b> (second account): the same rule on QQQ, the S&amp;P 500 and
semiconductors, each run on three half-hour clocks offset by 10 minutes (:00/:30, :10/:40,
:20/:50). Nine independent slices, about 16 trades a day spread across the hour.</li>
<li><b>Learning allocator</b> (third account): five unrelated daily strategies (Bitcoin trend, short
volatility, overnight TQQQ, RSI(2) dip buying, vol-targeted 3x trend). The Hedge online-learning
algorithm decides each one's share, starting from equal trust and moving money toward whatever has
been working (5% floor each, slow forgetting).</li>
</ul></section>

<section><h2>Making it trade more often (v2.1)</h2>
<p class="note">Goal: several trades an hour without giving up the edge. Trading more often on one
market only added costs: checking every 5 minutes turned the breakout from +26%/yr into a loss, and
every 5-minute VWAP strategy lost money after costs. Adding more markets didn't help either: the
breakout works on the big index and tech funds but not on small caps, Treasuries, banks or energy.
What worked was keeping the 30-minute rhythm and staggering it across three clocks and the three
markets where the edge showed up. Bold rows are live.</p>
{_table(activity_table, show_index=False)}</section>

<section><img alt="Growth of $100 for the three live strategies and SPY since 2017" src="data:image/png;base64,{growth}"></section>

<section><h2>Daily strategies tested</h2>
<p class="note">Bold = in the fund. Each used its published or practitioner default settings,
leveraged with 3x funds where the idea is about stocks. Costs: 3 bp per trade for ETFs, 0.30% for
crypto (Alpaca's fee plus spread).</p>{_table(daily_table, show_index=False)}</section>

<section><h2>Intraday strategies tested</h2>
<p class="note">Signals on QQQ; trades in TQQQ/SQQQ, paying half of each fund's one-cent spread plus
0.5 bp per trade. Development 2017–2022, holdout 2023–today. The opening-range breakout looks great
in its paper but, as independent replications also found, not after realistic costs.</p>
{_table(intraday_table, show_index=False)}</section>

<section><h2>Combining the five sleeves</h2>
<p class="note">Development July 2012–2017, holdout 2018–today. The sleeves are barely related
(crypto is uncorrelated with the rest), so combining them cuts risk sharply.</p>
{_table(alloc_table, show_index=False)}</section>

<section><h2>What I read</h2><ul>{sources}</ul>
<p class="note">Common thread in the research: most published edges shrink after publication and
after realistic costs, so everything here was re-tested on our own data with costs, and the
holdouts are the honest guide to expectations.</p></section>

<section><h2>Risks to know</h2><ul>
<li>Short volatility can lose most of its value in days (February 2018: -34% for this sleeve in a
year; the fund it belongs to was -23% that year).</li>
<li>Crypto trend had a -60% drawdown in the holdout.</li>
<li>The day trader holds 3x funds; a bad day can cost 5–10%.</li>
<li>All backtests ignore taxes. Fake money only.</li>
</ul></section>"""
    path = OUT / "v2_research.html"
    path.write_text(_page("Lab v2 Research", body), encoding="utf-8")
    print(json.dumps(summary["finalists"], indent=1, default=lambda v: round(v, 3)))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
