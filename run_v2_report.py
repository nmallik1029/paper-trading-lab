"""Rebuild the research write-up and the backtest files the dashboard reads.

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
               "Staggered breakout portfolio": res(finalists["staggered"].loc[start:]),
               "Learning allocator": res(learner.loc[start:])}
    growth = _save(growth_chart(charted, _colors(list(charted), "Buy & hold SPY"), 0,
                               title="Growth of $100 since 2017 (log scale, after trading costs)"),
                   OUT / "v2_growth.png")

    labels = {"Dev years": "Design years", "CAGR (dev)": "Design: return", "Sharpe (dev)": "Design: Sharpe",
              "Max drawdown (dev)": "Design: worst drop", "CAGR (holdout)": "Unseen: return",
              "Sharpe (holdout)": "Unseen: Sharpe", "Max drawdown (holdout)": "Unseen: worst drop",
              "Trades / day": "Trades a day"}
    live = pd.DataFrame([
        {"Strategy": name, "CAGR (dev)": f["dev"]["cagr"], "Sharpe (dev)": f["dev"]["sharpe"],
         "Max drawdown (dev)": f["dev"]["max_drawdown"], "CAGR (holdout)": f["holdout"]["cagr"],
         "Sharpe (holdout)": f["holdout"]["sharpe"], "Max drawdown (holdout)": f["holdout"]["max_drawdown"],
         "_default": key != "spy"}
        for key, name in [("daytrader", "QQQ breakout trader"), ("staggered", "Staggered breakout portfolio"),
                          ("learner", "Learning allocator"), ("spy", "S&P 500 (SPY), for comparison")]
        for f in [summary["finalists"][key]]])
    periods = (f"The intraday strategies were designed on 2017–2022 and tested on 2023 onward; the learning "
               f"allocator was designed on 2012–2017 and tested on 2018 onward. The S&amp;P 500 row uses the "
               f"learning allocator's periods.")

    sources = "".join(f'<li><a href="{u}">{html.escape(t)}</a></li>' for t, u in SOURCES)
    body = f"""
<h1>Research and backtests</h1>
<p class="meta">How the three live strategies were chosen: the ideas that were tested, how each one was
evaluated, what the results were, and the risks that come with the strategies that made the cut.</p>

<section><h2>Summary</h2>
<p>{n} strategy variants were built and tested on historical data with trading costs included. Each
was designed using one stretch of history and then evaluated once on later years it had never seen.
Three are live: two intraday breakout strategies based on Zarattini, Aziz and Barbon (2024), and a
daily portfolio of five strategies whose weights are set by an online-learning algorithm. None of the
{n} variants clears the significance bar once it is adjusted for the number of attempts
(t ≥ {t_needed(n):.2f}), so the results below are encouraging evidence rather than proof.</p>
{_table(live, show_index=False, labels=labels)}
<p class="note">Returns are per year. "Design" is the period the strategy was built on and "unseen" is the later period it was tested on once. {periods}</p></section>

<section><img alt="Growth of $100 for the three live strategies and SPY since 2017" src="data:image/png;base64,{growth}"></section>

<section><h2>How the strategies were tested</h2>
<p><b>Data.</b> Daily prices come from Yahoo Finance, and five-minute bars come from Alpaca's
consolidated market feed starting in 2016. Leveraged funds are rebuilt from their underlying index for
the years before they launched, which lets the daily tests start in 2003.</p>
<p><b>Costs.</b> Daily trades in exchange-traded funds pay 3 basis points each. Intraday trades in
leveraged funds pay half of a one-cent bid-ask spread at each fund's current price plus 0.5 basis points
of slippage, with a minimum of 1.5 basis points. Bitcoin trades pay 0.30%, which covers Alpaca's fee and
the spread.</p>
<p><b>Design and evaluation periods.</b> Every design decision was made on the earlier period. The later
period was looked at once, after the choices were final, so it gives an honest read on how the strategy
handles data it was not tuned to.</p>
<p><b>Significance.</b> Every variant is recorded, including the ones that failed. The t-statistic
needed to pass uses a Šidák correction: the more variants have been tried, the higher the bar, because
the best of many attempts can look good by luck alone.</p>
<p><b>Checking the live code.</b> Past trading days were replayed through the live trading code on a
simulated clock and compared with the backtest. More than 99% of decisions matched; the few that did not
were ties decided by a few cents of difference between the two price feeds.</p></section>

<section><h2>The live strategies</h2>
<p><b>QQQ breakout trader.</b> QQQ usually stays within a predictable distance of its opening price, and
how far it strays depends on the time of day. The strategy measures that distance as the average move
from the open to each time of day over the past 14 days, widened when the market gaps overnight. At each
half-hour mark, a price above the range buys TQQQ and a price below it buys SQQQ. The position is closed
when the price crosses back through the range or the day's volume-weighted average price, and
everything is sold before the close.</p>
<p><b>Staggered breakout portfolio.</b> The same rule runs on the Nasdaq-100, the S&amp;P 500 and the
semiconductor index, each with its own pair of 3x funds. Every market runs on three half-hour clocks
offset by ten minutes, which makes nine independent slices of the account. The result trades about 16
times a day without checking any single slice more often than every 30 minutes.</p>
<p><b>Learning allocator.</b> Five daily strategies share the account: Bitcoin trend-following, short
volatility that steps aside when the VIX futures curve inverts, TQQQ held overnight during uptrends,
TQQQ bought after sharp two-day drops, and a TQQQ trend position sized to its recent volatility. The
Hedge algorithm, a multiplicative-weights method from online learning, scores each strategy daily by its
return relative to its volatility and shifts money toward the ones doing well. It keeps at least 5% in
each and slowly forgets old results, so it can adapt when conditions change.</p></section>

<section><h2>Intraday strategies</h2>
<p>Both published intraday strategies were rebuilt with signals on QQQ and trades in TQQQ and SQQQ. The
volatility-band breakout held up after costs. The opening-range breakout, which looks strong in its
paper, did not, matching what independent replications have found.</p>
{_table(intraday_table, show_index=False, labels=labels)}
<p>The next question was how to trade more often without losing the edge. Checking the breakout every
five minutes instead of every thirty turned a profitable strategy into a losing one, and every
five-minute strategy built around the volume-weighted average price lost money once costs were
counted. Running the breakout on six other markets roughly broke even, because the edge shows up in the
large index and technology funds but not in small caps, Treasuries, banks or energy. Keeping the
30-minute rhythm and staggering it across three clocks and the three markets where the edge appeared
kept most of the returns while multiplying the number of trades.</p>
{_table(activity_table, show_index=False, labels=labels)}
<p class="note">Bold rows are live. The out-of-sample columns are only filled in for the candidates that
were evaluated on 2023 onward.</p></section>

<section><h2>Daily strategies</h2>
<p>Seven daily strategies were tested with the default settings from the research or trading practice
they came from, using 3x funds where the idea concerns stocks. Five went into the learning allocator;
the turn-of-the-month effect was left out because it was the weakest during the design period, and the
plain overnight strategy was replaced by the version that only holds in uptrends.</p>
{_table(daily_table, show_index=False, labels=labels)}
<p>The five strategies are only loosely related to each other, and Bitcoin trend-following is close to
uncorrelated with the rest, so combining them reduces risk considerably. The table compares three ways
of splitting the money between them.</p>
{_table(alloc_table, show_index=False, labels={**labels, "Allocator": "Weighting"})}
<p class="note">Bold rows were run live. The equal-weight version ran for one day before it was retired,
because it held the same strategies as the learning allocator.</p></section>

<section><h2>Risks</h2><ul>
<li>Short volatility can lose a large share of its value within days. In February 2018 this strategy
fell 34% for the year, and the combined portfolio was down 23% that year.</li>
<li>Bitcoin trend-following went through a 60% drawdown during the out-of-sample years.</li>
<li>The intraday strategies hold 3x leveraged funds, so a single bad day can cost 5% to 10% of an
account.</li>
<li>None of the backtests account for taxes. The accounts hold simulated money only.</li>
</ul></section>

<section><h2>Earlier experiments</h2>
<p>Before these strategies, the project tested simpler rules and machine-learning models: sector
momentum and trend following, ridge regression and gradient-boosted models that ranked sectors each
month, an intraday model that predicted the next 30 minutes, and a version of that model that learned
from scratch. None of them beat holding the S&amp;P 500 by a meaningful margin after costs, which is
what led to the research behind the current strategies. The original reports are kept for reference:
<a href="v1/report.html">baseline strategies</a>, <a href="v1/ml_dev.html">monthly machine learning</a>,
<a href="v1/daytrader_dev.html">intraday machine learning</a> and
<a href="v1/coldstart_dev.html">learning from scratch</a>.</p></section>

<section><h2>Sources</h2><ul>{sources}</ul></section>"""
    path = OUT / "v2_research.html"
    path.write_text(_page("Research and Backtests · Paper Trading Lab", body, home="../index.html"), encoding="utf-8")
    print(json.dumps(summary["finalists"], indent=1, default=lambda v: round(v, 3)))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
