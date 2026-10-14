"""Backtest the baseline strategies and write report/report.html.

    .venv\\Scripts\\python archive\\v1\\run.py            # uses cached prices in data/
    .venv\\Scripts\\python archive\\v1\\run.py --refresh  # re-downloads from Yahoo Finance
"""
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[2]))  # shared modules live at the repo root

import argparse

import pandas as pd

import strategies
from backtest import run_backtest
from data import load_prices
from metrics import TRADING_DAYS, return_stats, summarize, yearly_returns
from report import write_report

START = "1999-12-31"  # first month-end where every strategy has enough history
COST_BPS = 5.0        # main assumption; COST_LEVELS shows how much it matters
COST_LEVELS = [0.0, 5.0, 20.0]
BENCH = "Buy & hold SPY"
PERIODS = [("2000–2009", "2000", "2009"), ("2010–2019", "2010", "2019"), ("2020–now", "2020", None)]
OUT = Path(__file__).parent / "report"


def baselines(prices):
    return {
        BENCH: strategies.buy_and_hold(prices, START),
        "Equal-weight sectors": strategies.equal_weight(prices),
        "Sector momentum": strategies.sector_momentum(prices),
        "Trend following": strategies.trend_following(prices),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="re-download prices")
    args = parser.parse_args()

    prices, rf = load_prices(refresh=args.refresh)
    targets = baselines(prices)
    for name, t in targets.items():
        # Every strategy must start on the same day, or the comparison isn't fair.
        assert t.loc[START:].index[0] == pd.Timestamp(START), f"{name} has no signal on {START}"

    def backtest(t, cost=COST_BPS):
        return run_backtest(t, prices, rf, START, cost_bps=cost)

    results = {name: backtest(t) for name, t in targets.items()}
    bench = results[BENCH].returns
    summary = pd.DataFrame({n: summarize(r, rf, bench) for n, r in results.items()}).T

    cost_rows = {}
    for name, t in targets.items():
        stats = {c: return_stats(backtest(t, c).returns, rf) for c in COST_LEVELS}
        cost_rows[name] = {f"{k} @ {c:g} bps": stats[c][k]
                           for k in ("CAGR", "Sharpe") for c in COST_LEVELS}
    costs = pd.DataFrame(cost_rows).T

    def grid_row(t):
        s = summarize(backtest(t), rf, bench)
        return {k: s[k] for k in ("CAGR", "Sharpe", "Max drawdown", "t-stat vs SPY")}

    momentum_grid = pd.DataFrame([
        {"Lookback (months)": lb, "Sectors held": n, "_default": (lb, n) == (12, 3),
         **grid_row(strategies.sector_momentum(prices, lookback=lb, top_n=n))}
        for lb in (6, 9, 12) for n in (2, 3, 4)
    ])
    trend_grid = pd.DataFrame([
        {"Moving average (days)": w, "_default": w == 200,
         **grid_row(strategies.trend_following(prices, window=w))}
        for w in (100, 150, 200, 250)
    ])

    period_rows = []
    for label, a, b in PERIODS:
        for i, (name, res) in enumerate(results.items()):
            s = return_stats(res.returns.loc[a:b], rf)
            period_rows.append({"Period": label if i == 0 else "", "Strategy": name,
                                **{k: s[k] for k in ("CAGR", "Sharpe", "Max drawdown")}})
    periods = pd.DataFrame(period_rows)

    yearly = pd.DataFrame({n: yearly_returns(r.returns) for n, r in results.items()})
    end = bench.index[-1]
    yearly.index = [f"{y} YTD" if y == end.year and end.month < 12 else str(y) for y in yearly.index]

    OUT.mkdir(exist_ok=True)
    summary.to_csv(OUT / "summary.csv")
    costs.to_csv(OUT / "costs.csv")
    momentum_grid.drop(columns="_default").to_csv(OUT / "momentum_grid.csv", index=False)
    trend_grid.drop(columns="_default").to_csv(OUT / "trend_grid.csv", index=False)
    yearly.to_csv(OUT / "yearly_returns.csv")
    pd.DataFrame({n: r.returns for n, r in results.items()}).to_csv(OUT / "daily_returns.csv")

    meta = {"benchmark": BENCH, "cost_bps": COST_BPS, "start": bench.index[0].date(),
            "end": end.date(), "years": len(bench) / TRADING_DAYS}
    path = write_report(OUT, meta=meta, summary=summary, results=results, costs=costs,
                        momentum_grid=momentum_grid, trend_grid=trend_grid,
                        periods=periods, yearly=yearly)

    view = summary[["CAGR", "Sharpe", "Sharpe t-stat", "Max drawdown", "t-stat vs SPY", "Turnover / yr"]]
    print(f"{meta['start']} to {meta['end']}, {COST_BPS:g} bps costs\n")
    print(view.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nReport: {path}")


if __name__ == "__main__":
    main()
