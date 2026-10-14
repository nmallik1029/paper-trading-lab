"""Score the staggered breakout portfolio: the noise-area rules on QQQ, SPY and SMH, each
run on three half-hour clocks offset by 10 minutes (checks at :00/:30, :10/:40, :20/:50),
nine equal slices. Keeps the 30-minute sampling that filters noise while spreading the
trades through the hour.

    .venv\\Scripts\\python research_staggered.py           # development 2017-2022
    .venv\\Scripts\\python research_staggered.py --final   # one-time holdout 2023-today
"""
import argparse

import pandas as pd

import intraday_v2 as iv
from backtest import Result
from intraday_data import load_bars, panels
from metrics import return_stats
from research_multi import load
from scorecard import log_variants, score, t_needed

MARKETS = {"QQQ": ("TQQQ", "SQQQ", 3.0), "SPY": ("UPRO", "SPXU", 3.0), "SMH": ("SOXL", "SOXS", 3.0)}
OFFSETS = (0, 2, 4)  # in 5-minute bars: :00/:30, :10/:40, :20/:50


def staggered(pans, funds, rf, markets=MARKETS, offsets=OFFSETS):
    sleeves = {(m, o): iv.breakout_sleeve(pans[m], funds[lf], funds[sf], lev, 6, o)
               for m, (lf, sf, lev) in markets.items() for o in offsets}
    rets = pd.DataFrame({k: s["ret"] for k, s in sleeves.items()}).dropna()
    port = pd.DataFrame({"ret": rets.mean(axis=1),
                         "trades": pd.DataFrame({k: s["trades"] for k, s in sleeves.items()}).reindex(rets.index).sum(axis=1),
                         "exposure": pd.DataFrame({k: s["exposure"] for k, s in sleeves.items()}).reindex(rets.index).mean(axis=1)})
    port["ret"] = (1 + port["ret"]) * (1 + rf.reindex(port.index).fillna(0.0)) - 1
    port["turnover"] = port["trades"].astype(float)
    return port, sleeves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--final", action="store_true")
    args = p.parse_args()
    d, rf, funds, pans = load()
    funds = pd.concat([funds, d["close"][["TQQQ", "SQQQ"]]], axis=1)
    pans = {**pans, **panels(load_bars(symbols=["TQQQ", "SQQQ"], name="lev"))}
    spy = d["close"]["SPY"].dropna()
    spy_r = spy / spy.shift(1) - 1
    log = log_variants(["v2.1 intraday | Staggered breakout portfolio | QQQ SPY SMH x clocks :00/:10/:20, 1/9 each"])
    n = len(log)
    a, b = ("2023-01-01", None) if args.final else ("2017-01-01", "2022-12-31")
    port, sleeves = staggered(pans, funds, rf)
    port.to_csv("report/v21_staggered_daily.csv")
    w = port.loc[a:b]
    s = score(Result(w["ret"], w[["exposure"]], w["turnover"], w["trades"]), spy_r.reindex(w.index), rf, -1.0, n)
    print(f"{'HOLDOUT' if args.final else 'DEVELOPMENT'} {a[:4]}-{(b or 'today')[:4]} | {n} variants tried, needed t = {t_needed(n):.2f}\n")
    print(f"Staggered breakout portfolio: trades/day {w['trades'].mean():.1f} (busiest 10% of days {w['trades'].quantile(.9):.0f}+)  "
          f"CAGR {s['CAGR']:.1%}  Sharpe {s['Sharpe']:.2f}  worst {s['Max drawdown']:.0%}  t {s['t-stat vs risk-matched SPY']:+.2f}")
    for o in OFFSETS:
        r = pd.DataFrame({m: sleeves[(m, o)]["ret"] for m in MARKETS}).dropna().mean(axis=1)
        st = return_stats(((1 + r) * (1 + rf.reindex(r.index).fillna(0)) - 1).loc[a:b], rf)
        print(f"    clock :{o * 5:02d}/:{30 + o * 5:02d} alone: CAGR {st['CAGR']:.1%}  Sharpe {st['Sharpe']:.2f}")
    yearly = (1 + w["ret"]).groupby(w.index.year).prod() - 1
    print("    by year: " + "  ".join(f"{y}: {v:+.0%}" for y, v in yearly.items()))


if __name__ == "__main__":
    main()
