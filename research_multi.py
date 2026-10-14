"""Score the multi-market breakout (development periods; --final for the one-time holdout).

    .venv\\Scripts\\python research_multi.py
    .venv\\Scripts\\python research_multi.py --final
"""
import argparse

import pandas as pd

import intraday_v2 as iv
from backtest import Result
from intraday_data import load_bars, panels
from metrics import return_stats
from scorecard import log_variants, score, t_needed
from universe import load_daily

CANDIDATES = {"Multi-market breakout, 30-min checks": 6, "Multi-market breakout, 15-min checks": 3}


def load():
    d = load_daily()
    rf = (d["close"]["^IRX"].ffill().shift(1) / 100 / 252).fillna(0.0)
    funds = load_daily(["UPRO", "SPXU", "TNA", "TZA", "SOXL", "SOXS", "TMF", "TMV", "FAS", "FAZ", "ERX", "ERY"],
                       name="funds")["close"]
    pans = {**panels(load_bars()), **panels(load_bars(symbols=["IWM", "SMH", "TLT", "XLF", "XLE"], name="multi"))}
    return d, rf, funds, pans


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--final", action="store_true")
    args = p.parse_args()
    d, rf, funds, pans = load()
    spy = d["close"]["SPY"].dropna()
    spy_r = spy / spy.shift(1) - 1
    log = log_variants([f"v2.1 intraday | {n} | SPY IWM SMH TLT XLF XLE, leveraged fund pairs, 1/6 each"
                        for n in CANDIDATES])
    n = len(log)
    a, b = ("2023-01-01", None) if args.final else ("2017-01-01", "2022-12-31")
    print(f"{'HOLDOUT' if args.final else 'DEVELOPMENT'} {a[:4]}-{(b or 'today')[:4]} | {n} variants tried, "
          f"needed t = {t_needed(n):.2f}\n")
    for name, every in CANDIDATES.items():
        port, sleeves = iv.multi_breakout(pans, funds, rf, check_every=every)
        port.to_csv(f"report/v21_multi_every{every}_daily.csv")
        w = port.loc[a:b]
        s = score(Result(w["ret"], w[["exposure"]], w["turnover"], w["trades"]), spy_r.reindex(w.index), rf, -1.0, n)
        print(f"{name:<40} trades/day {w['trades'].mean():5.1f}  CAGR {s['CAGR']:7.1%}  Sharpe {s['Sharpe']:5.2f}  "
              f"worst {s['Max drawdown']:5.0%}  t {s['t-stat vs risk-matched SPY']:+.2f}")
        for m, sl in sleeves.items():
            st = return_stats(((1 + sl["ret"]) * (1 + rf.reindex(sl.index).fillna(0)) - 1).loc[a:b], rf)
            print(f"    {m:<4} alone: CAGR {st['CAGR']:7.1%}  Sharpe {st['Sharpe']:5.2f}  trades/day {sl['trades'].loc[a:b].mean():.1f}")
        corr = pd.DataFrame({m: sl["ret"] for m, sl in sleeves.items()}).loc[a:b].corr()
        print(f"    average correlation between markets: {(corr.sum().sum() - len(corr)) / (len(corr) ** 2 - len(corr)):.2f}")


if __name__ == "__main__":
    main()
