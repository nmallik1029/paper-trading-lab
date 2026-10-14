"""Backtest the learn-from-scratch day trader: how fast does it learn, and what's normal?

    .venv\\Scripts\\python archive\\v1\\run_coldstart.py

1. One continuous run that starts from scratch at the start of the development period
   (January 2017) and keeps learning through 2022: this is what gets scored.
2. Sixty fresh starts, one at the beginning of every month from 2017 to 2021, each
   followed for a year: the learning curve, and the normal range the dashboard shows.
"""
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[2]))  # shared modules live at the repo root

import json

import numpy as np
import pandas as pd

import daytrader as dt
from backtest import Result
from data import load_prices
from intraday_data import load_bars, panels
from metrics import TRADING_DAYS
from report import write_coldstart_report
from run_daytrader import as_result
from scorecard import log_variants, score, t_needed

DEV_START, DEV_END = "2017-01-03", "2022-12-30"
STARTS = pd.date_range("2017-01-01", "2021-12-01", freq="MS")
HORIZON = 250          # trading days followed after each fresh start (about a year)
COST_BPS = 1.0
BENCH = "Buy & hold SPY"
CONFIG = (f"Day trader | ridge | learns from scratch: {dt.SEED_DAYS}-day seed, retrained daily, "
          f"never forgets | SPY+QQQ 30-min blocks | features {dt.FEATURE_VERSION}")
OUT = Path(__file__).parent / "report"
LIVE_REFERENCE_END = "2026-10-01"  # the long-history model the dashboard compares against


def run_path(X, y, rf, first, last):
    preds, weights = dt.cold_start(X, y, first, last)
    daily, _ = dt.simulate(dt.decide(preds, COST_BPS), y, COST_BPS, rf)
    return daily, weights


def reference_weights(X, y, before):
    keep = np.asarray(X.index.get_level_values("date") < pd.Timestamp(before))
    return pd.Series(dt.train(X[keep], y[keep], "ridge")[-1].coef_ * 1e4, index=X.columns)


def main():
    prices, rf = load_prices()
    X, y = dt.features(panels(load_bars()))
    days = X.index.get_level_values("date").unique().sort_values()
    ref = reference_weights(X, y, "2023-01-01")  # what six years of data settles on

    print("continuous run from scratch, 2017-2022...")
    daily, W = run_path(X, y, rf, DEV_START, DEV_END)
    print("comparison: the full-history day trader over the same dates...")
    preds_full, _ = dt.walk_forward(X, y, "ridge")
    full, _ = dt.simulate(dt.decide(preds_full.loc[DEV_START:DEV_END], COST_BPS), y, COST_BPS, rf)
    full = full.loc[daily.index]

    spy = (prices["SPY"] / prices["SPY"].shift(1) - 1).reindex(daily.index)
    results = {
        "Day trader, from scratch": as_result(daily),
        "Day trader, full history": as_result(full),
        BENCH: Result(spy, pd.DataFrame({"invested": 1.0}, index=daily.index),
                      pd.Series(0.0, index=daily.index), pd.Series(0, index=daily.index)),
    }
    log = log_variants([CONFIG])
    n = len(log)
    card = pd.DataFrame({
        "Day trader, from scratch": score(results["Day trader, from scratch"], spy, rf, -1.0, n),
        "Day trader, full history": score(results["Day trader, full history"], spy, rf, None, n),
        BENCH: score(results[BENCH], spy, rf, None, n),
    }).T.infer_objects()
    card.insert(4, "Trades / day", [r.trades.mean() for r in results.values()])

    print(f"{len(STARTS)} fresh starts, a year each...")
    equity, trades, error, change = [], [], [], []
    for s in STARTS:
        first = days[days.searchsorted(s)]
        path_days = days[days >= first][:HORIZON]
        d, w = run_path(X, y, rf, first, path_days[-1])
        equity.append(np.r_[1000.0, 1000 * (1 + d["ret"]).cumprod().to_numpy()])
        trades.append(d["trades"].to_numpy())
        error.append((w - ref).abs().mean(axis=1).to_numpy())
        change.append(np.r_[np.nan, w.diff().abs().mean(axis=1).to_numpy()[1:]])
    E, T, R, C = (np.array(a) for a in (equity, trades, error, change))
    band = pd.DataFrame({"p10": np.percentile(E, 10, axis=0), "p50": np.median(E, axis=0),
                         "p90": np.percentile(E, 90, axis=0)})
    band.index.name = "day"
    curve = pd.DataFrame({
        "error_p25": np.percentile(R, 25, axis=0), "error_p50": np.median(R, axis=0),
        "error_p75": np.percentile(R, 75, axis=0),
        "change_p50": np.r_[np.nan, np.median(C[:, 1:], axis=0)],  # day 1 has no yesterday
        "trades": pd.Series(T.mean(axis=0)).rolling(5, min_periods=1).mean().to_numpy(),
    }, index=pd.RangeIndex(1, HORIZON + 1, name="day"))
    band.to_csv(OUT / "coldstart_band.csv")
    curve.to_csv(OUT / "coldstart_learning.csv")

    # How the continuous run's beliefs evolved, next to where six years of data lands.
    checkpoints = {"After 1 week": 5, "1 month": 21, "3 months": 63, "1 year": 250,
                   f"{len(W) / TRADING_DAYS:.0f} years": len(W)}
    evolution = pd.DataFrame({k: W.iloc[i - 1] for k, i in checkpoints.items()})
    evolution["Six-year model"] = ref
    evolution = evolution.reindex(ref.abs().sort_values(ascending=False).index)
    evolution.index = [dt.FEATURE_NAMES[f] for f in evolution.index]

    live_ref = reference_weights(X, y, LIVE_REFERENCE_END)
    (OUT / "daytrader_reference_weights.json").write_text(json.dumps(
        {"trained_before": LIVE_REFERENCE_END, "weights": live_ref.round(4).to_dict(),
         "names": dt.FEATURE_NAMES}, indent=1))

    meta = {"start": daily.index[0].date(), "end": daily.index[-1].date(),
            "years": len(daily) / TRADING_DAYS, "cost_bps": COST_BPS, "benchmark": BENCH,
            "n_variants": n, "t_needed": t_needed(n), "seed_days": dt.SEED_DAYS,
            "n_starts": len(STARTS), "horizon": HORIZON}
    path = write_coldstart_report(OUT, meta=meta, card=card, results=results, band=band,
                                  curve=curve, evolution=evolution)
    pd.DataFrame({k: r.returns for k, r in results.items()}).to_csv(OUT / "coldstart_daily_returns.csv")
    card.to_csv(OUT / "coldstart_scorecard.csv")

    cols = ["CAGR", "Sharpe", "Max drawdown", "Trades / day", "t-stat vs risk-matched SPY", "Needed t", "Result"]
    print(f"\n{meta['start']} to {meta['end']} | {n} variants tried\n")
    print(card[cols].to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nlearning curve (median across fresh starts):")
    for d in (1, 5, 21, 63, 125, 250):
        r = curve.loc[d]
        print(f"  day {d:>3}: beliefs off by {r['error_p50']:.2f} bps, change since yesterday "
              f"{r['change_p50']:.3f} bps, {r['trades']:.1f} trades/day, "
              f"value p10-p90 ${band.loc[d, 'p10']:,.0f}-${band.loc[d, 'p90']:,.0f}")
    print(f"\nReport: {path}")


if __name__ == "__main__":
    main()
