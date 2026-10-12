"""Backtest the ML day trader and judge it on the scorecard.

    .venv\\Scripts\\python run_daytrader.py           # development period (2017-2022)
    .venv\\Scripts\\python run_daytrader.py --final   # the one-time test on 2023-today

Design everything on the development period; the holdout stays locked until the
design is final.
"""
import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import daytrader as dt
from backtest import Result
from data import load_prices
from intraday_data import load_bars, panels
from metrics import TRADING_DAYS, return_stats, yearly_returns
from report import write_daytrader_report
from scorecard import log_variants, score, t_needed

DEV = ("2017-01-01", "2022-12-31")
HOLDOUT = ("2023-01-01", None)
COST_BPS = 1.0              # per dollar traded: half the bid-ask spread plus slippage
COST_LEVELS = [0.0, 1.0, 3.0]
DD_LIMIT = -1.0             # ultra risky: any drawdown is allowed
MODELS = ["ridge", "gbm"]   # ridge is the main model; gbm is a second variant
BENCH = "Buy & hold SPY"
OUT = Path(__file__).parent / "report"
HOLDOUT_MARK = Path(__file__).parent / "HOLDOUT_USED_DAYTRADER.txt"


def as_result(daily):
    return Result(returns=daily["ret"], weights=daily[["invested"]],
                  turnover=daily["trades"].astype(float), trades=daily["trades"])


def monthly_ic(preds, y):
    """Rank correlation between predicted and actual block returns, month by month."""
    df = pd.DataFrame({"pred": preds, "actual": y.reindex(preds.index)})
    month = df.index.get_level_values("date").to_period("M")
    return df.groupby(month).apply(lambda g: g["pred"].rank().corr(g["actual"].rank()))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--final", action="store_true",
                        help="run the one-time holdout test (2023-today)")
    args = parser.parse_args()

    prices, rf = load_prices()
    X, y = dt.features(panels(load_bars()))
    preds, coefs = {}, {}
    for kind in MODELS:
        print(f"training {kind} walk-forward (retrained monthly)...")
        preds[kind], coefs[kind] = dt.walk_forward(X, y, kind)

    first = max(p.index.get_level_values("date").min() for p in preds.values())
    a, b = HOLDOUT if args.final else (max(pd.Timestamp(DEV[0]), first), DEV[1])

    def window(s):
        return s.loc[pd.Timestamp(a):(pd.Timestamp(b) if b else None)]

    configs = {f"Day trader ({k})": f"Day trader | {k} | SPY+QQQ 30-min blocks | enter above "
               f"round-trip cost | features {dt.FEATURE_VERSION}" for k in MODELS}
    holds = {f"Day trader ({k})": dt.decide(window(preds[k]), COST_BPS) for k in MODELS}
    holds |= {"SPY, daytime only": dt.always_hold(window(y)),
              "Last-half-hour momentum": dt.last_half_hour_momentum(X, window(y))}

    results, held = {}, {}
    for name, hold in holds.items():
        daily, held[name] = dt.simulate(hold, y, COST_BPS, rf)
        results[name] = as_result(daily)
    days = results["Day trader (ridge)"].returns.index
    spy = prices["SPY"] / prices["SPY"].shift(1) - 1
    results[BENCH] = Result(spy.reindex(days), pd.DataFrame({"invested": 1.0}, index=days),
                            pd.Series(0.0, index=days), pd.Series(0, index=days))
    bench = results[BENCH].returns
    missing = len(prices.loc[days[0]:days[-1]]) - len(days)
    if missing:
        print(f"note: {missing} trading days have no intraday data and are skipped")

    log = log_variants(list(configs.values()))
    n_variants = len(log)
    card = pd.DataFrame(
        {n: score(results[n], bench, rf, DD_LIMIT if n in configs else None, n_variants)
         for n in [*configs, "SPY, daytime only", "Last-half-hour momentum", BENCH]}).T.infer_objects()

    activity = pd.DataFrame({n: {
        "Trades / day": results[n].trades.mean(),
        "Time in market": results[n].weights["invested"].mean(),
        "Win rate (blocks held)": (held[n] > 0).mean() if len(held[n]) else np.nan,
        "Avg gain per block held (bps)": held[n].mean() * 1e4 if len(held[n]) else np.nan,
        "Cost / day (bps)": results[n].trades.mean() * COST_BPS,
    } for n in holds}).T

    cost_rows = {}
    for name, hold in holds.items():
        stats = {c: return_stats(dt.simulate(hold, y, c, rf)[0]["ret"], rf) for c in COST_LEVELS}
        cost_rows[name] = {f"{k} @ {c:g} bps": stats[c][k] for k in ("CAGR", "Sharpe") for c in COST_LEVELS}
    costs = pd.DataFrame(cost_rows).T

    quality = {}
    for kind in MODELS:
        p = window(preds[kind])
        ic = monthly_ic(p, y)
        quality[f"Day trader ({kind})"] = {
            "Avg monthly IC": ic.mean(),
            "IC t-stat": ic.mean() / ic.std() * np.sqrt(len(ic)),
            "Direction right": (np.sign(p) == np.sign(y.reindex(p.index))).mean(),
        }
    quality = pd.DataFrame(quality).T

    weights = coefs["ridge"].loc[:b].mean() * 1e4
    coef_table = pd.DataFrame({
        "Average weight (bps)": weights,
        "Meaning": ["no real effect" if abs(w) < 0.05 else
                    "predicts higher returns when this is high" if w > 0 else
                    "predicts lower returns when this is high" for w in weights],
    }).reindex(weights.abs().sort_values(ascending=False).index)
    coef_table.index = [dt.FEATURE_NAMES[f] for f in coef_table.index]

    if args.final and not HOLDOUT_MARK.exists():
        HOLDOUT_MARK.write_text(f"Day-trader holdout first viewed on {date.today()}. From here on "
                                "it is no longer an unseen test.\n")
    holdout_note = HOLDOUT_MARK.read_text().strip() if HOLDOUT_MARK.exists() else \
        "Holdout (2023 to today) is still locked and unseen."

    meta = {
        "benchmark": BENCH, "cost_bps": COST_BPS,
        "label": "Final test: holdout period" if args.final else "Development period",
        "start": days[0].date(), "end": days[-1].date(), "years": len(days) / TRADING_DAYS,
        "n_variants": n_variants, "t_needed": t_needed(n_variants), "holdout_note": holdout_note,
    }
    name = "daytrader_final" if args.final else "daytrader_dev"
    charted = [BENCH, "Day trader (ridge)", "Day trader (gbm)", "SPY, daytime only"]
    path = write_daytrader_report(
        OUT, name, meta=meta, card=card, results={n: results[n] for n in charted},
        activity=activity, costs=costs, quality=quality, coefs=coef_table,
        yearly=pd.DataFrame({n: yearly_returns(r.returns) for n, r in results.items()}), log=log)
    card.to_csv(OUT / f"{name}_scorecard.csv")
    pd.DataFrame({n: r.returns for n, r in results.items()}).to_csv(OUT / f"{name}_daily_returns.csv")

    cols = ["CAGR", "Sharpe", "Max drawdown", "t-stat vs risk-matched SPY", "Needed t", "Result"]
    print(f"\n{meta['label']}: {meta['start']} to {meta['end']} | {n_variants} variants tried\n")
    print(card[cols].to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n" + activity.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n" + quality.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\n{holdout_note}\nReport: {path}")


if __name__ == "__main__":
    main()
