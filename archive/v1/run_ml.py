"""Walk-forward ML sector strategies, judged on the scorecard.

    .venv\\Scripts\\python archive\\v1\\run_ml.py           # development period only (2003-2017)
    .venv\\Scripts\\python archive\\v1\\run_ml.py --final   # the one-time test on 2018-today

Design everything on the development period. The holdout stays locked until the design
is final: every look at it followed by a tweak turns it into more training data.
"""
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[2]))  # shared modules live at the repo root

import argparse
from datetime import date

import pandas as pd

import strategies
from backtest import Result, run_backtest
from data import load_prices
from metrics import TRADING_DAYS, yearly_returns
from ml import FEATURE_NAMES, FEATURE_VERSION, ml_sectors, walk_forward
from report import write_ml_report
from scorecard import log_variants, score, t_needed

START = "2002-12-31"  # first month-end with 36 months of training history behind it
DEV = ("2003-01-01", "2017-12-31")
HOLDOUT = ("2018-01-01", None)
COST_BPS = 5.0
BENCH = "Buy & hold SPY"
# The worst drop each risk level may have and still pass. These are your call: change
# them freely, but before looking at results, not after.
DD_LIMITS = {"ML moderate": -0.60, "ML safe": -0.30}
MODELS = ["ridge", "gbm"]  # ridge is the main model; gbm is a second variant
OUT = Path(__file__).parent / "report"
HOLDOUT_MARK = Path(__file__).parent / "HOLDOUT_USED.txt"


def window(res, a, b):
    s = slice(a, b)
    return Result(res.returns.loc[s], res.weights.loc[s], res.turnover.loc[s], res.trades.loc[s])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--final", action="store_true",
                        help="run the one-time holdout test (2018-today)")
    args = parser.parse_args()

    prices, rf = load_prices()
    scores, coefs = {}, {}
    for kind in MODELS:
        print(f"training {kind} walk-forward...")
        scores[kind], coefs[kind] = walk_forward(prices, rf, kind)

    # name -> (target weights, config recorded in the variant log, risk level)
    candidates = {}
    for kind in MODELS:
        suffix = "" if kind == "ridge" else f" ({kind})"
        candidates[f"ML moderate{suffix}"] = (
            ml_sectors(scores[kind], prices),
            f"ML moderate | {kind} | top 3 | features {FEATURE_VERSION}", "ML moderate")
        candidates[f"ML safe{suffix}"] = (
            ml_sectors(scores[kind], prices, crash_filter=True),
            f"ML safe | {kind} | top 3 | 200-day crash filter | features {FEATURE_VERSION}", "ML safe")
    references = {"Sector momentum": strategies.sector_momentum(prices),
                  BENCH: strategies.buy_and_hold(prices, START)}

    targets = {n: c[0] for n, c in candidates.items()} | references
    for name, t in targets.items():
        # Every strategy must start on the same day, or the comparison isn't fair.
        assert t.loc[START:].index[0] == pd.Timestamp(START), f"{name} has no signal on {START}"

    a, b = HOLDOUT if args.final else DEV
    results = {n: window(run_backtest(t, prices, rf, START, cost_bps=COST_BPS), a, b)
               for n, t in targets.items()}
    bench = results[BENCH].returns

    log = log_variants([c[1] for c in candidates.values()])
    n_variants = len(log)
    card = pd.DataFrame(
        {n: score(results[n], bench, rf, DD_LIMITS[c[2]], n_variants) for n, c in candidates.items()}
        | {n: score(results[n], bench, rf, None, n_variants) for n in references}).T.infer_objects()

    if args.final and not HOLDOUT_MARK.exists():
        HOLDOUT_MARK.write_text(f"Holdout first viewed on {date.today()}. From here on it is "
                                "no longer an unseen test.\n")
    holdout_note = HOLDOUT_MARK.read_text().strip() if HOLDOUT_MARK.exists() else \
        "Holdout (2018 to today) is still locked and unseen."

    weights = coefs["ridge"].loc[:b].mean()
    coef_table = pd.DataFrame({
        "Average weight": weights,
        "Meaning": ["no real effect" if abs(w) < 5e-4 else
                    "favors sectors high on this" if w > 0 else "favors sectors low on this"
                    for w in weights],
    }).reindex(weights.abs().sort_values(ascending=False).index)
    coef_table.index = [FEATURE_NAMES[f] for f in coef_table.index]

    charted = [BENCH, "ML moderate", "ML safe", "Sector momentum"]
    meta = {
        "benchmark": BENCH, "cost_bps": COST_BPS,
        "label": "Final test: holdout period" if args.final else "Development period",
        "start": bench.index[0].date(), "end": bench.index[-1].date(),
        "years": len(bench) / TRADING_DAYS, "n_variants": n_variants,
        "t_needed": t_needed(n_variants), "holdout_note": holdout_note,
    }
    name = "ml_final" if args.final else "ml_dev"
    path = write_ml_report(
        OUT, name, meta=meta, card=card,
        results={n: results[n] for n in charted},
        yearly=pd.DataFrame({n: yearly_returns(r.returns) for n, r in results.items()}),
        coefs=coef_table, log=log)
    card.to_csv(OUT / f"{name}_scorecard.csv")
    pd.DataFrame({n: r.returns for n, r in results.items()}).to_csv(OUT / f"{name}_daily_returns.csv")

    cols = ["CAGR", "Sharpe", "Max drawdown", "t-stat vs risk-matched SPY", "Needed t", "Result"]
    print(f"\n{meta['label']}: {meta['start']} to {meta['end']} | {n_variants} variants tried\n")
    print(card[cols].to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\n{holdout_note}\nReport: {path}")


if __name__ == "__main__":
    main()
