# Baseline strategy backtest

Tests simple monthly strategies against buy-and-hold SPY from 2000 to today. Every
strategy runs through the same backtest engine, so comparisons are apples to apples.

## Run

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python run.py            # uses cached prices in data/
.venv\Scripts\python run.py --refresh  # re-downloads from Yahoo Finance
```

Writes `report/report.html` (open it in a browser), plus CSVs of every table and the
daily returns of each strategy.

## ML strategies

```
.venv\Scripts\python run_ml.py           # development period only (2003-2017)
.venv\Scripts\python run_ml.py --final   # the one-time test on 2018-today
```

Writes `report/ml_dev.html`: a pass/fail scorecard for each ML variant. Rules:

- **Design on 2003–2017 only.** Run `--final` once, when the design is done. It leaves a
  `HOLDOUT_USED.txt` marker, because after that the holdout is no longer unseen.
- **Every configuration ever tried is logged in `experiments.csv`.** Never delete rows.
  The t-stat needed to pass rises with the count (about 2.5 at 4 variants, 3.5 at 100).
- **If you change the features, bump `FEATURE_VERSION` in `ml.py`** so the change counts
  as a new variant.

## ML day trader

```
.venv\Scripts\python run_daytrader.py           # development period (2017-2022)
.venv\Scripts\python run_daytrader.py --final   # the one-time test on 2023-today
```

Needs Alpaca keys in `.env` (copy `.env.example`) for the 5-minute price data. Writes
`report/daytrader_dev.html`. Same rules as above; its feature version is in `daytrader.py`,
and its holdout marker is `HOLDOUT_USED_DAYTRADER.txt`.

## Day trader that learns from scratch (the live version)

```
.venv\Scripts\python run_coldstart.py
```

The live day trader starts out knowing only the 5 trading days before launch, retrains
every morning on everything since, and never forgets. This backtest runs one continuous
from-scratch start (2017–2022) plus 60 fresh starts, each followed for a year. It writes
`report/coldstart_dev.html` (the learning curve), the normal range the dashboard shades,
and the long-history weights that the dashboard compares the live model against.

## Live paper trading

Three Alpaca paper accounts, $1,000 each, trade automatically on GitHub Actions:

| Strategy | Account (.env prefix) | Runs |
|---|---|---|
| ML safe | `SAFE_` | daily at 15:40 ET; trades when a new month's signal is ready |
| ML moderate | `MODERATE_` | same |
| ML day trader | `ALPACA_` | starts at 9:20 ET and retrains; decides every 30 min from 10:00 to 15:30 ET; sells everything by the close |

The trading runs are started by cron-job.org through GitHub's API (a fine-grained token
limited to starting Actions runs in this repo), because GitHub's own scheduled runs can
fire hours late. The dashboard also keeps a GitHub schedule as a harmless backup.

The dashboard (`dashboard.py`, published to GitHub Pages) rebuilds hourly and after every
trading run. Every decision is logged to `logs/<account>.jsonl`, and the day trader's
weights after each morning's retrain go to `logs/daytrader_model.jsonl`. All logs are committed.

```
.venv\Scripts\python live.py monthly --dry-run          # what the monthly bots would trade now
.venv\Scripts\python live.py daytrader --replay 2026-10-09   # rerun a past day, no orders
.venv\Scripts\python live.py flatten daytrader          # emergency: sell everything
.venv\Scripts\python dashboard.py                       # build site/index.html locally
```

Dry runs log to `logs_dryrun/`, never the real logs. Keys live in `.env` locally and in
the repository's GitHub secrets for the scheduled runs; never commit them.

## Files

| File | What it does |
|---|---|
| `data.py` | Downloads and caches dividend-adjusted prices and the T-bill rate |
| `strategies.py` | Each strategy returns the weights it wants on its decision dates |
| `backtest.py` | Turns weights into daily returns: next-day fills, drift between rebalances, costs |
| `metrics.py` | CAGR, Sharpe, drawdown, t-stats |
| `report.py` | Charts and the HTML reports |
| `run.py` | Runs the baselines; the settings (start date, costs, grids) are at the top |
| `ml.py` | Features, models (ridge, gradient boosting) and walk-forward training |
| `scorecard.py` | Pass/fail checks, risk-matched comparison, the variant log |
| `run_ml.py` | Runs the ML strategies; drawdown limits per risk level are at the top |
| `broker.py` | Alpaca connections (paper trading only), keys read from `.env` |
| `intraday_data.py` | Downloads and caches 5-minute bars and the market calendar |
| `daytrader.py` | Day-trader features, walk-forward training, trading rule, simulator |
| `run_daytrader.py` | Runs the day trader; costs and the holdout split are at the top |
| `run_coldstart.py` | Backtests the learn-from-scratch day trader and its learning curve |
| `live.py` | Live paper trading: monthly rebalances, the day trader, safety close-outs |
| `dashboard.py` | Builds the dashboard website from the accounts and logs |
| `.github/workflows/` | The schedules: day trader, monthly rebalance, dashboard |

## Adding a strategy (including an ML model later)

Write a function that returns a DataFrame of target weights: one row per decision
date, one column per ticker, values summing to at most 1 (the rest sits in T-bills).
Each row may only use data up to and including its own date. Add it to `baselines()`
in `run.py`. The engine handles execution timing and costs, so a model that leaks
future data into its features is the main way to fool yourself here.
