# Paper Trading Lab

Three strategies, each with $1,000 of fake money in its own Alpaca paper account, trade
automatically. Dashboard: https://nmallik1029.github.io/paper-trading-lab/

## Live now (lab v2, since Oct 13, 2026)

| Strategy | Account (.env prefix) | What it does | Runs (New York time) |
|---|---|---|---|
| Breakout day trader | `ALPACA_` | Buys TQQQ (3x long) when QQQ breaks above its normal range for that time of day, SQQQ (3x short) when it breaks below; trailing stop at the band or VWAP; out by the close | decisions every 30 min, 10:00–15:30 |
| Multi-strategy fund | `MODERATE_` | Five sleeves, 20% each: Bitcoin trend, short volatility, overnight TQQQ, RSI(2) dip buying, vol-targeted 3x trend | rebalances 15:40; sells the overnight TQQQ at 9:30 |
| Learning allocator | `SAFE_` | The same five sleeves; the Hedge online-learning algorithm sets each one's share, starting from equal trust on launch day | same as the fund |

`report/v2_research.html` explains what was researched and tested, and why these three won.

```
.venv\Scripts\python run_v2_report.py               # rebuild the v2 research report + dashboard data
.venv\Scripts\python research_v2.py                 # score the daily sleeves (development periods)
.venv\Scripts\python live.py daily --dry-run        # what the two daily accounts would trade now
.venv\Scripts\python live.py daytrader --replay 2026-10-08   # rerun a past day vs the backtest
.venv\Scripts\python live.py flatten daytrader      # emergency: sell everything in one account
.venv\Scripts\python dashboard.py                   # build site/index.html locally
```

Dry runs log to `logs_dryrun/`, never the real logs. Keys live in `.env` locally and in
the repository's GitHub secrets for the scheduled runs; never commit them.

### How it runs
cron-job.org starts the GitHub Actions workflows through GitHub's API, using a fine-grained
token that can only start Actions runs in this repo. (GitHub's own schedules fired hours
late, so only the dashboard keeps one, as a harmless backup.)

| cron-job.org alarm (weekdays) | Workflow | Does |
|---|---|---|
| 9:20 | `daytrader.yml`, session morning | unwind the overnight TQQQ at the open, then day-trade until 13:00 |
| 13:05 | `daytrader.yml`, session afternoon | day-trade 13:30–15:30, rebalance the fund and the learner at 15:40, sell the day trader out at 15:57 |
| 15:40 | `monthly.yml` ("Daily rebalance") | backup only: waits for the afternoon run, then skips the already-logged rebalance |
| :50 past each hour 10–16 | `dashboard.yml` | rebuild the dashboard; sell the day trader's position if its session died |

Every decision is logged to `logs/<account>.jsonl` and committed.

## Research rules

- **Design on development periods only;** each holdout gets looked at once. Markers:
  `HOLDOUT_USED*.txt`. v2's holdouts (daily 2018+, intraday 2023+) were used on Oct 13, 2026.
- **Every configuration ever tried is logged in `experiments.csv`.** Never delete rows.
  The t-stat needed to pass rises with the count (`scorecard.t_needed`).
- **Costs are always in:** 3 bp per ETF trade, half of a one-cent spread + 0.5 bp for the
  intraday 3x funds, 0.30% for crypto.

## Lab v1 (retired Oct 13, 2026)

Kept for the record; each has its own report in `report/`.

```
.venv\Scripts\python run.py            # baseline strategies vs SPY, 2000-today
.venv\Scripts\python run_ml.py         # ML sector strategies (2003-2017 development)
.venv\Scripts\python run_daytrader.py  # ML day trader on 30-minute blocks
.venv\Scripts\python run_coldstart.py  # ML day trader that learns from scratch
```

## Files

| File | What it does |
|---|---|
| `universe.py` | Daily data for leveraged/inverse ETFs, volatility products, crypto; synthetic 3x history |
| `sleeves.py` | The v2 daily strategies, each returning daily returns and today's decision |
| `research_v2.py` | Scores the daily sleeves; logs every variant |
| `intraday_v2.py` | Noise-area breakout and opening-range breakout on 5-minute bars |
| `allocators.py` | Equal weight, risk parity, and the Hedge learning allocator |
| `run_v2_report.py` | The v2 research report and the dashboard's backtest data |
| `live.py` | Live trading: daily rebalances, overnight unwind, the day trader, safety close-outs |
| `dashboard.py` | Builds the dashboard website from the accounts and logs |
| `broker.py` | Alpaca connections (paper trading only), keys read from `.env` |
| `intraday_data.py` | Downloads and caches 5-minute bars and the market calendar |
| `backtest.py`, `metrics.py`, `scorecard.py`, `report.py` | Shared engine, statistics, pass/fail checks, charts |
| `data.py`, `strategies.py`, `ml.py`, `daytrader.py`, `run*.py` | Lab v1 |
| `.github/workflows/` | Day trader, daily rebalance, dashboard |
