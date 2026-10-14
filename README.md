# Paper Trading Lab

Three algorithmic strategies, each with $1,000 of fake money in its own Alpaca paper account,
trade automatically. Dashboard: https://nmallik1029.github.io/paper-trading-lab/

## Live now (lab v2.1, from Oct 14, 2026)

| Strategy | Account (.env prefix) | What it does | Runs (New York time) |
|---|---|---|---|
| QQQ breakout trader | `ALPACA_` | Buys TQQQ (3x long) when QQQ breaks above its normal range for that time of day, SQQQ (3x short) when it breaks below; trailing stop at the band or VWAP; out by the close | checks every 30 min, 10:00–15:30 |
| Staggered breakout portfolio | `SAFE_` | The same rule on QQQ, the S&P 500 and semiconductors (TQQQ/SQQQ, UPRO/SPXU, SOXL/SOXS), each on three half-hour clocks offset by 10 minutes: nine slices, ~16 trades a day | checks every 10 min, 10:00–15:50 |
| Learning allocator | `MODERATE_` | Five daily sleeves (Bitcoin trend, short volatility, overnight TQQQ, RSI(2) dip buying, vol-targeted 3x trend); the Hedge online-learning algorithm sets each one's share | rebalances 15:40; sells the overnight TQQQ at 9:30 |

`report/v2_research.html` explains what was researched and tested, and why these three won.
(The equal-weight multi-strategy fund ran Oct 13 only; it was retired because it held the same
sleeves as the learner.)

```
.venv\Scripts\python run_v2_report.py               # rebuild the research report + dashboard data
.venv\Scripts\python research_v2.py                 # score the daily sleeves (development periods)
.venv\Scripts\python research_staggered.py          # score the staggered breakout portfolio
.venv\Scripts\python rehearse.py 2026-10-08         # run a past day through the live code vs the backtest
.venv\Scripts\python live.py daily --dry-run        # what the learning allocator would trade now
.venv\Scripts\python live.py flatten safe           # emergency: sell everything in one account
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
| 9:20 | `daytrader.yml`, session morning | unwind the learner's overnight TQQQ at the open; both day traders until 13:10 |
| 13:05 | `daytrader.yml`, session afternoon | both day traders from 13:20; the learner's rebalance at 15:40; everything sold at 15:57 |
| 15:40 | `monthly.yml` ("Daily rebalance") | backup only: waits for the afternoon run, then skips the already-logged rebalance |
| :50 past each hour 10–16 | `dashboard.yml` | rebuild the dashboard; sell the day traders' positions if their session died |

The trading job also asks for a dashboard rebuild every 30 minutes. Every decision is logged
to `logs/<account>.jsonl`; the staggered slices' open positions are in `logs/safe_slices.json`.

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
| `intraday_v2.py` | Intraday strategies on 5-minute bars: noise-area breakout (single, multi-market, staggered), opening-range breakout, VWAP scalers |
| `research_multi.py`, `research_staggered.py` | Score the multi-market and staggered breakout portfolios |
| `rehearse.py` | Runs a past day through the live day-trading code with a fake clock, vs the backtest |
| `allocators.py` | Equal weight, risk parity, and the Hedge learning allocator |
| `run_v2_report.py` | The v2 research report and the dashboard's backtest data |
| `live.py` | Live trading: daily rebalances, overnight unwind, the day trader, safety close-outs |
| `dashboard.py` | Builds the dashboard website from the accounts and logs |
| `broker.py` | Alpaca connections (paper trading only), keys read from `.env` |
| `intraday_data.py` | Downloads and caches 5-minute bars and the market calendar |
| `backtest.py`, `metrics.py`, `scorecard.py`, `report.py` | Shared engine, statistics, pass/fail checks, charts |
| `data.py`, `strategies.py`, `ml.py`, `daytrader.py`, `run*.py` | Lab v1 |
| `.github/workflows/` | Day trader, daily rebalance, dashboard |
