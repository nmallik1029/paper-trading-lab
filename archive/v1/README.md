# Version 1 (October 2026)

The project's first round of strategies, kept for reference. None of them beat holding the
S&P 500 by a meaningful margin after trading costs, which led to the research behind the current
strategies.

| Script | What it tested | Report |
|---|---|---|
| `run.py` | Baseline strategies from 2000: buy and hold, equal-weight sectors, sector momentum and trend following | `report/report.html` |
| `run_ml.py` | Ridge regression and gradient-boosted models that rank the nine sector ETFs each month | `report/ml_dev.html` |
| `run_daytrader.py` | An intraday model that predicts QQQ's and SPY's next 30 minutes from five-minute bars | `report/daytrader_dev.html` |
| `run_coldstart.py` | The same intraday model, trained from five days of data and retrained every morning | `report/coldstart_dev.html` |

The scripts still run from the repository root, for example `.venv\Scripts\python archive\v1\run.py`.
They use the shared modules at the root (`backtest.py`, `metrics.py`, `scorecard.py`, `report.py`)
and log their variants to the same `experiments.csv`.
