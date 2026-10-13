"""Score the v2 daily sleeves on their development periods (holdouts stay locked).

    .venv\\Scripts\\python research_v2.py           # development periods
    .venv\\Scripts\\python research_v2.py --final   # one-time holdout test
"""
import argparse

import numpy as np
import pandas as pd

import sleeves as sl
from backtest import Result
from scorecard import log_variants, score, t_needed
from universe import load_daily

# name -> (function, kwargs, development window). Holdout = everything after it.
CANDIDATES = {
    "Leveraged trend + vol target": (sl.trend_voltarget, {}, ("2003-01-01", "2017-12-31")),
    "Short volatility (VIX curve)": (sl.short_vol, {}, ("2012-01-01", "2017-12-31")),
    "Overnight TQQQ": (sl.overnight, {}, ("2003-01-01", "2017-12-31")),
    "Overnight TQQQ, uptrend only": (sl.overnight, {"trend_filter": True}, ("2003-01-01", "2017-12-31")),
    "Turn of month, 3x": (sl.turn_of_month, {}, ("2003-01-01", "2017-12-31")),
    "RSI(2) dip buying, 3x": (sl.rsi2_dip, {}, ("2003-01-01", "2017-12-31")),
    "Crypto trend (BTC)": (sl.crypto_trend, {}, ("2015-01-01", "2020-12-31")),
}


def to_trading_days(df, days):
    """Compound returns from any calendar (crypto trades on weekends) onto stock trading
    days, so every sleeve can be compared with SPY day by day."""
    key = pd.Series(days, index=days).reindex(df.index, method="bfill")
    grouped = df.groupby(key.to_numpy())
    out = pd.DataFrame({"ret": grouped["ret"].apply(lambda r: (1 + r).prod() - 1),
                        "exposure": grouped["exposure"].mean(), "turnover": grouped["turnover"].sum(),
                        "target": grouped["target"].last()})
    out.index = pd.DatetimeIndex(out.index)
    return out


def as_result(df):
    return Result(df["ret"], df[["exposure"]], df["turnover"], (df["turnover"] > 0).astype(int))


def build(d, rf):
    days = d["close"]["SPY"].dropna().index
    out = {}
    for name, (fn, kw, window) in CANDIDATES.items():
        df = fn(d, rf, **kw)
        if not df.index.isin(days).all():
            df = to_trading_days(df, days)
        out[name] = (df, window)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--final", action="store_true")
    args = p.parse_args()

    d = load_daily()
    rf = (d["close"]["^IRX"].ffill().shift(1) / 100 / 252).fillna(0.0)
    spy = d["close"]["SPY"].dropna()
    spy_r = spy / spy.shift(1) - 1
    sleeves = build(d, rf)

    configs = [f"v2 daily | {n} | {fn.__name__} {kw} | defaults from literature"
               for n, (fn, kw, _) in CANDIDATES.items()]
    log = log_variants(configs)
    n = len(log)
    rows = {}
    for name, (df, (a, b)) in sleeves.items():
        if args.final:
            a, b = (pd.Timestamp(b) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), None
        w = df.loc[a:b]
        res = as_result(w)
        s = score(res, spy_r.reindex(w.index), rf, -1.0, n)
        s["Window"] = f"{w.index[0]:%Y}-{w.index[-1]:%Y}"
        s["Time invested"] = w["exposure"].mean()
        s["SPY CAGR (same years)"] = (1 + spy_r.reindex(w.index).fillna(0)).prod() ** (252 / len(w)) - 1
        rows[name] = s
    card = pd.DataFrame(rows).T.infer_objects()
    cols = ["Window", "CAGR", "SPY CAGR (same years)", "Sharpe", "Max drawdown", "Turnover / yr",
            "Time invested", "t-stat vs risk-matched SPY", "Needed t", "Result"]
    pd.set_option("display.width", 220)
    print(f"{'HOLDOUT' if args.final else 'DEVELOPMENT'} | {n} variants tried, needed t = {t_needed(n):.2f}\n")
    print(card[cols].to_string(float_format=lambda v: f"{v:.3f}"))
    return sleeves, card


if __name__ == "__main__":
    main()
