"""Performance statistics."""
import numpy as np

TRADING_DAYS = 252


def drawdown(returns):
    """How far below its running peak the portfolio is, each day."""
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1


def return_stats(returns, rf):
    years = len(returns) / TRADING_DAYS
    excess = returns - rf.reindex(returns.index)
    sharpe = excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)
    return {
        "CAGR": (1 + returns).prod() ** (1 / years) - 1,
        "Volatility": returns.std() * np.sqrt(TRADING_DAYS),
        "Sharpe": sharpe,
        # t-stat of the mean excess return = Sharpe * sqrt(years); |t| >= 2 is the usual bar.
        "Sharpe t-stat": sharpe * np.sqrt(years),
        "Max drawdown": drawdown(returns).min(),
    }


def summarize(result, rf, benchmark):
    """Full stats for one backtest, including whether it beat the benchmark."""
    r = result.returns
    years = len(r) / TRADING_DAYS
    stats = return_stats(r, rf)
    stats["Turnover / yr"] = result.turnover.sum() / years
    stats["Trades / yr"] = result.trades.sum() / years
    stats["Time invested"] = result.weights.sum(axis=1).mean()

    active = r - benchmark.reindex(r.index)
    if active.std() > 0:  # zero for the benchmark itself
        stats["Excess vs SPY / yr"] = active.mean() * TRADING_DAYS
        stats["t-stat vs SPY"] = active.mean() / active.std() * np.sqrt(len(active))
    else:
        stats["Excess vs SPY / yr"] = stats["t-stat vs SPY"] = np.nan
    return stats


def yearly_returns(returns):
    return (1 + returns).groupby(returns.index.year).prod() - 1
