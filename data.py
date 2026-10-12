"""Download and cache daily prices from Yahoo Finance."""
from pathlib import Path

import pandas as pd
import yfinance as yf

BENCHMARK = "SPY"
# The nine original Select Sector SPDR ETFs, all trading since December 1998.
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
TBILL = "^IRX"  # 13-week T-bill yield, annualized percent

CACHE = Path(__file__).parent / "data" / "prices.csv"


def load_prices(start="1998-12-01", refresh=False):
    """Return (prices, rf).

    prices: dividend-adjusted daily closes for SPY and the sectors, so day-over-day
            changes are total returns (ETF expense ratios are already in the price).
    rf:     daily T-bill return, which is what uninvested cash earns.
    Downloads once and caches to data/prices.csv; pass refresh=True to re-download.
    """
    tickers = [BENCHMARK, *SECTORS, TBILL]
    if refresh or not CACHE.exists():
        try:
            raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
            # Keep complete days only: until the close, Yahoo includes today's unfinished bar.
            now = pd.Timestamp.now(tz="America/New_York")
            if now.hour * 60 + now.minute < 16 * 60 + 15:
                raw = raw[raw.index < now.normalize().tz_localize(None)]
            missing = [t for t in tickers if t not in raw or raw[t].isna().all()]
            if missing:
                raise RuntimeError(f"Yahoo Finance returned no data for {missing}")
            CACHE.parent.mkdir(exist_ok=True)
            raw.to_csv(CACHE)
        except Exception as e:  # Yahoo sometimes rate-limits cloud servers
            if not CACHE.exists():
                raise
            print(f"warning: price download failed ({e}); using cached prices")
    raw = pd.read_csv(CACHE, index_col=0, parse_dates=True)

    # The ETF trading calendar is the master calendar.
    prices = raw[[BENCHMARK, *SECTORS]].dropna(how="all")
    # Bond-market holidays differ slightly, so carry the last yield forward;
    # yesterday's yield is what earns today's interest.
    yld = raw[TBILL].reindex(prices.index).ffill()
    rf = (yld.shift(1) / 100 / 252).fillna(0.0)
    return prices, rf
