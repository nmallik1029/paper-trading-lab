"""Alpaca connections. Keys come from .env, which never leaves this machine."""
import os
from pathlib import Path

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

ENV = Path(__file__).parent / ".env"

# One paper account per strategy; the names are the key prefixes used in .env.
ACCOUNTS = {"daytrader": "ALPACA", "moderate": "MODERATE", "safe": "SAFE"}


def _keys(prefix="ALPACA"):
    """Read <prefix>_API_KEY / <prefix>_SECRET_KEY from the environment, falling back to
    .env. (GitHub Actions will supply them as environment variables.)"""
    values = {}
    if ENV.exists():
        for line in ENV.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip().strip('"').strip("'")
    names = (f"{prefix}_API_KEY", f"{prefix}_SECRET_KEY")
    key, secret = (os.environ.get(n) or values.get(n) for n in names)
    if not key or not secret:
        raise RuntimeError(f"Alpaca keys missing: fill in {names[0]} and {names[1]} in .env")
    return key, secret


def trading_client(account="daytrader"):
    """Paper trading only. Switching to real money should be a deliberate code change."""
    return TradingClient(*_keys(ACCOUNTS[account]), paper=True)


def data_client():
    return StockHistoricalDataClient(*_keys())
