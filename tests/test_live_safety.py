"""Checks for the live bot's failure handling, using a fake broker (no orders, no network
except Alpaca's market calendar).

    .venv\\Scripts\\python tests\\test_live_safety.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import live  # noqa: E402

TZ = live.TZ


class FakeClient:
    """Just enough of Alpaca's TradingClient for the code paths under test."""

    def __init__(self, positions):
        self.positions = {s: SimpleNamespace(symbol=s, qty=q, market_value=q * 50.0,
                                             asset_class=SimpleNamespace(value="us_equity"))
                          for s, q in positions.items()}
        self.closed = []

    def get_all_positions(self):
        return list(self.positions.values())

    def get_open_position(self, symbol):
        if symbol not in self.positions:
            raise RuntimeError("position does not exist")
        return self.positions[symbol]

    def close_position(self, symbol):
        p = self.positions.pop(symbol)
        self.closed.append(symbol)
        return SimpleNamespace(id=symbol, symbol=symbol, qty=p.qty)

    def get_order_by_id(self, order_id):
        return SimpleNamespace(status=SimpleNamespace(value="filled"), symbol=order_id,
                               filled_qty="1.0", filled_avg_price="50.0")


def entries(name):
    path = live.LOGS / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_reconcile_slices():
    client = FakeClient({"UPRO": 1.5, "XLE": 5.0})
    st = {"date": "2026-10-15", "budget": 111.0, "slices": {
        "SPY :00/:30": {"side": 1, "fund": "UPRO", "qty": 2.0},
        "SPY :10/:40": {"side": 1, "fund": "UPRO", "qty": 1.0},
        "QQQ :00/:30": {"side": -1, "fund": "SQQQ", "qty": 3.0}}}
    live.reconcile_slices(client, st, dry_run=False)
    assert abs(st["slices"]["SPY :00/:30"]["qty"] - 1.0) < 1e-9, st
    assert abs(st["slices"]["SPY :10/:40"]["qty"] - 0.5) < 1e-9, st
    assert st["slices"]["QQQ :00/:30"] == {"side": 0}, st
    assert client.closed == ["XLE"], client.closed
    print("ok: reconcile shrinks claims to real shares, clears missing ones, sells unclaimed funds")


def test_sell_missing_position():
    client = FakeClient({})
    assert live.sell_shares(client, "safe", "SOXL", 2.0, dry_run=False, why="test") is None
    assert entries("safe")[-1]["why"] == "position already closed"
    print("ok: selling a position that's already gone is a note, not a crash")


def test_watchdog():
    day = pd.Timestamp("2026-10-15", tz=TZ)  # a Thursday the market is open
    calls = []

    def fake_run(cmd, capture_output=False, text=False, check=False):
        calls.append(cmd)
        if cmd[:3] == ["gh", "run", "list"]:
            status = cmd[cmd.index("--status") + 1]
            return SimpleNamespace(returncode=0, stdout=json.dumps(running.get(status, [])))
        return SimpleNamespace(returncode=0, stdout="")

    live.subprocess.run = fake_run

    def run_at(clock, last_decision):
        calls.clear()
        live.now_et = lambda: day + pd.Timedelta(clock)
        (live.LOGS / "safe.jsonl").write_text(
            "" if last_decision is None else json.dumps(
                {"time": (day + pd.Timedelta(last_decision)).isoformat(), "action": "decision"}) + "\n")
        live.watchdog()
        return [c for c in calls if c[:3] == ["gh", "workflow", "run"]]

    running = {}
    assert run_at("10:50:00", "10:40:15") == [], "healthy session must be left alone"
    started = run_at("10:50:00", "10:10:15")
    assert started and started[0][-1] == "session=morning", started
    assert run_at("14:50:00", None)[0][-1] == "session=afternoon"
    running = {"in_progress": [{"databaseId": 1}]}
    assert run_at("10:50:00", "10:10:15") == [], "must not start a second run"
    running = {}
    assert run_at("09:50:00", None) == [], "too early: the morning run may still be starting"
    assert run_at("15:50:00", None) == [], "too late: nothing left to rescue"
    print("ok: watchdog restarts a dead session, and only then")


if __name__ == "__main__":
    real_run, real_now = subprocess.run, live.now_et
    with tempfile.TemporaryDirectory() as tmp:
        live.LOGS = Path(tmp)
        try:
            test_reconcile_slices()
            test_sell_missing_position()
            test_watchdog()
        finally:
            live.subprocess.run, live.now_et = real_run, real_now
    print("all checks passed")
