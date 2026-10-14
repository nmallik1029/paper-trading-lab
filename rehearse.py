"""Rehearse a full past trading day through the live day-trading code, with a fake clock
and no real orders, then compare its decisions with the backtest's.

    .venv\\Scripts\\python rehearse.py 2026-10-08

The live code runs unchanged (same timeline, same data path: full-market history plus
that day's IEX bars), except that time is simulated and every order is a dry run.
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

import intraday_v2 as iv
import live
from intraday_data import TZ, load_bars, panels
from research_multi import load


def main(day, morning_start="09:20"):
    """morning_start: when the morning session starts; later simulates a restart."""
    day = pd.Timestamp(day)
    clock = {"now": pd.Timestamp(f"{day.date()} 09:20", tz=TZ)}
    live.now_et = lambda: clock["now"]
    live.sleep_until = lambda t: clock.update(now=max(clock["now"], t))
    live.daily = lambda dry_run: print("   (15:40: the learning allocator would rebalance here)")
    test_logs = Path("logs_rehearsal")
    shutil.rmtree(test_logs, ignore_errors=True)
    live.LOGS = test_logs
    for session in ("morning", "afternoon"):
        clock["now"] = pd.Timestamp(f"{day.date()} {morning_start if session == 'morning' else '13:07'}", tz=TZ)
        live.daytrader(session, dry_run=True)

    rows = [json.loads(line) for line in (test_logs / "safe.jsonl").read_text().splitlines()]
    live_dec = {(r["decision"], m): v["after"] for r in rows if r.get("action") == "decision"
                for m, v in r["markets"].items()}
    qqq_rows = [json.loads(line) for line in (test_logs / "daytrader.jsonl").read_text().splitlines()]
    qqq_live = {r["decision"]: r["after"] for r in qqq_rows if r.get("action") == "decision"}

    # The backtest's decisions for the same day.
    d, rf, funds, pans = load()
    funds = pd.concat([funds, d["close"][["TQQQ", "SQQQ"]]], axis=1)
    agree = total = 0
    print(f"\n{'clock':<9} {'market':<5} live vs backtest positions at each check")
    for m, (lf, sf, lev) in live.STAGGER_MARKETS.items():
        for o in live.STAGGER_OFFSETS:
            trace = []
            iv.breakout_sleeve({k: v.loc[:day] for k, v in pans[m].items()}, funds[lf], funds[sf], lev, 6, o, trace)
            names = {1: lf, -1: sf, 0: "cash"}
            bt = {(pd.Timestamp(f"{day.date()} 09:30") + pd.Timedelta(minutes=5 * k)).strftime("%H:%M"): names[w]
                  for dt, k, _, w in trace if dt == day}
            line = []
            for t, pos in sorted(bt.items()):
                lv = live_dec.get((t, m), "?")
                total += 1
                agree += lv == pos
                line.append(f"{t} {lv}" + ("" if lv == pos else f"(bt {pos})"))
            print(f"{live.clock_name(o):<9} {m:<5} " + " | ".join(line))
    print(f"\nstaggered slices: {agree}/{total} decisions match the backtest")
    print("QQQ trader:", " | ".join(f"{t} {p}" for t, p in sorted(qqq_live.items())))
    trades = [json.loads(l) for f in ("safe.jsonl", "daytrader.jsonl") for l in (test_logs / f).read_text().splitlines()]
    print(f"orders the live code would have placed: {sum(r['action'].startswith('would') for r in trades)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2026-10-08", *sys.argv[2:3])
