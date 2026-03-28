#!/usr/bin/env python3
"""
Run the engine from the repo root and print wall-clock duration + winner line.

Usage (from repo root, venv active):
  python3 daniel_bot_v2/run_benchmark.py

Set PLAYER_*_PATH in config.py before running. Tournament bots often have a
per-player game clock (e.g. 180s in STARTING_GAME_CLOCK); wall time here is
how long the full match took on your machine.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENGINE = REPO / "engine.py"
GAMELOG = REPO / "results" / "gamelog.txt"


def main() -> int:
    if not ENGINE.is_file():
        print("Could not find engine.py at", ENGINE, file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, str(ENGINE)], cwd=str(REPO))
    elapsed = time.perf_counter() - t0

    final_line = None
    if GAMELOG.is_file():
        for line in GAMELOG.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("Final"):
                final_line = line

    print()
    print("--- Match benchmark ---")
    print(f"Wall time: {elapsed:.2f} s")
    print(
        "Note: each bot's game clock is decremented by thinking time (see "
        "STARTING_GAME_CLOCK in config.py); disqualification is usually when "
        "that clock hits 0, not wall time."
    )
    if final_line:
        print("Result:", final_line)
    else:
        print("Result: (could not read Final line from results/gamelog.txt)")
    print("---")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
