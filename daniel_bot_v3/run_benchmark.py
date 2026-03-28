#!/usr/bin/env python3
"""
Run engine.py from repo root, print wall time + winner, append timing line to gamelog.

Usage (repo root, venv active):
  python3 daniel_bot_v3/run_benchmark.py

Before running, set in config.py (repo root), e.g.:
  PLAYER_1_PATH = "./daniel_bot_v3"
  PLAYER_2_PATH = "./daniel_bot_v2"

This appends a line to results/gamelog.txt after the engine finishes, similar to the
printed benchmark summary.
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone
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
        text = GAMELOG.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if line.startswith("Final"):
                final_line = line

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    append_line = (
        f"Benchmark wall time: {elapsed:.3f} s (match ended {stamp}). "
        "Per-bot game clock is STARTING_GAME_CLOCK in config.py, not this wall time."
    )

    GAMELOG.parent.mkdir(parents=True, exist_ok=True)
    with open(GAMELOG, "a", encoding="utf-8") as f:
        f.write("\n" + append_line + "\n")

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
    print("Appended to gamelog:", append_line)
    print("---")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
