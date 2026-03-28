"""
Baseline opponent: uniform random legal betting action (no redraw).
Use for smoke-testing your bot from repo root: set PLAYER_*_PATH in config.py.
"""
import os
import random
import sys

# Use any bot folder that ships the shared `skeleton` package (same layout as check_call → python_skeleton).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "daniel_bot"))

from skeleton.actions import (
    FoldAction,
    CallAction,
    CheckAction,
    RaiseAction,
    RedrawAction,
)
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot


class Player(Bot):
    def handle_new_round(self, game_state, round_state, active):
        _ = game_state
        _ = round_state
        _ = active

    def handle_round_over(self, game_state, terminal_state, active):
        _ = game_state
        _ = terminal_state
        _ = active

    def get_action(self, game_state, round_state, active):
        _ = game_state
        legal = round_state.legal_actions()
        # Ignore RedrawAction so we never send malformed redraw targets.
        candidates = [a for a in legal if a is not RedrawAction]
        kind = random.choice(candidates)
        if kind is FoldAction:
            return FoldAction()
        if kind is CallAction:
            return CallAction()
        if kind is CheckAction:
            return CheckAction()
        mn, mx = round_state.raise_bounds()
        amt = random.randint(mn, mx) if mn <= mx else mn
        return RaiseAction(amt)


if __name__ == "__main__":
    run_bot(Player(), parse_args())
