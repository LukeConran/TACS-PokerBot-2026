"""
Monte Carlo equity estimator using pkrbot's hand evaluator.

pkrbot.evaluate() is the same evaluator the engine uses, so equity
estimates are ground-truth accurate. Higher return value = stronger hand.
"""

import numpy as np
import pkrbot

RANKS = '23456789TJQKA'
SUITS = 'hdcs'
FULL_DECK = [r + s for r in RANKS for s in SUITS]


def _eval(hole, board):
    """Evaluate best hand from hole + board using pkrbot. Lower = better."""
    cards = [str(c) for c in hole + board if c and str(c) != '??']
    if len(cards) < 2:
        return 10**9  # worst possible — loses to everything
    try:
        return pkrbot.evaluate([pkrbot.Card(c) for c in cards])
    except Exception:
        return 10**9


def estimate_equity(hole_cards, board_cards, n_simulations=500):
    """
    Monte Carlo win probability estimate.

    hole_cards:    list of 2 known card strings
    board_cards:   list of 0-4 known community card strings
    n_simulations: rollout count — caller should scale with game clock

    Returns float in [0.0, 1.0]. Ties count as 0.5.
    """
    clean_board = [c for c in board_cards if c and c != '??']
    known = set(hole_cards) | set(clean_board)
    remaining = [c for c in FULL_DECK if c not in known]

    cards_needed = 5 - len(clean_board)
    total_needed = cards_needed + 2  # 2 opponent hole cards

    if total_needed > len(remaining):
        return 0.5

    deck_arr = np.array(remaining)
    wins = 0.0

    for _ in range(n_simulations):
        idxs = np.random.choice(len(deck_arr), size=total_needed, replace=False)
        sampled = deck_arr[idxs]

        opp_hole = list(sampled[:2])
        runout   = list(sampled[2:])
        full_board = clean_board + runout

        my_score  = _eval(hole_cards, full_board)
        opp_score = _eval(opp_hole,   full_board)

        if my_score > opp_score:    # higher = better in pkrbot
            wins += 1.0
        elif my_score == opp_score:
            wins += 0.5

    return wins / n_simulations
