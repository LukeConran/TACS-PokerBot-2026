"""
Monte Carlo equity estimator for heads-up Hold'em + Redraw.

Usage:
    equity = estimate_equity(hole_cards, board_cards, n_simulations=1000)

Cards are strings like 'Ah', 'Kd', 'Tc', '9s'.
Unknown cards (redrawn placeholders '??') are treated as not in the deck.
"""

import numpy as np

RANKS = '23456789TJQKA'
SUITS = 'hdcs'
RANK_VAL = {r: i for i, r in enumerate(RANKS)}

FULL_DECK = [r + s for r in RANKS for s in SUITS]


def _parse(card):
    """Return (rank_int, suit_int) for a card string."""
    return RANK_VAL[card[0]], SUITS.index(card[1])


def _hand_rank(cards):
    """
    Evaluate the best 5-card hand from up to 7 cards.
    Returns a comparable tuple — higher is better.
    cards: list of card strings, length 5-7.
    """
    parsed = [_parse(c) for c in cards]
    ranks = sorted([r for r, s in parsed], reverse=True)
    suits = [s for r, s in parsed]

    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1

    # Check flush (5+ of same suit)
    flush_suit = None
    for s in SUITS:
        if suits.count(s) >= 5:
            flush_suit = s
            break

    flush_cards = None
    if flush_suit is not None:
        flush_cards = sorted(
            [r for r, s in parsed if s == flush_suit], reverse=True
        )[:5]

    # Check straight (in a sorted unique rank list)
    def best_straight(rank_list):
        unique = sorted(set(rank_list), reverse=True)
        # Ace-low straight: A-2-3-4-5
        if 12 in unique:
            unique.append(-1)
        for i in range(len(unique) - 4):
            window = unique[i:i + 5]
            if window[0] - window[4] == 4 and len(set(window)) == 5:
                return window[0]
        return None

    straight_high = best_straight(ranks)

    # Straight flush
    if flush_suit is not None:
        sf_high = best_straight(flush_cards)
        if sf_high is not None:
            return (8, sf_high)

    # Four of a kind
    quads = [r for r, cnt in rank_counts.items() if cnt == 4]
    if quads:
        q = max(quads)
        kicker = max(r for r in ranks if r != q)
        return (7, q, kicker)

    # Full house
    trips = [r for r, cnt in rank_counts.items() if cnt >= 3]
    pairs = [r for r, cnt in rank_counts.items() if cnt >= 2]
    if trips and len(pairs) >= 2:
        t = max(trips)
        p = max(r for r in pairs if r != t)
        return (6, t, p)

    # Flush
    if flush_cards is not None:
        return (5,) + tuple(flush_cards)

    # Straight
    if straight_high is not None:
        return (4, straight_high)

    # Three of a kind
    if trips:
        t = max(trips)
        kickers = sorted([r for r in ranks if r != t], reverse=True)[:2]
        return (3, t) + tuple(kickers)

    # Two pair
    pairs_only = [r for r, cnt in rank_counts.items() if cnt == 2]
    if len(pairs_only) >= 2:
        p1, p2 = sorted(pairs_only, reverse=True)[:2]
        kicker = max(r for r in ranks if r != p1 and r != p2)
        return (2, p1, p2, kicker)

    # One pair
    if len(pairs_only) == 1:
        p = pairs_only[0]
        kickers = sorted([r for r in ranks if r != p], reverse=True)[:3]
        return (1, p) + tuple(kickers)

    # High card
    return (0,) + tuple(ranks[:5])


def estimate_equity(hole_cards, board_cards, n_simulations=1000):
    """
    Estimate win probability via Monte Carlo simulation.

    hole_cards:  list of 2 card strings (your hole cards)
    board_cards: list of 0-4 card strings (known community cards, no '??')
    n_simulations: number of rollouts

    Returns float in [0.0, 1.0] — probability of winning (ties count as 0.5).
    """
    known = set(hole_cards) | set(board_cards)
    remaining_deck = [c for c in FULL_DECK if c not in known]

    cards_needed = 5 - len(board_cards)  # community cards still to come
    opp_cards_needed = 2

    total_needed = cards_needed + opp_cards_needed
    if total_needed > len(remaining_deck):
        return 0.5  # degenerate case

    wins = 0.0
    deck_arr = np.array(remaining_deck)

    for _ in range(n_simulations):
        idxs = np.random.choice(len(deck_arr), size=total_needed, replace=False)
        sampled = deck_arr[idxs]

        opp_hole = list(sampled[:opp_cards_needed])
        runout = list(sampled[opp_cards_needed:])

        full_board = board_cards + runout

        my_rank = _hand_rank(hole_cards + full_board)
        opp_rank = _hand_rank(opp_hole + full_board)

        if my_rank > opp_rank:
            wins += 1.0
        elif my_rank == opp_rank:
            wins += 0.5

    return wins / n_simulations


def best_redraw_hole(hole_cards, board_cards, n_simulations=500):
    """
    Returns (index, equity_gain) for the hole card whose removal
    maximizes equity. Returns None if neither redraw improves equity.
    """
    base = estimate_equity(hole_cards, board_cards, n_simulations)
    best_idx, best_gain = None, 0.0

    for i in range(2):
        # Temporarily remove hole card i — simulate as if we got a fresh card
        remaining = [c for c in FULL_DECK if c not in hole_cards and c not in board_cards]
        n_samples = min(10, len(remaining))
        sampled_replacements = np.random.choice(remaining, size=n_samples, replace=False)

        avg_eq = 0.0
        for replacement in sampled_replacements:
            new_hole = list(hole_cards)
            new_hole[i] = replacement
            avg_eq += estimate_equity(new_hole, board_cards, n_simulations // 4)
        avg_eq /= n_samples

        gain = avg_eq - base
        if gain > best_gain:
            best_gain = gain
            best_idx = i

    return (best_idx, best_gain) if best_idx is not None else None
