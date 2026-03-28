"""
Equity-driven poker bot with:
  - Preflop hand strength heuristic (fast, no simulation)
  - Monte Carlo equity estimation postflop
  - Draw detection + semi-bluff raising
  - Hole card AND board card redraw logic
  - Opponent modeling (VPIP, aggression factor, fold frequency)
"""
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, RedrawAction
from skeleton.states import BIG_BLIND, STARTING_STACK
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

from equity import estimate_equity
from opponent_model import OpponentModel

RANKS = '23456789TJQKA'
SUITS = 'hdcs'
RANK_VAL = {r: i for i, r in enumerate(RANKS)}

# Monte Carlo simulation budget per street (postflop only)
SIMS = {3: 800, 4: 1000, 5: 1000}

# Redraw: minimum equity gain required to justify swapping a hole card
REDRAW_GAIN_THRESHOLD = {3: 0.04, 4: 0.06}  # higher bar on turn (less upside)

# Equity below which we consider hole card redraw at all
REDRAW_EQUITY_CEIL = {3: 0.52, 4: 0.44}


def _clean(cards):
    return [c for c in cards if c and c != '??']


def _rank(card):
    return RANK_VAL.get(card[0], -1)


# ── Preflop heuristic ──────────────────────────────────────────────────────────

def preflop_strength(hole):
    """Fast heuristic equity estimate for 2 hole cards, no simulation needed."""
    vals = sorted((_rank(c) for c in hole), reverse=True)
    high, low = vals
    suited = hole[0][1] == hole[1][1]
    gap = high - low

    # Pocket pair
    if gap == 0:
        return min(0.56 + (high / 24.0), 0.96)

    strength = 0.33 + (high / 20.0) + (low / 35.0)
    if suited:
        strength += 0.04
    if gap == 1:
        strength += 0.05
    elif gap == 2:
        strength += 0.025
    elif gap >= 4:
        strength -= 0.05
    if high >= RANK_VAL['Q'] and low >= RANK_VAL['T']:
        strength += 0.05
    return max(0.12, min(strength, 0.90))


# ── Draw detection ─────────────────────────────────────────────────────────────

def has_flush_draw(hole, board):
    """True if 4+ cards of the same suit between hole and board."""
    all_cards = _clean(hole + board)
    suit_counts = {}
    for c in all_cards:
        suit_counts[c[1]] = suit_counts.get(c[1], 0) + 1
    return max(suit_counts.values(), default=0) >= 4


def has_straight_draw(hole, board):
    """True if 4 cards within a 5-card window (open-ended or gutshot)."""
    vals = sorted(set(_rank(c) for c in _clean(hole + board)))
    if 12 in vals:
        vals = [-1] + vals
    for start in range(-1, 10):
        window = set(range(start, start + 5))
        if len(window.intersection(vals)) >= 4:
            return True
    return False


def made_hand_strength(hole, board):
    """
    Returns hand category (0=high card … 8=straight flush) using
    the best 5 from hole+board.
    """
    from itertools import combinations

    all_cards = _clean(hole + board)
    if len(all_cards) < 2:
        return -1

    def _eval5(cards):
        rv = sorted((_rank(c) for c in cards), reverse=True)
        from collections import Counter
        rc = Counter(rv)
        counts = sorted(rc.items(), key=lambda x: (x[1], x[0]), reverse=True)
        is_flush = len({c[1] for c in cards}) == 1
        unique = sorted(set(rv))
        if 12 in unique:
            unique = [-1] + unique
        straight_high = None
        run = 1
        for i in range(1, len(unique)):
            if unique[i] == unique[i - 1] + 1:
                run += 1
                if run >= 5:
                    straight_high = unique[i]
            else:
                run = 1
        if is_flush and straight_high is not None:
            return (8, straight_high)
        if counts[0][1] == 4:
            return (7,)
        if counts[0][1] == 3 and counts[1][1] == 2:
            return (6,)
        if is_flush:
            return (5,)
        if straight_high is not None:
            return (4, straight_high)
        if counts[0][1] == 3:
            return (3,)
        if counts[0][1] == 2 and counts[1][1] == 2:
            return (2,)
        if counts[0][1] == 2:
            return (1,)
        return (0,)

    if len(all_cards) < 5:
        return -1

    best = max(_eval5(combo) for combo in combinations(all_cards, 5))
    return best[0]


# ── Board texture analysis ─────────────────────────────────────────────────────

def board_flush_threat(board):
    """
    Returns (suit, [indices]) if 3+ board cards share a suit, else None.
    """
    suit_indices = {}
    for i, c in enumerate(board):
        s = c[1]
        suit_indices.setdefault(s, []).append(i)
    for suit, idxs in suit_indices.items():
        if len(idxs) >= 3:
            return suit, idxs
    return None


def board_straight_threat(board):
    """
    Returns list of board indices forming a 4-card straight draw, else None.
    """
    val_to_idx = {}
    for i, c in enumerate(board):
        val_to_idx.setdefault(_rank(c), []).append(i)
    vals = sorted(val_to_idx.keys())
    if 12 in vals:
        vals_ext = [-1] + vals
    else:
        vals_ext = vals
    for start in range(-1, 10):
        window = list(range(start, start + 5))
        matching_vals = [v for v in window if v in val_to_idx]
        if len(matching_vals) >= 4:
            indices = []
            for v in matching_vals:
                indices.extend(val_to_idx[v])
            return indices[:4]
    return None


# ── Redraw logic ───────────────────────────────────────────────────────────────

def best_hole_redraw(hole, board, equity, street):
    """
    Returns hole card index (0 or 1) to redraw, or None.
    Only redraw if equity is below threshold and gain is large enough.
    """
    if equity >= REDRAW_EQUITY_CEIL.get(street, 0.52):
        return None

    # Don't redraw if we have a strong made hand
    strength = made_hand_strength(hole, board)
    if strength >= 2:  # two pair or better
        return None

    # Don't redraw if we have a strong draw
    if has_flush_draw(hole, board) or has_straight_draw(hole, board):
        if strength >= 1:
            return None

    # Find which hole card contributes least
    weakest = 0 if _rank(hole[0]) <= _rank(hole[1]) else 1

    # Verify equity gain is worth it by quick sampling
    gain = _estimate_hole_redraw_gain(hole, board, weakest, equity)
    if gain >= REDRAW_GAIN_THRESHOLD.get(street, 0.04):
        return weakest
    return None


def _estimate_hole_redraw_gain(hole, board, swap_idx, base_equity, n_replacements=8):
    """Average equity improvement from replacing hole[swap_idx]."""
    import numpy as np
    known = set(_clean(hole + board))
    deck = [r + s for r in RANKS for s in SUITS if (r + s) not in known]
    if not deck:
        return 0.0
    replacements = np.random.choice(deck, size=min(n_replacements, len(deck)), replace=False)
    total = 0.0
    for rep in replacements:
        new_hole = list(hole)
        new_hole[swap_idx] = rep
        total += estimate_equity(new_hole, board, n_simulations=150)
    avg = total / len(replacements)
    return avg - base_equity


def best_board_redraw(hole, board, street, made_strength):
    """
    Returns board card index to redraw when protecting a made hand, or None.

    Strategy: if the board has a dangerous flush or straight draw and we have
    a made hand (pair+), remove the most threatening board card.
    """
    if made_strength < 1:
        return None
    if not board:
        return None

    max_board_idx = 2 if street == 3 else 3  # flop: 0-2, turn: 0-3

    # Flush threat — remove highest ranked card of the threatening suit
    flush_result = board_flush_threat(board)
    if flush_result is not None:
        suit, idxs = flush_result
        # Filter to legal indices
        legal = [i for i in idxs if i <= max_board_idx]
        if legal:
            return max(legal, key=lambda i: _rank(board[i]))

    # Straight threat on the board — remove a connector
    straight_idxs = board_straight_threat(board)
    if straight_idxs is not None:
        legal = [i for i in straight_idxs if i <= max_board_idx]
        if legal:
            # Remove the middle connector (hardest to replace)
            return sorted(legal)[len(legal) // 2]

    return None


# ── Bet sizing ─────────────────────────────────────────────────────────────────

def compute_raise(equity, pot, my_stack, continue_cost, round_state, opp_multiplier=1.0):
    """Scale raise size by equity strength, adjusted for opponent type."""
    min_raise, max_raise = round_state.raise_bounds()

    if equity >= 0.85:
        fraction = 1.8
    elif equity >= 0.75:
        fraction = 1.2
    elif equity >= 0.65:
        fraction = 0.75
    else:
        fraction = 0.45  # semi-bluff / thin value

    base = round_state.pips[round_state.button % 2] + continue_cost
    target = int(base + max(BIG_BLIND * 2, pot * fraction * opp_multiplier))

    # Shove if we're short-stacked or very strong
    if equity >= 0.90 or my_stack <= BIG_BLIND * 6:
        target = max_raise

    return max(min_raise, min(max_raise, target))


# ── Main bot ───────────────────────────────────────────────────────────────────

class Player(Bot):

    def __init__(self):
        self.opp = OpponentModel()

        # Per-hand state
        self._preflop_opp_pip_seen = False   # did we observe opp raise preflop?
        self._my_raised_street = set()       # streets where I raised (for 3bet tracking)
        self._opp_bet_streets = set()        # streets where opp showed aggression
        self._last_street = -1

    def handle_new_round(self, game_state, round_state, active):
        self.opp.new_hand()
        self._preflop_opp_pip_seen = False
        self._my_raised_street = set()
        self._opp_bet_streets = set()
        self._last_street = -1

    def handle_round_over(self, game_state, terminal_state, active):
        delta = terminal_state.deltas[active]
        opp_delta = terminal_state.deltas[1 - active]

        # Infer if opponent folded: they lost chips but only the blind amount
        prev = terminal_state.previous_state
        if prev is not None:
            last_street = prev.street
            # If hand ended preflop or flop, opp likely folded early
            if last_street <= 3 and opp_delta < 0:
                after_my_raise = bool(self._my_raised_street)
                self.opp.saw_fold(after_my_raise=after_my_raise)

    def _observe_opp_action(self, round_state, active, continue_cost):
        """Update opponent model based on observable state."""
        street = round_state.street
        opp = 1 - active

        if continue_cost > 0 and street not in self._opp_bet_streets:
            self._opp_bet_streets.add(street)
            if street == 0:
                # Preflop raise: opp put in more than the big blind
                if round_state.pips[opp] > BIG_BLIND:
                    self.opp.saw_preflop_raise()
                else:
                    self.opp.saw_preflop_call()
            else:
                self.opp.saw_postflop_bet()
        elif continue_cost == 0 and street > 0:
            # Opponent checked
            if street not in self._opp_bet_streets:
                self.opp.saw_postflop_check()

        # 3-bet tracking: if I raised this street and opp is calling
        if street in self._my_raised_street and continue_cost == 0:
            self.opp.saw_call_to_raise()

    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()
        street = round_state.street

        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        my_stack = round_state.stacks[active]
        continue_cost = opp_pip - my_pip
        pot = (STARTING_STACK - round_state.stacks[0]) + (STARTING_STACK - round_state.stacks[1])

        hole = _clean(round_state.hands[active])
        board = _clean(round_state.board)

        self._observe_opp_action(round_state, active, continue_cost)

        # ── Equity ────────────────────────────────────────────────────
        if street == 0:
            equity = preflop_strength(hole)
        else:
            equity = estimate_equity(hole, board, n_simulations=SIMS.get(street, 800))

        # ── Opponent adjustments ──────────────────────────────────────
        call_adj = self.opp.call_equity_adj()
        raise_adj = self.opp.raise_equity_adj()
        size_mult = self.opp.bet_size_multiplier()

        pot_odds = continue_cost / (pot + continue_cost) if continue_cost > 0 else 0.0

        # ── Redraw decision ───────────────────────────────────────────
        if RedrawAction in legal_actions and not round_state.redraws_used[active]:
            redraw = self._decide_redraw(hole, board, equity, street, made_hand_strength(hole, board))
            if redraw is not None:
                target_type, target_idx = redraw
                bet = self._decide_bet(
                    equity, pot, my_stack, continue_cost, pot_odds,
                    call_adj, raise_adj, size_mult, legal_actions, round_state, street, hole, board
                )
                return RedrawAction(target_type, target_idx, bet)

        # ── Betting decision ──────────────────────────────────────────
        return self._decide_bet(
            equity, pot, my_stack, continue_cost, pot_odds,
            call_adj, raise_adj, size_mult, legal_actions, round_state, street, hole, board
        )

    def _decide_redraw(self, hole, board, equity, street, made_strength):
        """
        Returns (target_type, target_index) or None.
        Prioritizes board redraw (protecting made hand) over hole card swap.
        """
        if street not in (3, 4):
            return None

        # Board redraw: protect a made hand from dangerous board texture
        # Only worth it on the flop (2 cards left) or early turn
        if made_strength >= 1:
            board_idx = best_board_redraw(hole, board, street, made_strength)
            if board_idx is not None:
                return ('board', board_idx)

        # Hole card redraw: swap a weak card when equity is low
        hole_idx = best_hole_redraw(hole, board, equity, street)
        if hole_idx is not None:
            return ('hole', hole_idx)

        return None

    def _decide_bet(self, equity, pot, my_stack, continue_cost, pot_odds,
                    call_adj, raise_adj, size_mult, legal_actions, round_state, street, hole, board):

        flush_draw = has_flush_draw(hole, board)
        straight_draw = has_straight_draw(hole, board)
        has_draw = flush_draw or straight_draw

        # No cost to continue (check or open bet)
        if CheckAction in legal_actions:
            if RaiseAction in legal_actions:
                # Value bet threshold
                value_threshold = 0.62 if street == 0 else 0.60
                # Semi-bluff: raise draws with decent equity
                semi_bluff = (
                    has_draw and
                    equity >= (0.48 + raise_adj) and
                    street in (3, 4)
                )
                if equity >= (value_threshold + raise_adj) or semi_bluff:
                    amount = compute_raise(equity, pot, my_stack, 0, round_state, size_mult)
                    self._my_raised_street.add(street)
                    return RaiseAction(amount)
            return CheckAction()

        # Facing a bet/raise — decide call/raise/fold
        if RaiseAction in legal_actions:
            # Strong re-raise
            reraise_threshold = max(0.70, pot_odds + 0.18)
            # Semi-bluff re-raise on draws (not vs calling stations)
            semi_reraise = (
                has_draw and
                continue_cost <= max(20, pot // 2) and
                equity >= (0.44 + raise_adj) and
                street in (3, 4)
            )
            if equity >= (reraise_threshold + raise_adj) or semi_reraise:
                amount = compute_raise(equity, pot, my_stack, continue_cost, round_state, size_mult)
                self._my_raised_street.add(street)
                return RaiseAction(amount)

        # Call if profitable (pot odds + adjustment)
        call_threshold = pot_odds + call_adj
        if CallAction in legal_actions:
            if equity >= call_threshold - 0.02:
                return CallAction()
            # Cheap call — don't fold for tiny bets even with weak equity
            if continue_cost <= BIG_BLIND * 2 and equity >= 0.28:
                return CallAction()

        return FoldAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
