"""
Equity-driven heads-up poker bot.

Improvements over v1:
  1. pkrbot.evaluate() for ground-truth hand ranking (same as engine)
  2. Clock-aware simulation budget -- never times out
  3. Preflop HU equity lookup table (calibrated, not formula)
  4. Simulation-based redraw for both hole and board cards
  5. Check-raise logic (value and bluff)
  6. River-specific strategy: no semi-bluffs, pure bluff option
  7. Opponent model: aggressive opponents trigger counter-aggression
"""

import random
import numpy as np

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, RedrawAction
from skeleton.states import BIG_BLIND, STARTING_STACK
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

from equity import estimate_equity, FULL_DECK
from opponent_model import OpponentModel

RANKS = '23456789TJQKA'
RANK_VAL = {r: i for i, r in enumerate(RANKS)}

# ── Preflop HU equity lookup table ────────────────────────────────────────────
# Keys: (hi_rank, lo_rank, suited:bool) for unpaired; (rank, rank, None) for pairs.

_PF = {}

def _suited(hi, lo, eq):
    _PF[(hi, lo, True)]  = eq
    _PF[(hi, lo, False)] = max(0.33, eq - 0.035)

# Pocket pairs 22 to AA
for _i, (_r, _eq) in enumerate(zip(RANKS, [
    0.56, 0.57, 0.59, 0.61, 0.63, 0.65, 0.67, 0.69, 0.72, 0.75, 0.80, 0.83, 0.85
])):
    _PF[(_r, _r, None)] = _eq

for _lo, _eq in [('K',0.67),('Q',0.66),('J',0.65),('T',0.64),('9',0.62),('8',0.61),
                  ('7',0.60),('6',0.59),('5',0.60),('4',0.59),('3',0.58),('2',0.57)]:
    _suited('A', _lo, _eq)
for _lo, _eq in [('Q',0.63),('J',0.62),('T',0.61),('9',0.59),('8',0.57),('7',0.56),
                  ('6',0.55),('5',0.54),('4',0.53),('3',0.52),('2',0.51)]:
    _suited('K', _lo, _eq)
for _lo, _eq in [('J',0.59),('T',0.58),('9',0.56),('8',0.54),('7',0.52),('6',0.51),
                  ('5',0.50),('4',0.49),('3',0.48),('2',0.47)]:
    _suited('Q', _lo, _eq)
for _lo, _eq in [('T',0.57),('9',0.55),('8',0.53),('7',0.51),('6',0.49),
                  ('5',0.48),('4',0.47),('3',0.46),('2',0.45)]:
    _suited('J', _lo, _eq)
for _lo, _eq in [('9',0.55),('8',0.53),('7',0.51),('6',0.49),
                  ('5',0.47),('4',0.46),('3',0.45),('2',0.44)]:
    _suited('T', _lo, _eq)
for _hi, _entries in [
    ('9', [('8',0.53),('7',0.51),('6',0.49),('5',0.47),('4',0.45),('3',0.44),('2',0.43)]),
    ('8', [('7',0.51),('6',0.49),('5',0.47),('4',0.45),('3',0.44),('2',0.43)]),
    ('7', [('6',0.49),('5',0.47),('4',0.45),('3',0.43),('2',0.42)]),
    ('6', [('5',0.47),('4',0.45),('3',0.43),('2',0.42)]),
    ('5', [('4',0.45),('3',0.43),('2',0.42)]),
    ('4', [('3',0.43),('2',0.41)]),
    ('3', [('2',0.41)]),
]:
    for _lo, _eq in _entries:
        _suited(_hi, _lo, _eq)


def preflop_equity(hole):
    c0, c1 = hole[0], hole[1]
    if not c0 or not c1 or c0 == '??' or c1 == '??':
        return 0.50
    i0 = RANK_VAL.get(c0[0], -1)
    i1 = RANK_VAL.get(c1[0], -1)
    if i0 < 0 or i1 < 0:
        return 0.50
    suited = c0[1] == c1[1]
    hi = RANKS[max(i0, i1)]
    lo = RANKS[min(i0, i1)]
    if hi == lo:
        return _PF.get((hi, lo, None), 0.52)
    return _PF.get((hi, lo, suited),
           _PF.get((hi, lo, not suited), 0.48) + (0.035 if suited else 0))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clean(cards):
    return [c for c in cards if c and c != '??']


def has_flush_draw(hole, board):
    counts = {}
    for c in _clean(hole + board):
        counts[c[1]] = counts.get(c[1], 0) + 1
    return max(counts.values(), default=0) >= 4


def has_straight_draw(hole, board):
    vals = sorted(set(RANK_VAL[c[0]] for c in _clean(hole + board)))
    if 12 in vals:
        vals = [-1] + vals
    for start in range(-1, 10):
        window = set(range(start, start + 5))
        if len(window.intersection(vals)) >= 4:
            return True
    return False


def _num_sims(clock):
    """Scale simulation count with remaining game clock."""
    if clock > 100: return 700
    if clock >  60: return 500
    if clock >  30: return 300
    if clock >  10: return 180
    return 80


def _redraw_inner_sims(clock):
    return max(50, _num_sims(clock) // 6)


# ── Bot ────────────────────────────────────────────────────────────────────────

class Player(Bot):

    def __init__(self):
        self.opp              = OpponentModel()
        self._last_street     = -1
        self._i_checked       = False
        self._raised_streets  = set()
        self._opp_bet_streets = set()

    def handle_new_round(self, game_state, round_state, active):
        self.opp.new_hand()
        self._last_street     = -1
        self._i_checked       = False
        self._raised_streets  = set()
        self._opp_bet_streets = set()

    def handle_round_over(self, game_state, terminal_state, active):
        prev      = terminal_state.previous_state
        opp_delta = terminal_state.deltas[1 - active]
        if prev is not None and opp_delta < 0:
            self.opp.saw_fold(after_my_raise=bool(self._raised_streets))

    # ── Opponent observation ───────────────────────────────────────────

    def _observe_opp(self, round_state, active, continue_cost):
        street = round_state.street
        opp    = 1 - active
        if continue_cost > 0 and street not in self._opp_bet_streets:
            self._opp_bet_streets.add(street)
            if street == 0:
                if round_state.pips[opp] > BIG_BLIND:
                    self.opp.saw_preflop_raise()
                else:
                    self.opp.saw_preflop_call()
            else:
                self.opp.saw_postflop_bet()
        elif continue_cost == 0 and street > 0 and street not in self._opp_bet_streets:
            self.opp.saw_postflop_check()
        if street in self._raised_streets and continue_cost == 0:
            self.opp.saw_call_to_raise()

    # ── Simulation-based redraw ────────────────────────────────────────

    def _redraw_gain(self, hole, board, t_type, t_idx, base_eq, clock):
        known = set(_clean(hole + board))
        deck  = [c for c in FULL_DECK if c not in known]
        if not deck:
            return 0.0
        n     = 6 if clock > 20 else 4
        inner = _redraw_inner_sims(clock)
        picks = np.random.choice(deck, size=min(n, len(deck)), replace=False)
        total = 0.0
        for rep in picks:
            if t_type == 'hole':
                h2 = list(hole); h2[t_idx] = rep
                b2 = list(board)
            else:
                h2 = list(hole)
                b2 = list(board)
                if t_idx < len(b2):
                    b2[t_idx] = rep
            total += estimate_equity(_clean(h2), _clean(b2), n_simulations=inner)
        return total / len(picks) - base_eq

    def _best_redraw(self, hole, board, street, equity, round_state, active, clock):
        if round_state.redraws_used[active] or street >= 5:
            return None
        if equity >= 0.88:
            return None
        min_gain  = 0.055 if clock > 25 else 0.07
        best      = None
        best_gain = min_gain
        for idx in (0, 1):
            g = self._redraw_gain(hole, board, 'hole', idx, equity, clock)
            if g > best_gain:
                best_gain, best = g, ('hole', idx)
        max_bi = 2 if street == 3 else 3
        for idx in range(min(len(board), max_bi + 1)):
            if board[idx] and board[idx] != '??':
                g = self._redraw_gain(hole, board, 'board', idx, equity, clock)
                if g > best_gain:
                    best_gain, best = g, ('board', idx)
        return best

    # ── Raise sizing ───────────────────────────────────────────────────

    def _size_raise(self, equity, pot, my_stack, continue_cost, round_state, mult):
        mn, mx = round_state.raise_bounds()
        if   equity >= 0.85: frac = 1.8
        elif equity >= 0.75: frac = 1.2
        elif equity >= 0.65: frac = 0.75
        else:                frac = 0.45
        active = round_state.button % 2
        base   = round_state.pips[active] + continue_cost
        target = int(base + max(BIG_BLIND * 2, pot * frac * mult))
        if equity >= 0.90 or my_stack <= BIG_BLIND * 6:
            target = mx
        return max(mn, min(mx, target))

    # ── Main action ────────────────────────────────────────────────────

    def get_action(self, game_state, round_state, active):
        legal  = round_state.legal_actions()
        street = round_state.street
        clock  = float(game_state.game_clock)

        if street != self._last_street:
            self._i_checked   = False
            self._last_street = street

        my_pip        = round_state.pips[active]
        opp_pip       = round_state.pips[1 - active]
        continue_cost = opp_pip - my_pip
        my_stack      = round_state.stacks[active]
        pot           = (STARTING_STACK - round_state.stacks[0]) + \
                        (STARTING_STACK - round_state.stacks[1])
        pot_odds = continue_cost / (pot + continue_cost) if continue_cost > 0 else 0.0

        hole  = _clean(round_state.hands[active])
        board = _clean(round_state.board)

        self._observe_opp(round_state, active, continue_cost)

        equity = preflop_equity(hole) if street == 0 else \
                 estimate_equity(hole, board, n_simulations=_num_sims(clock))

        call_adj  = self.opp.call_equity_adj()
        raise_adj = self.opp.raise_equity_adj()
        size_mult = self.opp.bet_size_multiplier()
        has_draw  = has_flush_draw(hole, board) or has_straight_draw(hole, board)

        redraw = None
        if RedrawAction in legal and street in (3, 4):
            redraw = self._best_redraw(hole, board, street, equity, round_state, active, clock)

        if street == 5:
            action = self._river(equity, pot, my_stack, continue_cost, pot_odds,
                                 call_adj, raise_adj, size_mult, legal, round_state)
        else:
            action = self._preriver(equity, pot, my_stack, continue_cost, pot_odds,
                                    call_adj, raise_adj, size_mult, legal, round_state,
                                    street, has_draw)

        if redraw is not None and RedrawAction in legal:
            inner  = action.action if isinstance(action, RedrawAction) else action
            action = RedrawAction(redraw[0], redraw[1], inner)

        inner = action.action if isinstance(action, RedrawAction) else action
        if isinstance(inner, CheckAction):
            self._i_checked = True
        if isinstance(inner, RaiseAction):
            self._raised_streets.add(street)

        return action

    # ── Pre-river ──────────────────────────────────────────────────────

    def _preriver(self, equity, pot, my_stack, continue_cost, pot_odds,
                  call_adj, raise_adj, size_mult, legal, round_state, street, has_draw):

        if CheckAction in legal:
            if RaiseAction in legal:
                vt         = 0.62 if street == 0 else 0.58
                semi_bluff = has_draw and equity >= (0.46 + raise_adj) and street in (3, 4)
                if equity >= (vt + raise_adj) or semi_bluff:
                    amt = self._size_raise(equity, pot, my_stack, 0, round_state, size_mult)
                    return RaiseAction(amt)
            return CheckAction()

        if RaiseAction in legal:
            if self._i_checked and equity >= (0.68 + raise_adj):
                amt = self._size_raise(equity, pot, my_stack, continue_cost, round_state, size_mult)
                return RaiseAction(amt)

            reraise_thresh = max(0.62, pot_odds + 0.12) if self.opp.is_aggressive() \
                             else max(0.68, pot_odds + 0.15)
            semi_reraise = (
                has_draw and
                continue_cost <= max(24, pot // 2) and
                equity >= (0.42 + raise_adj) and
                street in (3, 4)
            )
            if equity >= (reraise_thresh + raise_adj) or semi_reraise:
                amt = self._size_raise(equity, pot, my_stack, continue_cost, round_state, size_mult)
                return RaiseAction(amt)

        if CallAction in legal:
            if equity >= pot_odds + call_adj - 0.02:
                return CallAction()
            if continue_cost <= BIG_BLIND * 2 and equity >= 0.28:
                return CallAction()

        if CheckAction in legal:
            return CheckAction()
        return FoldAction()

    # ── River ──────────────────────────────────────────────────────────

    def _river(self, equity, pot, my_stack, continue_cost, pot_odds,
               call_adj, raise_adj, size_mult, legal, round_state):
        """No semi-bluffs. Value bet strong hands. Pure bluff vs folders."""

        if CheckAction in legal:
            if RaiseAction in legal:
                if equity >= (0.60 + raise_adj):
                    amt = self._size_raise(equity, pot, my_stack, 0, round_state, size_mult)
                    return RaiseAction(amt)
                if equity < 0.25 and self.opp.fold_frequency > 0.42 and random.random() < 0.35:
                    mn, mx    = round_state.raise_bounds()
                    bluff_amt = max(mn, min(int(pot * 0.5), mx))
                    return RaiseAction(bluff_amt)
            return CheckAction()

        if RaiseAction in legal:
            if self._i_checked and equity >= (0.78 + raise_adj):
                amt = self._size_raise(equity, pot, my_stack, continue_cost, round_state, size_mult)
                return RaiseAction(amt)
            if equity >= max(0.75, pot_odds + 0.20):
                amt = self._size_raise(equity, pot, my_stack, continue_cost, round_state, size_mult)
                return RaiseAction(amt)

        if CallAction in legal:
            if equity >= pot_odds + call_adj:
                return CallAction()
            if continue_cost <= BIG_BLIND * 2 and equity >= 0.32:
                return CallAction()

        if CheckAction in legal:
            return CheckAction()
        return FoldAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
