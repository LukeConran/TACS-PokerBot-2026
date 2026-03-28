'''
Equity-driven pokerbot.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, RedrawAction
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

from equity import estimate_equity, best_redraw_hole


# Simulation budgets — stay well within the 180s match clock
PREFLOP_SIMS = 600
POSTFLOP_SIMS = 1000

# How much equity gain justifies using the redraw
REDRAW_GAIN_THRESHOLD = 0.04


class Player(Bot):

    def __init__(self):
        self.equity = 0.5  # cached per street

    def handle_new_round(self, game_state, round_state, active):
        self.equity = 0.5

    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()

        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        my_stack = round_state.stacks[active]
        pot = my_pip + opp_pip
        continue_cost = opp_pip - my_pip

        hole = [c for c in round_state.hands[active] if c != '??']
        board = [c for c in round_state.board if c]

        sims = PREFLOP_SIMS if round_state.street == 0 else POSTFLOP_SIMS
        equity = estimate_equity(hole, board, n_simulations=sims)
        self.equity = equity

        # ── Redraw decision ──────────────────────────────────────────────
        if RedrawAction in legal_actions:
            result = best_redraw_hole(hole, board, n_simulations=300)
            if result is not None and result[1] >= REDRAW_GAIN_THRESHOLD:
                idx, _ = result
                bet = _pick_bet(equity, pot, my_stack, continue_cost, legal_actions, round_state)
                return RedrawAction('hole', idx, bet)

        # ── Betting decision ─────────────────────────────────────────────
        return _pick_bet(equity, pot, my_stack, continue_cost, legal_actions, round_state)


def _pot_odds(continue_cost, pot):
    """Minimum equity needed to break even on a call."""
    total = pot + continue_cost
    return continue_cost / total if total > 0 else 0.0


def _pick_bet(equity, pot, my_stack, continue_cost, legal_actions, round_state):
    """Return the best action given equity and pot odds."""
    pot_odds = _pot_odds(continue_cost, pot)

    # --- No cost to continue (check available) ---
    if CheckAction in legal_actions:
        # Bet for value when strong
        if RaiseAction in legal_actions and equity >= 0.60:
            min_raise, max_raise = round_state.raise_bounds()
            amount = _size_bet(equity, pot, my_stack, min_raise, max_raise)
            return RaiseAction(amount)
        return CheckAction()

    # --- There is a bet to call ---
    if equity >= pot_odds + 0.05:
        # Re-raise when clearly ahead
        if RaiseAction in legal_actions and equity >= 0.65:
            min_raise, max_raise = round_state.raise_bounds()
            amount = _size_bet(equity, pot, my_stack, min_raise, max_raise)
            return RaiseAction(amount)
        return CallAction()

    if equity >= pot_odds:
        # Barely profitable — just call
        return CallAction()

    return FoldAction()


def _size_bet(equity, pot, my_stack, min_raise, max_raise):
    """
    Scale bet size with equity strength.
      0.60–0.70  → ~0.5x pot
      0.70–0.80  → ~1.0x pot
      0.80+      → ~1.5x pot (capped at stack)
    """
    if equity >= 0.80:
        target = int(pot * 1.5)
    elif equity >= 0.70:
        target = int(pot * 1.0)
    else:
        target = int(pot * 0.5)

    return max(min_raise, min(target, max_raise))


if __name__ == '__main__':
    run_bot(Player(), parse_args())
