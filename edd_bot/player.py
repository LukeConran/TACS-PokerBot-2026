'''
Simple example pokerbot, written in Python.
'''
import random

import pkrbot
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, RedrawAction
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
from skeleton.states import STARTING_STACK

RANK_ORDER = '23456789TJQKA'


class Player(Bot):

    def __init__(self):
        pass

    # --------------------------
    # UTIL
    # --------------------------

    def rank_value(self, card):
        if card == '??':
            return -1
        return RANK_ORDER.index(card[0])

    # --------------------------
    # HAND STRENGTH
    # --------------------------

    def evaluate_strength(self, hand, board):
        cards = hand + board
        ranks = [c[0] for c in cards if c != '??']
        suits = [c[1] for c in cards if c != '??']

        counts = {}
        for r in ranks:
            counts[r] = counts.get(r, 0) + 1

        values = sorted(counts.values(), reverse=True)

        if values and values[0] >= 3:
            return 0.9
        if len(values) > 1 and values[0] == 2 and values[1] == 2:
            return 0.8
        if values and values[0] == 2:
            return 0.6

        # Flush draw
        for s in suits:
            if suits.count(s) >= 4:
                return 0.55

        # Straight draw
        vals = sorted([self.rank_value(c) for c in cards if c != '??'])
        if len(vals) >= 4 and max(vals) - min(vals) <= 4:
            return 0.5

        if hand:
            high = max([self.rank_value(c) for c in hand])
            return 0.25 + high / 20.0

        return 0.3

    # --------------------------
    # BOARD ANALYSIS
    # --------------------------

    def board_dangerous(self, board):
        if len(board) < 3:
            return False

        suits = [c[1] for c in board if c != '??']
        ranks = sorted([self.rank_value(c) for c in board if c != '??'])

        if suits and max(suits.count(s) for s in suits) >= 2:
            return True

        if len(ranks) >= 3 and max(ranks) - min(ranks) <= 4:
            return True

        return False

    def weakest_hole_index(self, hand):
        if len(hand) < 2:
            return 0
        return 0 if self.rank_value(hand[0]) <= self.rank_value(hand[1]) else 1

    # --------------------------
    # SAFE BOARD INDEX (FIXED)
    # --------------------------

    def best_board_card_index(self, round_state):
        street = round_state.street
        board = round_state.board

        if street == 3:
            max_index = min(2, len(board) - 1)
        elif street == 4:
            max_index = min(3, len(board) - 1)
        else:
            return 0

        valid = []
        for i in range(max_index + 1):
            if board[i] != '??':
                valid.append((i, self.rank_value(board[i])))

        if not valid:
            return 0

        return max(valid, key=lambda x: x[1])[0]

    # --------------------------
    # REDRAW (FULLY FIXED)
    # --------------------------

    def should_redraw(self, round_state, active, strength):
        if round_state.redraws_used[active]:
            return False, None

        street = round_state.street
        hand = round_state.hands[active]
        board = round_state.board

        if street not in (3, 4):
            return False, None

        # Weak → fix hole
        if strength < 0.45:
            return True, ("hole", self.weakest_hole_index(hand))

        # Medium → sometimes fix
        if 0.45 <= strength < 0.65:
            if random.random() < 0.5:
                return True, ("hole", self.weakest_hole_index(hand))

        # Strong → attack board safely
        if strength > 0.7 and self.board_dangerous(board):
            return True, ("board", self.best_board_card_index(round_state))

        return False, None

    # --------------------------
    # PREFLOP
    # --------------------------

    def preflop_action(self, round_state, active, legal):
        my_cards = round_state.hands[active]
        ranks = sorted([self.rank_value(c) for c in my_cards], reverse=True)

        if RaiseAction in legal:
            min_raise, max_raise = round_state.raise_bounds()

        if round_state.button % 2 == active:
            if RaiseAction in legal and random.random() < 0.95:
                return RaiseAction(min_raise)
            if CallAction in legal:
                return CallAction()
            if CheckAction in legal:
                return CheckAction()
            return FoldAction()

        if ranks[0] >= 10 and ranks[1] >= 7:
            if RaiseAction in legal:
                return RaiseAction(max_raise)

        if ranks[0] >= 4:
            if CallAction in legal:
                return CallAction()

        return FoldAction()

    # --------------------------
    # POSTFLOP
    # --------------------------

    def postflop_action(self, round_state, active, strength, legal):
        continue_cost = round_state.pips[1 - active] - round_state.pips[active]

        pot = (
            sum(round_state.pips)
            + (STARTING_STACK - round_state.stacks[0])
            + (STARTING_STACK - round_state.stacks[1])
        )

        redraw_available = not round_state.redraws_used[active]

        if redraw_available:
            if round_state.street == 3:
                strength += 0.12
            elif round_state.street == 4:
                strength += 0.08

        strength = min(strength, 1.0)

        if continue_cost > 0:
            pot_odds = continue_cost / (pot + continue_cost)
        else:
            pot_odds = 0

        if RaiseAction in legal:
            min_raise, max_raise = round_state.raise_bounds()

        if continue_cost == 0:
            if strength > 0.7 and RaiseAction in legal and random.random() < 0.75:
                return RaiseAction(int((min_raise + max_raise) / 2))
            if strength < 0.4 and RaiseAction in legal and random.random() < 0.2:
                return RaiseAction(min_raise)
            if CheckAction in legal:
                return CheckAction()
            if CallAction in legal:
                return CallAction()
            return FoldAction()

        if strength > pot_odds:
            if strength > 0.8 and RaiseAction in legal:
                return RaiseAction(max_raise)
            if CallAction in legal:
                return CallAction()

        return FoldAction()

    # --------------------------
    # MAIN
    # --------------------------

    def get_action(self, game_state, round_state, active):
        legal = round_state.legal_actions()

        if round_state.street == 0:
            return self.preflop_action(round_state, active, legal)

        hand = round_state.hands[active]
        board = round_state.board

        strength = self.evaluate_strength(hand, board)

        if RedrawAction in legal:
            redraw, target = self.should_redraw(round_state, active, strength)
            if redraw:
                action = self.postflop_action(round_state, active, strength, legal)
                return RedrawAction(target[0], target[1], action)

        return self.postflop_action(round_state, active, strength, legal)

    def handle_new_round(self, game_state, round_state, active):
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        pass


if __name__ == '__main__':
    run_bot(Player(), parse_args())