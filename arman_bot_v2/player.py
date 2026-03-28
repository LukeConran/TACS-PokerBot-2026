"""
Stronger redraw-aware poker bot using pkrbot-backed equity rollouts.
"""
from collections import Counter
import random

import pkrbot

from skeleton.actions import CallAction, CheckAction, FoldAction, RaiseAction, RedrawAction
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
from skeleton.states import BIG_BLIND, STARTING_STACK


RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK = [rank + suit for rank in RANKS for suit in SUITS]


class Player(Bot):
    """
    Equity-driven redraw bot with conservative board manipulation.
    """

    def __init__(self):
        self.rng = random.Random(2026)
        self.round_num = 0
        self.match_delta = 0
        self._equity_cache = {}

    def handle_new_round(self, game_state, round_state, active):
        _ = round_state
        _ = active
        self.round_num = game_state.round_num
        self._equity_cache = {}

    def handle_round_over(self, game_state, terminal_state, active):
        _ = game_state
        self.match_delta += terminal_state.deltas[active]

    def _rank_value(self, card):
        if not card or card == "??":
            return -1
        return RANKS.index(card[0])

    def _clean_cards(self, cards):
        return [card for card in cards if card and card != "??"]

    def _pot_size(self, round_state):
        return (2 * STARTING_STACK) - round_state.stacks[0] - round_state.stacks[1]

    def _preflop_strength(self, my_cards):
        values = sorted((self._rank_value(card) for card in my_cards), reverse=True)
        high, low = values
        suited = my_cards[0][1] == my_cards[1][1]
        gap = high - low

        if high == low:
            return min(0.57 + (high / 23.0), 0.97)

        strength = 0.31 + (high / 18.0) + (low / 38.0)
        if suited:
            strength += 0.035
        if gap == 1:
            strength += 0.05
        elif gap == 2:
            strength += 0.025
        elif gap >= 4:
            strength -= 0.045
        if high >= RANKS.index("Q") and low >= RANKS.index("T"):
            strength += 0.055
        if high == RANKS.index("A"):
            strength += 0.02
        return max(0.1, min(strength, 0.92))

    def _sample_equity(self, my_cards, board, street):
        known = set(self._clean_cards(my_cards + board))
        deck = [card for card in FULL_DECK if card not in known]
        if len(deck) < 2:
            return 0.5

        if street == 3:
            trials = 120
        elif street == 4:
            trials = 90
        else:
            trials = 65

        board_cards = self._clean_cards(board)
        needed_board = 5 - len(board_cards)
        wins = 0.0

        for _ in range(trials):
            sample = self.rng.sample(deck, 2 + needed_board)
            opp_cards = sample[:2]
            future_board = board_cards + sample[2:]
            my_score = pkrbot.evaluate(my_cards + future_board)
            opp_score = pkrbot.evaluate(opp_cards + future_board)
            if my_score > opp_score:
                wins += 1.0
            elif my_score == opp_score:
                wins += 0.5
        return wins / trials

    def _equity_for_cards(self, my_cards, board, street):
        clean_hole = tuple(self._clean_cards(my_cards))
        clean_board = tuple(self._clean_cards(board))
        key = (clean_hole, clean_board, street)
        if key in self._equity_cache:
            return self._equity_cache[key]
        if street == 0:
            value = self._preflop_strength(list(clean_hole))
        else:
            value = self._sample_equity(list(clean_hole), list(clean_board), street)
        self._equity_cache[key] = value
        return value

    def _equity(self, round_state, active):
        return self._equity_for_cards(
            round_state.hands[active],
            round_state.board,
            round_state.street,
        )

    def _flush_draw(self, my_cards, board):
        suit_counts = Counter(card[1] for card in self._clean_cards(my_cards + board))
        return max(suit_counts.values(), default=0) >= 4

    def _straight_draw(self, my_cards, board):
        values = sorted(set(self._rank_value(card) for card in self._clean_cards(my_cards + board)))
        if 12 in values:
            values = [-1] + values
        for start in range(-1, 10):
            window = set(range(start, start + 5))
            if len(window.intersection(values)) >= 4:
                return True
        return False

    def _made_class(self, my_cards, board):
        cards = self._clean_cards(my_cards + board)
        if len(cards) < 5:
            return -1
        score = pkrbot.evaluate(cards)
        # Larger pkrbot scores are better; coarse buckets are enough for policy.
        if score >= 8_000_000:
            return 7
        if score >= 7_000_000:
            return 6
        if score >= 6_000_000:
            return 5
        if score >= 5_000_000:
            return 4
        if score >= 4_000_000:
            return 3
        if score >= 3_000_000:
            return 2
        if score >= 2_000_000:
            return 1
        return 0

    def _hole_card_is_expendable(self, my_cards, board, index):
        kept = my_cards[1 - index]
        card = my_cards[index]
        card_rank = self._rank_value(card)

        if card_rank >= RANKS.index("T"):
            return False
        if self._rank_value(kept) == card_rank:
            return False
        if any(board_card[0] == card[0] for board_card in board):
            return False
        if card[1] == kept[1]:
            suit_matches = sum(1 for board_card in board if board_card[1] == card[1])
            if suit_matches >= 2:
                return False
        return True

    def _board_redraw_candidate(self, my_cards, board, equity):
        if len(board) < 3 or equity >= 0.34:
            return None
        if self._made_class(my_cards, board) >= 1:
            return None
        if self._flush_draw(my_cards, board) or self._straight_draw(my_cards, board):
            return None

        rank_counts = Counter(card[0] for card in board)
        suit_counts = Counter(card[1] for card in board)

        paired = [idx for idx, card in enumerate(board) if rank_counts[card[0]] >= 2]
        if paired:
            return max(paired, key=lambda idx: self._rank_value(board[idx]))

        heavy_suits = {suit for suit, count in suit_counts.items() if count >= 3}
        if heavy_suits:
            suited = [idx for idx, card in enumerate(board) if card[1] in heavy_suits]
            return max(suited, key=lambda idx: self._rank_value(board[idx]))

        values = sorted((self._rank_value(card), idx) for idx, card in enumerate(board))
        for pos in range(len(values) - 2):
            window = values[pos : pos + 3]
            if window[2][0] - window[0][0] <= 4:
                return window[1][1]
        return None

    def _redraw_gain(self, round_state, active, target_type, index, base_equity):
        my_cards = self._clean_cards(round_state.hands[active])
        board = self._clean_cards(round_state.board)
        known = set(my_cards + board)
        deck = [card for card in FULL_DECK if card not in known]
        if not deck:
            return 0.0

        samples = min(8, len(deck))
        total = 0.0
        for replacement in self.rng.sample(deck, samples):
            new_hole = list(my_cards)
            new_board = list(board)
            if target_type == "hole":
                new_hole[index] = replacement
            else:
                new_board[index] = replacement
            total += self._equity_for_cards(new_hole, new_board, round_state.street)
        return (total / samples) - base_equity

    def _redraw_target(self, round_state, active, equity):
        if round_state.redraws_used[active] or round_state.street not in (3, 4):
            return None

        my_cards = self._clean_cards(round_state.hands[active])
        board = self._clean_cards(round_state.board)
        made_class = self._made_class(my_cards, board)
        has_draw = self._flush_draw(my_cards, board) or self._straight_draw(my_cards, board)

        if made_class >= 2:
            return None
        if round_state.street == 4 and made_class >= 1 and not self._board_redraw_candidate(my_cards, board, 0.0):
            return None

        weakest = 0 if self._rank_value(my_cards[0]) <= self._rank_value(my_cards[1]) else 1
        if equity < 0.56:
            if self._hole_card_is_expendable(my_cards, board, weakest) or (made_class <= 0 and not has_draw):
                hole_gain = self._redraw_gain(round_state, active, "hole", weakest, equity)
                if hole_gain >= 0.025:
                    return ("hole", weakest)

        board_candidate = self._board_redraw_candidate(my_cards, board, equity)
        if board_candidate is not None:
            board_gain = self._redraw_gain(round_state, active, "board", board_candidate, equity)
            if board_gain >= 0.05:
                return ("board", board_candidate)

        return None

    def _raise_amount(self, round_state, active, equity, continue_cost):
        min_raise, max_raise = round_state.raise_bounds()
        pot = self._pot_size(round_state)
        my_stack = round_state.stacks[active]

        if round_state.street == 0:
            pressure = 0.22 if equity >= 0.82 else 0.15 if equity >= 0.72 else 0.10
        elif round_state.street == 3:
            pressure = 0.52 if equity >= 0.82 else 0.36 if equity >= 0.68 else 0.24
        else:
            pressure = 0.64 if equity >= 0.82 else 0.44 if equity >= 0.68 else 0.28

        target = round_state.pips[active] + continue_cost + max(BIG_BLIND * 2, int(pot * pressure))
        if equity >= 0.91 or my_stack <= BIG_BLIND * 8:
            target = max_raise
        return max(min_raise, min(max_raise, target))

    def _wrap_redraw(self, redraw_target, action):
        if redraw_target is None:
            return action
        return RedrawAction(redraw_target[0], redraw_target[1], action)

    def get_action(self, game_state, round_state, active):
        _ = game_state
        legal_actions = round_state.legal_actions()

        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        continue_cost = opp_pip - my_pip
        pot = self._pot_size(round_state)
        equity = self._equity(round_state, active)
        redraw_target = self._redraw_target(round_state, active, equity)
        pot_odds = continue_cost / (pot + continue_cost) if continue_cost > 0 else 0.0
        has_draw = self._flush_draw(round_state.hands[active], round_state.board) or self._straight_draw(
            round_state.hands[active],
            round_state.board,
        )

        if continue_cost == 0:
            should_raise = False
            if RaiseAction in legal_actions:
                if round_state.street == 0:
                    should_raise = equity >= 0.61
                else:
                    should_raise = equity >= 0.65 or (equity >= 0.50 and has_draw)
            if should_raise:
                amount = self._raise_amount(round_state, active, equity, continue_cost)
                return self._wrap_redraw(redraw_target, RaiseAction(amount))
            return self._wrap_redraw(redraw_target, CheckAction())

        if round_state.street == 0:
            if RaiseAction in legal_actions and equity >= max(0.79, pot_odds + 0.18):
                amount = self._raise_amount(round_state, active, equity, continue_cost)
                return self._wrap_redraw(redraw_target, RaiseAction(amount))
            if CallAction in legal_actions and equity >= max(0.34, pot_odds + 0.02):
                return self._wrap_redraw(redraw_target, CallAction())
            if CallAction in legal_actions and continue_cost <= BIG_BLIND * 2 and equity >= 0.28:
                return self._wrap_redraw(redraw_target, CallAction())
            return self._wrap_redraw(redraw_target, FoldAction())

        if RaiseAction in legal_actions:
            strong_value = equity >= max(0.73, pot_odds + 0.17)
            semi_bluff = (
                continue_cost <= max(24, pot // 2)
                and equity >= max(0.43, pot_odds + 0.04)
                and round_state.street in (3, 4)
                and has_draw
            )
            if strong_value or semi_bluff:
                amount = self._raise_amount(round_state, active, equity, continue_cost)
                return self._wrap_redraw(redraw_target, RaiseAction(amount))

        if CallAction in legal_actions and equity >= pot_odds - 0.02:
            return self._wrap_redraw(redraw_target, CallAction())

        if CallAction in legal_actions and continue_cost <= BIG_BLIND * 3 and equity >= 0.29:
            return self._wrap_redraw(redraw_target, CallAction())

        if CheckAction in legal_actions:
            return self._wrap_redraw(redraw_target, CheckAction())
        return self._wrap_redraw(redraw_target, FoldAction())


if __name__ == "__main__":
    run_bot(Player(), parse_args())
