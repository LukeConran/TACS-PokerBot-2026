"""
Redraw-aware poker bot with lightweight equity estimation.
"""
from collections import Counter
from itertools import combinations
import random

from skeleton.actions import CallAction, CheckAction, FoldAction, RaiseAction, RedrawAction
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
from skeleton.states import BIG_BLIND, STARTING_STACK


RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK = [rank + suit for rank in RANKS for suit in SUITS]


class Player(Bot):
    """
    A poker bot tuned to beat simple baseline strategies while remaining fast.
    """

    def __init__(self):
        self.rng = random.Random(2026)
        self.round_num = 0
        self.match_delta = 0

    def handle_new_round(self, game_state, round_state, active):
        _ = round_state
        _ = active
        self.round_num = game_state.round_num

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

    def _board_is_paired(self, board):
        ranks = [card[0] for card in board]
        return len(set(ranks)) < len(ranks)

    def _straight_high(self, rank_values):
        unique = sorted(set(rank_values))
        if 12 in unique:
            unique = [-1] + unique
        best = None
        run = 1
        for idx in range(1, len(unique)):
            if unique[idx] == unique[idx - 1] + 1:
                run += 1
                if run >= 5:
                    best = unique[idx]
            else:
                run = 1
        return best

    def _evaluate_five(self, cards):
        rank_values = sorted((self._rank_value(card) for card in cards), reverse=True)
        rank_counts = Counter(rank_values)
        counts = sorted(rank_counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
        is_flush = len({card[1] for card in cards}) == 1
        straight_high = self._straight_high(rank_values)

        if is_flush and straight_high is not None:
            return (8, straight_high)
        if counts[0][1] == 4:
            kicker = max(rank for rank, count in counts if count == 1)
            return (7, counts[0][0], kicker)
        if counts[0][1] == 3 and counts[1][1] == 2:
            return (6, counts[0][0], counts[1][0])
        if is_flush:
            return (5,) + tuple(sorted(rank_values, reverse=True))
        if straight_high is not None:
            return (4, straight_high)
        if counts[0][1] == 3:
            kickers = sorted((rank for rank, count in counts if count == 1), reverse=True)
            return (3, counts[0][0]) + tuple(kickers)
        if counts[0][1] == 2 and counts[1][1] == 2:
            pair_ranks = sorted((rank for rank, count in counts if count == 2), reverse=True)
            kicker = max(rank for rank, count in counts if count == 1)
            return (2, pair_ranks[0], pair_ranks[1], kicker)
        if counts[0][1] == 2:
            pair_rank = counts[0][0]
            kickers = sorted((rank for rank, count in counts if count == 1), reverse=True)
            return (1, pair_rank) + tuple(kickers)
        return (0,) + tuple(sorted(rank_values, reverse=True))

    def _best_hand(self, cards):
        if len(cards) < 5:
            rank_values = sorted((self._rank_value(card) for card in cards), reverse=True)
            return (-1,) + tuple(rank_values)
        best = None
        for combo in combinations(cards, 5):
            score = self._evaluate_five(combo)
            if best is None or score > best:
                best = score
        return best

    def _made_strength(self, my_cards, board):
        return self._best_hand(self._clean_cards(my_cards + board))[0]

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

    def _preflop_strength(self, my_cards):
        values = sorted((self._rank_value(card) for card in my_cards), reverse=True)
        high, low = values
        suited = my_cards[0][1] == my_cards[1][1]
        gap = high - low

        if high == low:
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
        if high >= RANKS.index("Q") and low >= RANKS.index("T"):
            strength += 0.05
        return max(0.12, min(strength, 0.9))

    def _sample_equity(self, my_cards, board, street):
        known = set(self._clean_cards(my_cards + board))
        deck = [card for card in FULL_DECK if card not in known]
        if len(deck) < 2:
            return 0.5

        if street == 3:
            trials = 110
        elif street == 4:
            trials = 85
        else:
            trials = 60

        wins = 0.0
        board_cards = self._clean_cards(board)
        needed_board = 5 - len(board_cards)

        for _ in range(trials):
            sample = self.rng.sample(deck, 2 + needed_board)
            opp_cards = sample[:2]
            future_board = board_cards + sample[2:]
            my_score = self._best_hand(my_cards + future_board)
            opp_score = self._best_hand(opp_cards + future_board)
            if my_score > opp_score:
                wins += 1.0
            elif my_score == opp_score:
                wins += 0.5
        return wins / trials

    def _equity(self, round_state, active):
        my_cards = self._clean_cards(round_state.hands[active])
        board = self._clean_cards(round_state.board)
        if round_state.street == 0:
            return self._preflop_strength(my_cards)
        return self._sample_equity(my_cards, board, round_state.street)

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

    def _redraw_index(self, round_state, active, equity):
        if round_state.redraws_used[active] or round_state.street not in (3, 4):
            return None

        my_cards = self._clean_cards(round_state.hands[active])
        board = self._clean_cards(round_state.board)
        made_strength = self._made_strength(my_cards, board)
        has_draw = self._flush_draw(my_cards, board) or self._straight_draw(my_cards, board)

        if made_strength >= 1 and has_draw:
            return None
        if made_strength >= 2:
            return None
        if round_state.street == 4 and made_strength >= 1 and not self._board_is_paired(board):
            return None
        if equity >= 0.54:
            return None

        weakest = 0 if self._rank_value(my_cards[0]) <= self._rank_value(my_cards[1]) else 1
        if self._hole_card_is_expendable(my_cards, board, weakest):
            return weakest
        if made_strength == 0 and not has_draw:
            return weakest
        return None

    def _raise_amount(self, round_state, active, equity, continue_cost):
        min_raise, max_raise = round_state.raise_bounds()
        pot = self._pot_size(round_state)
        my_stack = round_state.stacks[active]
        pressure = 0.55 if equity >= 0.8 else 0.4 if equity >= 0.67 else 0.26
        target = round_state.pips[active] + continue_cost + max(BIG_BLIND * 2, int(pot * pressure))
        if equity >= 0.9 or my_stack <= BIG_BLIND * 8:
            target = max_raise
        return max(min_raise, min(max_raise, target))

    def _wrap_redraw(self, redraw_index, action):
        if redraw_index is None:
            return action
        return RedrawAction("hole", redraw_index, action)

    def get_action(self, game_state, round_state, active):
        _ = game_state
        legal_actions = round_state.legal_actions()

        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        continue_cost = opp_pip - my_pip
        pot = self._pot_size(round_state)
        equity = self._equity(round_state, active)
        redraw_index = self._redraw_index(round_state, active, equity)
        pot_odds = continue_cost / (pot + continue_cost) if continue_cost > 0 else 0.0

        if continue_cost == 0:
            should_raise = False
            if RaiseAction in legal_actions:
                if round_state.street == 0:
                    should_raise = equity >= 0.6
                else:
                    should_raise = equity >= 0.63 or (
                        equity >= 0.5
                        and (
                            self._flush_draw(round_state.hands[active], round_state.board)
                            or self._straight_draw(round_state.hands[active], round_state.board)
                        )
                    )
            if should_raise:
                amount = self._raise_amount(round_state, active, equity, continue_cost)
                return self._wrap_redraw(redraw_index, RaiseAction(amount))
            return self._wrap_redraw(redraw_index, CheckAction())

        if RaiseAction in legal_actions:
            strong_value = equity >= max(0.72, pot_odds + 0.18)
            semi_bluff = (
                continue_cost <= max(20, pot // 2)
                and equity >= max(0.42, pot_odds + 0.05)
                and round_state.street in (3, 4)
                and (
                    self._flush_draw(round_state.hands[active], round_state.board)
                    or self._straight_draw(round_state.hands[active], round_state.board)
                )
            )
            if strong_value or semi_bluff:
                amount = self._raise_amount(round_state, active, equity, continue_cost)
                return self._wrap_redraw(redraw_index, RaiseAction(amount))

        if CallAction in legal_actions and equity >= pot_odds - 0.03:
            return self._wrap_redraw(redraw_index, CallAction())

        if CallAction in legal_actions and continue_cost <= BIG_BLIND * 2 and equity >= 0.28:
            return self._wrap_redraw(redraw_index, CallAction())

        if CheckAction in legal_actions:
            return self._wrap_redraw(redraw_index, CheckAction())
        return self._wrap_redraw(redraw_index, FoldAction())


if __name__ == "__main__":
    run_bot(Player(), parse_args())
