"""
daniel_bot_v3 — speed-optimized NLHE + redraw (TACS PokerBots 2026).

- Street-based sim caps (flop 200 / turn 150 / river 100), capped by game clock.
- Two-pass equity: 50-sim screen, then full pass-2 only when the spot is not obvious.
- Early termination inside MC when running equity is clearly extreme.
- Rule-based hole redraw only (no nested simulations).
- Large equity cache; numpy sampling; lru_cache on 7-card pkrbot scores.
"""
from __future__ import annotations

import functools
import random
import time

import numpy as np
import pkrbot
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, RedrawAction
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
from skeleton.states import STARTING_STACK

RANK_ORDER = "23456789TJQKA"
# Rank index 0..5 = 2..7 → "7 or below" for simple redraw
_MAX_WEAK_RANK_IDX = 5


def _card_str(c):
    if c is None or c == "??":
        return c
    return str(c)


@functools.lru_cache(maxsize=100_000)
def _pkrbot_eval7_sorted(key: tuple[str, ...]) -> int:
    return pkrbot.evaluate([pkrbot.Card(c) for c in key])


class Player(Bot):
    def __init__(self):
        self.start_time = time.time()
        self.time_budget = 180.0
        self.hands_played = 0
        self.preflop_equity = self._build_preflop_table()
        self.equity_cache: dict[tuple, float] = {}
        self._cache_order: list[tuple] = []
        self.max_cache_size = 50_000
        self.sim_counts = {
            0: 0,
            3: 200,
            4: 150,
            5: 100,
        }
        self.opp_stats = {
            "hands": 0,
            "vpip": 0,
            "pfr": 0,
            "aggression": 0,
            "fold_to_raise": 0,
            "times_we_won": 0,
        }

    def handle_new_round(self, game_state, round_state, active):
        self.hands_played = game_state.round_num
        _ = round_state
        _ = active

    def handle_round_over(self, game_state, terminal_state, active):
        self.opp_stats["hands"] += 1
        if hasattr(terminal_state, "deltas") and terminal_state.deltas:
            my_delta = terminal_state.deltas[active]
            if my_delta > 0:
                self.opp_stats["times_we_won"] = self.opp_stats.get("times_we_won", 0) + 1
        _ = game_state

    def _rank_idx(self, card):
        s = _card_str(card)
        if not s or s == "??" or len(s) < 1:
            return -1
        try:
            return RANK_ORDER.index(s[0])
        except ValueError:
            return -1

    def _build_preflop_table(self):
        t = {}
        ranks = list(RANK_ORDER)

        def pair_eq(i):
            return 0.50 + 0.035 * i

        for i, r in enumerate(ranks):
            t[(r, r, None)] = min(0.86, pair_eq(i))

        def add_suited(hi, lo, eq):
            t[(hi, lo, True)] = eq
            t[(hi, lo, False)] = max(0.32, eq - 0.035)

        premiums = [
            ("A", "K", 0.67),
            ("A", "Q", 0.66),
            ("A", "J", 0.65),
            ("A", "T", 0.64),
            ("A", "9", 0.60),
            ("K", "Q", 0.63),
            ("K", "J", 0.61),
            ("K", "T", 0.59),
            ("Q", "J", 0.59),
            ("Q", "T", 0.57),
            ("J", "T", 0.57),
            ("T", "9", 0.55),
            ("9", "8", 0.53),
            ("8", "7", 0.51),
            ("7", "6", 0.49),
            ("6", "5", 0.47),
            ("5", "4", 0.45),
            ("4", "3", 0.43),
            ("3", "2", 0.41),
        ]
        for hi, lo, eq in premiums:
            add_suited(hi, lo, eq)

        for lo, eq in [
            ("2", 0.54),
            ("3", 0.53),
            ("4", 0.52),
            ("5", 0.52),
            ("6", 0.51),
            ("7", 0.52),
            ("8", 0.54),
        ]:
            add_suited("A", lo, eq)

        for lo, eq in [
            ("9", 0.58),
            ("T", 0.64),
            ("J", 0.65),
            ("Q", 0.66),
            ("K", 0.67),
        ]:
            add_suited("A", lo, eq)

        for lo, eq in [
            ("2", 0.48),
            ("3", 0.48),
            ("4", 0.49),
            ("5", 0.49),
            ("6", 0.50),
            ("7", 0.51),
            ("8", 0.53),
            ("9", 0.55),
        ]:
            add_suited("K", lo, eq)

        return t

    def get_preflop_equity(self, cards):
        c0, c1 = _card_str(cards[0]), _card_str(cards[1])
        if c0 == "??" or c1 == "??":
            return 0.50
        r0, s0, r1, s1 = c0[0], c0[1], c1[0], c1[1]
        suited = s0 == s1
        i0, i1 = self._rank_idx(c0), self._rank_idx(c1)
        if i0 < 0 or i1 < 0:
            return 0.50
        hi = RANK_ORDER[max(i0, i1)]
        lo = RANK_ORDER[min(i0, i1)]
        if hi == lo:
            return self.preflop_equity.get((hi, lo, None), 0.52)
        key = (hi, lo, True)
        keyo = (hi, lo, False)
        if suited:
            return self.preflop_equity.get(key, self.preflop_equity.get(keyo, 0.50) + 0.03)
        return self.preflop_equity.get(keyo, 0.48)

    def _full_deck_str(self):
        return [str(c) for c in pkrbot.Deck().cards]

    def _evaluate_7(self, hole, board):
        all_s = [_card_str(c) for c in hole + board if c and _card_str(c) != "??"]
        if len(all_s) < 2:
            return 0
        key = tuple(sorted(all_s))
        return _pkrbot_eval7_sorted(key)

    def _cache_put(self, key: tuple, value: float) -> None:
        if len(self.equity_cache) >= self.max_cache_size and self._cache_order:
            old = self._cache_order.pop(0)
            self.equity_cache.pop(old, None)
        self.equity_cache[key] = value
        self._cache_order.append(key)

    def estimate_equity(self, my_cards, board, street, num_sims=None, rng=None):
        if street == 0:
            return self.get_preflop_equity(my_cards)

        if num_sims is None:
            num_sims = self.sim_counts.get(street, 200)
        num_sims = max(num_sims, 1)

        if rng is None:
            rng = random

        def norm_key(h, b, ns: int):
            return (
                tuple(sorted(_card_str(x) for x in h)),
                tuple(sorted(_card_str(x) for x in b)),
                street,
                ns,
            )

        cache_key = norm_key(my_cards, board, num_sims)
        if cache_key in self.equity_cache:
            return self.equity_cache[cache_key]

        wins = 0.0
        ties = 0.0
        min_early = 40
        full_deck = self._full_deck_str()

        for i in range(num_sims):
            known = set()
            for c in my_cards + board:
                cs = _card_str(c)
                if cs and cs != "??":
                    known.add(cs)

            hole = [_card_str(c) for c in my_cards]
            if "??" in hole:
                idx = hole.index("??")
                avail = [c for c in full_deck if c not in known]
                if not avail:
                    continue
                hole[idx] = rng.choice(avail)
            for c in hole:
                if c and c != "??":
                    known.add(c)

            board_known = [_card_str(c) for c in board if c and _card_str(c) != "??"]
            need = 5 - len(board_known)
            avail = [c for c in full_deck if c not in known]
            if len(avail) < 2 + need:
                continue

            n_av = len(avail)
            k = 2 + need
            if n_av > 40:
                rest_idx = np.random.choice(n_av, size=k, replace=False)
                rest = [avail[j] for j in rest_idx]
            else:
                rest = rng.sample(avail, k)
            opp = rest[:2]
            fill = rest[2:]
            full_board = board_known + fill

            my_rank = self._evaluate_7(hole, full_board)
            opp_rank = self._evaluate_7(opp, full_board)
            if my_rank > opp_rank:
                wins += 1.0
            elif my_rank == opp_rank:
                ties += 0.5

            ran = i + 1
            if ran >= min_early:
                cur = (wins + ties) / ran
                # Early exit (do not cache: key assumes full num_sims).
                if cur > 0.82:
                    return cur
                if cur < 0.18:
                    return cur

        equity = (wins + ties) / max(num_sims, 1)
        self._cache_put(cache_key, equity)
        return equity

    def _num_sims_for_clock(self, game_state):
        gc = float(game_state.game_clock)
        if gc <= 0:
            gc = max(0.0, self.time_budget - (time.time() - self.start_time))
        if gc > 40:
            return 220
        if gc > 15:
            return 160
        if gc > 5:
            return 120
        return 80

    def _pass2_sims(self, street, game_state) -> int:
        base = self.sim_counts.get(street, 200)
        cap = self._num_sims_for_clock(game_state)
        return max(100, min(base, cap))

    def evaluate_redraw_simple(self, my_cards, board, street, redraws_used_active):
        if redraws_used_active or street >= 5 or street == 0:
            return None
        c0, c1 = _card_str(my_cards[0]), _card_str(my_cards[1])
        if c0 == "??" or c1 == "??":
            return None
        i0, i1 = self._rank_idx(c0), self._rank_idx(c1)
        if i0 < 0 or i1 < 0:
            return None
        if min(i0, i1) > _MAX_WEAK_RANK_IDX:
            return None
        worst_idx = 0 if i0 <= i1 else 1
        return ("hole", worst_idx)

    def _pot_total(self, round_state):
        return 2 * STARTING_STACK - round_state.stacks[0] - round_state.stacks[1]

    def _legalize(self, action, round_state, legal):
        basic = legal - {RedrawAction}

        if isinstance(action, RedrawAction):
            inner = self._legalize(action.action, round_state, basic)
            return RedrawAction(action.target_type, action.target_index, inner)

        if isinstance(action, RaiseAction):
            if RaiseAction not in basic:
                return CallAction() if CallAction in basic else CheckAction()
            mn, mx = round_state.raise_bounds()
            amt = max(mn, min(action.amount, mx))
            return RaiseAction(amt)

        if isinstance(action, FoldAction) and FoldAction in basic:
            return FoldAction()
        if isinstance(action, CallAction) and CallAction in basic:
            return CallAction()
        if isinstance(action, CheckAction) and CheckAction in basic:
            return CheckAction()

        if CheckAction in basic:
            return CheckAction()
        if CallAction in basic:
            return CallAction()
        if FoldAction in basic:
            return FoldAction()
        if RaiseAction in basic:
            mn, _ = round_state.raise_bounds()
            return RaiseAction(mn)
        return CheckAction()

    def _raise_pot_frac(self, round_state, my_pip, my_stack, pot, legal, frac):
        if RaiseAction not in legal:
            return None
        bet_size = int(pot * frac)
        raise_to = my_pip + max(bet_size, 1)
        raise_to = min(raise_to, my_stack + my_pip)
        mn, mx = round_state.raise_bounds()
        raise_to = max(mn, min(raise_to, mx))
        if raise_to <= my_pip:
            return None
        return RaiseAction(raise_to)

    def _apply_simple_redraw(self, base_action, round_state, legal, my_cards, board, street, active):
        if RedrawAction not in legal or round_state.redraws_used[active]:
            return base_action
        r = self.evaluate_redraw_simple(
            my_cards, board, street, round_state.redraws_used[active]
        )
        if r is None:
            return base_action
        inner = base_action.action if isinstance(base_action, RedrawAction) else base_action
        return RedrawAction(r[0], r[1], inner)

    def get_action(self, game_state, round_state, active):
        legal = round_state.legal_actions()
        rng = random

        my_cards = [_card_str(c) for c in round_state.hands[active]]
        board = [_card_str(c) for c in round_state.board]
        street = round_state.street

        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        to_call = opp_pip - my_pip
        my_stack = round_state.stacks[active]

        pot = self._pot_total(round_state)
        pot_odds = to_call / (pot + to_call) if to_call > 0 else 0.0

        in_position = active == 1
        if in_position:
            value_threshold = 0.55
            call_threshold = pot_odds + 0.03
        else:
            value_threshold = 0.65
            call_threshold = pot_odds + 0.08

        nh = self.opp_stats["hands"]
        if nh > 50:
            wr = self.opp_stats.get("times_we_won", 0) / max(nh, 1)
            if wr < 0.40:
                value_threshold += 0.05
            elif wr > 0.60:
                value_threshold -= 0.05
        value_threshold = max(0.50, min(0.72, value_threshold))

        # --- Preflop: table only, single pass ---
        if street == 0:
            equity = self.get_preflop_equity(my_cards)
            base_action = self._decide_from_equity(
                equity,
                legal,
                round_state,
                my_pip,
                my_stack,
                pot,
                to_call,
                pot_odds,
                in_position,
                value_threshold,
                call_threshold,
                rng,
            )
            base_action = self._legalize(base_action, round_state, legal)
            return self._apply_simple_redraw(
                base_action, round_state, legal, my_cards, board, street, active
            )

        # --- Postflop pass 1: 50 sims ---
        quick = self.estimate_equity(my_cards, board, street, num_sims=50, rng=rng)

        if quick > 0.75 and RaiseAction in legal and to_call < my_stack:
            act = self._raise_pot_frac(round_state, my_pip, my_stack, pot, legal, 0.8)
            if act is not None:
                out = self._legalize(act, round_state, legal)
                return self._apply_simple_redraw(
                    out, round_state, legal, my_cards, board, street, active
                )

        if quick < 0.25 and to_call > pot * 0.3 and to_call > 0:
            out = self._legalize(FoldAction(), round_state, legal)
            return self._apply_simple_redraw(
                out, round_state, legal, my_cards, board, street, active
            )

        if quick > pot_odds + 0.12 and to_call > 0:
            out = self._legalize(
                CallAction() if CallAction in legal else CheckAction(), round_state, legal
            )
            return self._apply_simple_redraw(
                out, round_state, legal, my_cards, board, street, active
            )

        if quick < pot_odds - 0.12:
            if to_call > 0:
                out = self._legalize(FoldAction(), round_state, legal)
            else:
                out = self._legalize(CheckAction(), round_state, legal)
            return self._apply_simple_redraw(
                out, round_state, legal, my_cards, board, street, active
            )

        # --- Pass 2: street-based sims (capped by clock) ---
        ns = self._pass2_sims(street, game_state)
        equity = self.estimate_equity(my_cards, board, street, num_sims=ns, rng=rng)

        base_action = self._decide_from_equity(
            equity,
            legal,
            round_state,
            my_pip,
            my_stack,
            pot,
            to_call,
            pot_odds,
            in_position,
            value_threshold,
            call_threshold,
            rng,
        )
        base_action = self._legalize(base_action, round_state, legal)
        return self._apply_simple_redraw(
            base_action, round_state, legal, my_cards, board, street, active
        )

    def _decide_from_equity(
        self,
        equity,
        legal,
        round_state,
        my_pip,
        my_stack,
        pot,
        to_call,
        pot_odds,
        in_position,
        value_threshold,
        call_threshold,
        rng,
    ):
        if equity > 0.75:
            if RaiseAction in legal:
                act = self._raise_pot_frac(round_state, my_pip, my_stack, pot, legal, 0.9)
                if act is not None:
                    return act
            return CallAction() if CallAction in legal else CheckAction()

        if equity > value_threshold:
            if RaiseAction in legal:
                act = self._raise_pot_frac(round_state, my_pip, my_stack, pot, legal, 0.6)
                if act is not None:
                    return act
            return CallAction() if CallAction in legal else CheckAction()

        if equity > call_threshold:
            return CallAction() if CallAction in legal else CheckAction()

        if equity > 0.25 and CheckAction in legal:
            return CheckAction()

        if to_call > 0:
            if in_position and rng.random() < 0.15 and RaiseAction in legal:
                act = self._raise_pot_frac(round_state, my_pip, my_stack, pot, legal, 0.5)
                return act if act is not None else FoldAction()
            return FoldAction()
        return CheckAction()


if __name__ == "__main__":
    run_bot(Player(), parse_args())
