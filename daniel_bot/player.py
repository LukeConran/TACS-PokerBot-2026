"""
Equity-driven NLHE bot with strategic redraw (TACS 2026).

- Preflop: rank/suited lookup (no simulation).
- Postflop: Monte Carlo equity vs one random opponent hand with caching.
- Redraw: compare expected equity after replacing weak hole or board cards.
- Betting: pot odds, value raises (~pot fraction), time-aware simulation counts.

Uses game_state.game_clock for remaining time (engine sends T{seconds} each action).
"""
import random
import time

import pkrbot
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, RedrawAction
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
from skeleton.states import STARTING_STACK

RANK_ORDER = "23456789TJQKA"


def _card_str(c):
    if c is None or c == "??":
        return c
    return str(c)


class Player(Bot):
    def __init__(self):
        self.start_time = time.time()
        self.time_budget = 180.0
        self.hands_played = 0
        self.preflop_equity = self._build_preflop_table()
        self.equity_cache = {}
        self._cache_order = []
        self._cache_max = 4000
        self.opp_stats = {
            "hands": 0,
            "vpip": 0,
            "pfr": 0,
            "aggression": 0,
            "fold_to_raise": 0,
        }

    def handle_new_round(self, game_state, round_state, active):
        self.hands_played = game_state.round_num
        _ = round_state
        _ = active

    def handle_round_over(self, game_state, terminal_state, active):
        self.opp_stats["hands"] += 1
        _ = game_state
        _ = terminal_state
        _ = active

    def _rank_idx(self, card):
        s = _card_str(card)
        if not s or s == "??" or len(s) < 1:
            return -1
        try:
            return RANK_ORDER.index(s[0])
        except ValueError:
            return -1

    def _build_preflop_table(self):
        """Heads-up vs random-ish equity approximations; keys (hi, lo, suited bool|None)."""
        t = {}
        ranks = list(RANK_ORDER)

        def pair_eq(i):
            return 0.50 + 0.035 * i

        for i, r in enumerate(ranks):
            t[(r, r, None)] = min(0.86, pair_eq(i))

        def add_suited(hi, lo, eq):
            t[(hi, lo, True)] = eq
            t[(hi, lo, False)] = max(0.32, eq - 0.035)

        # Premiums & broadways (suited / offsuit via add_suited)
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

        # Ax rag
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

        # Kx
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
        """hole: 2 strings, board: up to 5 strings; unknowns skipped (caller must complete)."""
        all_s = [_card_str(c) for c in hole + board if c and _card_str(c) != "??"]
        if len(all_s) < 2:
            return 0
        return pkrbot.evaluate([pkrbot.Card(c) for c in all_s])

    def estimate_equity(self, my_cards, board, street, num_sims=400, rng=None):
        if street == 0:
            return self.get_preflop_equity(my_cards)

        if rng is None:
            rng = random

        def norm_key(h, b):
            return (
                tuple(sorted(_card_str(x) for x in h)),
                tuple(sorted(_card_str(x) for x in b)),
                street,
            )

        cache_key = norm_key(my_cards, board)
        if cache_key in self.equity_cache:
            return self.equity_cache[cache_key]

        wins = 0.0
        ties = 0.0

        for _ in range(num_sims):
            known = set()
            for c in my_cards + board:
                cs = _card_str(c)
                if cs and cs != "??":
                    known.add(cs)

            hole = [_card_str(c) for c in my_cards]
            if "??" in hole:
                idx = hole.index("??")
                avail = [c for c in self._full_deck_str() if c not in known]
                if not avail:
                    continue
                hole[idx] = rng.choice(avail)
            for c in hole:
                if c and c != "??":
                    known.add(c)

            board_known = [_card_str(c) for c in board if c and _card_str(c) != "??"]
            need = 5 - len(board_known)
            avail = [c for c in self._full_deck_str() if c not in known]
            if len(avail) < 2 + need:
                continue
            rest = rng.sample(avail, 2 + need)
            opp = rest[:2]
            fill = rest[2:]
            full_board = board_known + fill

            my_rank = self._evaluate_7(hole, full_board)
            opp_rank = self._evaluate_7(opp, full_board)
            if my_rank > opp_rank:
                wins += 1.0
            elif my_rank == opp_rank:
                ties += 0.5

        equity = (wins + ties) / max(num_sims, 1)
        self._cache_put(cache_key, equity)
        return equity

    def _cache_put(self, key, value):
        if len(self.equity_cache) >= self._cache_max and self._cache_order:
            old = self._cache_order.pop(0)
            self.equity_cache.pop(old, None)
        self.equity_cache[key] = value
        self._cache_order.append(key)

    def _simulate_hole_redraw(self, my_cards, hole_idx, board, street, num_samples, sims_inner, rng):
        used = {_card_str(c) for c in my_cards + board if c and _card_str(c) != "??"}
        deck = [c for c in self._full_deck_str() if c not in used]
        if not deck:
            return 0.0
        samples = min(num_samples, len(deck))
        picks = rng.sample(deck, samples) if len(deck) >= samples else deck
        s = 0.0
        for nc in picks:
            new_hand = [_card_str(my_cards[0]), _card_str(my_cards[1])]
            new_hand[hole_idx] = str(nc)
            s += self.estimate_equity(new_hand, board, street, num_sims=sims_inner, rng=rng)
        return s / max(len(picks), 1)

    def _simulate_board_redraw(self, my_cards, board, board_idx, street, num_samples, sims_inner, rng):
        used = {_card_str(c) for c in my_cards + board if c and _card_str(c) != "??"}
        deck = [c for c in self._full_deck_str() if c not in used]
        if not deck:
            return 0.0
        samples = min(num_samples, len(deck))
        picks = rng.sample(deck, samples) if len(deck) >= samples else deck
        s = 0.0
        nb = [_card_str(c) for c in board]
        for nc in picks:
            nb2 = list(nb)
            nb2[board_idx] = str(nc)
            s += self.estimate_equity(my_cards, nb2, street, num_sims=sims_inner, rng=rng)
        return s / max(len(picks), 1)

    def evaluate_redraw_options(self, my_cards, board, street, rng, min_gain, cur_eq, num_samples, sims_inner):
        if street >= 5:
            return None

        best = None
        best_gain = min_gain

        for hole_idx in (0, 1):
            if my_cards[hole_idx] == "??":
                continue
            avg = self._simulate_hole_redraw(
                my_cards, hole_idx, board, street, num_samples, sims_inner, rng
            )
            g = avg - cur_eq
            if g > best_gain:
                best_gain = g
                best = ("hole", hole_idx, g)

        nboard = len([c for c in board if c])
        for bi in range(nboard):
            if board[bi] == "??":
                continue
            avg = self._simulate_board_redraw(
                my_cards, board, bi, street, num_samples, sims_inner, rng
            )
            g = avg - cur_eq
            if g > best_gain:
                best_gain = g
                best = ("board", bi, g)

        return best

    def _pot_total(self, round_state):
        return 2 * STARTING_STACK - round_state.stacks[0] - round_state.stacks[1]

    def _num_sims_for_clock(self, game_state):
        gc = float(game_state.game_clock)
        if gc <= 0:
            gc = max(0.0, self.time_budget - (time.time() - self.start_time))
        if gc > 40:
            return 380
        if gc > 15:
            return 260
        if gc > 5:
            return 160
        return 100

    def _legalize(self, action, round_state, legal):
        """legal is a set of action classes; return a concrete legal action."""
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

    def get_action(self, game_state, round_state, active):
        legal = round_state.legal_actions()
        rng = random

        my_cards = [_card_str(c) for c in round_state.hands[active]]
        board = [_card_str(c) for c in round_state.board]
        street = round_state.street

        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        continue_cost = opp_pip - my_pip
        my_stack = round_state.stacks[active]

        pot = self._pot_total(round_state)
        num_sims = self._num_sims_for_clock(game_state)

        equity = self.estimate_equity(my_cards, board, street, num_sims=num_sims, rng=rng)

        to_call = continue_cost
        pot_after = pot + to_call if to_call > 0 else pot
        pot_odds = (to_call / pot_after) if to_call > 0 and pot_after > 0 else 0.0

        # --- Base betting decision ---
        base = None
        if equity > 0.66 and RaiseAction in legal and continue_cost < my_stack:
            bet_frac = 0.72
            target = my_pip + int(max(2, bet_frac * pot))
            mn, mx = round_state.raise_bounds()
            raise_to = max(mn, min(target, mx))
            if raise_to > my_pip and RaiseAction in legal:
                base = RaiseAction(raise_to)
            elif CallAction in legal:
                base = CallAction()
            elif CheckAction in legal:
                base = CheckAction()
            else:
                base = FoldAction()
        elif equity > pot_odds + 0.09:
            if RaiseAction in legal and continue_cost == 0 and equity > 0.72 and my_stack > 0:
                mn, mx = round_state.raise_bounds()
                target = my_pip + int(max(2, 0.55 * pot))
                raise_to = max(mn, min(target, mx))
                if raise_to > my_pip:
                    base = RaiseAction(raise_to)
                elif CheckAction in legal:
                    base = CheckAction()
                else:
                    base = CallAction()
            elif CallAction in legal:
                base = CallAction()
            elif CheckAction in legal:
                base = CheckAction()
            else:
                base = FoldAction()
        elif equity > 0.34:
            if CheckAction in legal:
                base = CheckAction()
            elif CallAction in legal:
                base = CallAction() if to_call <= 0.12 * pot + 1 else FoldAction()
            else:
                base = FoldAction()
        else:
            if to_call == 0 and CheckAction in legal:
                base = CheckAction()
            elif to_call > 0 and CallAction in legal and equity > pot_odds - 0.05:
                base = CallAction()
            elif to_call > 0:
                base = FoldAction()
            else:
                base = CheckAction() if CheckAction in legal else CallAction()

        base = self._legalize(base, round_state, legal)

        # --- Redraw (optional) ---
        if RedrawAction in legal and not round_state.redraws_used[active] and street < 5:
            gc = float(game_state.game_clock)
            # Skip redraw search with premiums (saves time; redraw rarely helps).
            skip_redraw = (street == 0 and equity > 0.74) or (street >= 3 and equity > 0.90)
            if not skip_redraw and gc > 0.4:
                if gc > 25:
                    sm, inner_sims, min_g = 10, 120, 0.055
                elif gc > 8:
                    sm, inner_sims, min_g = 7, 80, 0.065
                else:
                    sm, inner_sims, min_g = 5, 60, 0.08

                opt = self.evaluate_redraw_options(
                    my_cards, board, street, rng, min_g, equity, sm, inner_sims
                )
                if opt is not None and opt[2] >= min_g:
                    inner = base.action if isinstance(base, RedrawAction) else base
                    return RedrawAction(opt[0], opt[1], inner)

        return base


if __name__ == "__main__":
    run_bot(Player(), parse_args())
