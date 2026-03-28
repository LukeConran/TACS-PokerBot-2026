"""
Opponent modeling across 300 hands.

Tracks VPIP, aggression factor, and fold frequency to adjust
bluff/value thresholds dynamically.
"""

from skeleton.states import BIG_BLIND


class OpponentModel:

    def __init__(self):
        self.hands_seen = 0

        # Preflop
        self.preflop_raises = 0      # raised above BB preflop
        self.preflop_calls = 0       # just called the BB or limped

        # Postflop
        self.postflop_bets = 0       # bet/raised postflop
        self.postflop_calls = 0      # called postflop
        self.postflop_checks = 0     # checked postflop (passive)

        # Folding
        self.folds_to_3bet = 0       # folded after I re-raised
        self.three_bet_chances = 0   # times I raised and they could re-raise or fold

        self.total_folds = 0

    # ── Per-hand tracking ──────────────────────────────────────────────

    def new_hand(self):
        self.hands_seen += 1

    def saw_preflop_raise(self):
        """Opponent raised preflop (pip > BIG_BLIND)."""
        self.preflop_raises += 1

    def saw_preflop_call(self):
        """Opponent just called preflop without raising."""
        self.preflop_calls += 1

    def saw_postflop_bet(self):
        self.postflop_bets += 1

    def saw_postflop_call(self):
        self.postflop_calls += 1

    def saw_postflop_check(self):
        self.postflop_checks += 1

    def saw_fold(self, after_my_raise=False):
        self.total_folds += 1
        if after_my_raise:
            self.folds_to_3bet += 1
            self.three_bet_chances += 1

    def saw_call_to_raise(self):
        self.three_bet_chances += 1

    # ── Derived stats ──────────────────────────────────────────────────

    @property
    def vpip(self):
        """Fraction of hands opponent voluntarily put money in preflop."""
        voluntary = self.preflop_raises + self.preflop_calls
        return voluntary / max(1, self.hands_seen)

    @property
    def preflop_raise_rate(self):
        return self.preflop_raises / max(1, self.hands_seen)

    @property
    def aggression_factor(self):
        """(bets + raises) / calls — >1.0 is aggressive, <0.7 is passive."""
        return self.postflop_bets / max(1, self.postflop_calls)

    @property
    def fold_to_3bet(self):
        return self.folds_to_3bet / max(1, self.three_bet_chances)

    @property
    def fold_frequency(self):
        return self.total_folds / max(1, self.hands_seen)

    def has_enough_data(self, n=25):
        return self.hands_seen >= n

    # ── Archetype classification ───────────────────────────────────────

    def is_calling_station(self):
        """Plays lots of hands, rarely raises — will call down with weak hands."""
        return self.has_enough_data() and self.vpip > 0.55 and self.aggression_factor < 0.8

    def is_tight_passive(self):
        """Folds a lot preflop, rarely bets when they play."""
        return self.has_enough_data() and self.vpip < 0.35 and self.aggression_factor < 0.7

    def is_aggressive(self):
        """Bets and raises frequently — apply pressure back."""
        return self.has_enough_data() and self.aggression_factor > 1.5

    def is_bluff_catcher(self):
        """Low fold-to-3bet — don't try to push them off hands."""
        return self.has_enough_data() and self.fold_to_3bet < 0.30

    # ── Strategy adjustments ───────────────────────────────────────────

    def call_equity_adj(self):
        """
        Adjust the minimum equity to call.
        Calling stations: thin down, call more (they pay off weak hands too).
        Tight folders: need stronger hand to commit chips.
        """
        if self.is_calling_station():
            return -0.04   # call more liberally, value bet thinner
        if self.is_tight_passive():
            return 0.03    # they only continue with strength
        return 0.0

    def raise_equity_adj(self):
        """
        Adjust the equity threshold to raise/semi-bluff.
        Don't bluff callers. Bluff more vs folders.
        """
        if self.is_calling_station() or self.is_bluff_catcher():
            return 0.10    # much higher bar to bluff
        if self.is_tight_passive() and self.fold_frequency > 0.45:
            return -0.06   # lower bar — they fold a lot
        return 0.0

    def bet_size_multiplier(self):
        """
        Scale bet sizes based on opponent type.
        Calling stations: bet bigger for value (they'll call).
        Tight folders: bet smaller to keep them in.
        """
        if self.is_calling_station():
            return 1.3
        if self.is_tight_passive():
            return 0.8
        return 1.0
