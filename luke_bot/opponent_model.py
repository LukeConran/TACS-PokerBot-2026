"""
Opponent modeling across 300 hands.

Tracks VPIP, aggression factor, and fold frequency to adjust
bluff/value thresholds dynamically.
"""


class OpponentModel:

    def __init__(self):
        self.hands_seen = 0

        # Preflop
        self.preflop_raises = 0
        self.preflop_calls  = 0

        # Postflop
        self.postflop_bets   = 0
        self.postflop_calls  = 0
        self.postflop_checks = 0

        # Folding
        self.folds_to_3bet    = 0
        self.three_bet_chances = 0
        self.total_folds       = 0

    # ── Per-hand tracking ──────────────────────────────────────────────

    def new_hand(self):
        self.hands_seen += 1

    def saw_preflop_raise(self):
        self.preflop_raises += 1

    def saw_preflop_call(self):
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
            self.folds_to_3bet    += 1
            self.three_bet_chances += 1

    def saw_call_to_raise(self):
        self.three_bet_chances += 1

    # ── Derived stats ──────────────────────────────────────────────────

    @property
    def vpip(self):
        return (self.preflop_raises + self.preflop_calls) / max(1, self.hands_seen)

    @property
    def aggression_factor(self):
        """postflop bets per call — >1.2 is aggressive, <0.7 is passive."""
        return self.postflop_bets / max(1, self.postflop_calls)

    @property
    def fold_to_3bet(self):
        return self.folds_to_3bet / max(1, self.three_bet_chances)

    @property
    def fold_frequency(self):
        return self.total_folds / max(1, self.hands_seen)

    def has_enough_data(self, n=15):
        """Lowered from 25 to 15 for earlier adaptation."""
        return self.hands_seen >= n

    # ── Archetype classification ───────────────────────────────────────

    def is_calling_station(self):
        return self.has_enough_data() and self.vpip > 0.55 and self.aggression_factor < 0.8

    def is_tight_passive(self):
        return self.has_enough_data() and self.vpip < 0.35 and self.aggression_factor < 0.7

    def is_aggressive(self):
        """Bets and raises frequently — fight back, don't fold to pressure."""
        return self.has_enough_data() and self.aggression_factor > 1.2

    def is_bluff_catcher(self):
        return self.has_enough_data() and self.three_bet_chances >= 5 and self.fold_to_3bet < 0.30

    # ── Strategy adjustments ───────────────────────────────────────────

    def call_equity_adj(self):
        """
        Shift minimum equity to call.
        vs calling station : call wider (thin value; they pay off)
        vs aggressive      : call wider (they bluff; don't fold to pressure)
        vs tight-passive   : tighten up (they only continue strong)
        """
        if self.is_calling_station():
            return -0.04
        if self.is_aggressive():
            return -0.03   # they bluff — don't overfold
        if self.is_tight_passive():
            return 0.03
        return 0.0

    def raise_equity_adj(self):
        """
        Shift equity threshold to raise/semi-bluff.
        vs calling station / bluff-catcher : don't bluff (they call)
        vs aggressive                      : lower bar — fight back, 3-bet light
        vs tight-passive + folder          : lower bar — they fold
        """
        if self.is_aggressive():
            return -0.05   # counter-aggression: re-raise more freely
        if self.is_calling_station() or self.is_bluff_catcher():
            return 0.10
        if self.is_tight_passive() and self.fold_frequency > 0.45:
            return -0.06
        return 0.0

    def bet_size_multiplier(self):
        """
        Scale bet sizes.
        vs calling station : bet bigger for value
        vs tight-passive   : smaller to keep them in
        """
        if self.is_calling_station():
            return 1.3
        if self.is_tight_passive():
            return 0.8
        return 1.0
