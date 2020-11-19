"""
Utility for accumulating the probability of a sequence of events.
"""


class SuccessAccumulator:
    """ Utility for accumulating success probabilities for a sequence of events.
    It helps reducing code clutter when computing
    P[w] = P[w0] + P[f0 and w1] + P[f0 and f1 and w2] + ...
         = P[w0] + P[w1|f0] * P[f0] + P[w2|f0 and f1] P[f0 and f1] + ...

    One only needs to provide the values of P[wi|fi-1 and ... and f0].

    TESTS:
        >>> # tests initialisation
        >>> sa = SuccessAccumulator(target_probability=.6)
        >>> assert(sa.success_probability() == 0)
        >>> assert(sa.failure_probability() == 1)
        >>> # tests some accumulation
        >>> sa.accumulate(0.5)
        False
        >>> sa.accumulate(0.2)
        True
        >>> assert(sa.success_probability() == 0.6)
        >>> assert(sa.failure_probability() == 0.4)
        >>> sa.accumulate(1)
        True
        >>> assert(sa.success_probability() == 1)
        >>> assert(sa.failure_probability() == 0)
        >>> # test accumulating lots of events, should not raise ValueError
        >>> sa = SuccessAccumulator()
        >>> import random
        >>> for _ in range(100): _ = sa.accumulate(random.random())
    """

    def __init__(self, target_probability=1):
        self.p_win = 0
        self.target = target_probability

    def accumulate(self, p):
        """ Method for providing the values of P[wi|fi-1 and ... and f0].
            Accumulating p = 1 will cause the resulting success probability to be 1.
        """
        if p < 0 or p > 1:
            raise ValueError(
                f"Input additional probability outside of valid range {p}")
        p_fail = 1 - self.p_win
        self.p_win += p_fail * p
        if self.p_win < 0 or self.p_win > 1:
            raise ValueError(
                f"Success probability outside of valid range: {self.p_win}")
        return self.p_win >= self.target

    def success_probability(self):
        """ Getter for value of p_win.
        """
        return self.p_win

    def failure_probability(self):
        """ Getter for value of 1 - p_win.
        """
        return 1 - self.p_win
