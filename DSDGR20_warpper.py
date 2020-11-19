"""
Wrapper to [DSDGR20] uSVP simulator.
Using commit 948dbf0f89d57a712b442fec7fa0ecbe6736a57b
of https://github.com/lducas/leaky-LWE-Estimator with minor changes
from leaky-LWE-Estimator-utils.patch.
"""


from sage.all import *


try:
    load("./leaky-LWE-Estimator/framework/utils.sage")
except OSError:
    print("[DSDGR20] simulator code missing. Run `make dsdgr20` to fix.")
    exit(1)


def simulate_pbkz(params):
    """ Simulate the success probability of pBKZ.
    """

    n, q, sd, m, nu, _, beta_min, beta_max = params
    d = n+m+1

    assert (sd == 1)
    assert (nu == 1)
    assert (d > beta_max)
    assert (beta_min > 0)
    assert (d > 0)
    assert (beta_max >= beta_min)

    p_b_first_viable, _ = compute_beta_delta(d, log(q**m))

    CMF_b_first_viable = {}
    for b in p_b_first_viable:
        CMF_b_first_viable[b] = sum(p_b_first_viable[_]
                                    for _ in range(min(p_b_first_viable.keys()), b+1))

    return CMF_b_first_viable
