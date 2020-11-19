"""
Wrapper for BKZ simulation code. It allows simulation caching to disk,
and deals with the appropriate calls to experiments.py
"""


from sage.all import load, save
from experiments import singleThreadExperiment


try:
    already_simulated = load("already_simulated.sobj")
except:
    pass
try:
    _ = already_simulated is None
except NameError:
    already_simulated = {}


def get_simulation(n, sd, q, m, beta, tours, nu=1, cache=True, float_type="d",
                   precision=None, simulate_also_lll=False, simulator="CN11",
                   prng_seed=0xdeadbeef):
    """Returns a simulation of a BKZ-β reduced basis for an LWE [BG14b] embedding
    lattice, using FPYLLL's implementation of [CN11]. It will attempt to load an
    already cached simulation first. If `cache` is set to true, it will also cache
    any uncached returned simulations.

    :param n:                   LWE secret dimension
    :param sd:                  LWE {error, secret} standard dedviation
    :param q:                   LWE module
    :param m:                   number of LWE samples
    :param beta:                block size
    :param tours:               number of BKZ-β tours. Default: 20
    :param nu:                  Bai and Galbraith's embedding factor
    :param cache:               cache new returned simulations. Default: True
    :param float_type:          type of float, used so LLL terminates for crypto params
    :param precision:           mpfr precision required for crypto sized params
    :param simulate_also_lll:   simulate rather than run the first call to LLL on the basis
    :param simulator:           "CN11", "BSW18", or "averagedBSW18"
    :param prng_seed:           PRNG seed

    :returns simulation:        simulation output
    """

    assert(tours > 0)
    if not simulate_also_lll and ((n, sd, q, m, beta, tours) in already_simulated):
        # if actually running LLL, one needs to construct a basis and reduce it
        # this takes a while, so caching simulations to disk is an option
        simulation = already_simulated[(n, sd, q, m, beta, tours)]
    else:
        simulation = singleThreadExperiment(prng_seed, n, q, sd, m, beta,
                                            nu=nu, max_tours=tours, simulate=True, float_type=float_type,
                                            mpfr_precision=precision, simulate_also_lll=simulate_also_lll,
                                            simulator=simulator)
    if cache:
        already_simulated[(n, sd, q, m, beta, tours)] = simulation
        save(already_simulated, "already_simulated.sobj")
    return simulation
