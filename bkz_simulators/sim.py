import bkz_simulators.CN11 as CN11# this is identical to fpylll.tools.bkz_simulator
import bkz_simulators.BSW18 as BSW18
from fpylll import BKZ
from sage.all import line, log, e
from lll_sim import lll_simulator


def LLLProfile(n, q, m, nu=1, embedding="baigal", use_gsa=False):
    """
    Returns simulated LLL profile with Z-shape due to q-vectors being at the
    beginning of the input basis.

    :returns:   list of squared norms
    """
    if embedding == "kannan":
        _dim = m+1
        _k = m-n
        scale = 1
        if nu != 1:
            raise ValueError("ν != 1 makes sense only using baigal.")
    elif embedding == "baigal":
        _dim = n+m+1
        _k = m
        scale = nu
        assert(nu > 0)

    if use_gsa:
        log_delta = float(log(1.021900))
        log_vol = float(_k * log(q) + n * log(scale))
        log_alpha = 2 * _dim * log_delta/(1-_dim)
        # log_alpha = -2 * float(log(delta))
        log_bi = lambda i: (i-1) * log_alpha + log_vol/_dim + _dim * log_delta # i from 1 to _dim
        # vvol = sum([log_bi(i+1) for i in range(_dim)])
        # print("original logvol", log_vol)
        # print("recomputed lvol", vvol)
        return [e**(2*log_bi(i+1)) for i in range(_dim)]

    return lll_simulator(_dim, _k, q, scale=scale)


class Sim:
    """
    Class to simulate BKZ reduction and variants on random q-ary lattices.

    TESTS:
        >>> import bkz_simulators.CN11 as CN11
        >>> from fpylll import BKZ
        >>> from bkz_simulators.sim import Sim, LLLProfile
        >>> n, q, m = 50, 2**10, 50
        >>> beta, tours = 40, 16
        >>> lll_prof = LLLProfile(n, q, m)
        >>> r1, l1 = CN11.simulate(lll_prof, BKZ.Param(block_size=beta, max_loops=tours))
        >>> sim = Sim(lll_prof)
        >>> sim(beta, 8)
        >>> len(sim.tours)
        1
        >>> sim.tours[0] == (beta, 8)
        True
        >>> sim(beta, tours-8)
        >>> len(sim.tours)
        2
        >>> sim.tours[1] == (beta, tours-8)
        True
        >>> sim.profile == r1
        True
        >>> sim(40, 2)
        >>> len(sim.tours)
        3
        >>> sim.tours[2] == (40, 2)
        True
    """
    def __init__(self, initial_profile):
        self.profile = initial_profile[::]
        self.tours = []

    def __call__(self, beta, tours, sim="CN11", prng_seed=0xdeadbeef):
        if sim == "CN11":
            simulate = CN11.simulate
        elif sim == "BSW18":
            simulate = lambda prof, pars: BSW18.simulate(prof, pars, prng_seed=prng_seed)
        pars = BKZ.Param(block_size=beta, max_loops=tours)
        l, r = simulate(self.profile, pars)
        self.tours.append((beta, tours))
        self.profile = l

    def plot(self, base=2):
        g = line(
                [(i, log(self.profile[i], base)/2) for i in range(len(self.profile))],
                axes_labels = ['$i$','$\\log_{%s}\\|b_{i}^*\\|$' % ("" if base == e else base)]
            )
        return g

