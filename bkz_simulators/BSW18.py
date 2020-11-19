# -*- coding: utf-8 -*-
"""
BKZ simulation algorithm as proposed in

- Bai, S., & Stehlé, D. & Wen, W. (2018). Measuring, simulating and exploiting
  the head concavity phenomenon in BKZ. In T. Peyrin, & S. Galbraith,
  ASIACRYPT~2018 : Springer, Heidelberg.

The code is based on the implementation of the simulation algorithm proposed by
Chen and Nguyen in "BKZ 2.0: Better Lattice Security Estimates" available at
https://github.com/fplll/fpylll/blob/master/src/fpylll/tools/bkz_simulator.py
by Michael Walter and Martin R. Albrecht.

.. moduleauthor:: Fernando Virdia <fernando.virdia.2016@rhul.ac.uk> (2020)

"""
from copy import copy
from math import log, sqrt, lgamma, pi
from collections import OrderedDict
import random

from fpylll.tools.quality import basis_quality
from fpylll.tools.bkz_stats import pretty_dict
from fpylll.fplll.bkz import BKZ
from fpylll.fplll.integer_matrix import IntegerMatrix
from fpylll.fplll.gso import MatGSO, GSO
from fpylll import FPLLL


rk = (
    0.789527997160000,
    0.780003183804613,
    0.750872218594458,
    0.706520454592593,
    0.696345241018901,
    0.660533841808400,
    0.626274718790505,
    0.581480717333169,
    0.553171463433503,
    0.520811087419712,
    0.487994338534253,
    0.459541470573431,
    0.414638319529319,
    0.392811729940846,
    0.339090376264829,
    0.306561491936042,
    0.276041187709516,
    0.236698863270441,
    0.196186341673080,
    0.161214212092249,
    0.110895134828114,
    0.0678261623920553,
    0.0272807162335610,
    -0.0234609979600137,
    -0.0320527224746912,
    -0.0940331032784437,
    -0.129109087817554,
    -0.176965384290173,
    -0.209405754915959,
    -0.265867993276493,
    -0.299031324494802,
    -0.349338597048432,
    -0.380428160303508,
    -0.427399405474537,
    -0.474944677694975,
    -0.530140672818150,
    -0.561625221138784,
    -0.612008793872032,
    -0.669011014635905,
    -0.713766731570930,
    -0.754041787011810,
    -0.808609696192079,
    -0.859933249032210,
    -0.884479963601658,
    -0.886666930030433,
)


def simulate(r, param, prng_seed=0xdeadbeef):
    """
    BKZ simulation algorithm as proposed by Bai and Stehlé and Wen in "Measuring, simulating and
    exploiting the head concavity phenomenon in BKZ".  Returns the reduced squared norms of the
    GSO vectors of the basis and the number of BKZ tours simulated.  This version terminates when
    no substantial progress is made anymore or at most ``max_loops`` tours were simulated.
    If no ``max_loops`` is given, at most ``d`` tours are performed, where ``d`` is the dimension
    of the lattice.

    :param r: squared norms of the GSO vectors of the basis.
    :param param: BKZ parameters

    EXAMPLE:

        >>> from fpylll import IntegerMatrix, GSO, LLL, FPLLL, BKZ
        >>> FPLLL.set_random_seed(1337)
        >>> A = LLL.reduction(IntegerMatrix.random(100, "qary", bits=30, k=50))
        >>> M = GSO.Mat(A)

        >>> from BSW18 import simulate
        >>> _ = simulate(M, BKZ.Param(block_size=40, max_loops=4, flags=BKZ.VERBOSE))
        {"i":        0,  "r_0":   2^33.5,  "r_0/gh": 6.623076,  "rhf": 1.018751,  "/": -0.07015,  "hv/hv": 2.426074}
        {"i":        1,  "r_0":   2^32.7,  "r_0/gh": 3.905694,  "rhf": 1.016064,  "/": -0.06171,  "hv/hv": 2.162941}
        {"i":        2,  "r_0":   2^32.4,  "r_0/gh": 3.222033,  "rhf": 1.015087,  "/": -0.05777,  "hv/hv": 2.055214}
        {"i":        3,  "r_0":   2^32.1,  "r_0/gh": 2.536317,  "rhf": 1.013873,  "/": -0.05587,  "hv/hv": 2.005962}


    """

    if not prng_seed:
        prng_seed = FPLLL.randint(0, 2**32-1)

    random.seed(prng_seed)

    if isinstance(r, IntegerMatrix):
        r = GSO.Mat(r)
    if isinstance(r, MatGSO):
        r.update_gso()
        r = r.r()

    n = len(r)

    # code uses log2 of norms, FPLLL uses squared norms
    r = list(map(lambda x: log(x, 2) / 2.0, r))

    l = copy(r)
    l̂ = copy(r)
    c = [rk[-j] - sum(rk[-j:]) / j for j in range(1, 46)]
    c += [
        (lgamma(d / 2.0 + 1) * (1.0 / d) - log(sqrt(pi))) / log(2.0)
        for d in range(46, param.block_size + 1)
    ]

    if param.max_loops:
        N = param.max_loops
    else:
        N = n

    t0 = [True for _ in range(n)]
    for j in range(N):
        t1 = [False for _ in range(n)]
        for k in range(n - min(45, param.block_size)):
            d = min(param.block_size, n - k - 1)
            e = k + d + 1
            tau = False
            for kp in range(k, e):
                tau |= t0[kp]
            logV = sum(l[:e-1]) - sum(l̂[:k])
            if tau:
                X = random.expovariate(.5)
                g = (log(X, 2) + logV) / d + c[d - 1]
                if g < l[k]:
                    l̂[k] = g
                    l̂[k+1] = l[k] + log(sqrt(1-1./d), 2)
                    γ = (l[k] + l[k+1]) - (l̂[k] + l̂[k+1])
                    for kp in range(k+2, e):
                        l̂[kp] = l[kp] + γ/(d-2.)
                        t1[kp] = True
                    tau = False
            for idx in range(k, e-1):
                l[idx] = l̂[idx]


        # early termination
        if True not in t1 or l == l̂:
            break
        else:
            d = min(45, param.block_size)
            logV = sum(l) - sum(l̂[:-d])

            if param.block_size < 45:
                tmp = sum(rk[-param.block_size :]) / param.block_size
                rk1 = [r_ - tmp for r_ in rk[-param.block_size :]]
            else:
                rk1 = rk

            for k, r in zip(range(n - d, n), rk1):
                l̂[k] = logV / d + r
                t1[kp] = True
            l = copy(l̂)
            t0 = copy(t1)

        if param.flags & BKZ.VERBOSE:
            r = OrderedDict()
            r["i"] = j
            for k, v in basis_quality(list(map(lambda x: 2.0 ** (2 * x), l))).items():
                r[k] = v
            print(pretty_dict(r))

    l = list(map(lambda x: 2.0 ** (2 * x), l))
    return l, j + 1



def averaged_simulate(L, params_fplll, tries=10):
    print("WARNING: you may want to run multiple simulations and average results instead.")

    if tries < 1:
        raise ValueError("Need to average over positive number of tries.")

    from sage.all import vector, RR

    for _ in range(tries):
        x, y = simulate(L, params_fplll, prng_seed=_+1)
        if _ == 0:
            i = vector(RR, x)
            j = y
        else:
            i += vector(RR, x)
            j += y

    res = (i/tries, j/tries)
    return res

