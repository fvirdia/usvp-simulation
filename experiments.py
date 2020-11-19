# -*- coding: utf-8 -*-
"""
Subset of the code necessary to reproduce and analyse the results from [AGVW17],
ported to Python 3.7, Sagemath 9.0, fpylll 0.5.1dev.

Extended to investigate distribution of successful block sizes around the 2016-estimate'd one.
The experiment instances can be defined using a list of parameter sets. An example is given in
the variable `parameter_sets`, below.

AUTHOR:

    Fernando Virdia - 2017, 2020

REFERENCES:

    [AGVW17] Albrecht, Martin R., Florian Göpfert, Fernando Virdia,
    and Thomas Wunderer. "Revisiting the Expected Cost of Solving uSVP and
    Applications to LWE." ASIACRYPT, 2017. https://ia.cr/2017/815
"""


import random
import time
import sys
import json
from sage.all import matrix, next_prime, sqrt, pi, parallel, set_random_seed, \
    variance, ZZ, identity_matrix, prod, RR, vector, save, floor, round, ceil, \
    sample, GF, QQ
from sage.stats.basic_stats import mean, std
from sage.crypto.lwe import LWE, DiscreteGaussianDistributionIntegerSampler
import fpylll
from fpylll.fplll.bkz_param import BKZParam
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.util import ReductionError
try:
    from g6k.siever_params import SieverParams
    from g6k.algorithms.bkz import naive_bkz_tour
    from g6k.siever import Siever
    from g6k.utils.stats import dummy_tracer, SieveTreeTracer
except ImportError:
    pass


try:
    from config import NCPUS
except ImportError:
    NCPUS = 2

LABEL = "tmp_out"


try:
    from params import parameter_sets
except ImportError:
    # Example params
    parameter_sets = [
        # init seed, instaces,        n,                 q,             sd,   m, beta, float_type, mpfr_precision, max_tours, embedding, nu, secret_dist, label
        (1,        2,       65,  next_prime(2**9), 8.0/sqrt(2*pi), 182, 16,
         "d",           None,        20,  "kannan",  1,     "noise", "asd"),
    ]


def instances(
    params_list,
    left_wing_percent=0.,
    right_wing_percent=0.,
    progressive=False,
    progressive_skip=1
):
    """
    :param params_list:         List of tuples encoding an experiment.
    :param left_wing_percent:   Default: 0.
    :param right_wing_percent:  Default: 0.
    :param progressive:         Sets whether to use BKZ or progressive BKZ. Default: False
    """
    for params in params_list:
        init_seed, trials, n, q, sd, m, base_bs, float_type, mpfr_precision, max_tours, embedding, nu, secret_dist, label = params
        if progressive:
            seed = init_seed
            while seed < init_seed + trials:
                yield ((seed, n, q, sd, m, base_bs, float_type, mpfr_precision, max_tours, embedding, nu, secret_dist, label, progressive, progressive_skip),)
                seed += 1
        else:
            left_bs = int(round(base_bs * (1. - left_wing_percent/100)))
            right_bs = int(round(base_bs * (1. + right_wing_percent/100)))
            bs = left_bs
            while bs <= right_bs:
                seed = init_seed
                while seed < init_seed + trials:
                    yield ((seed, n, q, sd, m, bs, float_type, mpfr_precision, max_tours, embedding, nu, secret_dist, label, progressive, progressive_skip),)
                    seed += 1
                bs += 1


class TernaryLWE:
    """Class simulating the interface of sage.crypto.lwe.LWE but generating instances
    with ternary secret and error.
    """

    _LWE__s = None
    _LWE__e = None

    def __init__(self, n, q, secret_dist="ignored", D="ignored"):
        from sage.all import vector, randint
        self.n = n
        self.q = q
        self._LWE__s = self.s = vector([randint(-1, 1) for _ in range(self.n)])
        self._LWE__e = []

    def __call__(self):
        from sage.all import vector, randint
        a = vector([randint(0, self.q-1) for _ in range(self.n)])
        ei = randint(-1, 1)
        self._LWE__e.append(ei)
        c = (a * self.s + ei) % self.q
        return a, c


class AnySecretLWE:
    """Class simulating the interface of sage.crypto.lwe.LWE but generating instances
    with any secret and error error.
    """

    _LWE__s = None
    _LWE__e = None

    def __init__(self, n, q, secret_dist_gen, error_dist_gen):
        from sage.all import vector, randint
        self.n = n
        self.q = q
        self.error_dist_gen = error_dist_gen
        self._LWE__s = self.s = vector(
            [secret_dist_gen() for _ in range(self.n)])
        self._LWE__e = []

    def __call__(self):
        from sage.all import vector, randint
        a = vector([randint(0, self.q-1) for _ in range(self.n)])
        ei = self.error_dist_gen()
        self._LWE__e.append(ei)
        c = (a * self.s + ei) % self.q
        return a, c


def genLWEInstance(n,
                   q,
                   sd,
                   m,
                   float_type="d",
                   mpfr_precision=None,
                   embedding="baigal",
                   secret_dist="noise",
                   nu=1,
                   lib='fplll',
                   moot=False,
                   use_suboptimal_embedding=False):
    """Generate lattices from LWE instances using Kannan's embedding.

    :param n:                   secret dimension
    :param q:                   lwe modulo
    :param sd:                  standard deviation
    :param m:                   number of lwe samples ('baigal' only uses m-n)
    :param float_type:          floating point type
    :param mpfr_precision:      floating point precision (if using mpfr)
    :param embedding:           "kannan" or "baigal"
    :param nu:                  scaling factor for "baigal" embedding
    :param moot:                if true, c is uniform rather than As+e
    :param use_suboptimal_embedding: if False, put the q vectors on top of basis

    :returns:                   the lwe generator, the samples, the lattice and
                                its volume
    """

    # generate LWE instance
    if isinstance(secret_dist, tuple) and secret_dist[0] == 'custom':
        domain = secret_dist[1]
        # print("DEBUG: sampling uniform from %s" % domain)
        lwe = AnySecretLWE(n=n, q=q,
                           secret_dist_gen=lambda: sample(domain, 1)[0],
                           error_dist_gen=lambda: sample(domain, 1)[0])
        #  error_dist_gen=DiscreteGaussianDistributionIntegerSampler(sd))
    else:
        lwe = LWE(n=n, q=q, secret_dist=secret_dist,
                  D=DiscreteGaussianDistributionIntegerSampler(sd))

    # get m different LWE samples
    samples = [lwe() for i in range(m)]

    A = matrix(a for a, _ in samples)
    if moot:
        from sage.all import randint
        C = matrix(randint(0, q-1) for _ in range(len(samples)))
    else:
        C = matrix(c for _, c in samples)

    # print(lwe._LWE__s)
    # print(lwe._LWE__e)
    # print(vector(GF(q), C[0]) == vector(GF(q), A * lwe._LWE__s + vector(lwe._LWE__e)))

    if embedding == "kannan":
        # generate kannan's embedding lattice
        AT = A.T.echelon_form()
        qs = matrix(ZZ, m-n, n).augment(q*identity_matrix(m-n))
        if use_suboptimal_embedding:
            B = AT.change_ring(ZZ).stack(qs)
        else:
            B = qs.stack(AT.change_ring(ZZ))
        # embed the ciphertext to the lattice, so that error vector
        # becomes the (most likely unique) SVP in the lattice
        BC = B.stack(matrix(C).change_ring(ZZ))
        BC = BC.augment(matrix(m+1, 1))
        BC[-1, -1] = max(floor(sd), 1)
    elif embedding == "baigal":
        # generate scaled Bai-and-Galbraith's embedding lattice
        assert(nu > 0)
        nu_rat = QQ(round(nu*100)/100)
        nu_num, nu_denom = nu_rat.numerator(), nu_rat.denominator()
        if nu_denom != 1:
            print(
                f"WARNING: due to fractional ν, output lengths are scaled by {nu_denom}")
        AT = (nu_num*identity_matrix(n)).augment(nu_denom*A.change_ring(ZZ).T)
        qs = matrix(ZZ, m, n).augment(nu_denom*q*identity_matrix(m))
        if use_suboptimal_embedding:
            B = AT.change_ring(ZZ).stack(qs)
        else:
            B = qs.stack(AT.change_ring(ZZ))
        # embed the ciphertext to the lattice, so that error vector
        # becomes the (most likely unique) SVP in the lattice
        BC = B.stack(matrix(1, n).augment(nu_denom*matrix(C).change_ring(ZZ)))
        BC = BC.augment(matrix(m+n+1, 1))
        BC[-1, -1] = nu_denom*max(floor(sd), 1)
    else:
        raise ValueError("embedding can only be 'kannan' or 'baigal'")

    # preprocess basis for low precision, else after GSO has float_type
    BC = fpylll.IntegerMatrix.from_matrix(BC)
    if float_type == "d":
        fpylll.LLL.reduction(BC)

    # set floating point precision
    if float_type == "mpfr":
        _ = fpylll.FPLLL.set_precision(mpfr_precision)

    if lib == 'fplll':
        BC_GSO = fpylll.GSO.Mat(BC, float_type=float_type)
    elif lib == 'g6k':
        BC = fpylll.IntegerMatrix.from_matrix(BC, int_type="long")
        BC_GSO = fpylll.GSO.Mat(BC, float_type=float_type,
                                U=fpylll.IntegerMatrix.identity(
                                    BC.nrows, int_type=BC.int_type),
                                UinvT=fpylll.IntegerMatrix.identity(BC.nrows, int_type=BC.int_type))
    else:
        raise NotImplementedError("'%s' unknown, use 'fplll' or 'g6k'" % lib)

    BC_GSO.update_gso()

    if float_type != "d":
        lll = fpylll.LLL.Reduction(BC_GSO)
        lll()

    # get lattice volume
    vol = sqrt(prod([RR(BC_GSO.get_r(i, i)) for i in range(m+1)]))

    return (lwe, samples, A, C, BC_GSO, vol)


def runBKZ(L, b, max_tours, lib='fplll', simulate=False, simulator="CN11", sim_prng_seed=0xdeadbeef):
    """Set up and run Algorithm 2 from the paper, recording detailed statistics.

    :param L:           lattice basis
    :param b:           BKZ block size
    :param max_tours:   max number of BKZ tours

    :returns:           the BKZ object and the tracer containing statistics
    """

    # set up BKZ
    params_fplll = BKZParam(block_size=b,
                            strategies=fpylll.load_strategies_json(
                                b"strategies-2020-06-03-master-8e44f91.json"),  # fpylll.BKZ.DEFAULT_STRATEGY
                            flags=0
                            # | fpylll.BKZ.VERBOSE
                            | fpylll.BKZ.AUTO_ABORT
                            | fpylll.BKZ.GH_BND
                            | fpylll.BKZ.MAX_LOOPS,
                            max_loops=max_tours)
    if simulate:
        if simulator == "CN11":
            from fpylll.tools.bkz_simulator import simulate
        elif simulator == "BSW18":
            from bkz_simulators.BSW18 import simulate as derand_simulate
            def simulate(prof, pars): return derand_simulate(
                prof, pars, prng_seed=sim_prng_seed)
        elif simulator == "averagedBSW18":
            from bkz_simulators.BSW18 import averaged_simulate as simulate
        else:
            raise ValueError(f"Simulator {simulator} not available.")
        # print(f"runBKZ; Simulator: {simulator}")

        bkz = simulate(L, params_fplll)
        return bkz
    elif lib == 'fplll':
        bkz = BKZReduction(L)
        bkz(params_fplll)
        return bkz
    elif lib == 'g6k':
        params_g6k = SieverParams()
        g6k = Siever(L, params=params_g6k)
        # tracer = dummy_tracer
        tracer = SieveTreeTracer(g6k, root_label=("bkz", L.B.nrows),
                                 start_clocks=True)
        for t in range(max_tours):
            with tracer.context("tour", t, dump_gso=True):
                extra_dim4free = 0
                dim4free_fun = 'default_dim4free_fun'
                workout_params = {}
                pump_params = {'down_sieve': True}
                naive_bkz_tour(g6k, tracer, b,
                               extra_dim4free=extra_dim4free,
                               dim4free_fun=dim4free_fun,
                               workout_params=workout_params,
                               pump_params=pump_params)
        tracer.exit()
        return tracer
    else:
        raise NotImplementedError("'%s' unknown, use 'fplll' or 'g6k'" % lib)


def singleThreadExperiment(
    seed,
    n,
    q,
    sd,
    m,
    b,
    float_type="d",
    mpfr_precision=None,
    max_tours=20,
    embedding="baigal",
    nu=1,
    secret_dist="noise",
    label=None,
    progressive=False,
    progressive_skip=1,
    verbose=False,
    profile=True,
    simulate=False,
    lib='fplll',
    simulate_also_lll=False,
    simulator="CN11"
):
    """Generates an LWE instance, calls BKZ, tides up statistics and returns them.

    :param n:               secret dimension
    :param q:               lwe modulo
    :param sd:              standard deviation
    :param m:               number of lwe samples
    :param b:               bkz block size
    :param max_tours:       maximum number of BKZ tours to allow
    :param float_type:      floating point type
    :param mpfr_precision:  floating point precision (if using mpfr)
    :param seed:            seed for all the corresponding PNRGs
    :param verbose:         if True, print some extra information

    :returns:
    """

    try:
        # prepare seed and prngs
        if not seed:
            seed = ceil(10000000000*random.random())

        seed = int(seed)  # serialization safe
        set_random_seed(seed)  # sets sage prng
        fpylll.FPLLL.set_random_seed(seed)  # sets underlying fplll prng
        random.seed(seed)  # used by fpylll pruning
        if verbose:
            print("Seed", seed)

        # basis/profile generation
        if simulate and simulate_also_lll:
            # simulation assumes q-vectors at the beginning of the basis
            assert(nu == 1)
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
            from lll_sim import lll_simulator
            BC = lll_simulator(_dim, _k, q, scale=scale)

        if not simulate or not simulate_also_lll:
            sample_variance = 0
            sampl_var_error = 2/100
            while abs(sample_variance-sd**2) > sampl_var_error * sd**2:
                # generate lwe instance
                (oracle, samples, A, C, BC, vol) = genLWEInstance(n,
                                                                  q,
                                                                  sd,
                                                                  m,
                                                                  lib=lib,
                                                                  float_type=float_type,
                                                                  mpfr_precision=mpfr_precision,
                                                                  embedding=embedding,
                                                                  nu=nu,
                                                                  secret_dist=secret_dist)

                # calculate the short embedded vector
                evc = (C[0] - A*oracle._LWE__s) % q
                if embedding == "kannan":
                    sol = vector(evc.list()+[max(floor(sd), 1)])
                    target = list(map(lambda x: ZZ(x) if ZZ(x)
                                      < q//2 else ZZ(x)-q, sol))

                    def success_check(M):
                        return sol in (M[0], -M[0])
                elif embedding == "baigal":
                    sol = nu * oracle._LWE__s
                    target = list(map(lambda x: ZZ(x) if ZZ(x)
                                      < q//2 else ZZ(x)-q, sol))
                    target += list(map(lambda x: ZZ(x) if ZZ(x) <
                                       q//2 else ZZ(x)-q, evc))+[max(floor(sd), 1)]

                    def success_check(M):
                        return sol in (M[0][:n], (-M[0])[:n])
                else:
                    raise ValueError(
                        "'%s' not a valid embedding. Available: 'kannan', 'baigal'" % embedding)
                sol = vector(map(lambda x: ZZ(x) if ZZ(x)
                                 < q//2 else ZZ(x)-q, sol))

                # compute sample variance to see if need to resample
                # sample_variance = float(variance(target))
                sample_variance = sd**2

            if verbose:
                print("sol: %s" % sol)

            # run BKZ
            dim = BC.d
            if verbose:
                print("Blocksize %d Samples %d Dimension %d" % (b, m, BC.d))

        recorded_tours = {}
        if progressive:
            for pb in range(3, min(dim+1, 71), progressive_skip):
                try:
                    # if G6K, only gives tracer for last tour
                    tracer = runBKZ(BC, pb, max_tours, lib=lib)
                except Exception as e:
                    if str(e) in ["b'infinite loop in babai'", "math domain error"]:
                        # try again with higher precision
                        if float_type == "d":
                            np = "dd"
                        elif float_type == "dd":
                            np = "qd"
                        else:
                            raise e
                        if verbose:
                            print("Increasing precision to %s" % np)
                        return singleThreadExperiment(seed, n, q, sd, m, b,
                                                      float_type=np, mpfr_precision=mpfr_precision,
                                                      max_tours=max_tours, embedding=embedding,
                                                      nu=nu, secret_dist=secret_dist, label=label, verbose=verbose,
                                                      lib=lib, progressive=progressive, progressive_skip=progressive_skip)
                    else:
                        raise e
                recorded_tours[pb] = int(len(tracer.trace.children)-1)

                # add some metadata to the statistics
                M = matrix(nrows=BC.B.nrows, ncols=BC.B.ncols)
                BC.B.to_matrix(M)

                if success_check(M):
                    break
        else:
            try:
                if simulate:
                    tracer = runBKZ(BC, b, max_tours, simulate=simulate,
                                    simulator=simulator, sim_prng_seed=seed)
                else:
                    tracer = runBKZ(BC, b, max_tours, lib=lib)
            except Exception as e:
                if str(e) in ["b'infinite loop in babai'", "math domain error"]:
                    # try again with higher precision
                    if float_type == "d":
                        np = "dd"
                    elif float_type == "dd":
                        np = "qd"
                    else:
                        raise e
                    if verbose:
                        print("Increasing precision to %s" % np)
                    return singleThreadExperiment(seed, n, q, sd, m, b,
                                                  float_type=np, mpfr_precision=mpfr_precision,
                                                  max_tours=max_tours, embedding=embedding,
                                                  nu=nu, secret_dist=secret_dist, label=label, verbose=verbose,
                                                  )
                else:
                    raise e
            if simulate:
                recorded_tours[b] = int(max_tours)
            else:
                recorded_tours[b] = int(len(tracer.trace.children)-1)

            # add some metadata to the statistics
            if not simulate_also_lll:
                M = matrix(nrows=BC.B.nrows, ncols=BC.B.ncols)
                BC.B.to_matrix(M)

        if verbose:
            print("Five shortest basis vectors:")
            for _ in range(5):
                print(M[_])

        stats = {
            "seed": int(seed),
            "success": False if simulate_also_lll else success_check(M),
            "bs": int(b) if not progressive else int(pb),
            "tours": recorded_tours,
        }

        if not simulate:
            # g6k tracer contains different things
            stats["trace"] = {'cputime': float(tracer.trace['cputime'])}

        if profile:
            if simulate:
                stats['profile'] = list(map(float, tracer[0]))
            else:
                BC.update_gso()
                stats['profile'] = [float(BC.get_r(_, _))
                                    for _ in range(M.dimensions()[0])]

        # show some stats
        if verbose:
            print("Success", stats["success"])
            print(stats)

        return stats

    except Exception as e:
        print("exception")
        ret = str(e)
        return ret


@parallel(ncpus=NCPUS)
def parallelExperiment(batch):
    """Run in parallel a batch of single thread experiments
    :param batch:   an entry from the experiment_parameters dictionary

    :returns:       return values from the batch
    """
    seed, n, q, sd, m, bs, float_type, mpfr_precision, max_tours, embedding, nu, secret_dist, label, progressive, progressive_skip = batch

    print(seed, n, q, sd, m, bs, float_type, mpfr_precision,
          max_tours, embedding, nu, secret_dist, label)
    res = singleThreadExperiment(
        seed,
        n,
        q,
        sd,
        m,
        bs,
        float_type=float_type,
        mpfr_precision=mpfr_precision,
        max_tours=max_tours,
        embedding=embedding,
        nu=nu,
        secret_dist=secret_dist,
        progressive=progressive,
        progressive_skip=progressive_skip
    )

    try:
        results = {}
        name = label if label else "n%d" % n
        bs_s = str(res['bs'])
        if name not in results:
            results[name] = {}
        if bs_s not in results[name]:
            results[name][bs_s] = {
                "tries": 0,
                "successes": 0,
                "timings": []
            }
        results[name][bs_s]["tries"] += int(1)
        results[name][bs_s]["successes"] += int(res['success'] == True)
        results[name][bs_s]["timings"].append(float(res['trace']['cputime']))
        # print(results)
    except:
        pass

    # dump emergency copy
    try:
        fn = LABEL
        with open(fn, "a") as f:
            f.write(str(batch))
            f.write("\n")
            # print(res)
            f.write(json.dumps(res))
            f.write("\n")
    except:
        pass

    return res


def statisticsExtrapolation(ret, label):
    # collect results
    results = {}
    n_of_tours = {}
    for inp, res in ret:
        seed, n, q, sd, m, bs, float_type, mpfr_precision, max_tours, embedding, nu, secret_dist, tag, progressive, progressive_skip = inp[
            0][0]
        name = tag if tag else "n%d" % n

        # find average n of tours per block size in batch of experiments
        if name not in n_of_tours:
            n_of_tours[name] = {}

        for tour_bs in res["tours"]:
            if tour_bs not in n_of_tours[name]:
                n_of_tours[name][tour_bs] = [0, 0]
            n_of_tours[name][tour_bs][0] += res["tours"][tour_bs]
            n_of_tours[name][tour_bs][1] += 1

        # extract other statistics
        try:
            bs_s = str(res['bs'])
            if name not in results:
                results[name] = {}
            if bs_s not in results[name]:
                results[name][bs_s] = {
                    "tries": 0,
                    "successes": 0,
                    "timings": []
                }
                if "profile" in res:
                    results[name][bs_s]["profile"] = [0] * len(res["profile"])
            results[name][bs_s]["tries"] += 1
            results[name][bs_s]["successes"] += int(res['success'] == True)
            results[name][bs_s]["timings"].append(RR(res['trace']['cputime']))
            for _ in range(len(res["profile"])):
                results[name][bs_s]["profile"][_] += float(res["profile"][_])
        except TypeError as e:
            print("got exception")
            raise(e)
            # probably some experiment failed and res is an error string
            print(inp, res)

    # generate statistics
    for name in results:
        for bs in results[name]:
            results[name][bs]['succ_ratio'] = results[name][bs]['successes'] / \
                results[name][bs]['tries']
            results[name][bs]['avg_time'] = float(
                mean(results[name][bs]['timings']))
            results[name][bs]['std_time'] = min(
                0, float(std(results[name][bs]['timings'])))
            results[name][bs]['timings'] = list(
                map(float, results[name][bs]['timings']))
            for _ in range(len(results[name][bs]["profile"])):
                results[name][bs]["profile"][_] /= float(
                    results[name][bs]['tries'])

        for tour_bs in n_of_tours[name]:
            n_of_tours[name][tour_bs][0] = float(
                n_of_tours[name][tour_bs][0]/n_of_tours[name][tour_bs][1])

    print(results)
    print(n_of_tours)
    save(results, "results_%s.sobj" % (label))
    with open("results_%s.json" % (label), 'w') as f:
        f.write(json.dumps(results))

    save(n_of_tours, "tours_%s.sobj" % (label))
    with open("tours_%s.json" % (label), 'w') as f:
        f.write(json.dumps(n_of_tours))


def loadStatisticsFromEmergencyDump(fn):
    with open(fn) as f:
        lines = f.readlines()

    ret = []
    for row in range(0, len(lines), 2):
        # reconstruct the input instance
        tup = lines[row][1:-2].split(', ')
        seed = int(tup[0])
        n = int(tup[1])
        q = int(tup[2])
        sd = float(tup[3])
        m = int(tup[4])
        bs = int(tup[5])
        float_type = tup[6].split("'")[1]
        mpfr_precision = None if tup[7] == 'None' else tup[7]
        max_tours = int(tup[8])
        embedding = tup[9].split("'")[1]
        nu = float(tup[10])
        secret_dist = tup[11].split("'")[1]
        tag = tup[-3].split("'")[1]
        progressive = tup[-2] == 'True'
        progressive_skip = int(tup[-1])
        inst = (seed, n, q, sd, m, bs, float_type, mpfr_precision, max_tours,
                embedding, nu, secret_dist, tag, progressive, progressive_skip)

        # reconstruct experimental result
        result = json.loads(lines[row+1])

        ret.append((((inst,),), result))

    return ret


def win_beta_distribution(left_wing_percent,
                          right_wing_percent,
                          label="",
                          parallel=True,
                          progressive=False,
                          progressive_skip=1):
    """Function that sets up the instances to run parametrised
    by their block size. If the function is called as

        win_beta_distribution(a, b)

    on a set of parameters with block size beta (see `parameter_sets` example at
    the top of the file), then experiments will be run with block sizes in

        [(1-a/100) * beta, (1+b/100) * beta].

    :params left_wing_percent:    `a` in the example above
    :params right_wing_percent:   `b` in the example above
    """
    global LABEL

    # run instances
    instance_generator = instances(
        parameter_sets,
        left_wing_percent=left_wing_percent,
        right_wing_percent=right_wing_percent,
        progressive=progressive,
        progressive_skip=progressive_skip
    )
    if parallel:
        LABEL = f"tmp_out_{label}"
        ret = list(parallelExperiment(instance_generator))
    else:
        ret = []
        for inst in instance_generator:
            print(inst)
            ret.append(((inst,), singleThreadExperiment(*inst[0])))

    statisticsExtrapolation(ret, label)


"""
if __name__ == "__main__":
    import warnings

    # ignore a DeprecationWarning for @parallel in Sagemath 9.0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Progressive BKZ
        win_beta_distribution(1, 1, label="test-pbkz-experiments", parallel=True, progressive=True, progressive_skip=1)

        # BKZ
        # win_beta_distribution(25, 8, label="test-bkz-experiments", parallel=True, progressive=False)
"""
