"""
Implementation of the uSVP simulators from 
  On the Success Probability of Solving Unique SVP via BKZ
  Eamonn W. Postlethwaite, Fernando Virdia, https://eprint.iacr.org/2020/1308
"""


from sage.all import sqrt
from bkz_simulators.sim import Sim, LLLProfile
from success_accumulator import SuccessAccumulator
from stddev_in_practice import get_concrete_var_distribution
from probabilities import pdf_chi_sq, cmf_chi_sq
from bkz_simulations_wrapper import get_simulation


def real_sd_simulate(usvp_simulator,
                     params,
                     tours=20,
                     tour_distr=None,
                     skip=1,
                     simulate_also_lll=True,
                     simulator="CN11",
                     target_probability=.999,
                     prng_seed=0xdeadbeef,
                     verbose=True,
                     account_for_sample_variance=False,
                     embedding="baigal",
                     suboptimal_embedding=False):

    n, q, sd, m, nu, secret_dist, beta_min, beta_max = params
    if embedding == "baigal":
        d = n + m + 1
    elif embedding == "kannan":
        d = m + 1

    if account_for_sample_variance:
        secret_var_distribution = get_concrete_var_distribution(
            sd, d, experimental=False)
    else:
        secret_var_distribution = {sd**2: 1.}

    p_beta_winner = {}
    for beta in range(beta_min, beta_max+1):
        p_beta_winner[beta] = 0

    for real_var, p_real_sd in secret_var_distribution.items():
        sd_params = (n, q, sqrt(real_var), m, nu,
                     secret_dist, beta_min, beta_max)
        p_beta_winner_sd = usvp_simulator(sd_params, tours=tours, tour_distr=tour_distr, skip=skip,
                                          simulate_also_lll=simulate_also_lll, simulator=simulator,
                                          embedding=embedding, target_probability=target_probability,
                                          prng_seed=prng_seed, verbose=verbose,
                                          suboptimal_embedding=suboptimal_embedding)

        for beta in p_beta_winner_sd:
            p_beta_winner[beta] += p_beta_winner_sd[beta] * p_real_sd
    return p_beta_winner


def simulate_bkz(params,
                 tours=20,
                 tour_distr=None,
                 skip=1,
                 simulate_also_lll=True,
                 simulator="CN11",
                 target_probability=.999,
                 prng_seed=0xdeadbeef,
                 verbose=True,
                 embedding="baigal",
                 suboptimal_embedding=False):
    """ Simulate the success probability of BKZ on uSVP instance.
    """

    n, q, sd, m, nu, secret_dist, beta_min, beta_max = params
    if embedding == "baigal":
        d = n + m + 1
    elif embedding == "kannan":
        d = m + 1

    if verbose:
        print(n, q, sd, m, "tours", tours)

    # validate params
    assert (d > beta_max)
    assert (beta_min > 0)
    assert (d > 0)
    assert (beta_max >= beta_min)

    # do avg bsw
    if simulator == "averagedBSW18":
        p_beta_winner = {}
        tries = 10
        for BSW18_seed in range(tries):
            out = simulate_bkz(params, tours=tours, simulate_also_lll=simulate_also_lll,
                               simulator="BSW18", target_probability=target_probability,
                               prng_seed=1+BSW18_seed, embedding=embedding)
            for key in out:
                if key not in p_beta_winner:
                    p_beta_winner[key] = 0
                p_beta_winner[key] += out[key]
        for key in p_beta_winner:
            p_beta_winner[key] /= float(tries)
        return p_beta_winner

    # compute prediction

    if simulate_also_lll:
        lll_profile = LLLProfile(n, q, m, nu=nu, embedding=embedding)
    else:
        assert(nu == 1)
        from experiments import genLWEInstance
        from sage.all import seed as _sage_seed
        import random
        import fpylll
        with _sage_seed(1):
            fpylll.FPLLL.set_random_seed(2)  # sets underlying fplll prng
            random.seed(3)  # used by fpylll pruning
            (_, _, _, _, BC, _) = genLWEInstance(n, q, sd, m, nu=1,
                embedding=embedding, use_suboptimal_embedding=suboptimal_embedding)
            lll_profile = [BC.get_r(i, i) for i in range(d)]

    p_beta_winner = {}
    for beta in range(beta_min, beta_max+1):
        sa = SuccessAccumulator()

        for tour in range(1, tours+1):
            sim = Sim(lll_profile)
            sim(beta, tour, sim=simulator, prng_seed=prng_seed)
            profile = sim.profile

            # option 1: just use d-beta+1
            # seems to work better (looking at averages) for many tours
            new_p = cmf_chi_sq(profile[d - beta]/sd**2, beta)
            assert new_p > 0
            sa.accumulate(new_p)
            # print(f"β {beta} τ {tour}: {sa.success_probability()}")

        p_beta_winner[beta] = sa.success_probability()

        if p_beta_winner[beta] > target_probability:
            if target_probability >= .999:
                for _ in range(beta+1, beta_max+1):
                    p_beta_winner[_] = 1
                break

    return p_beta_winner


def simulate_pbkz(params,
                  tours=20,
                  tour_distr=None,
                  skip=1,
                  simulate_also_lll=True,
                  simulator="CN11",
                  target_probability=.999,
                  verbose=True,
                  prng_seed=0xdeadbeef,
                  use_gsa_for_lll=False,
                  embedding="baigal",
                  suboptimal_embedding=False):
    """ Simulate the success probability of progressive BKZ on uSVP instance.
    """

    if embedding != "baigal":
        raise NotImplementedError("Kannan embedding not implemented for PBKZ")
    if suboptimal_embedding:
        raise NotImplementedError("Suboptimal embedding not implemented for PBKZ")

    n, q, sd, m, nu, secret_dist, beta_min, beta_max = params
    d = n+m+1

    if verbose:
        print(n, q, sd, m, "tours", tours)

    # validate params
    assert (d > beta_max)
    assert (beta_min > 0)
    assert (d > 0)
    assert (beta_max >= beta_min)

    if simulator == "averagedBSW18":
        p_beta_winner = {}
        tries = 10
        for prng_seed in range(tries):
            out = simulate_pbkz(params, tours=tours, simulate_also_lll=simulate_also_lll,
                                simulator="BSW18", target_probability=target_probability, verbose=verbose,
                                prng_seed=prng_seed+1)
            for key in out:
                if key not in p_beta_winner:
                    p_beta_winner[key] = 0
                p_beta_winner[key] += out[key]
        for key in p_beta_winner:
            p_beta_winner[key] /= float(tries)
        return p_beta_winner

    # for crypto simulations
    # initial_sim_bs = beta_min
    initial_sim_bs = 40

    # compute prediction
    p_beta_winner = {}
    sa = SuccessAccumulator()

    # just simulate pbkz
    if simulate_also_lll:
        profile = LLLProfile(n, q, m, nu=nu, use_gsa=use_gsa_for_lll)
    else:
        assert(nu == 1)
        profile = get_simulation(n, sd, q, m, 2, 1,
                                 simulate_also_lll=simulate_also_lll, simulator=simulator)['profile']

    sim = Sim(profile)
    for beta in range(3, initial_sim_bs, skip):
        if tour_distr and str(beta) in tour_distr:
            ttours = tour_distr[str(beta)][0]
        else:
            ttours = tours
        if verbose:
            print(f"β {beta}, τ {ttours}")
        sim(beta, round(ttours), sim=simulator, prng_seed=prng_seed)

    initial_sim_bs = beta  # this is the last simulated beta, considering the skip

    for beta in range(initial_sim_bs + skip, beta_max+1, skip):
        # accumulate multiplet tries
        if verbose:
            print(f"{beta}/{beta_max}", end="\r")

        if tour_distr and str(beta) in tour_distr:
            ttours = tour_distr[str(beta)][0]
        else:
            ttours = tours
        if verbose:
            print(f"β {beta}, τ {ttours}")

        for repeat in range(round(ttours)):
            sim(beta, 1, sim=simulator, prng_seed=prng_seed)
            profile = sim.profile

            new_p = cmf_chi_sq(float(profile[d - beta]/sd**2), int(beta))
            assert new_p > 0
            sa.accumulate(new_p)

        if beta >= beta_min:
            p_beta_winner[beta] = sa.success_probability()
            if p_beta_winner[beta] > target_probability:
                if target_probability >= .999:
                    for _ in range(beta+1, beta_max+1):
                        p_beta_winner[_] = 1
                break

    return p_beta_winner
