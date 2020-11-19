"""
Code for running simulations and reproducing plots.
See main function at the bottom.
"""

import json
from sage.all import sqrt, next_prime, save, line, point, pi
# from fpylll import BKZ
from utilities import bcolors
from experimental_data_list import bkz_data, pbkz_data
from probabilities import pmf_from_cmf, kth_moment
from usvp_simulation import simulate_bkz, simulate_pbkz, real_sd_simulate

def plot_experiments(fn, label, progressive=False, color=None, marker="x", return_cmf=False):
    colors = (x for x in ['red', 'blue', 'orange', 'green', 'purple', 'black',
                          'brown', 'violet'])

    # generate plots
    g = line([])

    with open(fn) as f:
        res = json.load(f)
        if progressive:
            acc = 0
            tries = 0
        for exp in res:
            prof = []
            if progressive:
                # compute total tries
                for bs in sorted(res[exp]):
                    tries += res[exp][bs]['tries']

                # compute success probabilities
                for bs in sorted(res[exp]):
                    acc += res[exp][bs]['successes']
                    prof.append((int(bs), acc/tries))
                if acc != tries:
                    print(f"WARNING: only {acc} out of {tries} solved")
            else:
                for bs in sorted(res[exp]):
                    prof.append((int(bs), res[exp][bs]['succ_ratio']))
            if label:
                tag = label
            else:
                tag = exp
            g += line(prof, color=next(colors) if not color else color,
                      linestyle=" ", marker=marker)  # , legend_label=tag)

    if return_cmf:
        cmf = {}
        for beta, prob in prof:
            cmf[beta] = prob
        return g, cmf

    return g


def gen_plots(data, usvp_simulator, output_dir,
              plot_by_dimension=False,
              plot_mean=False,
              simulate_also_lll=True,
              format_string=None,
              progressive=False,
              simulator="CN11",
              max_tau=None,
              save_plots=True,
              mixed_plots=False,
              account_for_sample_variance=False):

    axes_labels = ['$\\beta$', '$P[B\\leq\\beta]$']
    figsize = [6, 4]
    font_size = 15

    if plot_by_dimension:
        plots_by_tour = {}
        tour_col = {
            1: 'black',
            5: 'blue',
            10: 'red',
            15: 'green',
            20: 'brown',
            30: 'orange',
        }
        tour_marker = {
            1: '|',
            5: 'd',
            10: '*',
            15: '+',
            20: 'x',
        }

    if not format_string:
        if plot_by_dimension:
            format_string = "{dir}/n-{n}.png"
        else:
            format_string = "{dir}/{tag}.png"

    for entry in data:
        try:
            fn, tours_fn, label, pars = entry
        except:
            fn, label, pars = entry
            tours_fn = None

        if len(pars) == 6:
            n, q, sd, m, nu, tau = pars
            beta_min, beta_max = 45, 65
            skip = 1
        elif len(pars) == 8:
            n, q, sd, m, nu, tau, beta_min, beta_max = pars
            skip = 1
        else:
            n, q, sd, m, nu, tau, skip, beta_min, beta_max = pars

        if max_tau and tau > max_tau:
            continue

        if mixed_plots:
            label = "${%sBKZ},\\,\\tau = %d$" % (
                "Prog.\\," if progressive else "", tau)

        secret_dist = "noise"
        params = n, q, sd, m, nu, secret_dist, beta_min, beta_max

        print("τ\tδ\tn\tq\tσ_s\tσ_e\tsimE(β)\tsimS(β)",
              end="\texpE(β)\texpS(β)\n" if fn else "\n")

        if plot_by_dimension and n not in plots_by_tour:
            plots_by_tour[n] = line([])

        if not plot_by_dimension:
            g = line([])
            if fn:
                gg, cmf_exp = plot_experiments(
                    fn, f"Exp, {label}", return_cmf=True, progressive=progressive)
                # print("extrapolating mean from data")
                pmf_exp = pmf_from_cmf(cmf_exp, force=True)
                exp_avg_beta = kth_moment(pmf_exp, 1)
                exp_avg_beta_sqr = kth_moment(pmf_exp, 2)
                exp_stddev_beta = sqrt(exp_avg_beta_sqr - exp_avg_beta**2)
                g += gg

        for tours in [tau]:
            import json
            if tours_fn:
                tour_distr = json.load(open(tours_fn))
                tour_distr = tour_distr[list(tour_distr.keys())[0]]
            else:
                tour_distr = None
            p_beta_winner = real_sd_simulate(usvp_simulator, params,
                                             tours=tours, tour_distr=tour_distr, skip=skip,
                                             simulate_also_lll=simulate_also_lll, simulator=simulator,
                                             verbose=False, account_for_sample_variance=account_for_sample_variance)
            # print("extrapolating mean from simulation")
            pmf = pmf_from_cmf(p_beta_winner, force=True)
            sim_avg_beta = kth_moment(pmf, 1)
            sim_avg_beta_sqr = kth_moment(pmf, 2)
            sim_stddev_beta = sqrt(sim_avg_beta_sqr - sim_avg_beta**2)
            if plot_by_dimension:
                if fn:
                    gg, cmf_exp = plot_experiments(fn,
                                                   f"Exp, {label}", color=tour_col[tours],
                                                   marker='x' if skip == 1 else tour_marker[tours],
                                                   return_cmf=True, progressive=progressive)
                    # print("extrapolating mean from data")
                    pmf_exp = pmf_from_cmf(cmf_exp, force=True)
                    exp_avg_beta = kth_moment(pmf_exp, 1)
                    exp_avg_beta_sqr = kth_moment(pmf_exp, 2)
                    exp_stddev_beta = sqrt(exp_avg_beta_sqr - exp_avg_beta**2)
                    plots_by_tour[n] += gg
                if skip == 1:
                    plots_by_tour[n] += line(p_beta_winner.items(),
                                             color=tour_col[tours],  legend_label=f"{label}", linestyle="--")
                else:
                    plots_by_tour[n] += line(p_beta_winner.items(),
                                             color=tour_col[tours],  legend_label=f"{label}", linestyle=" ", marker="o",
                                             # markerfacecolor="white", markeredgecolor=tour_col[tours]
                                             alpha=.5
                                             )
                print("%d\t%d\t%d\t%d\t%.2f\t%.2f" %
                      (tours, skip, n, q, sd, sd*nu), end="\t")
                print("%.2f" % sim_avg_beta, end="\t")
                print("%.2f" % sim_stddev_beta, end="\t")
                # print(f"Sim mean {sim_avg_beta}")
                # print(f"Sim stdv {sim_stddev_beta}")
                if fn:
                    print("%.2f" % exp_avg_beta, end="\t")
                    print("%.2f" % exp_stddev_beta, end="\t")
                    # print(f"Exp mean {exp_avg_beta}")
                    # print(f"Exp stdv {exp_stddev_beta}")
                print(label)
                if plot_mean:
                    plots_by_tour[n] += line([(sim_avg_beta, 0), (sim_avg_beta, 1)],
                                             color=tour_col[tours], linestyle="--")
                    if fn:
                        plots_by_tour[n] += line([(exp_avg_beta, 0),
                                                  (exp_avg_beta, 1)], color=tour_col[tours])
            else:
                if skip == 1:
                    g += line(p_beta_winner.items(),
                              legend_label=f"{label}", color="green", linestyle="--")
                else:
                    g += line(p_beta_winner.items(), legend_label=f"{label}", color="green", linestyle=" ", marker="o",
                              # markerfacecolor="white", markeredgecolor="green",
                              alpha=.5,
                              )
                print(f"Sim mean {sim_avg_beta}")
                print(f"Sim stdv {sim_stddev_beta}")
                if fn:
                    print(f"Exp mean {exp_avg_beta}")
                    print(f"Exp stdv {exp_stddev_beta}")
                if plot_mean:
                    g += line([(sim_avg_beta, 0), (sim_avg_beta, 1)],
                              color="green")
                    if fn:
                        g += line([(exp_avg_beta, 0),
                                   (exp_avg_beta, 1)], color="red")

        if not plot_by_dimension and save_plots:
            safe_tag = label.replace(
                '/', '-').replace('τ', 'tau').replace('\\', '')
            g.set_legend_options(font_size=font_size)
            save(g, format_string.format(dir=output_dir, tag=safe_tag),
                 dpi=300, ymin=0, ymax=1, xmax=beta_max+1, axes_labels=axes_labels, figsize=figsize)

    if plot_by_dimension:
        if save_plots:
            for n, g in plots_by_tour.items():
                g.set_legend_options(font_size=font_size)
                save(g, format_string.format(dir=output_dir, n=n),
                     dpi=300, ymin=0, ymax=1, xmax=beta_max+1, axes_labels=axes_labels, figsize=figsize)
        return plots_by_tour


def bkz_plots(experiments, simulator="CN11", plot_mean=False, max_tau=None, account_for_sample_variance=False):
    # problem parameters
    data = bkz_data(experiments)

    gen_plots(data, simulate_bkz, "plots/plain/",
              plot_by_dimension=True, plot_mean=plot_mean, simulate_also_lll=True,
              format_string="{dir}/n-{n}-%s-using-%s%s%s.pdf" % (
                  experiments, simulator, "-max-tau-%d" % max_tau if max_tau else "",
                  "-acc-samp-var" if account_for_sample_variance else ""
              ),
              progressive=False, simulator=simulator, max_tau=max_tau, account_for_sample_variance=account_for_sample_variance)

    # One by one plots
    # gen_plots(data, simulate_bkz, "plots/plain/",
    #     plot_by_dimension=False, plot_mean=plot_mean, simulate_also_lll=True,
    #     format_string="{dir}/{tag}-%s-using-%s%s%s.pdf" % (
    #         experiments, simulator, "-max-tau-%d" % max_tau if max_tau else "",
    #         "-acc-samp-var" if account_for_sample_variance else ""
    #     ),
    #     progressive=False, simulator=simulator, max_tau=max_tau, account_for_sample_variance=account_for_sample_variance)


def pbkz_plots(experiments, simulator="CN11", plot_mean=False, max_tau=None, account_for_sample_variance=False):
    # problem parameters
    data = pbkz_data(experiments)

    print(f"{bcolors.HEADER}Progressive BKZ{bcolors.ENDC}")
    gen_plots(data, simulate_pbkz, "plots/progressive/",
              plot_by_dimension=True, plot_mean=plot_mean, simulate_also_lll=True,
              format_string="{dir}/n-{n}-%s-using-%s%s%s.pdf" % (
                  experiments, simulator, "-max-tau-%d" % max_tau if max_tau else "",
                  "-acc-samp-var" if account_for_sample_variance else ""
              ),
              progressive=True, simulator=simulator, max_tau=max_tau, account_for_sample_variance=account_for_sample_variance)

    # One by one plots
    # gen_plots(data, simulate_pbkz, "plots/progressive/",
    #     plot_by_dimension=False, plot_mean=plot_mean, simulate_also_lll=True,
    #     format_string="{dir}/{tag}-%s-using-%s%s%s.pdf" % (
    #         experiments, simulator, "-max-tau-%d" % max_tau if max_tau else "",
    #         "-acc-samp-var" if account_for_sample_variance else ""
    #     ),
    #     progressive=True, simulator=simulator, max_tau=max_tau, account_for_sample_variance=account_for_sample_variance)


def bkz_and_pbkz_plots(experiments, simulator="CN11", plot_mean=False, max_tau=None, account_for_sample_variance=False):
    data_bkz = bkz_data(experiments)
    data_pbkz = pbkz_data(experiments)
    data_pbkz_tau1 = []
    for entry in data_pbkz:
        try:
            fn, tours_fn, label, pars = entry
        except:
            fn, label, pars = entry
            tours_fn = None

        if len(pars) == 6:
            n, q, sd, m, nu, tau = pars
            beta_min, beta_max = 45, 65
            skip = 1
        elif len(pars) == 8:
            n, q, sd, m, nu, tau, beta_min, beta_max = pars
            skip = 1
        else:
            n, q, sd, m, nu, tau, skip, beta_min, beta_max = pars
        if tau == 1:
            data_pbkz_tau1.append(entry)

    print(f"{bcolors.HEADER}Mixed plots{bcolors.ENDC}")
    bkz_plots = {}
    bkz_plots = gen_plots(data_bkz, simulate_bkz, "plots/plain/",
                          plot_by_dimension=True, plot_mean=plot_mean, simulate_also_lll=True,
                          format_string="{dir}/n-{n}-%s-using-%s.pdf" % (
                              experiments, simulator),
                          progressive=False, simulator=simulator, max_tau=max_tau, save_plots=False,
                          mixed_plots=True, account_for_sample_variance=account_for_sample_variance)

    pbkz_plots = gen_plots(data_pbkz_tau1, simulate_pbkz, "plots/progressive/",
                           plot_by_dimension=True, plot_mean=plot_mean, simulate_also_lll=True,
                           format_string="{dir}/n-{n}-%s-using-%s.pdf" % (
                               experiments, simulator),
                           progressive=True, simulator=simulator, max_tau=max_tau, save_plots=False,
                           mixed_plots=True, account_for_sample_variance=account_for_sample_variance)

    font_size = 15
    axes_labels = ['$\\beta$', '$P[B\\leq\\beta]$']
    figsize = [6, 4]
    plots = {}
    format_string = "plots/mixed/n-{n}-%s-using-%s%s.pdf" % (
        experiments, simulator, "-acc-samp-var" if account_for_sample_variance else ""
    )

    for plot_set in [pbkz_plots, bkz_plots]:
        for n, g in plot_set.items():
            if n not in plots:
                plots[n] = line([])
            g.set_legend_options(font_size=font_size)
            plots[n] += g

    for n, g in plots.items():
        save(g, format_string.format(n=n),
             ymin=0, ymax=1, xmin=38, xmax=70,
             axes_labels=axes_labels, figsize=figsize,
             #  title=f"$n = {n}$",
             )


def compare_vs_lwe_side_channel(output_dir="plots/progressive/vs-leaky/", simulate_also_lll=True, simulator="CN11"):
    from DSDGR20_warpper import simulate_pbkz as sim_pbkz
    import timeit
    # allow timeit to return function return value, https://stackoverflow.com/a/40385994
    timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        retval = {stmt}
    _t1 = _timer()
    return _t1 - _t0, retval
"""

    data = [
        ("$n = 72, \\tau = 1$", (72, next_prime(1.5*2**6), 1, 87, 1, 1, 45, 70)),
        ("$n = 93, \\tau = 1$", (93, next_prime(2**8), 1, 105, 1, 1, 45, 70)),
    ]

    # Kyber
    from estimates import primal_estimate
    crypto_pars = [
        # (n, q, sd, nu, secret_dist)
        (256*2, 3329, 1, 1, "noise", "kyber512", 370, 400),
        # (256*3, 3329, 1, 1, "noise", "kyber768", 620, 650),
        (256*4, 3329, 1, 1, "noise", "kyber1024", 875, 905),
    ]
    for par in crypto_pars:
        n, q, sd, nu, secret_dist, tag, beta_min, beta_max = par
        est = primal_estimate(n, q, sd)
        m = est['m']
        beta = est['beta']
        d = est['d']
        tau = 1
        params = n, q, sd, m, nu, tau, beta_min, beta_max
        data.append((f"{tag}, $\\tau = {tau}$", params))

    axes_labels = ['$\\beta$', '$P[B\\leq\\beta]$']
    figsize = [6, 4]
    font_size = 15

    format_string = "{dir}/{tag}-vs-D-SDGR20-using-%s.pdf" % simulator

    for label, pars in data:
        try:
            n, q, sd, m, nu, tau = pars
            beta_min, beta_max = 45, 65
        except:
            n, q, sd, m, nu, tau, beta_min, beta_max = pars
        if tau != 1:
            continue
        secret_dist = "noise"
        params = n, q, sd, m, nu, secret_dist, beta_min, beta_max

        g = line([])

        print("Simulator\ttime\tmean succ. β")

        # our simulator
        line_thickness = 3
        for use_gsa_for_lll in [False, True]:
            timer = timeit.Timer(lambda: simulate_pbkz(params, tau,
                                                       simulate_also_lll=simulate_also_lll, simulator=simulator,
                                                       verbose=False, use_gsa_for_lll=use_gsa_for_lll))
            time, p_beta_winner = timer.timeit(number=1)
            g += line(p_beta_winner.items(),
                      legend_label="this work%s" % (
                          " (GSA for LLL)" if use_gsa_for_lll else ""),
                      linestyle="--",
                      color="purple" if use_gsa_for_lll else "green",
                      thickness=line_thickness)
            line_thickness -= 1
            pmf = pmf_from_cmf(p_beta_winner)
            print("ours%s\t%.1f s\t%.2f" % (
                " (gsa)" if use_gsa_for_lll else "\t", time, kth_moment(pmf, 1)
            ))

        # [DSDGR20] simluator
        timer = timeit.Timer(lambda: sim_pbkz(params))
        time, old_p_beta_winner = timer.timeit(number=1)
        g += line(old_p_beta_winner.items(),
                  legend_label=f"[D-SDGR20]",
                  linestyle="--",
                  color="blue",
                  thickness=line_thickness)
        pmf = pmf_from_cmf(old_p_beta_winner)
        print("[DSDGR20]\t%.1f s\t%.2f" % (time, kth_moment(pmf, 1)))

        safe_tag = label.replace(
            '/', '-').replace('τ', 'tau').replace('\\', '').replace(' ', '_').replace('$', '')
        g.set_legend_options(font_size=font_size)
        save(g, format_string.format(dir=output_dir, tag=safe_tag), dpi=300,
             ymin=0, ymax=1, xmin=beta_min, xmax=beta_max,
             axes_labels=axes_labels, figsize=figsize)
        """
        Simulator	time	mean succ. β
        ours		0.0 s	57.25
        ours (gsa)	0.0 s	57.26
        WARNING: extrapolating pmf from incomplete cmf
                pmf aggiusted by factor 1.0000000000
        [DSDGR20]	3.6 s	57.85
        Simulator	time	mean succ. β
        ours		0.0 s	58.93
        ours (gsa)	0.0 s	58.97
        [DSDGR20]	6.1 s	59.85
        Simulator	time	mean succ. β
        ours		2.6 s	388.65
        ours (gsa)	2.7 s	388.66
        [DSDGR20]	503.7 s	388.85
        Simulator	time	mean succ. β
        ours		20.0 s	894.29
        ours (gsa)	21.6 s	894.29
        WARNING: extrapolating pmf from incomplete cmf
                pmf aggiusted by factor 1.0000000000
        [DSDGR20]	3196.7 s	894.38

        """


def plot_previous_literature(simulator="CN11", simulate_also_lll=True, account_for_sample_variance=False):

    colors = (x for x in ['red', 'blue', 'orange', 'green', 'purple', 'black',
                          'brown', 'violet'])

    # AGVW17 experiments
    exp_cmfs = {
        65: {46: .048, 51: .528, 56: .933},
        80: {45: .002, 50: .089, 55: .606, 60: .942},
        100: {52: .002, 57: .058, 62: .396, 67: .888}
    }

    params = [
        # AGVW17
        (65, 521, float(8/sqrt(2*pi)), 182, 1, "noise", 40, 70),
        (80, 1031, float(8/sqrt(2*pi)), 204, 1, "noise", 40, 70),
        (100, 2053, float(8/sqrt(2*pi)), 243, 1, "noise", 40, 70),
    ]

    # reset color generator to get same colors
    colors = (x for x in ['red', 'blue', 'orange', 'green', 'purple', 'black',
                          'brown', 'violet'])

    g = line([])
    for param in params:
        n, q, sd, m, nu, secret_dist, beta_min, beta_max = param
        tours = 20
        p_beta_winner = real_sd_simulate(simulate_bkz, param, tours=tours,
                                         simulate_also_lll=simulate_also_lll, simulator=simulator,
                                         account_for_sample_variance=account_for_sample_variance, verbose=False)

        sim_pmf = pmf_from_cmf(p_beta_winner, force=True)
        sim_avg_beta = kth_moment(sim_pmf, 1)
        sim_avg_beta_sqr = kth_moment(sim_pmf, 2)
        sim_stddev_beta = sqrt(sim_avg_beta_sqr - sim_avg_beta**2)
        exp_pmf = pmf_from_cmf(exp_cmfs[n], force=True)
        exp_avg_beta = kth_moment(exp_pmf, 1)
        exp_avg_beta_sqr = kth_moment(exp_pmf, 2)
        exp_stddev_beta = sqrt(exp_avg_beta_sqr - exp_avg_beta**2)

        print("τ\tn\tq\tσ_s\tσ_e\tsimE(β)\tsimS(β)\texpE(β)\texpS(β)")
        print("%d\t%d\t%d\t%.2f\t%.2f" % (tours, n, q, sd, sd*nu), end="\t")
        print("%.2f" % sim_avg_beta, end="\t")
        print("%.2f" % sim_stddev_beta, end="\t")
        print("%.2f" % exp_avg_beta, end="\t")
        print("%.2f" % exp_stddev_beta, end="\n")

        col = next(colors)
        g += point(list(exp_cmfs[n].items()),
                   color=col, marker="X", size=75)
        g += line(p_beta_winner.items(), color=col,
                  linestyle="--", legend_label="$n = %d, \\tau = %d$" % (n, tours))
        g.set_legend_options(font_size=20)

    save(g, "plots/plain/previous-exps-simlll-%s%s.pdf" % (
        simulate_also_lll, "-acc-samp-var" if account_for_sample_variance else ""),
        figsize=[10, 3], axes_labels=['$\\beta$', '$P[B\\leq\\beta]$'])


def comparing_gsa_vs_sim():
    """ The GSA results in a translated profile, causing bad predictions.
    """
    from bkz_simulators.sim import LLLProfile
    from experiments import genLWEInstance
    from sage.all import seed as _sage_seed
    import random
    import fpylll
    from sage.all import log

    params = [
        # This paper
        (72, 97, 1, 87, 1, "noise", 40, 70),
        (93, 257, 1, 87, 105, "noise", 40, 70),
        (100, 257, sqrt(2/3), 104, 1, "noise", 40, 70),
        # AGVW17 but with optimal number of samples
        # (65, 521, float(8/sqrt(2*pi)), 112, 1, "noise", 40, 70),
        # (80, 1031, float(8/sqrt(2*pi)), 122, 1, "noise", 40, 70),
        # (100, 2053, float(8/sqrt(2*pi)), 159, 1, "noise", 40, 70),
        # AGVW17
        # (65, 521, float(8/sqrt(2*pi)), 182, 1, "noise", 40, 70),
        # (80, 1031, float(8/sqrt(2*pi)), 204, 1, "noise", 40, 70),
        # (100, 2053, float(8/sqrt(2*pi)), 243, 1, "noise", 40, 70),
    ]

    for n, q, sd, m, nu, _, _, _ in params:
        print(f"d = {n+m+1}")
        print(f"n = {n}")
        sim = LLLProfile(n, q, m, nu=nu, embedding="baigal",
                         use_gsa=False, use_mine=False)
        my_sim = LLLProfile(n, q, m, nu=nu, embedding="baigal",
                            use_gsa=False, use_mine=True)
        print("sim", len(sim))
        print("my_sim", len(my_sim))
        gsa = LLLProfile(n, q, m, nu=nu, embedding="baigal", use_gsa=True)
        print("gsa", len(gsa))
        with _sage_seed(1):
            fpylll.FPLLL.set_random_seed(2)  # sets underlying fplll prng
            random.seed(3)  # used by fpylll pruning
            BC = genLWEInstance(n, q, sd, m, nu=nu,
                                use_suboptimal_embedding=False)[4]
            print("lll", BC.d)
            lll = [BC.get_r(i, i) for i in range(BC.d)]
            BC = genLWEInstance(n, q, sd, m, nu=nu,
                                use_suboptimal_embedding=True)[4]
            print("lll bad q", BC.d)
            lll_bad_q = [BC.get_r(i, i) for i in range(BC.d)]
        g = line([])
        # g += line([(i, log(sim[i])) for i in range(len(sim))], legend_label="sim", color="red")
        g += line([(i, log(lll_bad_q[i])) for i in range(len(lll_bad_q))],
                  legend_label="lll bad q", color="black")
        g += line([(i, log(lll[i])) for i in range(len(lll))],
                  legend_label="lll", color="green")
        g += line([(i, log(my_sim[i])) for i in range(len(my_sim))],
                  legend_label="my sim", color="purple")
        g += line([(i, log(gsa[i])) for i in range(len(gsa))],
                  legend_label="gsa", color="blue")
        save(g, "deleteme-%d.png" % n, dpi=300)


def reproduce_paper_data():

    # # Figure 1 -- takes the longest, disabled by default!
    # compare_vs_lwe_side_channel(simulator="CN11")

    # Figure 2
    bkz_and_pbkz_plots("full-lll", simulator="CN11")

    # Figure 3
    print("Fig 5, Progressive BKZ data from here")
    pbkz_plots("full-lll", simulator="CN11", max_tau=None)

    # Figure 4
    plot_previous_literature(simulator="CN11", simulate_also_lll=True)

    # Figure 5
    print("Fig 5, BKZ 2.0 data from here")
    bkz_plots("full-lll", simulator="CN11", max_tau=None)

    # Figure 6
    pbkz_plots("bu-full-lll", simulator="CN11", max_tau=None)

    # Figure 8
    pbkz_plots("good-sample-variance-skip-1", simulator="CN11", max_tau=10)

    # Figure 9
    bkz_plots("full-lll", simulator="BSW18", max_tau=None)
    bkz_plots("full-lll", simulator="averagedBSW18", max_tau=None)

    # Table 2
    bkz_plots("crypto", simulator="CN11")
    pbkz_plots("crypto", simulator="CN11")


if __name__ == "__main__":
    reproduce_paper_data()
