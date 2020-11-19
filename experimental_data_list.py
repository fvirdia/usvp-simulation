"""
Interface to the raw data from the experiments.
"""

from sage.all import sqrt, next_prime
from settings import NCPUS, HEADLESS, RESULTS_PATH


def bkz_data(experiments):
    if experiments == "full-lll":
        data = [

            (RESULTS_PATH + 'full-lll/results_bkz-seed6000-tries100-n72-tours-5.json', '$n = 72, \\tau = 5$', (72, next_prime(1.5*2**6), 1, 87, 1, 5, 45, 75)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed6000-tries100-n72-tours-10.json', '$n = 72, \\tau = 10$', (72, next_prime(1.5*2**6), 1, 87, 1, 10, 40, 70)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed6000-tries100-n72-tours-15.json', '$n = 72, \\tau = 15$', (72, next_prime(1.5*2**6), 1, 87, 1, 15, 40, 70)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed6000-tries100-n72-tours-20.json', '$n = 72, \\tau = 20$', (72, next_prime(1.5*2**6), 1, 87, 1, 20, 40, 70)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed6000-tries100-n72-tours-30.json', '$n = 72, \\tau = 30$', (72, next_prime(1.5*2**6), 1, 87, 1, 30, 40, 70)),

            (RESULTS_PATH + 'full-lll/results_bkz-seed6000-tries100-n93-tours-5.json', '$n = 93, \\tau = 5$', (93, next_prime(2**8), 1, 105, 1, 5, 45, 75)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed6000-tries100-n93-tours-10.json', '$n = 93, \\tau = 10$', (93, next_prime(2**8), 1, 105, 1, 10, 40, 70)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed6000-tries100-n93-tours-15.json', '$n = 93, \\tau = 15$', (93, next_prime(2**8), 1, 105, 1, 15, 40, 70)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed6000-tries100-n93-tours-20.json', '$n = 93, \\tau = 20$', (93, next_prime(2**8), 1, 105, 1, 20, 40, 70)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed6000-tries100-n93-tours-30.json', '$n = 93, \\tau = 30$', (93, next_prime(2**8), 1, 105, 1, 30, 40, 70)),

            (RESULTS_PATH + 'full-lll/results_bkz-seed6000-tries100-n100-tours-5.json', '$n = 100, \\tau = 5$', (100, next_prime(2**8), sqrt(2/3), 104, 1, 5, 45, 75)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed6000-tries100-n100-tours-10.json', '$n = 100, \\tau = 10$', (100, next_prime(2**8), sqrt(2/3), 104, 1, 10, 40, 70)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed6000-tries100-n100-tours-15.json', '$n = 100, \\tau = 15$', (100, next_prime(2**8), sqrt(2/3), 104, 1, 15, 40, 70)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed6000-tries100-n100-tours-20.json', '$n = 100, \\tau = 20$', (100, next_prime(2**8), sqrt(2/3), 104, 1, 20, 40, 70)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed6000-tries100-n100-tours-30.json', '$n = 100, \\tau = 30$', (100, next_prime(2**8), sqrt(2/3), 104, 1, 30, 40, 70)),
        ]
    elif experiments == "crypto":
        from estimates import primal_estimate

        crypto_pars = [
            # (n, q, sd, nu, secret_dist)
            # (72, next_prime(1.5*2**6), 1, 1, "noise", "toy"),
            (256*2, 3329, 1, 1, "noise", "kyber512"),
            (256*3, 3329, 1, 1, "noise", "kyber768"),
            (256*4, 3329, 1, 1, "noise", "kyber1024"),
        ]

        data = []
        for par in crypto_pars:
            n, q, sd, nu, secret_dist, tag = par
            est = primal_estimate(n, q, sd)
            m = est['m']
            beta = est['beta']
            d = est['d']
            nu = 1
            beta_min, beta_max = int(beta-20), int(beta+40)
            for tau in [15]:
                params = n, q, sd, m, nu, tau, beta_min, beta_max
                data.append((None, f"${tag}, \\tau = {tau}$", params))

        # Saber
        for tau in [15]:
            data += [
                # (tag, (n, q, sd, m, nu, tau, beta_min, beta_max))
                (None, "LightSaber", (512, 8192, 2.29, 507, 2.29/1.58, tau, 360, 440)), # secret_sd = 1.58, nu = 2.29/1.58
                (None, "Saber",      (768, 8192, 2.29, 736, 2.29/1.41, tau, 600, 700)), # secret_sd = 1.41, nu = 2.29/1.41
                (None, "FireSaber",  (1024, 8192, 2.29, 891, 2.29/1.22, tau, 850, 950)), # secret_sd = 1.22, nu = 2.29/1.22
            ]

        # NTRU
        for tau in [15]:
            data += [
                # (tag, (n, q, sd, m, nu, tau, beta_min, beta_max))
                (None, "ntruhps2048509", (508, 2048 , sqrt(2/3), 508, sqrt(2/3)/sqrt(1/2), tau, 350, 400)),
                (None, "ntruhps2048677", (676, 2048, sqrt(2/3), 676, sqrt(2/3)/sqrt(127/338), tau, 480, 540)),
                (None, "ntruhps4096821", (820, 4096, sqrt(2/3), 820, sqrt(2/3)/sqrt(51/82), tau, 595, 655)),
                (None, "ntruhrss701", (700, 8192, sqrt(2/3), 700, 1, tau, 455, 495)),
            ]
    elif experiments == "good-sample-variance":
        data = [
            (RESULTS_PATH + 'full-lll/results_bkz-seed6000-tries100-n72-tours-5-20-good-samp-var.json-bkz-small-gauss-n72-tau-5.json', '$n = 72, \\tau = 5$', (72, next_prime(1.5*2**6), 1, 87, 1, 5, 45, 75)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed6000-tries100-n72-tours-5-20-good-samp-var.json-bkz-small-gauss-n72-tau-10.json', '$n = 72, \\tau = 10$', (72, next_prime(1.5*2**6), 1, 87, 1, 10, 40, 70)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed6000-tries100-n72-tours-5-20-good-samp-var.json-bkz-small-gauss-n72-tau-15.json', '$n = 72, \\tau = 15$', (72, next_prime(1.5*2**6), 1, 87, 1, 15, 40, 70)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed6000-tries100-n72-tours-5-20-good-samp-var.json-bkz-small-gauss-n72-tau-20.json', '$n = 72, \\tau = 20$', (72, next_prime(1.5*2**6), 1, 87, 1, 20, 40, 70)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed6000-tries100-n72-tours-30-good-samp-var.json', '$n = 72, \\tau = 30$', (72, next_prime(1.5*2**6), 1, 87, 1, 30, 40, 70)),
        ]
    elif experiments == "bu-full-lll":
        data = [
            (RESULTS_PATH + 'full-lll/results_bkz-seed5000-tries100-n72-tours-5-30-bu.json-small-bg-n72-tau-5.json', "$n = 72, \\tau = 5$", (72, next_prime(1.5*2**6), 1, 87, 1, 5, 40, 70)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed5000-tries100-n72-tours-5-30-bu.json-small-bg-n72-tau-10.json', "$n = 72, \\tau = 10$", (72, next_prime(1.5*2**6), 1, 87, 1, 10, 40, 70)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed5000-tries100-n72-tours-5-30-bu.json-small-bg-n72-tau-15.json', "$n = 72, \\tau = 15$", (72, next_prime(1.5*2**6), 1, 87, 1, 15, 40, 70)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed5000-tries100-n72-tours-5-30-bu.json-small-bg-n72-tau-20.json', "$n = 72, \\tau = 20$", (72, next_prime(1.5*2**6), 1, 87, 1, 20, 40, 70)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed5000-tries100-n72-tours-5-30-bu.json-small-bg-n72-tau-30.json', "$n = 72, \\tau = 30$", (72, next_prime(1.5*2**6), 1, 87, 1, 30, 40, 70)),

            (RESULTS_PATH + 'full-lll/results_bkz-seed7000-tries100-n93-tours-5-30-bu.json-small-bu-n93-tau-5.json', "$n = 93, \\tau = 5$", (93, next_prime(2**8), 1, 105, 1, 5, 40, 70)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed7000-tries100-n93-tours-5-30-bu.json-small-bu-n93-tau-10.json', "$n = 93, \\tau = 10$", (93, next_prime(2**8), 1, 105, 1, 10, 40, 70)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed7000-tries100-n93-tours-5-30-bu.json-small-bu-n93-tau-15.json', "$n = 93, \\tau = 15$", (93, next_prime(2**8), 1, 105, 1, 15, 40, 70)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed7000-tries100-n93-tours-5-30-bu.json-small-bu-n93-tau-20.json', "$n = 93, \\tau = 20$", (93, next_prime(2**8), 1, 105, 1, 20, 40, 70)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed7000-tries100-n93-tours-5-30-bu.json-small-bu-n93-tau-30.json', "$n = 93, \\tau = 30$", (93, next_prime(2**8), 1, 105, 1, 30, 40, 70)),

            (RESULTS_PATH + 'full-lll/results_bkz-seed50000-tries100-n100-tour-5-30-bu.json-small-bg-n100-tau-5.json', "$n = 100, \\tau = 5$", (100, next_prime(2**8), sqrt(2/3), 104, 1, 5, 40, 70)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed50000-tries100-n100-tour-5-30-bu.json-small-bg-n100-tau-10.json', "$n = 100, \\tau = 10$", (100, next_prime(2**8), sqrt(2/3), 104, 1, 10, 40, 70)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed50000-tries100-n100-tour-5-30-bu.json-small-bg-n100-tau-15.json', "$n = 100, \\tau = 15$", (100, next_prime(2**8), sqrt(2/3), 104, 1, 15, 40, 70)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed50000-tries100-n100-tour-5-30-bu.json-small-bg-n100-tau-20.json', "$n = 100, \\tau = 20$", (100, next_prime(2**8), sqrt(2/3), 104, 1, 20, 40, 70)),
            (RESULTS_PATH + 'full-lll/results_bkz-seed50000-tries100-n100-tour-5-30-bu.json-small-bg-n100-tau-30.json', "$n = 100, \\tau = 30$", (100, next_prime(2**8), sqrt(2/3), 104, 1, 30, 40, 70)),
        ]
    return data


def pbkz_data(experiments):
    if experiments == "full-lll":
        data = [
            (RESULTS_PATH + 'progressive/full-lll/results_retrying-prog-seed5000-tries100-n72-tours-1.json', "$n = 72, \\tau = 1$", (72, next_prime(1.5*2**6), 1, 87, 1, 1, 40, 70)),
            (RESULTS_PATH + 'progressive/full-lll/results_retrying-prog-seed5000-tries100-n72-tours-5.json', "$n = 72, \\tau = 5$", (72, next_prime(1.5*2**6), 1, 87, 1, 5, 40, 70)),
            (RESULTS_PATH + 'progressive/full-lll/results_retrying-prog-seed5000-tries100-n72-tours-10.json', "$n = 72, \\tau = 10$", (72, next_prime(1.5*2**6), 1, 87, 1, 10, 40, 70)),
            (RESULTS_PATH + 'progressive/full-lll/results_retrying-prog-seed5000-tries100-n72-tours-15.json', "$n = 72, \\tau = 15$", (72, next_prime(1.5*2**6), 1, 87, 1, 15, 40, 70)),
            (RESULTS_PATH + 'progressive/full-lll/results_retrying-prog-seed5000-tries100-n72-tours-20.json', "$n = 72, \\tau = 20$", (72, next_prime(1.5*2**6), 1, 87, 1, 20, 40, 70)),

            (RESULTS_PATH + 'progressive/full-lll/results_retrying-prog-seed5000-tries100-n93-tours-1.json', "$n = 93, \\tau = 1$", (93, next_prime(2**8), 1, 105, 1, 1, 40, 70)),
            (RESULTS_PATH + 'progressive/full-lll/results_retrying-prog-seed5000-tries100-n93-tours-5.json', "$n = 93, \\tau = 5$", (93, next_prime(2**8), 1, 105, 1, 5, 40, 70)),
            (RESULTS_PATH + 'progressive/full-lll/results_retrying-prog-seed5000-tries100-n93-tours-10.json', "$n = 93, \\tau = 10$", (93, next_prime(2**8), 1, 105, 1, 10, 40, 70)),
            (RESULTS_PATH + 'progressive/full-lll/results_retrying-prog-seed5000-tries100-n93-tours-15.json', "$n = 93, \\tau = 15$", (93, next_prime(2**8), 1, 105, 1, 15, 40, 70)),
            (RESULTS_PATH + 'progressive/full-lll/results_retrying-prog-seed5000-tries100-n93-tours-20.json', "$n = 93, \\tau = 20$", (93, next_prime(2**8), 1, 105, 1, 20, 40, 70)),

            (RESULTS_PATH + 'progressive/full-lll/results_retrying-prog-seed5000-tries100-n100-tours-1.json', "$n = 100, \\tau = 1$", (100, next_prime(2**8), sqrt(2/3), 104, 1, 1, 40, 70)),
            (RESULTS_PATH + 'progressive/full-lll/results_retrying-prog-seed5000-tries100-n100-tours-5.json', "$n = 100, \\tau = 5$", (100, next_prime(2**8), sqrt(2/3), 104, 1, 5, 40, 70)),
            (RESULTS_PATH + 'progressive/full-lll/results_retrying-prog-seed5000-tries100-n100-tours-10.json', "$n = 100, \\tau = 10$", (100, next_prime(2**8), sqrt(2/3), 104, 1, 10, 40, 70)),
            (RESULTS_PATH + 'progressive/full-lll/results_retrying-prog-seed5000-tries100-n100-tours-15.json', "$n = 100, \\tau = 15$", (100, next_prime(2**8), sqrt(2/3), 104, 1, 15, 40, 70)),
            (RESULTS_PATH + 'progressive/full-lll/results_retrying-prog-seed5000-tries100-n100-tours-20.json', "$n = 100, \\tau = 20$", (100, next_prime(2**8), sqrt(2/3), 104, 1, 20, 40, 70)),
        ]
    elif experiments == "full-lll-tour_map-skip-1":
        data = [
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed5000-tries100-n72-tours5-20-skip-1.json-small-gauss-n72-tau-5.json",
                RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed5000-tries100-n72-tours5-20-skip-1.json-small-gauss-n72-tau-5.json",
                "$n = 72, \\tau = 5, \\delta = 1$", (72, next_prime(1.5*2**6), 1, 87, 1, 5, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed5000-tries100-n72-tours5-20-skip-1.json-small-gauss-n72-tau-10.json",
                RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed5000-tries100-n72-tours5-20-skip-1.json-small-gauss-n72-tau-10.json",
                "$n = 72, \\tau = 10, \\delta = 1$", (72, next_prime(1.5*2**6), 1, 87, 1, 10, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed5000-tries100-n72-tours5-20-skip-1.json-small-gauss-n72-tau-15.json",
                RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed5000-tries100-n72-tours5-20-skip-1.json-small-gauss-n72-tau-15.json",
                "$n = 72, \\tau = 15, \\delta = 1$", (72, next_prime(1.5*2**6), 1, 87, 1, 15, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed5000-tries100-n72-tours5-20-skip-1.json-small-gauss-n72-tau-20.json",
                RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed5000-tries100-n72-tours5-20-skip-1.json-small-gauss-n72-tau-20.json",
                "$n = 72, \\tau = 20, \\delta = 1$", (72, next_prime(1.5*2**6), 1, 87, 1, 20, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed5000-tries100-n93-tours5-20-skip-1.json-small-bg-n93-tau-5.json",
                RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed5000-tries100-n93-tours5-20-skip-1.json-small-bg-n93-tau-5.json",
                "$n = 93, \\tau = 5, \\delta = 1$", (93, next_prime(2**8), 1, 105, 1, 5, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed5000-tries100-n93-tours5-20-skip-1.json-small-bg-n93-tau-10.json",
                RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed5000-tries100-n93-tours5-20-skip-1.json-small-bg-n93-tau-10.json",
                "$n = 93, \\tau = 10, \\delta = 1$", (93, next_prime(2**8), 1, 105, 1, 10, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed5000-tries100-n93-tours5-20-skip-1.json-small-bg-n93-tau-15.json",
                RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed5000-tries100-n93-tours5-20-skip-1.json-small-bg-n93-tau-15.json",
                "$n = 93, \\tau = 15, \\delta = 1$", (93, next_prime(2**8), 1, 105, 1, 15, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed5000-tries100-n93-tours5-20-skip-1.json-small-bg-n93-tau-20.json",
                RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed5000-tries100-n93-tours5-20-skip-1.json-small-bg-n93-tau-20.json",
                "$n = 93, \\tau = 20, \\delta = 1$", (93, next_prime(2**8), 1, 105, 1, 20, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed5000-tries100-n100-tours5-20-skip-1.json-small-bg-n100-tau-5.json",
                RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed5000-tries100-n100-tours5-20-skip-1.json-small-bg-n100-tau-5.json",
                "$n = 100, \\tau = 5, \\delta = 1$", (100, next_prime(2**8), sqrt(2/3), 104, 1, 5, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed5000-tries100-n100-tours5-20-skip-1.json-small-bg-n100-tau-10.json",
                RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed5000-tries100-n100-tours5-20-skip-1.json-small-bg-n100-tau-10.json",
                "$n = 100, \\tau = 10, \\delta = 1$", (100, next_prime(2**8), sqrt(2/3), 104, 1, 10, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed5000-tries100-n100-tours5-20-skip-1.json-small-bg-n100-tau-15.json",
                RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed5000-tries100-n100-tours5-20-skip-1.json-small-bg-n100-tau-15.json",
                "$n = 100, \\tau = 15, \\delta = 1$", (100, next_prime(2**8), sqrt(2/3), 104, 1, 15, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed5000-tries100-n100-tours5-20-skip-1.json-small-bg-n100-tau-20.json",
                RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed5000-tries100-n100-tours5-20-skip-1.json-small-bg-n100-tau-20.json",
                "$n = 100, \\tau = 20, \\delta = 1$", (100, next_prime(2**8), sqrt(2/3), 104, 1, 20, 1, 40, 70)
            ),
        ]
    elif experiments == "full-lll-bu-tour_map-skip-1":
        data = [
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed5000-tries100-n72-bin-tours1-20-skip-1.json-small-bg-n72-tau-5.json",
                RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed5000-tries100-n72-bin-tours1-20-skip-1.json-small-bg-n72-tau-5.json",
                "$n = 72, \\tau = 5, \\delta = 1$", (72, next_prime(1.5*2**6), 1, 87, 1, 5, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed5000-tries100-n72-bin-tours1-20-skip-1.json-small-bg-n72-tau-10.json",
                RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed5000-tries100-n72-bin-tours1-20-skip-1.json-small-bg-n72-tau-10.json",
                "$n = 72, \\tau = 10, \\delta = 1$", (72, next_prime(1.5*2**6), 1, 87, 1, 10, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed5000-tries100-n72-bin-tours1-20-skip-1.json-small-bg-n72-tau-15.json",
                RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed5000-tries100-n72-bin-tours1-20-skip-1.json-small-bg-n72-tau-15.json",
                "$n = 72, \\tau = 15, \\delta = 1$", (72, next_prime(1.5*2**6), 1, 87, 1, 15, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed5000-tries100-n72-bin-tours1-20-skip-1.json-small-bg-n72-tau-20.json",
                RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed5000-tries100-n72-bin-tours1-20-skip-1.json-small-bg-n72-tau-20.json",
                "$n = 72, \\tau = 20, \\delta = 1$", (72, next_prime(1.5*2**6), 1, 87, 1, 20, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed7000-tries100-n93-bin-tours1-20-skip-1.json-small-bg-n93-tau-5.json",
                RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed7000-tries100-n93-bin-tours1-20-skip-1.json-small-bg-n93-tau-5.json",
                "$n = 93, \\tau = 5, \\delta = 1$", (93, next_prime(2**8), 1, 105, 1, 5, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed7000-tries100-n93-bin-tours1-20-skip-1.json-small-bg-n93-tau-10.json",
                RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed7000-tries100-n93-bin-tours1-20-skip-1.json-small-bg-n93-tau-10.json",
                "$n = 93, \\tau = 10, \\delta = 1$", (93, next_prime(2**8), 1, 105, 1, 10, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed7000-tries100-n93-bin-tours1-20-skip-1.json-small-bg-n93-tau-15.json",
                RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed7000-tries100-n93-bin-tours1-20-skip-1.json-small-bg-n93-tau-15.json",
                "$n = 93, \\tau = 15, \\delta = 1$", (93, next_prime(2**8), 1, 105, 1, 15, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed7000-tries100-n93-bin-tours1-20-skip-1.json-small-bg-n93-tau-20.json",
                RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed7000-tries100-n93-bin-tours1-20-skip-1.json-small-bg-n93-tau-20.json",
                "$n = 93, \\tau = 20, \\delta = 1$", (93, next_prime(2**8), 1, 105, 1, 20, 1, 40, 70)
            ),
        ]
    elif experiments == "good-sample-variance-skip-1":
        data = [
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed5000-tries100-n72-tours1-20-skip-1-good-samp-var.json-small-gauss-n72-tau-1.json",
                # RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed5000-tries100-n72-tours1-20-skip-1-good-samp-var.json-small-gauss-n72-tau-1.json",
                None,
                "$n = 72, \\tau = 1$", (72, next_prime(1.5*2**6), 1, 87, 1, 1, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed5000-tries100-n72-tours1-20-skip-1-good-samp-var.json-small-gauss-n72-tau-5.json",
                # RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed5000-tries100-n72-tours1-20-skip-1-good-samp-var.json-small-gauss-n72-tau-5.json",
                None,
                "$n = 72, \\tau = 5$", (72, next_prime(1.5*2**6), 1, 87, 1, 5, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed5000-tries100-n72-tours1-20-skip-1-good-samp-var.json-small-gauss-n72-tau-10.json",
                # RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed5000-tries100-n72-tours1-20-skip-1-good-samp-var.json-small-gauss-n72-tau-10.json",
                None,
                "$n = 72, \\tau = 10$", (72, next_prime(1.5*2**6), 1, 87, 1, 10, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed5000-tries100-n72-tours1-20-skip-1-good-samp-var.json-small-gauss-n72-tau-15.json",
                # RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed5000-tries100-n72-tours1-20-skip-1-good-samp-var.json-small-gauss-n72-tau-15.json",
                None,
                "$n = 72, \\tau = 15$", (72, next_prime(1.5*2**6), 1, 87, 1, 15, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed5000-tries100-n72-tours1-20-skip-1-good-samp-var.json-small-gauss-n72-tau-20.json",
                # RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed5000-tries100-n72-tours1-20-skip-1-good-samp-var.json-small-gauss-n72-tau-20.json",
                None,
                "$n = 72, \\tau = 20$", (72, next_prime(1.5*2**6), 1, 87, 1, 20, 1, 40, 70)
            ),


            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed5000-tries100-n100-tours1-20-skip-1-good-samp-var.json-small-gauss-n100-tau-1.json",
                # RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed5000-tries100-n100-tours1-20-skip-1-good-samp-var.json-small-gauss-n100-tau-1.json",
                None,
                "$n = 100, \\tau = 1$", (100, next_prime(2**8), sqrt(2/3), 104, 1, 1, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed5000-tries100-n100-tours1-20-skip-1-good-samp-var.json-small-gauss-n100-tau-5.json",
                # RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed5000-tries100-n100-tours1-20-skip-1-good-samp-var.json-small-gauss-n100-tau-5.json",
                None,
                "$n = 100, \\tau = 5$", (100, next_prime(2**8), sqrt(2/3), 104, 1, 5, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed5000-tries100-n100-tours1-20-skip-1-good-samp-var.json-small-gauss-n100-tau-10.json",
                # RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed5000-tries100-n100-tours1-20-skip-1-good-samp-var.json-small-gauss-n100-tau-10.json",
                None,
                "$n = 100, \\tau = 10$", (100, next_prime(2**8), sqrt(2/3), 104, 1, 10, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed5000-tries100-n100-tours1-20-skip-1-good-samp-var.json-small-gauss-n100-tau-15.json",
                # RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed5000-tries100-n100-tours1-20-skip-1-good-samp-var.json-small-gauss-n100-tau-15.json",
                None,
                "$n = 100, \\tau = 15$", (100, next_prime(2**8), sqrt(2/3), 104, 1, 15, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/delta/results_prog-seed5000-tries100-n100-tours1-20-skip-1-good-samp-var.json-small-gauss-n100-tau-20.json",
                # RESULTS_PATH + "progressive/full-lll/delta/tours_prog-seed5000-tries100-n100-tours1-20-skip-1-good-samp-var.json-small-gauss-n100-tau-20.json",
                None,
                "$n = 100, \\tau = 20$", (100, next_prime(2**8), sqrt(2/3), 104, 1, 20, 1, 40, 70)
            ),
        ]
    elif experiments == "crypto":

        from estimates import primal_estimate

        crypto_pars = [
            # (n, q, sd, nu, secret_dist)
            # (72, next_prime(1.5*2**6), 1, 1, "noise", "toy"),
            (256*2, 3329, 1, 1, "noise", "kyber512"),
            (256*3, 3329, 1, 1, "noise", "kyber768"),
            (256*4, 3329, 1, 1, "noise", "kyber1024"),
        ]

        data = []
        for par in crypto_pars:
            n, q, sd, nu, secret_dist, tag = par
            est = primal_estimate(n, q, sd)
            m = est['m']
            beta = est['beta']
            d = est['d']
            nu = 1
            beta_min, beta_max = int(beta-20), int(beta+40)
            for tau in [1, 5]:
                params = n, q, sd, m, nu, tau, beta_min, beta_max
                data.append((None, f"${tag}, \\tau = {tau}$", params))

        # Saber
        for tau in [1, 5]:
            data += [
                # (tag, (n, q, sd, m, nu, tau, beta_min, beta_max))
                (None, "LightSaber", (512, 8192, 2.29, 507, 2.29/1.58, tau, 360, 440)), # secret_sd = 1.58, nu = 2.29/1.58
                (None, "Saber",      (768, 8192, 2.29, 736, 2.29/1.41, tau, 600, 700)), # secret_sd = 1.41, nu = 2.29/1.41
                (None, "FireSaber",  (1024, 8192, 2.29, 891, 2.29/1.22, tau, 850, 950)), # secret_sd = 1.22, nu = 2.29/1.22
            ]
        
        # NTRU
        for tau in [1, 5]:
            data += [
                # (tag, (n, q, sd, m, nu, tau, beta_min, beta_max))
                (None, "ntruhps2048509", (508, 2048 , sqrt(2/3), 508, sqrt(2/3)/sqrt(1/2), tau, 350, 400)),
                (None, "ntruhps2048677", (676, 2048, sqrt(2/3), 676, sqrt(2/3)/sqrt(127/338), tau, 480, 540)),
                (None, "ntruhps4096821", (820, 4096, sqrt(2/3), 820, sqrt(2/3)/sqrt(51/82), tau, 595, 655)),
                (None, "ntruhrss701", (700, 8192, sqrt(2/3), 700, 1, tau, 455, 495)),
            ]
    elif experiments == "bu-full-lll":
        data = [
            (RESULTS_PATH + 'progressive/full-lll/results_prog-seed5000-tries100-n72-bin-tours-1.json', "$n = 72, \\tau = 1$", (72, next_prime(1.5*2**6), 1, 87, 1, 1, 40, 70)),
            (RESULTS_PATH + 'progressive/full-lll/results_prog-seed5000-tries100-n72-bin-tours-5.json', "$n = 72, \\tau = 5$", (72, next_prime(1.5*2**6), 1, 87, 1, 5, 40, 70)),
            (RESULTS_PATH + 'progressive/full-lll/results_prog-seed5000-tries100-n72-bin-tours-10.json', "$n = 72, \\tau = 10$", (72, next_prime(1.5*2**6), 1, 87, 1, 10, 40, 70)),
            (RESULTS_PATH + 'progressive/full-lll/results_prog-seed5000-tries100-n72-bin-tours-15.json', "$n = 72, \\tau = 15$", (72, next_prime(1.5*2**6), 1, 87, 1, 15, 40, 70)),
            (RESULTS_PATH + 'progressive/full-lll/results_prog-seed5000-tries100-n72-bin-tours-20.json', "$n = 72, \\tau = 20$", (72, next_prime(1.5*2**6), 1, 87, 1, 20, 40, 70)),

            (RESULTS_PATH + 'progressive/full-lll/results_prog-seed7000-tries100-n93-bin-tours-1.json', "$n = 93, \\tau = 1$", (93, next_prime(2**8), 1, 105, 1, 1, 40, 70)),
            (RESULTS_PATH + 'progressive/full-lll/results_prog-seed7000-tries100-n93-bin-tours-5.json', "$n = 93, \\tau = 5$", (93, next_prime(2**8), 1, 105, 1, 5, 40, 70)),
            (RESULTS_PATH + 'progressive/full-lll/results_prog-seed7000-tries100-n93-bin-tours-10.json', "$n = 93, \\tau = 10$", (93, next_prime(2**8), 1, 105, 1, 10, 40, 70)),
            (RESULTS_PATH + 'progressive/full-lll/results_prog-seed7000-tries100-n93-bin-tours-15.json', "$n = 93, \\tau = 15$", (93, next_prime(2**8), 1, 105, 1, 15, 40, 70)),
            (RESULTS_PATH + 'progressive/full-lll/results_prog-seed7000-tries100-n93-bin-tours-20.json', "$n = 93, \\tau = 20$", (93, next_prime(2**8), 1, 105, 1, 20, 40, 70)),

            (RESULTS_PATH + 'progressive/full-lll/results_prog-seed50000-tries100-n100-ter-tours-1.json', "$n = 100, \\tau = 1$", (100, next_prime(2**8), sqrt(2/3), 104, 1, 1, 40, 70)),
            (RESULTS_PATH + 'progressive/full-lll/results_prog-seed50000-tries100-n100-ter-tours-5.json', "$n = 100, \\tau = 5$", (100, next_prime(2**8), sqrt(2/3), 104, 1, 5, 40, 70)),
            (RESULTS_PATH + 'progressive/full-lll/results_prog-seed50000-tries100-n100-ter-tours-10.json', "$n = 100, \\tau = 10$", (100, next_prime(2**8), sqrt(2/3), 104, 1, 10, 40, 70)),
            (RESULTS_PATH + 'progressive/full-lll/results_prog-seed50000-tries100-n100-ter-tours-15.json', "$n = 100, \\tau = 15$", (100, next_prime(2**8), sqrt(2/3), 104, 1, 15, 40, 70)),
            (RESULTS_PATH + 'progressive/full-lll/results_prog-seed50000-tries100-n100-ter-tours-20.json', "$n = 100, \\tau = 20$", (100, next_prime(2**8), sqrt(2/3), 104, 1, 20, 40, 70)),
        ]
    elif experiments == "big-q":
        data = [
            # big q
            (
                RESULTS_PATH + "progressive/full-lll/big-q/results_prog-seed4000-tries100-n100-tours1-skip-1.json",
                None,
                "$n, q, \\tau = 100, 1031, 1$", ( 100,      1031,         1.85,    128, 1, 1, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/big-q/results_prog-seed4000-tries100-n100-tours5-20-skip-1.json-big-q-gauss-n100-tau-5.json",
                # RESULTS_PATH + "progressive/full-lll/big-q/tours_prog-seed4000-tries100-n100-tours5-20-skip-1.json-big-q-gauss-n100-tau-5.json",
                None,
                "$n, q, \\tau = 100, 1031, 5$", ( 100,      1031,         1.85,    128, 1, 5, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/big-q/results_prog-seed4000-tries100-n100-tours5-20-skip-1.json-big-q-gauss-n100-tau-10.json",
                # RESULTS_PATH + "progressive/full-lll/big-q/tours_prog-seed4000-tries100-n100-tours5-20-skip-1.json-big-q-gauss-n100-tau-10.json",
                None,
                "$n, q, \\tau = 100, 1031, 10$", ( 100,      1031,         1.85,    128, 1, 10, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/big-q/results_prog-seed4000-tries100-n100-tours5-20-skip-1.json-big-q-gauss-n100-tau-15.json",
                # RESULTS_PATH + "progressive/full-lll/big-q/tours_prog-seed4000-tries100-n100-tours5-20-skip-1.json-big-q-gauss-n100-tau-15.json",
                None,
                "$n, q, \\tau = 100, 1031, 15$", ( 100,      1031,         1.85,    128, 1, 15, 1, 40, 70)
            ),
            (
                RESULTS_PATH + "progressive/full-lll/big-q/results_prog-seed4000-tries100-n100-tours5-20-skip-1.json-big-q-gauss-n100-tau-20.json",
                # RESULTS_PATH + "progressive/full-lll/big-q/tours_prog-seed4000-tries100-n100-tours5-20-skip-1.json-big-q-gauss-n100-tau-20.json",
                None,
                "$n, q, \\tau = 100, 1031, 20$", ( 100,      1031,         1.85,    128, 1, 20, 1, 40, 70)
            ),
        ]
    return data

