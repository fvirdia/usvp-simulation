"""
We define a wrapper around the LWE estimator.
"""


from sage.all import load, RR, ZZ, next_prime, sqrt
from math import log, ceil
from utilities import my_hsv_to_rgb
load("lwe-estimator/estimator.py")  # alphaf and primal_usvp


# we should not need this, but we leave it explicit
def reduction_cost_model(beta, d, B): return ZZ(2)**RR(0.292*beta)


def primal_estimate(n, q, sd, m=None, secret_distribution="normal", d=None):
    alpha = alphaf(sd, q, sigma_is_stddev=True)
    if m == None:
        m = 2*n
    res = primal_usvp(
        n, alpha, q,
        m=m,
        d=d,
        secret_distribution=secret_distribution,
        reduction_cost_model=reduction_cost_model
    )
    return res


def estimate_kyber():
    crypto_pars = [
        (256*2, 3329, 1, 1, "normal", "kyber512"),
        (256*3, 3329, 1, 1, "normal", "kyber768"),
        (256*4, 3329, 1, 1, "normal", "kyber1024"),
    ]

    for par in crypto_pars:
        n, q, sd, _, secret_distribution, tag = par
        print(tag)
        print(primal_estimate(n, q, sd, secret_distribution=secret_distribution, m=n))


def estimate_ntru():
    # ntru param sets

    # ntru-hps
    # t = (t1, t2, 1) <- (U(ZZ3)**(n-1), q/8-2 = 2*(q/16-1) ones, 1)

    # n = 509, q = 2048
    # submission: n=508, m=446, d=955 => beta 364
    # estimator:
    print("hps 509")
    n = 508
    q = 2048
    sd = sqrt(2/3)
    secret_distribution = ((-1, 1), int(q/8-2))
    print(primal_estimate(
        n, q, sd, secret_distribution=secret_distribution, m=n, d=2*n+1))
    # rop: 2^109.2, red: 2^109.2, delta_0: 1.004171, beta:  374, d: 1017, m: 508

    # n = 677, q = 2048
    # submission: n=676, m=551, d=1228 => beta 496
    # estimator:
    print("hps 677")
    n = 676
    q = 2048
    sd = sqrt(2/3)
    secret_distribution = ((-1, 1), int(q/8-2))
    print(primal_estimate(
        n, q, sd, secret_distribution=secret_distribution, m=n, d=2*n+1))
    # rop: 2^152.1, red: 2^152.1, delta_0: 1.003306, beta:  521, d: 1353, m: 676

    # n = 821, q = 4096
    # submission: n=820, m=705, d=1526 => beta 615 # 612 tab 5
    # estimator:
    print("hps 820")
    n = 820
    q = 4096
    sd = sqrt(2/3)
    secret_distribution = ((-1, 1), int(q/8-2))
    print(primal_estimate(
        n, q, sd, secret_distribution=secret_distribution, m=n, d=2*n+1))
    # rop: 2^181.3, red: 2^181.3, delta_0: 1.002912, beta:  621, d: 1641, m: 820

    # ntru-hrss
    # t = (f, g, 1) <- (U(ZZ3)**(n-1), U(ZZ3)**m, 1)  (ie dim 700 + m + 1)

    # n = 701
    # submission: n=700, m=627, d=1328 => beta = 466 # 470 tab 5
    # estimator:
    print("hrss 700")
    n = 700
    q = 8192
    sd = sqrt(2/3)
    secret_distribution = (-1, 1)
    print(primal_estimate(
        n, q, sd, secret_distribution=secret_distribution, m=n, d=2*n+1))
    # rop: 2^137.5, red: 2^137.5, delta_0: 1.003551, beta:  471, d: 1401, m: 700


def estimate_saber():
    print()
    print()
    print("To estimate Saber, run:")
    print("git clone https://bitbucket.org/malb/lwe-estimator.git saber_estimator")
    print("cd saber_estimator")
    print("git checkout fb7deba98e599df10b665eeb6a26332e43fb5004")
    print("cd ..")
    print("Then comment line 97 of estimates.py, and rerun it")

    return
    load("saber_estimator/estimator.py")

    def saber_primal_estimate(n, q, sd, m=None, secret_distribution="normal", d=None):
        alpha = alphaf(sd, q, sigma_is_stddev=True)
        if m == None:
            m = 2*n
        res = primal_usvp(n, alpha, q, m=m, d=d,
                          secret_distribution=secret_distribution,
                          reduction_cost_model=reduction_cost_model
                          )
        return res

    crypto_pars = [
        (512, 8192, 1.58, 2.29, "normal", "LightSaber"),
        (768, 8192, 1.41, 2.29, "normal", "Saber"),
        (1024, 8192, 1.22, 2.29, "normal", "FireSaber"),
    ]

    data = []
    for par in crypto_pars:
        n, q, error_sd, secret_sd, secret_distribution, tag = par
        print(tag)
        # NOTE: swap error and secret. This can be done up to m=n samples.
        print(saber_primal_estimate(n, q, secret_sd, secret_distribution=alphaf(
            error_sd, q, sigma_is_stddev=True), m=n))


if __name__ == "__main__":
    estimate_kyber()
    estimate_ntru()
    estimate_saber()
