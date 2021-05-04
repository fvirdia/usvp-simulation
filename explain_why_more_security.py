"""
Script for comparing the estimated required blocksize when using the GSA or the
[CN11] BKZ simulator.
"""


from sage.all import load, next_prime, gamma, gamma_inc_lower, RR, sqrt, line, save, log
from fpylll import BKZ as fpBKZ
from fpylll.tools.bkz_simulator import simulate as CN11_simulate
from bkz_simulators.sim import LLLProfile
from estimates import primal_estimate, delta_0f
from probabilities import cmf_chi_sq
from tikz import TikzPlot


def gsa(beta, d, n, q, i=False):
    """ GSA prediction for BKZ-β.
    """
    delta_0 = delta_0f(beta)
    if not i:
        i = beta
    _vol = q ** ((d - n - 1)/RR(d))
    _gsa = (delta_0 ** (2 * i - d)) * _vol
    return _gsa


def find_intersection(line1, line2):
    if line1[0][1] < line2[0][1]:
        line1, line2 = line2, line1

    # line 1 starts higher than line 2
    for i in range(len(line1)):
        if line1[i][1] < line2[i][1]:
            return i


params = [
    # Kyber
    (256*2, 3329, 1, None, None, "normal", "kyber512"),
    (256*3, 3329, 1, None, None, "normal", "kyber768"),
    (256*4, 3329, 1, None, None, "normal", "kyber1024"),
    # # Saber -- local version of the estimator does not support Saber
    # (512, 8192, 2.29, oo, "LightSaber"),
    # (768, 8192, 2.29, 709, tau, 600, 700, "noise", "Saber"),
    # (1024, 8192, 2.29, 920, tau, 850, 950, "noise", "FireSaber"),
    # # NTRU
    (508, 2048, sqrt(2/3), 508, 2*508+1, ((-1, 1), int(2048/8-2)), "ntruhps2048509"),
    (676, 2048, sqrt(2/3), 676, 2*676+1, ((-1, 1), int(2048/8-2)), "ntruhps2048677"),
    (820, 4096, sqrt(2/3), 820, 2*820+1, ((-1, 1), int(4096/8-2)), "ntruhps4096821"),
    (700, 8192, sqrt(2/3), 700, 2*700+1, (-1, 1), "ntruhrss701"),
]


def main():
    for p in params:
        n, q, sd, m, d, secret_distribution, tag = p
        print(f"--- {tag}")
        res = primal_estimate(
            n, q, sd, secret_distribution=secret_distribution, m=m, d=d)
        print(res)
        m = res['m']
        beta = res['beta']

        if d != None:
            assert(d == n+m+1)
        d = n+m+1
        max_loops = 16

        profile = LLLProfile(n, q, m, nu=1)
        fplll_cn11 = CN11_simulate(profile, fpBKZ.Param(
            block_size=beta, max_loops=max_loops))

        gsa_line = [(i, log(gsa(beta, d, n, q, i=d-i))) for i in range(d)]
        sim_line = [(i, log(fplll_cn11[0][i])/2) for i in range(d)]
        pit_line = [(i, log((d-i)*sd**2)/2) for i in range(d)]
        beta_from_gsa = d - find_intersection(gsa_line, pit_line)
        beta_from_sim = d - find_intersection(sim_line, pit_line)

        g = line(gsa_line, color='red', axes_labels=[
                 "$i$", "$\\log{\\|\\mathbf{b}_i^*\\|}$"], legend_label="GSA")
        g += line(sim_line, color='blue', legend_label="[CN11] simulation")
        g += line(pit_line, color='purple',
                  legend_label="$\\mathbf{\\mathbb{E}}(\\|\\pi_{d-i+1}(\\mathbf{t})\\|)$")
        g += line([(d-beta_from_gsa, 0), (d-beta_from_gsa, log(fplll_cn11[0][0])/2)],
                  color='black', legend_label="$\\beta$, as predicted with the GSA")
        g += line([(d-beta_from_sim, 0), (d-beta_from_sim, log(fplll_cn11[0][0])/2)],
                  color="green", legend_label="$\\beta$, as predicted using [CN11]")

        p = TikzPlot(grid="none")
        p.line(gsa_line, color='red', axes_labels=[
               "$i$", "$\\log{\\|\\mathbf{b}_i^*\\|}$"], legend_label="GSA")
        p.line(sim_line, color='blue', legend_label="[CN11] simulation")
        p.line(pit_line, color='violet',
               legend_label="$\\mathbf{\\mathbb{E}}(\\|\\pi_{d-i+1}(\\mathbf{t})\\|)$")
        p.line([(d-beta_from_gsa, 0), (d-beta_from_gsa, log(fplll_cn11[0][0])/2)],
               color='black', legend_label="$\\beta$, as predicted with the GSA")
        p.line([(d-beta_from_sim, 0), (d-beta_from_sim, log(fplll_cn11[0][0])/2)],
               color="green", legend_label="$\\beta$, as predicted using [CN11]")

        if tag == "kyber512":
            save(g, f"plots/lwe-estimator-with-cn11/n{n}-{tag}.pdf",
                 xmin=500, xmax=800, ymax=5, ymin=1
                 )
            p.save(f"plots/lwe-estimator-with-cn11/n{n}-{tag}.tikz",
                   xmin=500, xmax=800, ymax=5, ymin=1, figsize=[10, 8], xticks=5, yticks=7
                   )
            return

        win_at_2016_gsa = cmf_chi_sq(
            gsa(beta, d, n, q)**2/sd**2, beta_from_gsa)
        win_at_2016_sim = cmf_chi_sq(
            fplll_cn11[0][d-beta]/sd**2, beta_from_gsa)
        win_at_2016_sim_sim = cmf_chi_sq(
            fplll_cn11[0][d-beta_from_sim]/sd**2, beta_from_sim)
        print(f"n = {n}, q = {q}, σ = {sd}")
        print(f"m = {m}, d = {d}")
        print(f"β = {beta}, P[win gsa | gsa's beta] = {win_at_2016_gsa}")
        print(f"P[win sim | gsa's beta] = {win_at_2016_sim}")
        print(f"P[win sim | sim's beta] = {win_at_2016_sim_sim}")
        print(f"β from sim = {beta_from_sim}")


main()
