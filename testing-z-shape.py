"""
Compare the output of LLL on a q-ary lattice basis with the q-vectors at the
beginning of the basis, with the GSA and the Z-shape simulator.
"""


from experiments import genLWEInstance
from bkz_simulators.sim import LLLProfile
from sage.all import next_prime, line, show, save, log, sqrt, QQ
from tikz import TikzPlot


def testing_z_shape_sim():
    """ Generates a plot comparing LLL with the GSA and the Z-shape simulator.
    """
    # n, q, sd, m, nu = 93, next_prime(2**8), 1, 105, 2
    n, q, sd, m, nu, embedding = 100, next_prime(2**8), sqrt(2/3), 105, 1, "baigal"
    # n, q, sd, m, nu, embedding = 100, next_prime(2**8), sqrt(2/3), 105, 2, "baigal"
    # n, q, sd, m, nu, embedding = 100, next_prime(2**8), sqrt(2/3), 205, 1, "kannan"
    # n, q, sd, m, nu, embedding = 100, next_prime(2**8), sqrt(2/3), 255, 1, "kannan"
    # n, q, sd, nu = 72, next_prime(1.5*2**6), 1, 2.378
    nu_denom = QQ(round(nu*100)/100).denominator()
    # m = n
    d = m + 1
    if embedding == "baigal":
        d += n
    tries = 1  # 25
    exp = [0] * d
    for _ in range(tries):
        (lwe, samples, A, C, BC_GSO, vol) = genLWEInstance(
            n, q, sd, m, nu=nu, embedding=embedding)
        for i in range(d):
            exp[i] += BC_GSO.get_r(i, i)/tries

    sim = list(LLLProfile(n, q, m, nu=nu, embedding=embedding, use_gsa=False))
    gsa = list(LLLProfile(n, q, m, nu=nu, embedding=embedding, use_gsa=True))

    # g = line([])
    # g += line([(x, log(exp[x], 2)/2) for x in range(d)],
    #           color="red", legend_label="LLL output")
    # g += line([(x, log(sim[x]*nu_denom**2, 2)/2) for x in range(d)],
    #           color="blue", linestyle="dashed", legend_label="LLL simulator")
    # g += line([(x, log(gsa[x]*nu_denom**2, 2)/2) for x in range(d)],
    #           color="green", linestyle="dashed", legend_label="LLL GSA")
    # save(g, "qary-lll-sim-n100.pdf", dpi=150,  # figsize = [10, 3],
    #      axes_labels=['$i$', '$\\log_2{\\|\\mathbf{b}_i^*\\|}$'])

    g = TikzPlot()
    g.line([(x, log(exp[x], 2)/2) for x in range(d)],
           color="red", legend_label="LLL output")
    g.line([(x, log(sim[x]*nu_denom**2, 2)/2) for x in range(d)],
           color="blue", linestyle="dashed", legend_label="LLL simulator")
    g.line([(x, log(gsa[x]*nu_denom**2, 2)/2) for x in range(d)],
           color="green", linestyle="dashed", legend_label="LLL GSA")
    g.save("qary-lll-sim-n100.tikz", xmin=1, xmax=d, ymin=-2, dpi=150,  # figsize = [10, 3],
           axes_labels=['index $i$', '$\\log_2{\\|\\mathbf{b}_i^*\\|}$'])


def looking_at_lll_on_qary():
    """ Generates a q-ary basis, LLL reduces it, and prints the Gram-Schmidt
    vectors for the basis.
    """
    n, q, sd, m, nu, embedding = 30, 5, sqrt(2/3), 35, 1, "baigal"
    # n, q, sd, m, nu, embedding = 30, 5, sqrt(2/3), 35, 2, "baigal"
    nu_denom = QQ(round(nu*100)/100).denominator()
    d = m + 1
    if embedding == "baigal":
        d += n
    tries = 1
    exp = [0] * d
    for _ in range(tries):
        (lwe, samples, A, C, BC_GSO, vol) = genLWEInstance(
            n, q, sd, m, nu=nu, embedding=embedding)
        for i in range(d):
            exp[i] += BC_GSO.get_r(i, i)/tries

    from sage.all import matrix
    basis = matrix(d)
    BC_GSO.B.to_matrix(basis)
    gs = basis.gram_schmidt(orthonormal=False)[0]
    for i in range(d):
        print(gs[i])


# looking_at_lll_on_qary()
testing_z_shape_sim()
