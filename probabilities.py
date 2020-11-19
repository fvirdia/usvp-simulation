"""
Probability utilities.
"""


from sage.all import gamma_inc_lower, RR, exp, gamma, cached_function


@cached_function
def pdf_chi_sq(x, n):
    """ Return f(x), where f is the density function of a chi-squared distribution
    with n degrees of freedom.
    """
    x = RR(x)
    n = RR(n)
    return x**(n/2-1)*exp(-x/2)/(2**(n/2) * gamma(n/2))


@cached_function
def cmf_chi_sq(x, n):
    """ Return P[X <= x], where X is chi-squared distributed with n degrees of freedom.
    """
    x = RR(x)
    n = RR(n)
    return gamma_inc_lower(n/2, x/2)/gamma(n/2)


def pmf_from_cmf(cmf, force=False):
    """ Given a cumulativie mass function in the form of a dictionary { x: P[ X <= x] },
    return a probability mass function in the form of a dictionary { x: P[x] }.

    If `force == True`, we force probabilities to be non-negative, in case the input
    supposed cmf decreases at some point.

    NOTE: it normalises the PMF to add up to 1.
    """
    pmf = {}
    l = list(cmf.items())
    l.sort()

    pmf[l[0][0]] = l[0][1]
    total = pmf[l[0][0]]
    for i in range(1, len(l)):
        pmf[l[i][0]] = l[i][1]-l[i-1][1]
        if pmf[l[i][0]] < 0:
            print("WARNING: cmf decreasing")
            if force:
                pmf[l[i][0]] = 0

        total += pmf[l[i][0]]

    for k in pmf:
        pmf[k] /= total

    if total != 1:
        print("WARNING: extrapolating pmf from incomplete cmf")
        print("         pmf aggiusted by factor %.10f" % total)

    return pmf


def kth_moment(pmf, k):
    """ Given a probability mass function in the form of a dictionary { x: P[x] },
    return the k-th moment.
    """
    res = 0
    for x in pmf:
        res += x**k * pmf[x]
    return res
