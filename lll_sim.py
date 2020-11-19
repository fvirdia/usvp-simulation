"""
LLL Z-shape simulator.
"""


from math import log, floor, e


def lll_simulator(dim, qrank, q, scale=1):
    """LLL basis profile simulator for q-ary lattices.

    It simulates the output of LLL on a basis in HNF with the q-vectors positioned
    as first basis vectors.

    GSA: ||b_i^*|| = alpha^(i-1) ||b_1||
         log(b_i*) = log(b_1) + (i-1) * log(alpha)

    If basis B is derived from a [BG14b] embedding with scaling factor `scale`,
    it factors out the scaling factor, and adds it back on the last step:
    v = c * B = c * scale * (B/scale) = scale * c * (B/scale)
    - B/scale is HNF with module q/scale. compute LLL simulation for that, and
      scale output vectors by `scale`

    :params dim:    lattice rank
    :params qrank:  number of q-vectors in the basis
    :params q:      value of q
    :params scale:  if working with a [BG14b] basis, this is the scaling factor

    :returns r:     squared norms of the GS vectors for the LLL-reduced basis
    """

    # LLL root hermite factor. log(alpha) = -2 log(delta)
    delta = 1.0219
    log_alpha = -2 * log(delta)
    log_vol = qrank * log(q/scale)

    # assumption: after the q-vectors, the slope of the basis vectors will
    #  be the GSA slope until all the volume is covered. no vector has norm < 1.
    #  remaining GS vectors have unit norm

    # compute a GSA slope from q until vector norms reach 1
    slope = []
    cur_norm = log(q/scale)
    while cur_norm > 0:
        slope.append(cur_norm)
        cur_norm += log_alpha

    # head vectors contribute log(q/scale) to the log volume each
    # log(vol) = head * log(q/scale) + sum_i slope[i]
    head = floor((log_vol - sum(slope))/log(q/scale))

    # compute missing volume and distribute it on slope
    missing_vol = log_vol - head * log(q/scale) - sum(slope)
    slope = [slope[0]] + [slope_i + missing_vol /
                          (len(slope)-1) for slope_i in slope[1:]]

    # compute number of remaining vectors will have unit norm
    tail = dim - head - len(slope)

    # form log profile
    r = head * [log(q/scale)] + slope + tail * [0]

    # if tail < 0, we may have to accumulate any cut volume onto last vector
    #   this seems to happen for example when simulating the smallest of the AGVW17
    #   parameter sets
    r = r[:dim-1] + [sum(r[dim-1:])]

    # safety check
    assert(dim == len(r))
    # print("vol diff", log_vol - sum(r))

    # compute square norms, and rescale up vector norms to account for [BG14b]
    r = [scale**2 * e**(2*ri) for ri in r]

    return r
