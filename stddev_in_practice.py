"""
Sample variance experiments and plots.
"""


from sage.all import variance, seed, line, show, sqrt, ceil
from sage.crypto.lwe import DiscreteGaussianDistributionIntegerSampler
from collections import OrderedDict
import math


def discretisedGaussian(mean, variance, cutoff=2**-20):
    """ Return discrete pmf approximating a N(mean, variance) distribution.
    This is essentially "discrete Gaussian" distribution over a finite subset of [-6σ, 6σ],
    used for plotting Gaussian profiles using the `line` function.

    :params mean:
    :params variance:
    :params cutoff:
    """
    sd = sqrt(variance)
    gauss = {}
    for y in range(int(100*12*sd) + 1):
        x = float(mean + (y-int(100*6*sd))/100.)
        gauss[x] = math.e**(-(x-mean)**2/(2*variance)) / \
            (sqrt(2*math.pi*variance))

    # normalise: first need to keep all values
    # only so we can correctly cut off the tail
    to_cut = []
    area = sum(gauss.values())
    for k in gauss:
        gauss[k] /= area
        if gauss[k] < cutoff:
            to_cut.append(k)

    # cut tails
    for k in to_cut:
        del gauss[k]

    # renormalise after cutoff
    area = sum(_ if _ >= cutoff else 0 for _ in gauss.values())
    for k in gauss:
        gauss[k] /= area

    return gauss


class ExperimentalStddevDistribution:
    """
    EXAMPLE:
    >>> exp = ExperimentalStddevDistribution(sqrt(2/3), 204)
    >>> p_sd = exp.compute()
    >>> exp.plot()
    """

    def __init__(self, sd, dimension):
        """
        :param sd:         theoretical standard deviation, or sampler function
        :param dimension:  dimension of the vectors sampled from DG(sd)
        """
        if callable(sd):
            self.sd = -1
            self.D = sd
        else:
            self.sd = sd
            self.D = DiscreteGaussianDistributionIntegerSampler(self.sd)
        self.dimension = dimension
        self.pmf = None

    def compute(self, tries=2**12, prng_seed=1337, verbose=True):
        """
        :param tries:       number of example vectors to sample
        :param prng_seed:   prng seed for experiment
        """
        self.pmf = {}
        with seed(prng_seed):
            for t in range(tries):
                if verbose and t % 100 == 0:
                    print("Approximating distribution of sd: %02d%%" %
                          (100.*t/tries), end="\r")
                sample = float(
                    round(100*variance([self.D() for _ in range(self.dimension)]))/100.)
                if sample not in self.pmf:
                    self.pmf[sample] = 0
                self.pmf[sample] += 1
        for key in self.pmf:
            self.pmf[key] /= float(tries)
        self.pmf = OrderedDict(sorted(self.pmf.items()))
        return self.pmf

    def mean(self):
        if not self.pmf:
            raise ValueError("Probability mass function not yet computed.")
        self._measured_mean = 0
        for v, p in self.pmf.items():
            self._measured_mean += v * p
        return self._measured_mean

    def variance(self):
        mu = self.mean()
        self._measured_variance = 0
        for v, p in self.pmf.items():
            self._measured_variance += (v - mu)**2 * p
        return self._measured_variance

    def extrapolate_gaussian(self, cutoff=2**-20):
        mean = self.mean()
        variance = self.variance()
        return discretisedGaussian(mean, variance, cutoff=cutoff)

    def plot(self):
        gauss = self.extrapolate_gaussian()
        g = line(self.pmf.items(), legend_label="measured",
                 title="Distribution of the variance of %d gaussian samples with sd %.2f" % (self.dimension, self.sd))
        g += line(gauss.items(), legend_label="extrapolated", color="red")
        return g


def get_concrete_var_distribution(
        sd,
        d,
        distribution="gaussian",
        cutoff=2**-10,
        tries=2**16,
        prng_seed=1337,
        experimental=False
    ):
    """ Compute an approximation of the probability distribution of the variance
    of a list `d` elements sampled from a DiscreteGaussian(σ).

    NOTE: The returned distribution is a Discrete Gaussian fitting of the results.

    :param sd:              standard deviation σ of the DiscreteGaussian source
    :param d:               number of elements sampled from the source
    :param cutoff:          probabilities < cutoff are rounded to 0
    :param tries:           number of lists of length `d` sampled to estimate the
                            probability distribution
    :param prng_seed:       prng seed for the experiment

    :returns:               object of the form {vaiance: prob(variance)}
    """
    # prepare and run experiment
    if experimental:
        # NOTE: only gaussian secrets supported
        exp = ExperimentalStddevDistribution(sd, d)
        exp.compute(tries=tries, prng_seed=prng_seed)
        gauss = exp.extrapolate_gaussian(cutoff=cutoff)
    else:

        if distribution == "binary":
            mu2 = mu4 = 1
        elif distribution == "ternary":
            mu2 = mu4 = 2/3
        elif distribution == "gaussian":
            mu2 = sd**2
            mu4 = 3*sd**4
        else:
            raise ValueError(
                "Sample variance compuations only implement binary, ternary and Gaussian secrets")
        mean = sd**2
        variance = sample_variances_variance(mu2, mu4, d)
        gauss = discretisedGaussian(mean, variance, cutoff=cutoff)
    return gauss


def sample_variances_mean(mu2, N):
    # https://mathworld.wolfram.com/SampleVarianceDistribution.html
    return mu2 * (N-1)/N


def sample_variances_variance(mu2, mu4, N):
    # https://mathworld.wolfram.com/SampleVarianceDistribution.html
    return ((N-1)**2 * mu4)/N**3 - ((N-1)*(N-3)*(mu2**2))/N**3


def compute_and_plot():
    from sage.all import save, sample, sqrt
    distributions = [
        # name, sampler, 2nd central moment, 4th central moment
        ("dgauss", 1, 1, 3),  # sd, mu2=sd**2, mu4 = 3*sd**4
        ("binary", lambda: sample([-1, 1], 1)[0], 1, 1),
        ("ternary", lambda: sample([-1, 0, 1], 1)[0], 2/3, 2/3),
    ]

    for N in [200, 1000]:
        print(f"Computing sample variance distribution over {N} samples.")
        print("distr\tmean\tvar\tE[var]\tE[V[var]]")
        for name, sampler, mu2, mu4 in distributions:
            exp = ExperimentalStddevDistribution(sampler, N)
            exp.compute(tries=2**12, verbose=False)
            mean = exp.mean()
            var = exp.variance()
            print("%s\t%.5f\t%.5f\t%.5f\t%.5f" % (
                name,
                mean,
                var,
                sample_variances_mean(mu2, N),
                sample_variances_variance(mu2, mu4, N),
            ))
            save(exp.plot(), f"plots/sample_variance/{name}-{N}.pdf")
        print()


if __name__ == "__main__":
    compute_and_plot()
    """
    Computing sample variance distribution.
    distr   mean    var     E[var]  E[V[var]]
    dgauss  0.99857 0.01009 0.99500 0.00995
    binary  0.99825 0.00004 0.99500 0.00005
    ternary 0.66721 0.00111 0.66333 0.00112
    """
