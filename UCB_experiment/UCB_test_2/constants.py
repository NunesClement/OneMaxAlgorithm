"""
All constants useful for the UCB algorithm
"""

# import numpy.random as rd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

# number of arms
K = 3

# total time
T = 10000

# multiplicative constant in the UCB bias (\sqrt{UCB_constant * \log(t) / s})
UCB_constant = 1.5

# set of arms
ARMS = [0] * K

# set of means
MEANS = [0] * K

# choice of the arms' distributions
# ARMS[0] = st.truncnorm(0, 5)

# ARMS[0] = st.expon(scale=1 / 2.5)
ARMS[0] = st.expon(scale=1 / 2.5)
ARMS[1] = st.bernoulli(0.3)
ARMS[2] = st.poisson(0.1)

# ARMS[0] = st.randint(0, 1)  # constant = 0
# ARMS[1] = st.randint(1, 2)  # constant = 1

# print(ARMS[0])
# print(ARMS[1])
# print(ARMS[2])

# fig, ax = plt.subplots(1, 1)
# mu = 0.6
# mean, var, skew, kurt = st.poisson.stats(mu, moments='mvsk')
# x = np.arange(st.poisson.ppf(0.01, mu),
#               st.poisson.ppf(0.99, mu))
# ax.plot(x, st.poisson.pmf(x, mu), 'bo', ms=8, label='poisson pmf')
# ax.vlines(x, 0, st.poisson.pmf(x, mu), colors='b', lw=5, alpha=0.5)
# rv = st.poisson(mu)
# ax.vlines(x, 0, rv.pmf(x), colors='k', linestyles='-', lw=1,
#         label='frozen pmf')
# ax.legend(loc='best', frameon=False)
#
# plt.show()

# set of arm distributions
ARM_DISTRIBUTIONS = [0] * K

# set of mean distributions
MEAN_DISTRIBUTIONS = [0] * K

ARM_DISTRIBUTIONS[0] = lambda x: st.expon(scale=1. / x)
ARM_DISTRIBUTIONS[1] = lambda x: st.bernoulli(x)
ARM_DISTRIBUTIONS[2] = lambda x: st.poisson(x)

# true means
for k in range(K):
    MEANS[k] = ARMS[k].stats(moments='m').tolist()

# means of distributions
for k in range(K):
    MEAN_DISTRIBUTIONS[k] = lambda x, k=k: ARM_DISTRIBUTIONS[k](x).stats(moments='m').tolist()

# Best arm and best mean
BEST_ARM = np.argmax(np.array(MEANS))
BEST_MEAN = np.max(np.array(MEANS))

# Delta
DELTAS = np.array([BEST_MEAN] * K) - np.array(MEANS)

# Hyperparameters for priors

HYPERPARAMETERS = [0] * K

HYPERPARAMETERS[0] = (1, 1)
HYPERPARAMETERS[1] = (1, 1)
HYPERPARAMETERS[2] = (1, 1)

# Conjugate prior

PRIORS = [0] * K

PRIORS[0] = lambda a, b: st.gamma(a, scale=1. / b)
PRIORS[1] = lambda a, b: st.beta(a, b)
PRIORS[2] = lambda a, b: st.gamma(a, scale=1. / b)


def update_prior_exponential(a, b, x):
    """ Update prior params for exponential likelihood """

    return a + 1, b + x


def update_prior_bernoulli(a, b, x):
    """ Update prior params for exponential likelihood """

    return a + x, b + 1 - x


def update_prior_poisson(a, b, x):
    """ Update prior for poisson likelihood """

    return a + x, b + 1


UPDATE_PRIORS = [0] * K

UPDATE_PRIORS[0] = update_prior_exponential
UPDATE_PRIORS[1] = update_prior_bernoulli
UPDATE_PRIORS[2] = update_prior_poisson
