"""
Main file for Thompson Sampling
"""

from constants import *


def sample_param(arm):
    """ Sample parameter from prior distribution """

    a, b = HYPERPARAMETERS[arm]

    return float(PRIORS[arm](a, b).rvs())


def sample(arm):
    """ Sample arm """

    return float(ARMS[arm].rvs())


def compute_mean(arm, param):
    """ Compute the expected reward for an arm with parameter param """

    return MEAN_DISTRIBUTIONS[arm](param)


def update_prior(arm, reward):
    """ Update the prior hyperparameters """

    a, b = HYPERPARAMETERS[arm]
    a, b = UPDATE_PRIORS[arm](a, b, reward)
    HYPERPARAMETERS[arm] = (a, b)


def thompson_sampling_step():
    """ One step of Thompson Sampling algorithm """

    params = np.zeros(K)
    expected_rewards = np.zeros(K)

    for k in range(K):
        # sample parameters from the prior
        params[k] = sample_param(k)
        # compute expected reward for the arm
        expected_rewards[k] = compute_mean(k, params[k])

    # print("params:", params)
    # print(expected_rewards)

    # Choose arm
    best_arm = np.argmax(expected_rewards)

    # Sample best arm
    reward = sample(best_arm)

    # Update params (depends on the conjugate priors)
    update_prior(best_arm, reward)

    return best_arm, reward


def compute_param_means():
    """ Compute mean value of the parameters of the priors (after the full algo) """

    # means of prior parameters
    PARAM_MEANS = [0] * K

    for k in range(K):
        a, b = HYPERPARAMETERS[k]
        PARAM_MEANS[k] = PRIORS[k](a, b).stats(moments='m').tolist()

    return PARAM_MEANS


def thompson_sampling():
    """ Thompson Sampling algorithm """

    gain = 0
    counters = [0] * K

    for t in range(T):
        best_arm, reward = thompson_sampling_step()
        counters[best_arm] += 1
        gain += reward

    regret = T * BEST_MEAN - gain

    PARAM_MEANS = compute_param_means()

    print('K={:d}'.format(K))
    print('Best arm: {:d} with mean {:.2f}'.format(BEST_ARM, BEST_MEAN))
    print('Hyperparameters: ', HYPERPARAMETERS)
    print('Param means: ', PARAM_MEANS)
    print('Counters: ', counters)
    print('Regret={:.2f}'.format(regret))

    # print(HYPERPARAMETERS[0][0]/HYPERPARAMETERS[0][1])
    # print(HYPERPARAMETERS[1][0]/(HYPERPARAMETERS[1][0] + HYPERPARAMETERS[1][1]))
    # print(HYPERPARAMETERS[2][0]/HYPERPARAMETERS[2][1])



thompson_sampling()
