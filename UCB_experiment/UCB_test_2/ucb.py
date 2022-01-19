
"""
Main file for UCB algorithm
"""

from math import *
import pylab as plt

from constants import *


# array with the number of times each arm has been sampled
counters = [0] * K

# array of empirical means
emp_means = [0] * K

# Gain and expected gain
gains = [0, 0]


def new_empirical_mean(old_mean, old_nb, value):
    """
    Compute the new empirical mean of arm k when arm k is sampled again
    old_nb is the previous number of times arm k has been sampled
    """

    return (old_mean * old_nb + value) / (old_nb + 1)


def sample(k):
    """
    Sample arm k
    """
    return float(ARMS[k].rvs())


def update_arm(k):
    """
    Update arm k
    """

    val = sample(k)
    # nouvelle moyenne empirique par opérateur [0.41, 0.23, 0.50]
    emp_means[k] = new_empirical_mean(emp_means[k], counters[k], val)

    counters[k] += 1
    gains[0] += val
    gains[1] += MEANS[k]


def init():
    """
    Initialize by sampling each arm once
    """

    for k in range(K):
        update_arm(k)


def ucb_step(t):
    """
    Realize one UCB step
    """
    # optimistic_means est un biais utile pour faire décroitre le regret
    optimistic_means = np.array(emp_means) + np.sqrt(UCB_constant * log(t) / np.array(counters))
    k = np.argmax(optimistic_means)
    update_arm(k)


def ucb():
    """
    Run the full UCB algorithm
    """

    init()
    for t in range(K, T):
        ucb_step(t)


def regret(t=T):
    """
    Return the regret
    """

    return BEST_MEAN * t - gains[1]


def runUCB():
    """
    Run UCB algorithm on one instance and display results
    """

    ucb()
    R = regret()

    print('K={:d}'.format(K))
    print('Best arm: {:d} with mean {:.2f}'.format(BEST_ARM, BEST_MEAN))
    print('Means: ', MEANS)
    print('Counters: ', counters)
    print('Regret: {:.2f}'.format(R))


def plotUCB():
    """
    Plot the regret vs T curve
    """

    regrets = [0]
    times = [0]

    init()
    times.append(K)
    regrets.append(regret(K))

    for t in range(K + 1, T + 1):
        ucb_step(t)
        if t % 10 == 0:
            times.append(t)
            regrets.append(regret(t))

    times = np.array(times)
    regrets = np.array(regrets)

    regrets_log = regrets / np.log(times)
    regrets_Tlog = regrets / np.sqrt(np.log(times) * times)

    plt.figure(1)
    plt.plot(times, regrets)
    plt.figure(2)
    plt.plot(times, regrets_log)
    # plt.figure(3)
    # plt.plot(times, regrets_Tlog)
    plt.show()


if __name__ == "__main__":
    runUCB()
    # plotUCB()