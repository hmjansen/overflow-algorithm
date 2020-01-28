"""Creates overflow networks.

This module contains functions that create (parameters defining) overflow
networks. These networks are used in Example 1 and Example 2 in the paper at
https://arxiv.org/pdf/2001.09611v1, referred to as the overflow paper.
"""


import numpy as np


def square_network_example(m, delta, epsilon):
    """Creates the network in Example 1.

    Calculates the parameters that constitute the network in Example 1 of the
    overflow paper for a given set of parameters.

    Args:
        m: Number of cells for the network of Example 1.
        delta: The delta parameter for the network of Example 1.
        epsilon: The epsilon parameter for the network of Example 1.

    Returns:
        alpha: Array (row vector) of exogenous arrival rates.
        mu: Array (row vector) of service capacities.
        P: Substochastic routing matrix.
        Q: Substochastic overflow matrix.
    """
    n = 4*(m**2)
    alpha = np.zeros(n)
    alpha[0] = n
    mu = np.zeros(n) + 1.0
    P = square_network_P_matrix(m, delta)
    Q = square_network_Q_matrix(m, epsilon)
    return alpha, mu, P, Q


def square_network_P_matrix(m, delta):
    """Creates routing matrix of the network in Example 1.

    Calculates the routing matrix P of the network in Example 1 of the
    overflow paper for a given set of parameters.

    Args:
        m: Number of cells for the network of Example 1.
        delta: The delta parameter for the network of Example 1.

    Returns:
        P: Substochastic routing matrix.
    """
    n = 4*(m**2)
    P = np.zeros((n, n))
    for i in range(n):
        if i % 4 == 0:
            # Internal cell flows.
            P[i, i + 1] = 0.4
            P[i, i + 3] = 0.4
            # External cell flows.
            if not i % (4*m) == 0:
                P[i, i - 3] = 0.19 * delta
        elif i % 4 == 1:
            # Internal cell flows.
            P[i, i - 1] = 0.1
            P[i, i + 1] = 0.1
            # External cell flows.
            if not i % (4*m) == (-3) % (4*m):
                P[i, i + 3] = 0.79*delta
        elif i % 4 == 2:
            # Internal cell flows.
            P[i, i - 1] = 0.4
            P[i, i + 1] = 0.4
            # External cell flows.
            if i % (4*m) == (-2) % (4*m) and i + 4*m < n:
                P[i, i + (4*m) - 1] = 0.19*delta
        else:
            # i % 4 == 3
            # Internal cell flows.
            P[i, i - 3] = 0.5
            P[i, i - 1] = 0.5
    return P


def square_network_Q_matrix(m, epsilon):
    """Creates overflow matrix of the network in Example 1.

    Calculates the overflow matrix Q of the network in Example 1 of the
    overflow paper for a given set of parameters.

    Args:
        m: Number of cells for the network of Example 1.
        epsilon: The epsilon parameter for the network of Example 1.

    Returns:
        Q: Substochastic overflow matrix.
    """
    n = 4*(m**2)
    Q = np.zeros((n, n))
    for i in range(n):
        if i % 4 == 0:
            # Internal cell flows.
            Q[i, i + 1] = 0.05
            Q[i, i + 2] = 0.9
            Q[i, i + 3] = 0.05
        elif i % 4 == 1:
            # External cell flows.
            if not i % (4*m) == (-3) % (4*m):
                Q[i, i + 3] = epsilon
        elif i % 4 == 2:
            # Internal cell flows.
            Q[i, i - 1] = 0.0
            Q[i, i + 1] = 0.1
            Q[i, i - 2] = 0.9
        else:
            # i % 4 == 3
            # External cell flows.
            if i + 4*m < n:
                Q[i, i + 4*m - 3] = epsilon
    return Q


def worst_case_example(n):
    """Creates the network in Example 2.

    Calculates the parameters that constitute the network in Example 2 of the
    overflow paper for a given number of nodes.

    Args:
        n: Number of nodes in the network of Example 2.

    Returns:
        alpha: Array (row vector) of exogenous arrival rates.
        mu: Array (row vector) of service capacities.
        P: Substochastic routing matrix.
        Q: Substochastic overflow matrix.
    """
    alpha = np.zeros(n)
    alpha[0] = 1
    mu = np.zeros(n)
    mu[n - 1] = 1 / (2 * n)
    for i in range(n-1):
        mu[i] = 1 + (n - (i + 1)) / (n * (n + 1))
    P = np.eye(n, k=1)
    Q = np.eye(n, k=-1) * (1 - 2**(-(n + 1)))
    return alpha, mu, P, Q
