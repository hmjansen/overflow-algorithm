"""Overflow algorithm.

This module implements the overflow algorithm presented as Algorithm 2 in the
paper https://arxiv.org/pdf/2001.09611v1, referred to as the overflow paper.
The implementation is in two functions. The first function
solve_linearized_equation() solves the linearized overflow equation (6) of the
overflow paper and is used in the inner loop of the second function. The second
function solve_overflow_traffic_equation() implements the actual overflow
algorithm presented as Algorithm 2 in the overflow paper.
"""


import copy
import numpy as np
import scipy.linalg as linalg


def solve_linearized_equation(n, alpha, mu, P, Q, A_set, B_set):
    """Solves the linearized overflow equation.

    Solves the linearized overflow equation (6) of the overflow paper. It is
    assumed and not checked that the parameters satisfy Condition 2 of the
    overflow paper. If this condition is not satisfied, a solution may be
    returned anyway. This solution may not be nonnegative nor unique.

    Args:
        n: Number of nodes of the network.
        alpha: Array of size nx1 of exogenous arrival rates.
        mu: Array of size nx1 of service capacities.
        P: Substochastic routing matrix of size nxn.
        Q: Substochastic overflow matrix of size nxn.
        A_set: Set of potentially stable nodes.
        B_set: Set of overflowing nodes.

    Returns:
        y: Array of size nx1 that solves the linearized overflow equation (6).
    """
    matrix_dim = [n, n]
    alpha = np.asarray(alpha)
    mu = np.asarray(mu)
    P = np.asarray(P)
    Q = np.asarray(Q)
    P_A = np.zeros(matrix_dim)
    Q_B = np.zeros(matrix_dim)
    for i in range(n):
        if i in A_set:
            for j in range(n):
                P_A[i, j] = P[i, j]
        if i in B_set:
            for j in range(n):
                Q_B[i, j] = Q[i, j]
    # Zero out row i of the matrix P if i is in A.
    P_A_bar = P - P_A
    I = np.eye(n)
    # Solve linear equation Xy = z.
    X = np.transpose(I - P_A - Q_B)
    z = np.transpose(alpha + mu @ (P_A_bar - Q_B))
    y = linalg.solve(X, z)
    # Turn y into a row vector.
    y = np.transpose(y)
    return y


def solve_overflow_traffic_equation(alpha, mu, P, Q, count=False):
    """Solves the overflow equation.

    Solves the overflow equation (3) of the overflow paper. It is assumed and
    not checked that the parameters satisfy Condition 2 of the overflow paper.
    If this condition holds, then the solution is nonnegative and unique. If
    this condition does not hold, a solution may be returned anyway. In this
    case the solution is not necessarily nonnegative and unique.

    Args:
        n: Number of nodes of the network.
        alpha: Array of size nx1 of exogenous arrival rates.
        mu: Array of size nx1 of service capacities.
        P: Substochastic nxn routing matrix.
        Q: Substochastic nxn overflow matrix.
        count: If true, the return value also includes iteration counts.

    Returns:
        labda: Array of size nx1 that solves the overflow equation (3).
        solve_count: The number of iterations required to compute labda.
        max_count: The maximum number of iterations required to compute labda.
    """
    n = len(alpha)
    alpha = np.asarray(alpha)
    mu = np.asarray(mu)
    P = np.asarray(P)
    Q = np.asarray(Q)
    solve_count = 0
    N_set = set(range(n))
    B_set = set()  # empty set
    previous_B_set = copy.deepcopy(N_set)
    while B_set != previous_B_set:
        previous_B_set = copy.deepcopy(B_set)
        A_set = set()  # empty set
        previous_A_set = copy.deepcopy(N_set)
        while A_set != previous_A_set:
            previous_A_set = copy.deepcopy(A_set)
            labda = solve_linearized_equation(n, alpha, mu, P, Q, A_set, B_set)
            A_set = set([i for i in range(n) if labda[i] < mu[i]])
            solve_count += 1
        B_set = N_set - A_set
    # Check whether an actual solution of the overflow equation is found.
    zeros = np.zeros(n)
    labda_check = (alpha + np.minimum(labda, mu) @ P +
                   np.maximum(labda - mu, zeros) @ Q)
    if not np.allclose(labda, labda_check):
        print("\nWarning: solution may be incorrect\n")
    if count:
        max_count = int(1 + (0.5 * n * (n + 1)))
        return labda, solve_count, max_count
    else:
        return labda
