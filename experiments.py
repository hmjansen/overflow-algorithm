"""Code used to create the examples in the overflow paper.

This module contains the code used to create the examples in the paper at
https://arxiv.org/pdf/2001.09611v1, referred to as the overflow paper.
Execute this module as a script to reproduce the examples. The code has been
tested using Python 3.5.2, NumPy 1.11.1, and SciPy 0.18.1 via the
Anaconda 4.2.0 (64-bit) installation.
"""


import matplotlib.pyplot as plt
import numpy as np
from create_overflow_networks import square_network_example
from create_overflow_networks import worst_case_example
from overflow_algorithm import solve_overflow_traffic_equation


def progress_bar(i, n, width=20):
    """Provides a simple progress bar.

    Visualizes the how many iterations of a loop have been completed. Does not
    take actual or estimated remaining computation time into account.

    Args:
        i: Number of iterations of the loop that are completed.
        n: Total number of iterations of the loop.
        width: The number of characters used to visualize the progress.
    """
    STRING_L = "X"
    STRING_R = "."
    frac = int(np.floor((i / n) * width))
    str_l = STRING_L * frac
    str_r = STRING_R * (width - frac)
    print("\rSolving: [" + str_l + str_r + "]", end="")


def square_network_overflow_plot(m, grid_nr, show_progress=False):
    """Creates a heat map as in Example 1 of the overflow paper.

    First defines a grid of parameter values for the delta and epsilon
    parameters of the family of networks presented in Example 1. Computes the
    solution to the overflow equation for each pair of parameters delta and
    epsilon and determines the fraction of overflowing nodes for that pair.
    Plots a heat map of these fractions.

    Args:
        m: Number of cells for the network of Example 1.
        grid_nr: Number of parameter values used for each parameter not
            including 0.0. The grid consists of grid_nr + 1 evenly spaced
            points in [0, 1] with end points 0.0 and 1.0.
        show_progress: Indicates whether a progress bar is shown.
    """
    n = 4*(m**2)
    alpha = np.zeros(n)
    alpha[0] = n
    mu = np.zeros(n) + 1.0
    grid_nr = grid_nr + 1
    delta_space = np.linspace(0, 1.0, grid_nr)
    epsilon_space = np.linspace(0, 1.0, grid_nr)
    overflow_proportion = np.zeros([grid_nr, grid_nr])
    if show_progress:
        print("")
    for i in range(grid_nr):
        for j in range(grid_nr):
            delta = delta_space[i]
            epsilon = epsilon_space[j]
            alpha, mu, P, Q = square_network_example(m, delta, epsilon)
            sol = solve_overflow_traffic_equation(alpha, mu, P, Q)
            overflow_proportion[i, j] = sum(sol >= mu) / n
            if show_progress:
                progress_bar(i * grid_nr + (j + 1), grid_nr**2)
    if show_progress:
        print("")

    # Plot results.
    # Set font sizes for plots.
    SMALL_FONT = 8
    MEDIUM_FONT = 18
    LARGE_FONT = 22
    plt.rc('font', size=SMALL_FONT)  # Default text font size
    plt.rc('axes', titlesize=MEDIUM_FONT)  # Axes titles font size
    plt.rc('axes', labelsize=LARGE_FONT)  # Axes lables font size
    plt.rc('xtick', labelsize=MEDIUM_FONT)  # xtick lables font size
    plt.rc('ytick', labelsize=MEDIUM_FONT)  # ytick lables font size
    plt.rc('legend', fontsize=MEDIUM_FONT)  # legend font size
    plt.rc('figure', titlesize=MEDIUM_FONT)  # Figure title font size

    # Flip the matrix for better plotting of the vertical axis.
    overflow_proportion = np.flipud(overflow_proportion)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.matshow(overflow_proportion)

    # Set ticks at horizontal and vertical axes;
    # the yticks are flipped to correspond with the flipped matrix.
    ticks_max = 6
    ticks_step = np.floor((grid_nr - 1) / (ticks_max - 1))
    ticks_index = np.arange(start=0, stop=grid_nr, step=ticks_step, dtype=int)
    ax.set_xticks(ticks_index)
    ax.set_yticks(np.flipud(ticks_index))
    # Label ticks with the corresponding values.
    delta_labels = [delta_space[i] for i in ticks_index]
    epsilon_labels = [epsilon_space[i] for i in ticks_index]
    ax.set_xticklabels(delta_labels)
    ax.set_yticklabels(epsilon_labels)
    # Show ticks on horizontal axis at the bottom.
    ax.tick_params(axis="x", bottom=True, top=False,
                   labelbottom=True, labeltop=False)
    # Set x label and y label.
    plt.xlabel("$\delta$")
    plt.ylabel("$\epsilon$")
    # Plot the figure.
    fig.colorbar(im)
    plt.show()


if __name__ == "__main__":
    show_progress = True  # Progress bar will be shown if True

    # Number of grid points used for the parameter space of the square network.
    # Choose grid_nr >= 1.
    # Step size is 1 / grid_nr for a total number of 1 + grid_nr grid points.
    grid_nr = 100
    # Number of blocks in the horizontal direction of the square network.
    # Choose m >= 2.
    m = 3
    # Run the square network example with m = 3.
    square_network_overflow_plot(m, grid_nr, show_progress)

    # Run the square network example with m = 5.
    m = 5
    square_network_overflow_plot(m, grid_nr, show_progress)

    # Check that the worst case examples indeed require the maximum number of
    # iterations of the overflow algorithm.
    n = 75
    results = np.zeros(n)
    if show_progress:
        print("")
    for i in range(1, n + 1):
        alpha, mu, P, Q = worst_case_example(i)
        x = solve_overflow_traffic_equation(alpha, mu, P, Q, count=True)
        labda, solve_count, max_count = x
        results[i - 1] = (solve_count == max_count)
        if show_progress:
            progress_bar(i, n)
    if show_progress:
        print("")
    if results.all():  # Every solve_count == max_count
        print("All solutions required the maximum number of iterations.")
    else:
        print("At least one solution required less than the maximum number " +
              "of iterations.")
