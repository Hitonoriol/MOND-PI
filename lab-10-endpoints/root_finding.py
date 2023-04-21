from scipy.optimize import bisect
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy.optimize import fsolve

from io_utils import *


def plot(out, f, a, b):
    x = np.arange(a, b, (b - a) / 100)
    f_x = np.array([f(xi) for xi in x])
    fig, ax = plt.subplots()
    ax.fill_between(x, f_x, alpha=0.25)
    ax.plot(x, f_x, label="f(x)")
    ax.set(xlabel="x", ylabel="y")
    ax.grid()
    out.plot(fig)


def find_root(a=1.3, b=1.5, err=1e-8):
    out = Output()

    # 1. Write a program to determine an approximation of √2 by finding
    # a root x of the function f(x) = 2 - x^2 using the bisection algorithm.
    # Choose a tolerance for the approximation of the root of 10^(−8).
    f = lambda x: 2 - x ** 2  # Function f(x)
    x0 = bisect(f, a, b, xtol=err)  # Find the root using bisection method
    out.out("Approximating √2 via bisection:")
    out.out(f"Root of f(x) = 2 - x^2: x_0 ≈ {x0} ± {err}\n")
    plot(out, f, a, b)

    # 3. Compute the value of √2 using math.sqrt(2) and compare this with
    # the approximation of the root. How big is the absolute error of x?
    # How does this compare with xtol?
    exact_x0 = sqrt(2)
    exact_err = abs(exact_x0 - x0)

    out.out("Comparing sqrt(2) to its approximation via f(x) = 2 - x^2:")
    out.out(f"Exact error: {exact_err:.6e}")
    out.out(f"xtol - exact_err = {err - exact_err:.6e}\n")

    # 4. Write a program to determine an approximation of √2 by finding
    # a root x of the function f(x) = 2 − x^2 using the fsolve function.
    # Compare the result of the fsolve function with the result of the bisection algorithm.

    fsolve_x0 = fsolve(f, 1.3)  # Find the root using `fsolve`
    fsolve_exact_err = exact_x0 - fsolve_x0
    fsolve_bisect_err = exact_err - fsolve_exact_err

    out.out("fsolve():")
    out.out(f"Root of f(x) = 2 - x^2: x_0 ≈ {fsolve_x0}")
    out.out(f"Exact error: {fsolve_exact_err}")
    out.out(f"Error relative to bisection algorithm result: {fsolve_bisect_err}")
    return out.get()
