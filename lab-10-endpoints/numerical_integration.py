from math import cos, sin, exp, pi
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
import inspect

from io_utils import *


def print_result(out, result, interval):
    a, b = interval
    I, err = result
    out.out(
        f"Numerical integration result:\n∫[{a}..{b}]f(x)dx ≈ {I} ± {err}\n"
    )


# 4. It is a good practice to plot the integrand function to check whether it
# is “well behaved” before you attempt to integrate. Write a function with name
# plotquad which takes the same arguments as the quad command (i.e. f, a and b)
# and which (I) creates a plot of the integrand f(x) and (II) computes the integral
# numerically using the quad function. The return values should be as for the quad
# function.

def plotquad(out, f, a, b):
    # Integrate f(x) numerically on [a, b]
    I, err = quad(f, a, b)
    # Create plot point arrays and plot the function
    x = np.arange(a, b, (b - a) / 100)
    f_x = np.array([f(xi) for xi in x])
    fig, ax = plt.subplots()
    ax.plot(x, f_x, label="f(x)")
    ax.fill_between(x, f_x, alpha=0.25)
    ax.set(xlabel='x', ylabel="f(x)", title=f"{inspect.getsourcelines(f)[0][0]}")
    ax.grid()
    ax.legend()
    fig.text(0.25, -0.025, f"∫[{a}..{b}]f(x)dx ≈ {I:.6f} ± {err:.6E}")
    out.plot(fig)
    return I, err


def integrate_function(a=0, b=1):
    out = Output()
    f = lambda x: cos(2 * pi * x)

    # 1. Using scipy’s quad function, write a program that solves
    # the following integral numerically: I = ∫[0..1]cos(2*π*x)dx.
    print_result(out, quad(f, a, b), (a, b))

    # Test plotquad() with the function from task 1
    out.out("\nplotquad() demonstration:")
    print_result(out, plotquad(out, f, a, b), (a, b))

    # Test with other functions
    f = lambda x: 2 / (1 + exp(-x)) - 1
    print_result(out, plotquad(out, f, a, b), (a, b))

    f = lambda x: exp(x) * sin(pi * x)
    print_result(out, plotquad(out, f, a, b), (a, b))
    return out.get()
