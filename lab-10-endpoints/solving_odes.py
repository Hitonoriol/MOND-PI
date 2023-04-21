from scipy.integrate import odeint
from numpy import exp, sin, cos
import matplotlib.pyplot as plt
import numpy as np
import pylab

from io_utils import *


# 1. Write a program that computes the solution y(t) of this ODE
# using the odeint algorithm:
# dy / dt = −exp(−t) * (10 * sin(10 * t) + cos(10 * t)) for t = 0..10.
# The initial value is y(0) = 1.

def solve_ode(y0=1, a=0, b=10, step=0.01):
    out = Output()

    f = lambda y, t: -exp(-t) * (10 * sin(10 * t) + cos(10 * t))

    t = np.arange(a, b + step, step)  # Inclusive integration range [a; b]
    y = odeint(f, y0, t)  # Solve the ODE dy / dt = f(y, t)

    out.out(f"Solving dy / dt = −exp(−t) * (10 * sin(10 * t) + cos(10 * t)) for t = {a}..{b}; y0 = {y0}...")
    out.out(f"Computed points of y(t) for each t = {a}..{b} with step = {step}:")
    out.out(f"{y}\n")

    # 2. You should display the solution graphically at points
    # t = 0, t = 0.01, t = 0.02, ..., t = 9.99, t = 10.

    out.out(f"Values of `t` to plot y(t) for:\n{t}")
    fig, ax = plt.subplots()
    ax.plot(t, y, label="y(t)")
    ax.set(xlabel='t', ylabel="y(t)", title="ODE Solution")
    ax.grid()
    out.plot(fig)
    return out.get()
