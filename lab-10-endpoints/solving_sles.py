import math
import numpy as np
import numpy.linalg as lin
import time
from numpy.random import default_rng
from io_utils import *


# A function for measuring time spent executing arbitrary functions:
def measure_elapsed_time(action, *args):
    start = time.time_ns()
    action(*args)
    elapsed_ms = (time.time_ns() - start) / 1000000
    return elapsed_ms


# A function to print SLE solutions:
def print_solution(out, method, a, b, x):
    out.out(f"<h3>SLE solution ({method}) (A[{a.shape[0]}x{a.shape[1]}], b[{b.size}]):</h3> ", end="")
    for n in range(x.size):
        out.out(f"x{n + 1} = {x[n]}")
    out.out()


def load_coef_matrix(mat_str):
    numbers = mat_str.split()
    size = int(math.sqrt(len(numbers)))
    coefs = np.zeros((size, size))
    for idx in range(len(numbers)):
        i = idx // size
        j = idx % size
        coefs[i][j] = float(numbers[idx])
    return coefs


def load_ordinate_vector(vec_str):
    return np.array([float(n) for n in vec_str.split()])


# And solve the system using numpy.linalg.solve:
def solve_numpy(a, b):
    return lin.solve(a, b)


# 2. Solve the given SLE using Cramer's Rule and numpy.linalg.det
def solve_cramer(a, b):
    n = b.size  # Number of variables in the SLE
    x = []  # Solution vector x
    det_a = lin.det(a)  # Determinant of the coefficient matrix

    # Create matrices A_n by sequentially replacing the n'th column by the ordinate vector B
    # Then calculate det(A_n) and divide it by det(A) to get the n'th SLE solution
    for i in range(n):
        an = a.copy()
        for k in range(n):
            an[k][i] = b[k]
        x.append(lin.det(an) / det_a)

    return np.array(x)


# 3. Build a timesheet for different numbers of rows in a
# square matrix. I.e., compare computing time of both methods
# for a size with 10, 20, 30, ..., 500 rows.

def timesheet(out, n):
    elapsed = []
    rng = default_rng(1)  # Set seed for values to be the same on every generation
    methods = [solve_numpy, solve_cramer]

    min_size = 10
    max_size = n + min_size
    out.out(f"<h3>Timesheet for matrix sizes {min_size}x{min_size} to {n}x{n}:</h3>", end="")

    tab = "        "
    out.out(f"<b>N</b>    <b>solve_numpy</b>    <b>solve_cramer</b>    <b>delta% (second/first)</b>")

    # Compute the SLE solution for a randomly generated matrix A[NxN] and vector B[N]
    # using two methods defined above and measure execution time.
    for size in range(min_size, max_size, min_size):
        out.out(size, end=tab)
        method_elapsed = []
        for method in methods:
            a = rng.random((size, size))
            b = rng.random((size))
            elapsed_ms = measure_elapsed_time(method, a, b)
            method_elapsed.append(elapsed_ms)
            out.out(f"{elapsed_ms} ms", end=tab)
            elapsed.append(method_elapsed)
        delta = 100 * (method_elapsed[1] / max(1, method_elapsed[0]))
        out.out(f"{delta}%")


def solve_sle(coef_str, ord_str, n=250):
    out = Output()
    a = load_coef_matrix(coef_str)
    b = load_ordinate_vector(ord_str)
    out.out(f"<h3>System matrix and ordinate vector:</h3>A:\n{a}\nB:\n{b}\n")
    print_solution(out, "numpy.linalg.solve", a, b, solve_numpy(a, b))
    print_solution(out, "Cramer's Method", a, b, solve_cramer(a, b))
    timesheet(out, n)
    return out.get()
