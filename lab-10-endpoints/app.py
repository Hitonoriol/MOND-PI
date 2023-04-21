from flask import Flask, jsonify
import matplotlib
from io_utils import *

import solving_sles  # Exercise 1
import plotting  # Exercise 2
import numerical_integration  # Exercise 3
import solving_odes  # Exercice 4
import root_finding  # Exercise 5
import linear_regression  # Exercise 6
import tree_regressors  # Exercise 7
import sarima_models  # Exercise 8
import scikit_learn_classifiers  # Exercise 9

app = Flask(__name__)
matplotlib.use('agg')


@app.route("/")
def index():
    return jsonify({"response": "API is up."})


# Exercise 1
@app.route("/solve_sle")
def solve_sle():
    return solving_sles.solve_sle(
        arg("a"),  # Matrix `A`
        arg("b"),  # Ordinate vector `b`
        inum("n", 250)  # Max matrix size for algorithm performance timesheet
    )


# Exercise 2
@app.route("/plotting/plot_function")
def plot_function():
    return plotting.plot_function(
        num("a", 0), num("b", 4),  # Plotting range [a; b]
        num("n", 500)  # Number of points to equally divide the range [a; b] into
    )


@app.route("/plotting/plot_function_with_legend")
def plot_function_with_legend():
    return plotting.plot_function_with_legend(
        num("a", 0), num("b", 4),  # Plotting range [a; b]
        num("n", 500)  # Number of points to equally divide the range [a; b] into
    )


@app.route("/plotting/plot_3d")
def plot_3d():
    return plotting.plot_3d(
        num("a", -2), num("b", 2),  # Plotting range [a; b]
        num("n", 500)  # Number of points to equally divide the range [a; b] into
    )


@app.route("/plotting/plot_isosurface")
def plot_isosurface():
    return plotting.plot_isosurface(
        num("a", -10), num("b", 10),  # Plotting range x = [a; b], y = [a; b]
        inum("n", 150)  # Number of points to equally divide the 3d space into
    )


# Exercise 3
@app.route("/integrate_function")
def integrate_function():
    return numerical_integration.integrate_function(
        num("a", 0), num("b", 1)  # Integration range
    )


# Exercice 4
@app.route("/solve_ode")
def solve_ode():
    return solving_odes.solve_ode(
        num("y0", 1),  # Initial value
        num("a", 0), num("b", 10),  # Integration range
        num("step", 0.01)  # Integration step
    )


# Exercise 5
@app.route("/find_root")
def find_root():
    return root_finding.find_root(
        num("a", 1.3),
        num("b", 1.5),
        num("err", 1e-8)
    )


# Exercise 6
@app.route("/linear_regression")
def do_linear_regression():
    return linear_regression.linear_regression()


# Exercise 7
@app.route("/tree_regression")
def tree_regression():
    return tree_regressors.tree_regressors(
        inum("min_estimators", 1),  # Number of estimators to start demonstration with
        inum("max_estimators", 512),  # Max estimators
        inum("mul_step", 4)  # Number to multiply the number of estimators by after each iteration
    )


# Exercise 8
@app.route("/sarima_models")
def sarima():
    return sarima_models.sarima_models(
        inum("start_p", 0), inum("start_q", 0), inum("start_P", 0), inum("start_Q", 0),
        inum("max_p", 1000), inum("max_q", 1000), inum("max_P", 1000), inum("max_Q", 1000),
        inum("maxiter", 100),
        bln("stepwise", True)
    )


# Exercise 9
@app.route("/scikit_learn_classifiers")
def sklearn_classifiers():
    return scikit_learn_classifiers.scikit_learn_classifiers(
        num("test_size", 0.2), inum("n_estimators", 100),
        inum("max_depth", 1), num("learning_rate", 1.0)
    )
