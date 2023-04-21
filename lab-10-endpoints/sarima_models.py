import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pmdarima as pm
from pmdarima import model_selection

from io_utils import *

x_test, y_test = [], []
x_train, y_train = [], []
feature_names = []
x, y = [], []
y_name = ""
x_points = []
x_test_points, x_train_points = [], []


def join_date(df):
    return pd.to_datetime(
        df['month'].map(str) + '-' + df['year'].map(str),
        format='%m-%Y'
    ).dt.strftime('%m-%Y')


def test_model(
        out, method,
        start_p, start_q, start_P, start_Q,
        max_p, max_q, max_P, max_Q,
        maxiter,
        stepwise,
):
    global x, y, x_train, x_test, y_train, y_test, feature_names, y_name, x_points, x_test_points, x_train_points

    # Fit an ARIMA model using our time series
    mod = pm.auto_arima(
        y_train,
        seasonal=True, m=12,  # Our series contains monthly data
        start_p=start_p, start_q=start_q, start_P=start_P, start_Q=start_Q,
        max_p=max_p, max_q=max_q, max_P=max_P, max_Q=max_Q,
        method=method,
        stepwise=stepwise,
        maxiter=maxiter,
        trace=True,
        suppress_warnings=True, error_action='ignore'
    )

    out.out(mod.summary())

    # Use the model to generate predictions
    pred, conf_int = mod.predict(n_periods=y_test.shape[0], return_conf_int=True)

    # Evaluate the correctness of the resulting predictions
    out.out("\nActual vs Predicted data:\nMonth     Actual       Predicted")
    for i in range(len(pred)):
        out.out(f"{x_test_points[i]} | {y_test[i]:.3f}    {pred[i]:.3f}")
    out.out(f"Test Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, pred)):.3f}")

    # Plot raw data and predictions obtained from the model
    fig, ax = plt.subplots(figsize=(15, 7))
    plt.xticks(rotation=90)
    ax.set(
        xlabel="Month-Year",
        ylabel="Distance",
        title="Distance(Year, Month) for USA Monthly Domestic Jet Flights"
    )

    ax.plot(x_train_points, y_train, label="Training data")
    ax.plot(
        x_test_points,
        pred,
        label="Predicted data")
    ax.scatter(x_test_points, y_test, marker='x', label="Testing data")
    ax.grid()
    ax.legend()
    ax.fill_between(
        x_test_points,
        conf_int[:, 0], conf_int[:, 1],
        alpha=0.1, color='b'
    )
    out.plot(fig)


def sarima_models(
        start_p=0, start_q=0, start_P=0, start_Q=0,
        max_p=1000, max_q=1000, max_P=1000, max_Q=1000,
        maxiter=100,
        stepwise=True,
):
    global x, y, x_train, x_test, y_train, y_test, feature_names, y_name, x_points, x_test_points, x_train_points
    out = Output()

    dst_col, year_col, mon_col = "distance", "year", "month"
    # Load the monthly statistics dataset and select only the 3 needed columns from it
    data = pd.read_csv(
        "domestic-jet-flights-usa/monthly-stats.csv"
    )[[dst_col, year_col, mon_col]]

    # Prepare the data
    x_df = data[[year_col, mon_col]]
    feature_names = x_df.columns
    y_name = dst_col
    y = data[dst_col].values
    x = x_df.values

    data_train = data[(data["year"] == 2017) | (data["year"] == 2018)]
    data_test = data[data["year"] == 2019]
    y_train, x_train = data_train[dst_col].values, data_train[[year_col, mon_col]].values
    y_test, x_test = data_test[dst_col].values, data_test[[year_col, mon_col]].values

    x_points = join_date(data).values
    x_train_points = join_date(data_train).values
    x_test_points = join_date(data_test).values

    out.out(f"Training data:\n{data_train}\n")
    out.out(f"Testing data:\n{data_test}\n")
    out.out(f"Training y:\n{y_train}\n\nTraining x:\n{x_train_points}\n")
    out.out(f"Testing y:\n{y_test}\n\nTesting x:\n{x_test_points}\n")

    # a) Let's first test the limited-memory Broyden-Fletcher-Goldfarb-Shanno method
    # (default when using auto_arima with method keyword unspecified):
    out.out("<h3>Broyden-Fletcher-Goldfarb-Shanno:</h3>")
    test_model(
        out, "lbfgs",
        start_p, start_q, start_P, start_Q,
        max_p, max_q, max_P, max_Q,
        maxiter,
        stepwise,
    )

    # b) Then we'll use the Nelder-Mead method:
    out.out("<h3>Nelder-Mead:</h3>")
    test_model(
        out, "nm",
        start_p, start_q, start_P, start_Q,
        max_p, max_q, max_P, max_Q,
        maxiter,
        stepwise,
    )

    # c) Powell's method:
    out.out("<h3>Powell's method:</h3>")
    test_model(
        out, "powell",
        start_p, start_q, start_P, start_Q,
        max_p, max_q, max_P, max_Q,
        maxiter,
        stepwise,
    )

    # d) Conjugate gradient:
    out.out("<h3>Conjugate gradient:</h3>")
    test_model(
        out, "cg",
        start_p, start_q, start_P, start_Q,
        max_p, max_q, max_P, max_Q,
        maxiter,
        stepwise,
    )

    out.out("As a result, we can choose the Nelder-Mead SARIMA model as the most accurate one.")
    return out.get()
