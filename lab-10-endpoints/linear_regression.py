import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Lasso, BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from io_utils import *

x_test, y_test = [], []
x_train, y_train = [], []
feature_names = []
x, y = [], []
y_name = ""
x_points = []


def evaluate_model(
        out,
        method_name,  # Method by which the model was built
        model,  # The model to evaluate
        xlabel="", ylabel="", title="",  # Plot parameters
):
    global x, y, x_train, x_test, y_train, y_test, feature_names, y_name, x_points
    out.out(f"Model {title} built using {method_name}")
    # Use the model to predict data and create a dataset with
    # actual data compared to the predicted one.
    y_pred = model.predict(x_test)
    pred_comparison = pd.DataFrame({
        "Actual": y_test.squeeze(),
        "Predicted": y_pred.squeeze(),
    })

    # Calculate errors to determine the model's accuracy
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_train = model.score(x_train, y_train)
    r2_test = model.score(x_test, y_test)

    # Create a dataframe with all of the model's coefficients
    # with their corresponding feature names.
    coefficients = pd.DataFrame(
        data=model.coef_,
        index=feature_names,
        columns=['Coefficient value']
    )

    out.out(f"Intercept: {model.intercept_}\n")
    out.out(f"Model coefficients:\n{coefficients}\n")
    out.out(f"Actual vs Predicted Data:\n{pred_comparison}\n")
    out.out("Model Accuracy")
    out.out(f"R²: {r2_test} / {r2_train} (testing / training)")
    out.out(f"Mean absolute error: {mae:.12f}")
    out.out(f"Root mean squared error: {rmse:.12f}")

    tx = x.transpose()
    plot_data = lambda ax: ax.scatter(tx[i], y, label="Data", color="mediumblue", alpha=0.35)
    set_plot_text = lambda ax, feat_idx: ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        title=f"{y_name}({feature_names[feat_idx]}) Model [{method_name}]"
    )

    for i in range(x.shape[1]):
        fig, ax = plt.subplots()
        plot_data(ax)
        ax.plot(tx[i], model.predict(x), label="Model", color="red")
        set_plot_text(ax, i)
        ax.legend()
        out.plot(fig)
    return mae, rmse


def evaluate_multi_model(
        out,
        method_name,  # Method by which the model was built
        model,  # The model to evaluate
        xlabel="", ylabel="", title="",  # Plot parameters
):
    global x, y, x_train, x_test, y_train, y_test, feature_names, y_name, x_points
    out.out(f"Model {title} built using {method_name}")
    # Use the model to predict data and create a dataset with
    # actual data compared to the predicted one.
    y_pred = model.predict(x_test)
    pred_comparison = pd.DataFrame({
        "Actual": y_test.squeeze(),
        "Predicted": y_pred.squeeze(),
    })

    # Calculate errors to determine the model's accuracy
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_train = model.score(x_train, y_train)
    r2_test = model.score(x_test, y_test)

    # Create a dataframe with all of the model's coefficients
    # with their corresponding feature names.
    coefficients = pd.DataFrame(
        data=model.coef_,
        index=feature_names,
        columns=['Coefficient value']
    )

    out.out(f"Intercept: {model.intercept_}\n")
    out.out(f"Model coefficients:\n{coefficients}\n")
    out.out(f"Actual vs Predicted Data:\n{pred_comparison}\n")
    out.out("Model Accuracy")
    out.out(f"R²: {r2_test} / {r2_train} (testing / training)")
    out.out(f"Mean absolute error: {mae:.12f}")
    out.out(f"Root mean squared error: {rmse:.12f}")

    fig, ax = plt.subplots(figsize=(26, 6))
    ax.scatter(x_points, y, label="Data", color="mediumblue", alpha=0.35)
    ax.plot(
        x_points, model.predict(x),
        label="Model", color="red", linewidth=2
    )
    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        title=f"{y_name}({feature_names.values}) Model [{method_name}]"
    )
    ax.legend()
    out.plot(fig)
    return mae, rmse


def join_date(df):
    return pd.to_datetime(
        df['month'].map(str) + '-' + df['year'].map(str),
        format='%m-%Y'
    ).dt.strftime('%m-%Y')


def linear_regression():
    global x, y, x_train, x_test, y_train, y_test, feature_names, y_name, x_points
    out = Output()

    # 1. For data from daily-stats.csv, models should be built for
    # the dependency distance(flights), i.e. the FLIGHTS column is an array
    # of features, the DISTANCE column is a predicted value (an observed target).
    # The program must build models and line-plots
    # using Ordinary Least Squares, Lasso and Bayesian Ridge Regression.

    out.out("<h3>Linear regression for the dependency Distance(Flights):</h3>")
    seed = 81212

    # Load daily flight data and select only two columns: "DISTANCE" & "FLIGHTS"
    dst_col, flt_col = "DISTANCE", "FLIGHTS"
    data = pd.read_csv(
        "domestic-jet-flights-usa/daily-stats.csv",
        sep=";"
    )[[dst_col, flt_col]]

    out.out(f"Loaded dataframe:\n{data.head()}\n{'.' * 23}")

    feature_names = [flt_col]
    y_name = dst_col

    # Extract numpy arrays from dataframe columns
    x = data[flt_col].values.reshape(-1, 1)
    y = data[dst_col].values.reshape(-1, 1)

    sort_idx = x.flatten().argsort()

    # Split data into training and testing sets 70%/30%
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=seed
    )

    # Build a Linear Regression model using Ordinary Least Squares
    mod_ols = LinearRegression().fit(x_train, y_train)
    mae_ols, rmse_ols = evaluate_model(
        out,
        "Ordinary Least Squares", mod_ols,
        xlabel="Flights",
        ylabel="Distance",
        title="Distance(Flights)"
    )

    # Build our model using Lasso method
    mod_lasso = Lasso(alpha=0.1).fit(x_train, y_train)
    mae_lasso, rmse_lasso = evaluate_model(
        out,
        "Lasso", mod_lasso,
        xlabel="Flights",
        ylabel="Distance",
        title="Distance(Flights)"
    )
    out.out("Lasso & Ordinary Least Squares comparison:")
    out.out(f"ΔMean Absolute Error: {mae_lasso - mae_ols}")
    out.out(f"ΔRoot Mean Squared Error: {rmse_lasso - rmse_ols}")

    mod_brid = BayesianRidge().fit(x_train, y_train.ravel())
    mae_brid, rmse_brid = evaluate_model(
        out,
        "Bayesian Ridge", mod_brid,
        xlabel="Flights",
        ylabel="Distance",
        title="Distance(Flights)"
    )
    out.out("Bayesian Ridge vs Lasso vs Ordinary Least Squares comparison:")
    out.out(f"ΔLasso[Mean Absolute Error]: {mae_brid - mae_lasso}")
    out.out(f"ΔLasso[Root Mean Squared Error]: {rmse_brid - rmse_lasso}")
    out.out(f"ΔOrdinary Least Squares[Mean Absolute Error]: {mae_brid - mae_ols}")
    out.out(f"ΔOrdinary Least Squares[Root Mean Squared Error]: {rmse_brid - rmse_ols}")

    # 2. For data from monthly-stats.csv, models should be built for
    # the dependency distance(year,month) using values for years 2017 - 2018
    # as training data and for the year 2019 as validation data.

    out.out("<hr><h3>Linear regression for the dependency Distance(Year, Month):</h3>")

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
    out.out(f"Training y:\n{y_train}\n\nTraining x:\n{x_train}\n")
    out.out(f"Testing y:\n{y_test}\n\nTesting x:\n{x_test}\n")

    mod_ols = LinearRegression().fit(x_train, y_train)
    mae_ols, rmse_ols = evaluate_multi_model(
        out,
        "Ordinary Least Squares", mod_ols,
        xlabel="Year, Month",
        ylabel="Distance",
        title="Distance(Year, Month)"
    )

    # Build our model using Lasso method
    mod_lasso = Lasso(alpha=0.1).fit(x_train, y_train)
    mae_lasso, rmse_lasso = evaluate_multi_model(
        out,
        "Lasso", mod_ols,
        xlabel="Year, Month",
        ylabel="Distance",
        title="Distance(Year, Month)"
    )
    out.out("Lasso & Ordinary Least Squares comparison:")
    out.out(f"ΔMean Absolute Error: {mae_lasso - mae_ols}")
    out.out(f"ΔRoot Mean Squared Error: {rmse_lasso - rmse_ols}")

    mod_brid = BayesianRidge().fit(x_train, y_train)
    mae_brid, rmse_brid = evaluate_multi_model(
        out,
        "Lasso", mod_ols,
        xlabel="Year, Month",
        ylabel="Distance",
        title="Distance(Year, Month)"
    )
    out.out("Bayesian Ridge vs Lasso vs Ordinary Least Squares comparison:")
    out.out(f"ΔLasso[Mean Absolute Error]: {mae_brid - mae_lasso}")
    out.out(f"ΔLasso[Root Mean Squared Error]: {rmse_brid - rmse_lasso}")
    out.out(f"ΔOrdinary Least Squares[Mean Absolute Error]: {mae_brid - mae_ols}")
    out.out(f"ΔOrdinary Least Squares[Root Mean Squared Error]: {rmse_brid - rmse_ols}")

    out.out("As a result of accuracy comparisons we can see that all three methods produced identical inaccurate models. This may be possible to fix by increasing the sample size of our dataset or changing the way training / testing sets are created.")
    return out.get()
