import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

from io_utils import *

x_test, y_test = [], []
x_train, y_train = [], []
feature_names = []
x, y = [], []
y_name = ""
x_points = []


def join_date(df):
    return pd.to_datetime(
        df['month'].map(str) + '-' + df['year'].map(str),
        format='%m-%Y'
    ).dt.strftime('%m-%Y')


# A function to break output into sections
def line_delim(out):
    out.out(f"\n\n{'-' * 60}\n\n")


def demonstrate_model(
        out,
        method_name, model,
        title="", xlabel="", ylabel="",
):
    global x, y, x_train, x_test, y_train, y_test, feature_names, y_name, x_points
    for divide_by in range(1, 11, 2):  # All points, 1/2, 1/4, 1/6
        points = x.size // (divide_by - (0 if divide_by == 1 else 1))
        tx = x[:points]
        ty = y[:points]
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.scatter(tx, ty, label="Data", color="mediumblue", alpha=0.35)
        ax.plot(
            tx, model.predict(tx),
            label="Model", color="red", linewidth=2
        )
        ax.set(
            xlabel=xlabel,
            ylabel=ylabel,
            title=f"{y_name}({feature_names}) Model [{method_name}], {points} points"
        )
        ax.legend()
        out.plot(fig)
    plt.close(fig)

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

    out.out(f"Actual vs Predicted Data:\n{pred_comparison}\n")
    out.out("Model Accuracy")
    out.out(f"R²: {r2_test} / {r2_train} (testing / training)")
    out.out(f"Mean absolute error: {mae:.12f}")
    out.out(f"Root mean squared error: {rmse:.12f}")


def demonstrate_multi_model(
        out,
        method_name, model,
        title="", xlabel="", ylabel="",
):
    global x, y, x_train, x_test, y_train, y_test, feature_names, y_name, x_points
    for divide_by in range(1, 8, 2):  # All points, 1/2, 1/4, 1/6
        points = x.size // (divide_by - (0 if divide_by == 1 else 1))
        txp = x_points[:points]
        tx = x[:points]
        ty = y[:points]
        fig, ax = plt.subplots(figsize=(26, 6))
        ax.scatter(txp, ty, label="Data", color="mediumblue", alpha=0.35)
        ax.plot(
            txp, model.predict(tx),
            label="Model", color="red", linewidth=2
        )
        ax.set(
            xlabel=xlabel,
            ylabel=ylabel,
            title=f"{y_name}({feature_names.values}) Model [{method_name}], {points} points"
        )
        ax.legend()
        out.plot(fig)
    plt.close(fig)

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

    out.out(f"Actual vs Predicted Data:\n{pred_comparison}\n")
    out.out("Model Accuracy")
    out.out(f"R²: {r2_test} / {r2_train} (testing / training)")
    out.out(f"Mean absolute error: {mae:.12f}")
    out.out(f"Root mean squared error: {rmse:.12f}")


def tree_regressors(min_estimators=1, max_estimators=1024, mul_by_estimators=4):
    global x, y, x_train, x_test, y_train, y_test, feature_names, y_name, x_points
    out = Output()

    seed = 81212

    # 1. For data from daily-stats.csv, models should be built for
    # the dependency distance(flights), i.e. the FLIGHTS column is
    # an array of features, the DISTANCE column is a predicted value (an observed target).
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
    x = data[[flt_col]].values
    y = data[dst_col].values
    sort_idx = x.flatten().argsort()
    x = x[sort_idx]
    y = y[sort_idx]

    # Split data into training and testing sets 70%/30%
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=seed
    )

    # a) AdaBoost Regressor for Distance(Flights):
    method = "AdaBoost"
    estimators = min_estimators
    while estimators <= max_estimators:
        title = f"{method}, {estimators} estimators"
        out.out(title)
        mod_ada = AdaBoostRegressor(
            random_state=seed, n_estimators=estimators
        ).fit(x_train, y_train)
        demonstrate_model(
            out,
            title, mod_ada,
            xlabel="Flights",
            ylabel="Distance",
            title="Distance(Flights)"
        )
        line_delim(out)
        estimators *= mul_by_estimators

    # b) Extra-Trees Regressor for Distance(Flights):
    method = "Extra-Trees"
    estimators = min_estimators
    while estimators <= max_estimators:
        title = f"{method}, {estimators} estimators"
        out.out(title)
        mod_extra = ExtraTreesRegressor(
            random_state=seed, n_estimators=estimators
        ).fit(x_train, y_train)
        demonstrate_model(
            out,
            title, mod_extra,
            xlabel="Flights",
            ylabel="Distance",
            title="Distance(Flights)"
        )
        line_delim(out)
        estimators *= mul_by_estimators

    # c) Random Forest Regressor for Distance(Flights):
    method = "Random Forest"
    estimators = min_estimators
    while estimators <= max_estimators:
        title = f"{method}, {estimators} estimators"
        out.out(title)
        mod_forest = RandomForestRegressor(
            random_state=seed, n_estimators=estimators
        ).fit(x_train, y_train)
        demonstrate_model(
            out,
            title, mod_forest,
            xlabel="Flights",
            ylabel="Distance",
            title="Distance(Flights)"
        )
        line_delim(out)
        estimators *= mul_by_estimators

    # 2. For data from monthly-stats.csv, models should be built for the dependency
    # distance(year,month) using values for years 2017 - 2018 as training data and
    # for the year 2019 as validation data.

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

    # a) AdaBoost Regressor for Distance(Year, Month):
    method = "AdaBoost"
    estimators = min_estimators
    while estimators <= max_estimators:
        title = f"{method}, {estimators} estimators"
        out.out(title)
        mod_ada = AdaBoostRegressor(
            random_state=seed, n_estimators=estimators
        ).fit(x_train, y_train)
        demonstrate_multi_model(
            out,
            title, mod_ada,
            xlabel="Year, Month",
            ylabel="Distance",
            title="Distance(Year, Month)"
        )
        line_delim(out)
        estimators *= mul_by_estimators

    # b) Extra-Trees Regressor for Distance(Year, Month):
    method = "Extra-Trees"
    estimators = min_estimators
    while estimators <= max_estimators:
        title = f"{method}, {estimators} estimators"
        out.out(title)
        mod_extra = ExtraTreesRegressor(
            random_state=seed, n_estimators=estimators
        ).fit(x_train, y_train)
        demonstrate_multi_model(
            out,
            title, mod_extra,
            xlabel="Year, Month",
            ylabel="Distance",
            title="Distance(Year, Month)"
        )
        line_delim(out)
        estimators *= mul_by_estimators

    # c) Random Forest Regressor for Distance(Year, Month):
    method = "Random Forest"
    estimators = min_estimators
    while estimators <= max_estimators:
        title = f"{method}, {estimators} estimators"
        out.out(title)
        mod_forest = RandomForestRegressor(
            random_state=seed, n_estimators=estimators
        ).fit(x_train, y_train)
        demonstrate_multi_model(
            out,
            title, mod_forest,
            xlabel="Year, Month",
            ylabel="Distance",
            title="Distance(Year, Month)"
        )
        line_delim(out)
        estimators *= mul_by_estimators
        return out.get()
