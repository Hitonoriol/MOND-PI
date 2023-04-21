import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from io_utils import *

test = []
x_test, y_test = [], []
img_size = 0


def show_image(out, images_data, n, label="", end="<br>"):
    global test, x_test, y_test, img_size
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(
        images_data.iloc[n, :]
            .values
            .reshape(img_size, img_size)
            .astype('uint8')
    )
    ax.set_title(f"{label}")
    out.plot(fig, end)


def evaluate_classifier(out, name, mod, demonstrate_n=20):
    global test, x_test, y_test, img_size
    out.out(f"<h3>{name}</h3>")
    # Make predictions and evaluate accuracy
    pred = mod.predict(x_test)
    out.out(f"Accuracy Score: {accuracy_score(y_test, pred)}")

    # Predict using test dataset
    pred_test = mod.predict(test)

    # Demonstrate `demonstrate_n` random images with predicted labels
    out.out(f"Demonstrating {demonstrate_n} predictions for images from `test.csv`")
    for i in random.sample(range(0, len(pred_test)), demonstrate_n):
        show_image(out, test, i, f"[{name}]\nPredicted label: {pred_test[i]}", end="")
    out.out()


def scikit_learn_classifiers(
        test_size=0.2, n_estimators=100, max_depth=1, learning_rate=1.0
):
    global test, x_test, y_test, img_size
    out = Output()
    seed = 81212

    train = pd.read_csv("mnist-handwritten-digits/train.csv")
    test = pd.read_csv("mnist-handwritten-digits/test.csv")
    img_size = int(sqrt(test.shape[1]))

    out.out(f"Image size: {img_size}x{img_size} ({test.shape[1]} pixels total)")
    out.out(f"Training data contains {train.shape[0]} labeled images")
    out.out(f"Testing data contains {test.shape[0]} images")
    out.out(f"Splitting data {1 - test_size}/{test_size} (training/testing)")
    out.out(f"Fitting classifiers with {n_estimators} estimators")

    # Prepare training data
    x = train.iloc[:, 1:]  # Pixel Values
    y = train.iloc[:, 0]  # Labels

    # Split our data 80/20 for cross-validations
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed
    )

    # a. Now we can use the function defined above to evaluate the accuracy of RandomForestClassifier:
    mod = RandomForestClassifier(n_estimators=n_estimators).fit(x_train, y_train)
    evaluate_classifier(out, "RandomForestClassifier", mod)

    # b. Then we'll fit a GradientBoostingClassifier:
    mod = GradientBoostingClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, random_state=seed
    ).fit(x_train, y_train)

    evaluate_classifier(out, "GradientBoostingClassifier", mod)

    # c. After this, we'll build a MLPClassifier:
    mod = MLPClassifier(
        max_iter=n_estimators, random_state=seed
    ).fit(x_train, y_train)

    evaluate_classifier(out, "MLPClassifier", mod)
    return out.get()
