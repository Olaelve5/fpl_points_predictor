import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import numpy as np
import joblib
import lightgbm as lgb
from utils.get_training_test_data import get_train_test_data


def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Target Score")
    plt.ylabel("Predicted Target Score")
    plt.title("Actual vs Predicted Target Score")
    plt.show()

    # Save the plot
    plt.savefig("data/saved_plots/preds_vs_actual.png")


def plot_permutations(model, X_test, y_test):
    importances = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    feature_importances = importances.importances_mean

    sorted_idx = feature_importances.argsort() 

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align="center")
    plt.yticks(range(len(sorted_idx)), [X_test.columns[i] for i in sorted_idx])
    plt.xlabel("Mean Decrease in RMSE")
    plt.title("Top 28 Feature Importances (Permutation Importance)")
    plt.show()


def plot_target_distribution(y_test):
    plt.figure(figsize=(10, 6))
    plt.hist(y_test, bins=25, edgecolor="k", alpha=0.7)
    plt.xlabel("Target Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Target Scores")
    plt.show()


def plot_tree(model):
    lgb.plot_tree(model, figsize=(20, 10), show_info=["split_gain"])
    plt.title("LightGBM Decision Tree")
    plt.show()


def plot_data(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df["ewma_points"], df["target_score"], alpha=0.5)
    plt.xlabel("EWMA Minutes")
    plt.ylabel("Target Score")
    plt.title("EWMA Minutes vs Target Score")
    plt.show()


if __name__ == "__main__":
    model = joblib.load("data/saved_models/minutes_model.pkl")

    X_train, X_test, y_train_log, y_test_log = get_train_test_data(minutes_training=True)

    # plot_target_distribution(y_test_log)
    plot_permutations(model, X_test, y_test_log)
    # plot_tree(model)
    # plot_data(original_df)
