from lightgbm import LGBMRegressor
from utils.processing.get_training_test_data import get_train_test_data
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import pandas as pd
import joblib

# --- CONFIGURATION ---

reg_params = {
    "objective": "regression_l1",  # MAE Loss (so that it isn't scared to predict 90)
    "metric": "mae",
    "n_estimators": 1000,
    "learning_rate": 0.01,  # Slightly higher LR than classifier is often ok for regression
    "num_leaves": 50,
    "random_state": 42,
    "n_jobs": -1,
    "reg_alpha": 0.1,  # L1 Regularization
    "reg_lambda": 1.0,  # L2 Regularization
}

model = LGBMRegressor(**reg_params)


# --- PLOTTING FUNCTIONS ---


def plot_results(y_true, y_pred):
    # 1. Scatter Plot (Accuracy)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)

    # Dynamic red line (Perfect prediction)
    limit = max(y_true.max(), y_pred.max())
    plt.plot([0, limit], [0, limit], "r--")

    plt.xlabel("Actual Minutes")
    plt.ylabel("Predicted Minutes")
    plt.title("Actual vs Predicted Minutes (Raw)")
    plt.grid(True, alpha=0.3)
    plt.show()

    # 2. Distribution Plot (The '90 min spike' check)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(y_true, color="blue", label="Actual", fill=True, alpha=0.3)
    sns.kdeplot(y_pred, color="orange", label="Predicted", fill=True, alpha=0.3)
    plt.title("Distribution of Minutes")
    plt.xlabel("Minutes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_learning_curve(model):
    # 1. Get the history dictionary
    results = model.evals_result_

    print("Available eval results keys:", results.keys())
    print("Training metrics keys:", results["training"].keys())

    metric_key = "l1"
    if "l1" not in results["training"]:
        metric_key = "mae"

    train_err = results["training"][metric_key]

    val_key = "valid_0" if "valid_0" in results else "valid_1"
    val_err = results[val_key][metric_key]

    epochs = range(len(train_err))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_err, label="Training MAE")
    plt.plot(epochs, val_err, label="Validation MAE")
    plt.title("Learning Curve: Mean Absolute Error")
    plt.xlabel("Trees")
    plt.ylabel("Error (Minutes)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_importance(model):
    plt.figure(figsize=(10, 8))
    lgb.plot_importance(
        model, max_num_features=20, importance_type="gain", figsize=(10, 8)
    )
    plt.title("Feature Importance (Gain)")
    plt.show()


if __name__ == "__main__":
    training_data = get_train_test_data(minutes_training=True)

    X_train, X_test, y_train, y_test, _ = training_data

    trained_model = model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric="mae",
        callbacks=[
            lgb.early_stopping(100, verbose=True),
            lgb.log_evaluation(100),  # print progress every 100 trees
        ],
    )

    # Save model
    joblib.dump(trained_model, "data/saved_models/minutes/minutes_regression_model.pkl")

    model_preds = trained_model.predict(X_test)

    print("Max regressor prediction:", model_preds.max())
    print("Min regressor prediction:", model_preds.min())

    # Plotting
    plot_results(y_test, model_preds)
    plot_learning_curve(trained_model)
    plot_importance(trained_model)
