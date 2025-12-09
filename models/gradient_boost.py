from lightgbm import LGBMRegressor
from utils.get_training_test_data import get_train_test_data
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import pandas as pd

model_params = {
    "objective": "regression",
    "n_estimators": 500,
    "learning_rate": 0.01,
    "num_leaves": 50,
    "random_state": 42,
    "n_jobs": -1,
    "alpha": 0.8,
    "reg_lambda": 1.0,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
}


model = LGBMRegressor(**model_params)


# --- Plotting Functions ---
def plot_feature_importance(model):
    """
    Shows which features (xG, form, minutes) drive the points prediction.
    """
    plt.figure(figsize=(10, 8))
    # 'gain' measures how much the feature improved the loss (accuracy)
    lgb.plot_importance(
        model, max_num_features=20, importance_type="gain", figsize=(10, 8)
    )
    plt.title("Feature Importance (Gain) - What drives Points?")
    plt.show()


def plot_actual_vs_predicted(y_true, y_pred):
    """
    Checks if high predictions match high actuals.
    Ideally, dots should cluster around the red dashed diagonal line.
    """
    plt.figure(figsize=(10, 6))

    # Scatter plot with transparency to see density
    plt.scatter(y_true, y_pred, alpha=0.3, color="blue")

    # Perfect prediction line
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], "r--", label="Perfect Prediction")

    plt.xlabel("Actual Points")
    plt.ylabel("Predicted Points")
    plt.title("Actual vs Predicted Points")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_distribution_overlay(y_true, y_pred):
    """
    Checks if the model captures the 'shape' of FPL points.
    FPL points are usually 2, 6, or 0. Models tend to average them to 3.5.
    We want the Orange curve (Preds) to look somewhat like the Blue curve (Actuals).
    """
    plt.figure(figsize=(10, 6))

    sns.kdeplot(y_true, color="blue", label="Actual Points", fill=True, alpha=0.3)
    sns.kdeplot(y_pred, color="orange", label="Predicted Points", fill=True, alpha=0.3)

    plt.title("Distribution: Does the model understand Hauls vs Blanks?")
    plt.xlabel("Points")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # 1. Train the model
    training_data = get_train_test_data()
    X_train, X_test, y_train, y_test = training_data

    trained_model = model.fit(X_train, y_train)
    print("Model trained.")

    # save the model
    pd.to_pickle(trained_model, "data/saved_models/boosting_model.pkl")

    model_preds = trained_model.predict(X_test)
    print("Training complete.")

    # 3. Visualizations
    plot_feature_importance(trained_model)
    plot_actual_vs_predicted(y_test, model_preds)
    plot_distribution_overlay(y_test, model_preds)

