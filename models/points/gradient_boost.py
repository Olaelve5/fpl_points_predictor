
from utils.processing.get_train_test_data import get_train_test_data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from xgboost import XGBRegressor, plot_importance

model_params = {
    "objective": "reg:squarederror",
    "n_estimators": 500,
    "learning_rate": 0.01,
    "max_depth": 7,  # Controls complexity (similar to num_leaves)
    "random_state": 42,
    "n_jobs": -1,
    "reg_alpha": 0.8,  # L1 Regularization
    "reg_lambda": 1.0,  # L2 Regularization
    "colsample_bytree": 0.8,
    "subsample": 0.8,
}

model = XGBRegressor(**model_params)


# --- Plotting Functions ---
def plot_xgb_feature_importance(model):
    """
    Shows which features (xG, form, minutes) drive the points prediction.
    XGBoost specific plotting function.
    """
    plt.figure(figsize=(10, 8))
    plot_importance(
        model, max_num_features=20, importance_type="gain", height=0.5, grid=False
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
    X_train, X_test, y_train, y_test, _ = training_data

    trained_model = model.fit(X_train, y_train)
    print("Model trained.")

    # save the model
    pd.to_pickle(trained_model, "data/saved_models/points/boosting_model.pkl")

    model_preds = trained_model.predict(X_test)
    print("Training complete.")

    # 3. Visualizations
    plot_importance(trained_model)
    plot_actual_vs_predicted(y_test, model_preds)
    plot_distribution_overlay(y_test, model_preds)
