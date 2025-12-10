from utils.processing.get_training_test_data import get_train_test_data
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

model_params = {
    "n_estimators": 1000,
    "max_depth": 10,
    "subsample": 0.8,
    "reg_lambda": 1,
    "min_child_weight": 2,
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "random_state": 42,
}

model = xgb.XGBRFRegressor(**model_params)

training_data = get_train_test_data(minutes_training=False)
X_train, X_test, y_train, y_test, _ = training_data

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "data/saved_models/points/forest_model.pkl")


def plot_actual_vs_predicted(y_true, y_pred):
    """
    Checks if high predictions match high actuals.
    Ideally, dots should cluster around the red dashed diagonal line.
    Obviously this isn't possible with a Random Forest, but we want to see some correlation.
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


y_pred = model.predict(X_test)
plot_actual_vs_predicted(y_test, y_pred)
plot_distribution_overlay(y_test, y_pred)
