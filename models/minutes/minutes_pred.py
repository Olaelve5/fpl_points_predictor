from lightgbm import LGBMRegressor
from utils.processing.get_train_test_data import get_train_test_data
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import joblib

reg_params = {
    "objective": "regression_l1",
    "metric": "mae",
    "n_estimators": 1000,
    "learning_rate": 0.01, 
    "num_leaves": 50,
    "random_state": 42,
    "n_jobs": -1,
    "reg_alpha": 0.1,  # L1 Regularization
    "reg_lambda": 1.0,  # L2 Regularization
}

model = LGBMRegressor(**reg_params)


def plot_results(y_true, y_pred):
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
    results = model.evals_result_

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
    X_train_full, X_test_full, y_train_full, y_test_full, _ = training_data

    # --- Train Set Split ---
    # Filter to only players who played > 5 mins for regression training
    train_mask = y_train_full["regressor_target"] > 5
    X_train_reg = X_train_full.loc[train_mask]
    y_train_reg = y_train_full.loc[train_mask, "regressor_target"]

    # --- Test Set Split ---
    # We filter test set here so the MAE doesn't explode due to bench players
    test_mask = y_test_full["regressor_target"] > 5
    X_test_reg = X_test_full.loc[test_mask]
    y_test_reg = y_test_full.loc[test_mask, "regressor_target"]

    print(f"Training on {len(X_train_reg)} samples (Filtered > 5 mins)")
    print(f"Validating on {len(X_test_reg)} samples (Filtered > 5 mins)")

    # Train
    trained_model = model.fit(
        X_train_reg,
        y_train_reg,
        eval_set=[(X_train_reg, y_train_reg), (X_test_reg, y_test_reg)],
        eval_metric="mae",
        callbacks=[
            lgb.early_stopping(100, verbose=True),
            lgb.log_evaluation(100),
        ],
    )

    joblib.dump(trained_model, "data/saved_models/minutes/minutes_regression_model.pkl")

    # Predict and evaluate
    model_preds = trained_model.predict(X_test_reg)

    # Plotting
    plot_results(y_test_reg, model_preds)
    plot_learning_curve(trained_model)
    plot_importance(trained_model)
