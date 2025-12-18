from lightgbm import LGBMRegressor
from utils.processing.get_train_test_data import get_train_test_data
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from models.minutes.reg.features_to_drop import reg_features_to_drop

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


def plot_distribution(y_true, y_pred):
    # Distribution Plot (The '90 min spike' check)
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


# --- Evaluateion ---


def evaluate_regressor_performance(model, X_test, y_test):
    """
    Evaluates the Regressor against naive baselines (Last Match & Form).

    Assumes X_test contains lag features like 'minutes_lag_1' or similar.
    """
    print("--- 📊 Regressor Model Evaluation Report ---")

    # Get Model Predictions (Clipped to realistic bounds)
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, 98)

    # Baseline A: "Minutes last game" (Naive)
    baseline_last_match = X_test["minutes"]

    # Baseline B: "Average Minutes last X games" (Form)
    baseline_ewma = X_test["ewma_minutes"]

    # 3. Calculate Metrics Dictionary
    def get_metrics(y_true, y_est):
        return {
            "MAE": mean_absolute_error(y_true, y_est),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_est)),
            "R2": r2_score(y_true, y_est),
            # % of predictions within 10 minutes of actual
            "Acc +/-10m": np.mean(np.abs(y_true - y_est) <= 10),
        }

    metrics = {"Model": get_metrics(y_test, y_pred)}

    if baseline_last_match is not None:
        metrics["Baseline (Last Match)"] = get_metrics(y_test, baseline_last_match)

    if baseline_ewma is not None:
        metrics["Baseline (Form)"] = get_metrics(y_test, baseline_ewma)

    # Print Comparison Table
    results_df = pd.DataFrame(metrics).T
    print("\nMetric Comparison (Lower MAE/RMSE is better, Higher R2/Acc is better):")
    print(results_df.round(2))

    # Success Check
    target_baseline = (
        "Baseline (Form)" if baseline_ewma is not None else "Baseline (Last Match)"
    )

    if target_baseline in metrics:
        model_mae = metrics["Model"]["MAE"]
        base_mae = metrics[target_baseline]["MAE"]

        if model_mae < base_mae:
            improvement = ((base_mae - model_mae) / base_mae) * 100
            print(
                f"\n✅ SUCCESS: Your model beats {target_baseline} by {improvement:.1f}%!"
            )
        else:
            print(f"\n❌ WARNING: Your model is WORSE than {target_baseline}.")
            print(
                "Action: Check if you are predicting 90 for everyone or ignoring subs."
            )

    # Residual Plot (The "Safety Threshold" equivalent for regression)
    plot_regressor_residuals(y_test, y_pred)

    return results_df


def plot_regressor_residuals(y_true, y_pred):
    """
    Plots Actual vs Predicted and Residuals to spot bias.
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    jitter_y_true = y_true + np.random.normal(0, 0.5, size=len(y_true))

    axes[0].scatter(
        jitter_y_true, y_pred, alpha=0.2, color="blue", s=15, label="Samples"
    )
    axes[0].plot([0, 100], [0, 100], "r--", lw=2, label="Perfect Prediction")

    # Highlight the "Sub Hazard Zones"
    axes[0].axvline(60, color="gray", linestyle=":", alpha=0.5)
    axes[0].text(61, 5, "60 min mark", color="gray", fontsize=8)

    axes[0].set_xlabel("Actual Minutes")
    axes[0].set_ylabel("Predicted Minutes")
    axes[0].set_title("Prediction Accuracy (with Jitter)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-2, 105)
    axes[0].set_ylim(-2, 105)

    # Plot B: Residuals (Look for the "Smile" or patterns)
    # We want these dots to be centered around 0 randomly.
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.3, ax=axes[1], color="purple")
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Predicted Minutes")
    axes[1].set_ylabel("Error (Actual - Predicted)")
    axes[1].set_title("Residuals: Are errors random?")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    training_data = get_train_test_data(minutes_training=True)
    X_train_full, X_test_full, y_train_full, y_test_full, _ = training_data

    # Drop Unnecessary Features
    X_train_full = X_train_full.drop(columns=reg_features_to_drop())
    X_test_full = X_test_full.drop(columns=reg_features_to_drop())

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
    plot_learning_curve(trained_model)
    plot_importance(trained_model)
    plot_distribution(y_test_reg, model_preds)

    # Evaluation Report
    evaluate_regressor_performance(trained_model, X_test_reg, y_test_reg)
