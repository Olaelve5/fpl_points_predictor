import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    log_loss,
    brier_score_loss,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve
from utils.processing.get_train_test_data import get_train_test_data

clf_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 1342,
    "learning_rate": 0.01,
    "min_child_samples": 68,
    "num_leaves": 45,
    "random_state": 42,
    "lambda_l1": 9.149741040178033e-07,
    "lambda_l2": 0.0005516549939026048,
    "n_jobs": -1,
    "subsample": 0.7134422500910328,
    "colsample_bytree": 0.7096679855599513,
    "max_depth": 10,
    "subsample_freq": 1,
}


def plot_calibration_curve(y_true, y_prob):
    """
    Checks if predicted probabilities (e.g., 0.6) match actual frequency (60%).
    Crucial for your Expected Minutes calculation.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred, prob_true, marker="o", label="LGBM Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives (Actual)")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Did Not Play", "Played"]
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def plot_learning_curve(model):
    results = model.evals_result_
    train_auc = results["training"]["auc"]
    val_auc = results["valid_1"]["auc"]
    epochs = range(len(train_auc))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_auc, label="Training AUC")
    plt.plot(epochs, val_auc, label="Validation AUC")
    plt.title("Learning Curve (Overfitting Check)")
    plt.xlabel("Trees")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_importance(model):
    plt.figure(figsize=(10, 8))
    lgb.plot_importance(
        model, max_num_features=20, importance_type="gain", figsize=(10, 8)
    )
    plt.title("Feature Importance (Gain)")
    plt.show()


# --- 3. VALIDATION LOGIC ---


def evaluate_model_performance(model, X_test, y_test):
    """
    Evaluates the model against naive baselines and plots safety thresholds.
    """
    print("--- 📊 Model Evaluation Report ---")

    # [:, 1] gets the probability of class 1 (Playing)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 1. Baseline A: "Played last game" (Deterministic - Simple Baseline)
    baseline_last_game = (X_test["minutes"] > 0).astype(int)

    # Baseline B: "Percentage played last 5 games" (Probabilistic - Stronger Baseline)
    if "played_last_5_pct" in X_test.columns:
        baseline_form_prob = X_test["played_last_5_pct"]
    else:
        print("⚠️ 'played_last_5_pct' not found. Skipping Form Baseline.")
        baseline_form_prob = None

    # 3. Calculate Metrics Dictionary
    metrics = {
        "Model": {
            "LogLoss": log_loss(y_test, y_prob),
            "Brier": brier_score_loss(y_test, y_prob),
            "AUC": roc_auc_score(y_test, y_prob),
        },
        "Baseline (Last Game)": {
            "LogLoss": log_loss(y_test, baseline_last_game),
            "Brier": brier_score_loss(y_test, baseline_last_game),
            "AUC": roc_auc_score(y_test, baseline_last_game),
        },
    }

    if baseline_form_prob is not None:
        metrics["Baseline (Form)"] = {
            "LogLoss": log_loss(y_test, baseline_form_prob),
            "Brier": brier_score_loss(y_test, baseline_form_prob),
            "AUC": roc_auc_score(y_test, baseline_form_prob),
        }

    # 4. Print Comparison Table
    results_df = pd.DataFrame(metrics).T
    print("\nMetric Comparison (Lower LogLoss/Brier is better):")
    print(results_df.round(4))

    # Check if we beat the baseline
    if baseline_form_prob is not None:
        if metrics["Model"]["LogLoss"] < metrics["Baseline (Form)"]["LogLoss"]:
            print("\n✅ SUCCESS: Your model is beating the 'Form' baseline!")
        else:
            print(
                "\n❌ WARNING: Your model is WORSE than just using 'played_last_5_pct'."
            )
            print("Action: Check if you are overfitting or need better features.")

    # 5. Find Safety Threshold (The 'Expert' Metric)
    find_safety_threshold(y_test, y_prob)

    return results_df


def find_safety_threshold(y_test, y_prob, target_recall=0.98):
    """
    Finds the probability cutoff where the model capture 98% of actual starters.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

    # Find index where recall is closest to target
    idx = (np.abs(recalls - target_recall)).argmin()

    # Handle edge case if idx is out of bounds for thresholds
    if idx < len(thresholds):
        optimal_threshold = thresholds[idx]
        current_precision = precisions[idx]
    else:
        optimal_threshold = 0.0
        current_precision = 0.0

    print(f"\n--- 🛡️ Safety Threshold Analysis ---")
    print(f"Goal: Don't miss more than {(1-target_recall)*100:.0f}% of starters.")
    print(f"Recommended Cutoff: P(Play) >= {optimal_threshold:.3f}")
    print(f"At this cutoff, Precision is: {current_precision:.3f}")
    print(
        "(Meaning: Of the players you keep, {:.1f}% will actually play)".format(
            current_precision * 100
        )
    )

    # Simple Plot
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precisions[:-1], label="Precision (Trust)")
    plt.plot(thresholds, recalls[:-1], label="Recall (Safety)")
    plt.axvline(
        optimal_threshold,
        color="red",
        linestyle="--",
        label=f"Cutoff {optimal_threshold:.2f}",
    )
    plt.xlabel("Probability Threshold")
    plt.title("Trade-off: Safety vs Trust")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    X_train, X_test, y_train_full, y_test_full, _ = get_train_test_data(
        minutes_training=True
    )

    y_train = y_train_full["classifier_target"]
    y_test = y_test_full["classifier_target"]

    # Train model
    print("Training Final Model...")
    model = LGBMClassifier(**clf_params)
    trained_model = model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric=["logloss", "auc"],
        callbacks=[lgb.early_stopping(100, verbose=True)],
    )

    # Save model
    pd.to_pickle(
        trained_model, "data/saved_models/minutes/minutes_classifier_model.pkl"
    )

    model_preds = trained_model.predict(X_test)
    model_proba = trained_model.predict_proba(X_test)[:, 1]

    # Visualizations
    plot_confusion_matrix(y_test, model_preds)
    plot_importance(trained_model)
    plot_learning_curve(trained_model)
    plot_calibration_curve(y_test, model_proba)

    # Evaluate performance
    evaluate_model_performance(trained_model, X_test, y_test)
