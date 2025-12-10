import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
from utils.processing.get_training_test_data import get_train_test_data

# --- 1. CONFIGURATION ---
model_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "num_leaves": 63,
    "random_state": 42,
    "reg_alpha": 0.2,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "subsample": 0.63,
    "max_depth": 12,
    "colsample_bytree": 0.66,
}


# --- 2. PLOTTING FUNCTIONS ---


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


def plot_loss_curve(model):
    results = model.evals_result_
    # Note: LGBM calls it 'binary_logloss' in the results dict
    train_loss = results["training"]["binary_logloss"]
    val_loss = results["valid_1"]["binary_logloss"]
    epochs = range(len(train_loss))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Training LogLoss")
    plt.plot(epochs, val_loss, label="Validation LogLoss")
    plt.title("LogLoss Curve")
    plt.xlabel("Trees")
    plt.ylabel("LogLoss (Lower is Better)")
    plt.legend()
    plt.grid(True)
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


# --- 3. VALIDATION LOGIC ---


def run_cross_validation(X, y, params, n_splits=5):
    """
    Runs 5 separate training rounds on different data slices to check stability.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    auc_scores = []

    print(f"Starting Cross-Validation with {n_splits} folds...")

    fold_no = 1
    for train_index, val_index in skf.split(X, y):
        X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
        y_tr, y_val = y.iloc[train_index], y.iloc[val_index]

        clf = LGBMClassifier(**params)
        clf.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        score = clf.best_score_["valid_0"]["auc"]
        auc_scores.append(score)
        print(f"Fold {fold_no} AUC: {score:.4f}")
        fold_no += 1

    print(f"\nMean AUC: {np.mean(auc_scores):.4f} +/- {np.std(auc_scores):.4f}")
    print("------------------------------------------------")


# --- 4. MAIN EXECUTION ---

if __name__ == "__main__":
    # Load Data
    training_data = get_train_test_data(minutes_training=True, minutes_classifier=True)
    X_train, X_test, y_train, y_test, _ = training_data

    # Cross validation
    run_cross_validation(X_train, y_train, model_params)

    # Train model
    print("Training Final Model...")
    model = LGBMClassifier(**model_params)
    trained_model = model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=True)],
    )

    # Save model
    pd.to_pickle(
        trained_model, "data/saved_models/minutes/minutes_classifier_model.pkl"
    )

    # Evaluate
    model_preds = trained_model.predict(X_test)
    model_proba = trained_model.predict_proba(X_test)[:, 1]

    print(f"Max prediction prob: {model_proba.max():.4f}")
    print(f"Min prediction prob: {model_proba.min():.4f}")

    # Visualizations
    plot_loss_curve(model)
    plot_confusion_matrix(y_test, model_preds)
    plot_learning_curve(trained_model)
    plot_calibration_curve(y_test, model_proba)
