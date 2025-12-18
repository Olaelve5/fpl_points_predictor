import optuna
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from utils.processing.get_train_test_data import get_train_test_data
from models.minutes.clf.features_to_drop import clf_features_to_drop


def objective(trial, X, y):
    """
    Optuna objective function to minimize LogLoss.
    """

    # Define the Hyperparameter Search Space
    param_grid = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": 42,
        "n_jobs": -1,
        # Tree Structure (The most important part)
        "num_leaves": trial.suggest_int("num_leaves", 5, 60),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        # Regularization (Prevents overfitting to specific past games)
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        # Learning Speed
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 300, 3000),
        # Sampling (Speed & Generalization)
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
    }

    # Setup Time-Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)

    logloss_scores = []

    # We must reset index to ensure .iloc slicing works
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Cross-Validation Loop
    for train_index, val_index in tscv.split(X):
        X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
        y_tr, y_val = y.iloc[train_index], y.iloc[val_index]

        # Create LightGBM Dataset format (faster)
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        clf = lgb.train(
            param_grid,
            dtrain,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
            ],
        )

        # Predict & Score
        preds = clf.predict(X_val)
        score = log_loss(y_val, preds)
        logloss_scores.append(score)

    # Return the average LogLoss across all folds
    return np.mean(logloss_scores)


if __name__ == "__main__":
    x_train, x_test, y_train_full, y_test_full, _ = get_train_test_data(
        minutes_training=True
    )
    y_train = y_train_full["classifier_target"]

    # Drop useless features before analysis
    features_to_drop = clf_features_to_drop()
    x_train.drop(columns=features_to_drop, inplace=True, errors="ignore")
    x_test.drop(columns=features_to_drop, inplace=True, errors="ignore")

    print("Starting Optuna Optimization...")

    study = optuna.create_study(
        direction="minimize",
        study_name="LGBM_Minutes_Classifier",
        pruner=optuna.pruners.MedianPruner(),
    )

    # Run Optimization
    study.optimize(lambda trial: objective(trial, x_train, y_train), n_trials=50)

    print("\n--- 🏆 Best Trial Results ---")
    print(f"Best LogLoss: {study.best_value:.4f}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"    '{key}': {value},")
