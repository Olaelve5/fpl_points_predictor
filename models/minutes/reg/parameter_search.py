import optuna
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from utils.processing.get_train_test_data import get_train_test_data
from models.minutes.reg.features_to_drop import reg_features_to_drop


def objective(trial, X, y):
    """
    Optuna objective function to minimize MAE (Mean Absolute Error).
    """

    # Define the Hyperparameter Search Space
    param_grid = {
        "objective": "regression_l1",  # Optimizes Median (Robust to outliers)
        "metric": "mae",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": 42,
        "n_jobs": -1,
        # Tree Structure
        "num_leaves": trial.suggest_int(
            "num_leaves", 20, 80
        ),  # Slightly higher for regression
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        # Regularization
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        # Learning Speed
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
        # Sampling
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
    }

    # Setup Time-Series Cross-Validation
    # We use 5 splits to ensure stability across different seasons/periods
    tscv = TimeSeriesSplit(n_splits=5)

    mae_scores = []

    # Reset index to ensure .iloc slicing works
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Cross-Validation Loop
    for train_index, val_index in tscv.split(X):
        X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
        y_tr, y_val = y.iloc[train_index], y.iloc[val_index]

        # Create LightGBM Dataset format
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            param_grid,
            dtrain,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
            ],
        )

        # Predict & Score
        preds = model.predict(X_val)

        # Clip predictions to realistic bounds before scoring
        preds = np.clip(preds, 0, 100)

        score = mean_absolute_error(y_val, preds)
        mae_scores.append(score)

    # Return the average MAE across all folds
    return np.mean(mae_scores)


if __name__ == "__main__":
    # 1. Get Data
    x_train_full, _, y_train_full, _, _ = get_train_test_data(minutes_training=True)

    # 2. FILTER: Only optimize on players who actually played (> 10 mins)
    # The regressor's job is to predict duration, not appearance.
    # We define >10 to filter out garbage time and pure bench warmers (0 mins).
    mask = y_train_full["regressor_target"] > 10

    x_train_opt = x_train_full.loc[mask].copy()
    y_train_opt = y_train_full.loc[mask, "regressor_target"].copy()

    # 3. Drop useless features
    features_to_drop = reg_features_to_drop()
    # Check intersection to avoid KeyErrors
    existing_drops = [c for c in features_to_drop if c in x_train_opt.columns]

    x_train_opt.drop(columns=existing_drops, inplace=True, errors="ignore")

    print(f"Optimizing on {len(x_train_opt)} samples (Filtered > 10 mins)")
    print(f"Dropped {len(existing_drops)} noise features.")

    print("Starting Optuna Optimization...")

    study = optuna.create_study(
        direction="minimize",
        study_name="LGBM_Minutes_Regressor",
        pruner=optuna.pruners.MedianPruner(),
    )

    # Run Optimization
    study.optimize(
        lambda trial: objective(trial, x_train_opt, y_train_opt), n_trials=50
    )

    print("\n--- 🏆 Best Trial Results ---")
    print(f"Best MAE: {study.best_value:.4f}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"    '{key}': {value},")
