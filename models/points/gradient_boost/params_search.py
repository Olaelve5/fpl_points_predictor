import optuna
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from utils.processing.get_train_test_data import get_train_test_data
from models.points.gradient_boost.get_columns_to_drop import get_columns_to_drop
from models.evaluate_model import calculate_rolling_ndcg


def objective(trial, X, y, metadata):
    """
    Optuna objective function to MAXIMIZE NDCG.
    """

    # 1. Hyperparameter Search Space
    param = {
        "objective": "reg:squarederror",
        "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0),
        "n_jobs": -1,
        "random_state": 42,
        "early_stopping_rounds": 50,
    }

    # 2. Setup Time-Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)
    ndcg_scores = []

    # 3. Cross-Validation Loop
    for step, (train_index, val_index) in enumerate(tscv.split(X)):
        # Safe Slicing using .iloc (Requires reset_index in main block)
        X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
        y_tr, y_val = y.iloc[train_index], y.iloc[val_index]

        # We need metadata for the validation set to calculate rolling NDCG
        meta_val = metadata.iloc[val_index].copy()

        # Train
        model = xgb.XGBRegressor(**param)

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Predict
        preds = model.predict(X_val)

        # Score (NDCG)
        # Using window_size=5 to match your evaluation logic
        score = calculate_rolling_ndcg(meta_val, y_val.values, preds, window_size=5)
        ndcg_scores.append(score)

        # Pruning (Kill bad trials early)
        # We want to MAXIMIZE NDCG, so low scores are bad.
        trial.report(np.mean(ndcg_scores), step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(ndcg_scores)


if __name__ == "__main__":
    # 1. Load Data
    training_data = get_train_test_data(minutes_training=False, test_season="25_26")
    x_train, _, y_train, _, metadata_train, _ = training_data

    # 2. Filter Columns
    cols_to_drop = get_columns_to_drop()
    existing_drop_cols = [c for c in cols_to_drop if c in x_train.columns]
    x_train.drop(columns=existing_drop_cols, inplace=True)

    # 3. CRITICAL: Reset Indices for Alignment
    # This prevents the "IndexError: positional indexers are out-of-bounds"
    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    metadata_train = metadata_train.reset_index(drop=True)

    # Sanity Check
    if len(x_train) != len(metadata_train):
        raise ValueError("X_train and Metadata lengths do not match!")

    print(f"Data Loaded. Training Shape: {x_train.shape}")
    print("Starting Optuna Optimization (Target: MAXIMIZE NDCG)...")

    # 4. Create Study
    study = optuna.create_study(
        direction="maximize",  # Higher NDCG is better
        study_name="XGB_Points_Model_NDCG",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    # 5. Run Optimization
    # We pass the pre-aligned dataframes into the objective via lambda
    study.optimize(
        lambda trial: objective(trial, x_train, y_train, metadata_train), n_trials=50
    )

    print("\n--- 🏆 Best Trial Results ---")
    print(f"Best NDCG: {study.best_value:.4f}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"    '{key}': {value},")
