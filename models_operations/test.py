from utils.get_training_test_data import get_train_test_data
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)
from models.baseline_model import BaselineModel


def rmsle(y_true, y_pred):
    """Root Mean Squared Logarithmic Error"""
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def test_model(model_preds, y_test):
    mae = mean_absolute_error(y_test, model_preds)
    mse = mean_squared_error(y_test, model_preds)
    rmse = np.sqrt(mse)  # A more direct way to get RMSE
    rmsle_score = rmsle(y_test, model_preds)
    r2 = r2_score(y_test, model_preds)

    return mae, rmse, r2, rmsle_score


def compare_model_to_baseline(
    other_model_preds, y_test, X_test, is_minutes_model=False
):
    baseline_model = BaselineModel()
    if is_minutes_model:
        baseline_predictions_raw = baseline_model.predict_minutes(X_test)
    else:
        baseline_predictions_raw = baseline_model.predict(X_test)

    baseline_model_preds = np.expm1(baseline_predictions_raw)

    # Print Baseline Metrics
    baseline_mae, baseline_rmse, baseline_r2, baseline_rmsle = test_model(
        baseline_model_preds, y_test
    )
    print("\n--- Baseline Model Performance ---")
    print(f"MAE: {baseline_mae:.2f}")
    print(f"RMSE: {baseline_rmse:.2f}")
    print(f"R^2: {baseline_r2:.2f}")
    print(f"RMSLE: {baseline_rmsle:.2f}")

    print("\n--- Other Model Performance ---")
    other_mae, other_rmse, other_r2, other_rmsle = test_model(other_model_preds, y_test)
    print(f"MAE: {other_mae:.2f}")
    print(f"RMSE: {other_rmse:.2f}")
    print(f"R^2: {other_r2:.2f}")
    print(f"RMSLE: {other_rmsle:.2f}")

    print("\n--- Summary of Improvements Over Baseline ---")
    print(f"MAE Improvement: {baseline_mae - other_mae:.2f}")
    print(f"RMSE Improvement: {baseline_rmse - other_rmse:.2f}")
    print(f"R^2 Improvement: {other_r2 - baseline_r2:.2f}")
    print(f"RMSLE Improvement: {baseline_rmsle - other_rmsle:.2f}")
