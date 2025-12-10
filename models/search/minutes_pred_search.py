import optuna
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from utils.processing.get_training_test_data import get_train_test_data
import numpy as np

# --- Load the data for the regressor model ---
# IMPORTANT: For a two-model (classifier + regressor) approach to work best,
# this data should ONLY include players who actually played (minutes > 0).
# Please ensure your get_train_test_data function is filtering for this.
training_data = get_train_test_data(minutes_training=True)
X_train, _, y_train_log, _ = training_data


# 1. Define the objective function for Optuna
def objective(trial):
    """
    This function defines the search space and returns the score to be optimized.
    We maximize the negative RMSE, which is the same as minimizing RMSE.
    """
    # Define the hyperparameter search space
    params = {
        "objective": "regression_l1",  # Mean Absolute Error, often robust to outliers
        "metric": "rmse",
        "n_estimators": trial.suggest_int("n_estimators", 400, 2500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-2, 10.0, log=True
        ),  # L1 regularization
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-2, 10.0, log=True
        ),  # L2 regularization
        "random_state": 42,
        "n_jobs": -1,
    }

    model = LGBMRegressor(**params)

    # Use cross-validation for a robust error score.
    # We use 'neg_root_mean_squared_error' because Optuna's goal is to maximize.
    score = cross_val_score(
        model,
        X_train,
        y_train_log,
        n_jobs=-1,
        cv=3,
        scoring="neg_root_mean_squared_error",
    )
    rmse = score.mean()

    return rmse


# 2. Create a study object and run the optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)  # You can adjust the number of trials

# 3. Print the best results
print("\nBest trial:")
trial = study.best_trial
# The value is the negative RMSE, so we multiply by -1 to get the actual RMSE
print(f"  Value (RMSE): {-trial.value:.4f}")
print("  Best hyperparameters: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
