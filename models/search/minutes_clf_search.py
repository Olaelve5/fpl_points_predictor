import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from utils.processing.get_train_test_data import get_train_test_data

# --- Load your data once ---
training_data = get_train_test_data(minutes_training=True, minutes_classifier=True)
X_train, _, y_train, _ = (
    training_data  # We only need the training set for cross-validation
)


# 1. Define the objective function for Optuna
def objective(trial):
    """
    This function takes a trial object and returns the score to be maximized.
    """
    # Define the hyperparameter search space
    params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
        "n_jobs": -1,
    }

    model = LGBMClassifier(**params)

    # Use cross-validation to get a robust score
    # This is more reliable than a single train-test split
    score = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=3, scoring="roc_auc")
    auc = score.mean()

    return auc


# 2. Create a study object and run the optimization
study = optuna.create_study(direction="maximize")  # We want to maximize the AUC
study.optimize(objective, n_trials=50)  # Run 50 trials

# 3. Print out the best results
print("Best trial:")
trial = study.best_trial
print(f"  Value (AUC): {trial.value}")
print("  Best hyperparameters: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
