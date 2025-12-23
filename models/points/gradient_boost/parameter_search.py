import optuna
import numpy as np
from xgboost import XGBRegressor
from utils.processing.get_train_test_data import get_train_test_data
from models.evaluate_model import calculate_rolling_ndcg
from models.points.gradient_boost.get_columns_to_drop import get_columns_to_drop


N_TRIALS = 100
WINDOW_SIZE = 5


def search_params_optuna(window_size=WINDOW_SIZE, n_trials=N_TRIALS):
    test_seasons = ["23_24", "24_25", "25_26"]
    data_cache = {}

    # Load Data Once (Cache it)
    print("--- Loading Data into Cache ---")
    for season in test_seasons:
        print(f"Loading {season}...")
        x_tr, x_te, y_tr, y_te, meta = get_train_test_data(
            test_season=season, minutes_training=False
        )
        data_cache[season] = (
            x_tr.drop(columns=get_columns_to_drop()),
            x_te.drop(columns=get_columns_to_drop()),
            y_tr,
            y_te,
            meta,
        )
    print("--- Data Loaded ---\n")

    # Define the Objective Function for Optuna
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=50),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.05, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 25),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "objective": "reg:squarederror",
            "n_jobs": -1,
            "random_state": 42,
            "early_stopping_rounds": 50,
        }

        scores = []

        # Train and Evaluate on all cached seasons
        for i, season in enumerate(test_seasons):
            x_train, x_test, y_train, y_test, test_meta = data_cache[season]

            model = XGBRegressor(**params)
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_train, y_train), (x_test, y_test)],
                verbose=False,
            )
            predictions = model.predict(x_test)

            score = calculate_rolling_ndcg(
                test_meta, y_test.values, predictions, window_size
            )
            scores.append(round(score, 4))

            # Pruning based on intermediate results
            current_average_score = np.mean(scores)
            trial.report(current_average_score, step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)

    # 3. Create Study and Optimize
    print(f"Starting Optuna Optimization with {n_trials} trials...")
    sampler = optuna.samplers.TPESampler(n_startup_trials=20)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,  # Don't prune the first 5 trials (let it learn the baseline)
        n_warmup_steps=0,
        interval_steps=1,
    )

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    print("\n--- OPTIMIZATION FINISHED ---")
    print(f"Best Score: {study.best_value}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {round(value, 4) if isinstance(value, float) else value}")

    return study.best_params


if __name__ == "__main__":
    best_params = search_params_optuna()

    print("\nBest Hyperparameters Found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
