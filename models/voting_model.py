from sklearn.ensemble import (
    VotingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from lightgbm import LGBMRegressor
from models_operations.train import train_model
from models_operations.test import compare_model_to_baseline
from utils.get_training_test_data import get_train_test_data
import numpy as np
from utils.load_csv_to_df import load_csv_to_df
from models_operations.predict import make_predictions


base_models = [
    (
        "lgbm_huber",
        LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.005,
            num_leaves=50,
            random_state=42,
            n_jobs=-1,
            alpha=0.8,
            reg_lambda=1.0,
            colsample_bytree=0.8,
            subsample=0.8,
        ),
    ),
    (
        "lgbm_l1",
        LGBMRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=15,
            random_state=42,
            force_row_wise=True,
            n_jobs=-1,
            objective="regression_l1",
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
        ),
    ),
    (
        "hgb_quantile",
        HistGradientBoostingRegressor(
            random_state=50,
            max_leaf_nodes=50,
            learning_rate=0.05,
            min_samples_leaf=15,
            max_iter=1200,
            quantile=0.5,  # Median regression
        ),
    ),
]

model = VotingRegressor(estimators=base_models, n_jobs=-1, verbose=True)


if __name__ == "__main__":
    training_data = get_train_test_data()
    X_train, X_test, y_train_log, y_test_log = training_data

    trained_model = train_model(
        model, training_data, "data/saved_models/voting_model.pkl", plot=True
    )

    model_preds = np.expm1(trained_model.predict(X_test))
    compare_model_to_baseline(model_preds, np.expm1(y_test_log), X_test)

    print("Training complete.")
