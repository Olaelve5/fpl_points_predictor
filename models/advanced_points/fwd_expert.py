from xgboost import XGBRegressor


def get_fwd_model():
    model_params = {
        "objective": "reg:squarederror",
        "n_estimators": 2000,
        "learning_rate": 0.005,
        "max_depth": 6,
        "random_state": 42,
        "n_jobs": -1,
        "reg_alpha": 0.8,
        "reg_lambda": 1.0,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "early_stopping_rounds": 50,
    }

    return XGBRegressor(**model_params)
