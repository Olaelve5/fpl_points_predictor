from lightgbm import LGBMRegressor
from models_operations.train import train_model
import numpy as np
from utils.get_training_test_data import get_train_test_data
from models_operations.test import compare_model_to_baseline
from models_operations.predict import make_predictions
from utils.load_csv_to_df import load_csv_to_df

model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    num_leaves=50,
    random_state=42,
    n_jobs=-1,
    alpha=0.8,
    reg_lambda=1.0,
)


if __name__ == "__main__":
    training_data = get_train_test_data(minutes_training=True)

    X_train, X_test, y_train_log, y_test_log = training_data

    trained_model = train_model(
        model,
        training_data,
        "data/saved_models/minutes_model.pkl",
        plot=True,
    )

    model_preds = np.expm1(trained_model.predict(X_test))
    model_preds = np.clip(model_preds, 0, 90)  # Clip predictions to valid range

    # Compare to baseline
    compare_model_to_baseline(
        model_preds, np.expm1(y_test_log), X_test, is_minutes_model=True
    )

    # Make predictions on new data
    raw_df = load_csv_to_df(
        "/Users/ola/Documents/FPL_Price_Predictor/data/players_data/merged_gw_25_26.csv"
    )

    make_predictions(trained_model, raw_df, is_minutes_model=True)
