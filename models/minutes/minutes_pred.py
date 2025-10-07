from lightgbm import LGBMRegressor
from models_operations.train import train_model
import numpy as np
from utils.get_training_test_data import get_train_test_data
from utils.get_prediction_data import get_rows_to_predict
from utils.load_csv_to_df import load_csv_to_df
from models_operations.predict import make_predictions

model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.005,
    num_leaves=50,
    random_state=42,
    n_jobs=-1,
    alpha=0.80,
)


if __name__ == "__main__":
    training_data = get_train_test_data(minutes_training=True)

    X_train, X_test, y_train_log, y_test_log = training_data

    trained_model = train_model(
        model,
        training_data,
        "data/saved_models/minutes_pred_model.pkl",
        plot=True,
        is_minutes_model=True,
    )

    model_preds = trained_model.predict(X_test)

    print("Max regressor prediction:", model_preds.max())
    print("Min regressor prediction:", model_preds.min())
