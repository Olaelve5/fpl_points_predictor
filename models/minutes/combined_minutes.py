import joblib
import pandas as pd
from utils.get_training_test_data import get_prediction_data
from utils.load_csv_to_df import load_csv_to_df
import numpy as np


def combined_minutes_model():
    try:
        classifier_model = pd.read_pickle(
            "data/saved_models/minutes_classifier_model.pkl"
        )
        regressor_model = joblib.load("data/saved_models/minutes_pred_model.pkl")
        # Load the feature order saved during training
        feature_order = joblib.load("data/saved_models/minutes_feature_order.pkl")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure model and feature_order files exist.")
        exit()

    raw_df = load_csv_to_df(
        "/Users/ola/Documents/FPL_Price_Predictor/data/players_data/merged_gw_25_26.csv"
    )

    rows_to_predict, _ = get_prediction_data(raw_df, is_minutes_model=True)
    rows_to_predict = rows_to_predict[feature_order]

    classifier_preds = classifier_model.predict_proba(rows_to_predict)[:, 1]
    regressor_preds = regressor_model.predict(rows_to_predict)

    print("Max classifier prediction:", classifier_preds.max())
    print("Min classifier prediction:", classifier_preds.min())
    print("Max regressor prediction:", regressor_preds.max())
    print("Min regressor prediction:", regressor_preds.min())

    regressor_preds = regressor_preds.clip(0, 90)
    final_minutes_preds = classifier_preds * regressor_preds
    final_minutes_preds = np.round(classifier_preds * regressor_preds).astype(int)

    return final_minutes_preds


if __name__ == "__main__":
    minutes_predictions = combined_minutes_model()
    print("Combined Minutes Predictions:")
    print(minutes_predictions)
