# predict_final_points.py
import joblib
import pandas as pd
import numpy as np

# You'll need to import your custom functions
from models.minutes.combined_minutes import combined_minutes_model
from utils.get_training_test_data import get_prediction_data
from utils.load_csv_to_df import load_csv_to_df


def predict_all():
    """
    Orchestrates the full two-stage prediction pipeline:
    1. Predicts minutes.
    2. Uses those predictions as a feature to predict points.
    """

    try:
        points_model = joblib.load("data/saved_models/stacking_model.pkl")
        points_feature_order = joblib.load("data/saved_models/feature_order.pkl")
    except FileNotFoundError as e:
        print(
            f"Error: {e}. Please ensure the points model and its feature order are saved."
        )
        return

    # Predict minutes using the combined model
    minutes_predictions_df = combined_minutes_model()

    # Raw data for this season up to the prediction gameweek
    raw_df = load_csv_to_df(
        "/Users/ola/Documents/FPL_Price_Predictor/data/players_data/merged_gw_25_26.csv"
    )

    # Get the feature set for the prediction gameweek
    rows_to_predict, round_to_predict = get_prediction_data(raw_df)

    # Add minutes predictions to the feature set
    rows_to_predict["minutes_next"] = minutes_predictions_df["predicted_minutes"]

    # Reorder columns to match the training feature order
    rows_to_predict = rows_to_predict[points_feature_order]

    # Predict using the points model
    predicted_points_log = points_model.predict(rows_to_predict)
    predicted_points = np.expm1(predicted_points_log)

    # Use the identifiers from the minutes prediction df
    final_df = minutes_predictions_df[["name", "team", "position", "value"]].copy()
    final_df["round"] = round_to_predict
    final_df["predicted_minutes"] = minutes_predictions_df["predicted_minutes"]
    final_df["predicted_points"] = predicted_points.round(1)

    # Divide value by 10 to convert to standard FPL format
    final_df["value"] = final_df["value"] / 10.0

    final_df.sort_values(by="predicted_points", ascending=False, inplace=True)

    return final_df


if __name__ == "__main__":
    final_predictions_df = predict_all()
    if final_predictions_df is not None:
        print("\n--- Final Top 20 Predictions (Minutes and Points) ---")
        print(final_predictions_df.head(20))

        final_predictions_df.to_csv(
            "data/prediction_data/final_gw_predictions.csv", index=False
        )
        print(
            "\n✅ Final predictions saved to data/prediction_data/final_gw_predictions.csv"
        )
