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
    # --- LOAD THE TRAINED MODELS AND METADATA ---
    try:
        points_model = joblib.load("data/saved_models/voting_model.pkl")
        points_feature_order = joblib.load("data/saved_models/feature_order.pkl")
    except FileNotFoundError as e:
        print(
            f"Error: {e}. Please ensure the points model and its feature order are saved."
        )
        return

    # === STEP 1: GENERATE THE MINUTES PREDICTIONS ===
    print("--- Running Step 1: Generating Minutes Predictions ---")
    # This function returns a full DataFrame with player identifiers and predicted_minutes
    minutes_predictions_df = combined_minutes_model()

    # === STEP 2: PREPARE THE BASE FEATURE SET FOR THE POINTS MODEL ===
    print("\n--- Running Step 2: Preparing Base Features for Points Model ---")
    raw_df = load_csv_to_df(
        "/Users/ola/Documents/FPL_Price_Predictor/data/players_data/merged_gw_25_26.csv"
    )
    # Get the feature set for the prediction gameweek. Use the default is_minutes_model=False
    rows_to_predict, round_to_predict = get_prediction_data(raw_df)

    # === STEP 3: IMPUTE THE PREDICTED MINUTES AS A FEATURE ===
    print("\n--- Running Step 3: Imputing Predicted Minutes ---")
    # This is the crucial step. You add the prediction from the first model
    # as a feature for the second model. It MUST be named 'minutes' to match
    # the column the points model was trained on.
    rows_to_predict["minutes_next"] = minutes_predictions_df["predicted_minutes"]
    print(rows_to_predict.head(20))

    # === STEP 4: PREDICT THE FINAL POINTS ===
    print("\n--- Running Step 4: Predicting Points ---")
    # Ensure the column order is correct for the points model
    rows_to_predict = rows_to_predict[points_feature_order]

    # Predict using the points model
    predicted_points_log = points_model.predict(rows_to_predict)
    predicted_points = np.expm1(predicted_points_log)

    # === STEP 5: CREATE THE FINAL OUTPUT DATAFRAME ===
    print("\n--- Running Step 5: Finalizing Output ---")
    # Use the identifiers from the minutes prediction df
    final_df = minutes_predictions_df[["name", "team", "position", "value"]].copy()
    final_df["predicted_minutes"] = minutes_predictions_df["predicted_minutes"]
    final_df["predicted_points"] = predicted_points.round(1)

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
