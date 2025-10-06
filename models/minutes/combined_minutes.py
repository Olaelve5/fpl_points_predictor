import joblib
import pandas as pd
from utils.load_csv_to_df import load_csv_to_df
import numpy as np


def minutes_prediction_pipeline(rows_to_predict=None, last_completed_round=None):
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

    identifiers = raw_df[["name", "team", "position", "value", "status"]].copy()

    rows_to_predict = rows_to_predict[
        rows_to_predict["round"] == last_completed_round
    ].copy()

    # Clean rows to predict
    rows_to_predict.drop(columns=["name", "team", "opponent_team"], errors="ignore", inplace=True)

    rows_to_predict = rows_to_predict[feature_order]

    classifier_preds = classifier_model.predict_proba(rows_to_predict)[:, 1]
    regressor_preds = regressor_model.predict(rows_to_predict)

    # Apply full game threshold
    full_game_threshold = 85
    regressor_preds = np.where(
        regressor_preds > full_game_threshold, 90, regressor_preds
    )

    # Apply the no mins threshold
    no_mins_threshold = 2
    regressor_preds = np.where(regressor_preds < no_mins_threshold, 0, regressor_preds)

    # Apply the max probability threshold
    max_prob_threshold = 0.95
    classifier_preds = np.where(
        classifier_preds > max_prob_threshold, 1.0, classifier_preds
    )

    # Apply the min probability threshold
    min_prob_threshold = 0.05
    classifier_preds = np.where(
        classifier_preds < min_prob_threshold, 0.0, classifier_preds
    )

    print("Max classifier prediction:", classifier_preds.max())
    print("Min classifier prediction:", classifier_preds.min())
    print("Max regressor prediction:", regressor_preds.max())
    print("Min regressor prediction:", regressor_preds.min())

    regressor_preds = regressor_preds.clip(0, 90)
    final_minutes_preds = classifier_preds * regressor_preds
    final_minutes_preds = np.round(classifier_preds * regressor_preds).astype(int)

    results_df = identifiers.loc[rows_to_predict.index].copy()
    results_df["predicted_minutes"] = final_minutes_preds

    results_df["raw_classifier_proba"] = classifier_preds
    results_df["raw_regressor_minutes"] = regressor_preds

    # If the status is 'unavailable', set predicted minutes to 0
    results_df.loc[results_df["status"] == "unavailable", "predicted_minutes"] = 0

    # Set all GK predictions to 90 minutes if above threshold
    gk_threshold = 60
    gk_mask = results_df["position"] == "GK"
    results_df.loc[
        gk_mask & (results_df["predicted_minutes"] > gk_threshold), "predicted_minutes"
    ] = 90

    results_df.loc[
        gk_mask & (results_df["predicted_minutes"] <= gk_threshold), "predicted_minutes"
    ] = 0

    return results_df


if __name__ == "__main__":
    minutes_predictions = minutes_prediction_pipeline()

    # Sort here
    minutes_predictions.sort_values(
        by="predicted_minutes", ascending=False, inplace=True
    )

    print(minutes_predictions.head(20))

    # Save to CSV
    minutes_predictions.to_csv(
        "data/prediction_data/combined_minutes_predictions.csv", index=False
    )
    print("Combined minutes predictions saved to combined_minutes_predictions.csv")
