import pandas as pd
import numpy as np
import joblib
from utils.get_last_completed_round import get_last_completed_round
from utils.get_prediction_data import get_rows_to_predict


# --- 1. Model Loading ---
def load_models():
    """Loads models and the feature list used during training."""
    try:
        classifier = pd.read_pickle("data/saved_models/minutes_classifier_model.pkl")
        regressor = joblib.load("data/saved_models/minutes_pred_model.pkl")
        feature_order = joblib.load("data/saved_models/minutes_feature_order.pkl")
        return classifier, regressor, feature_order
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        exit()

def handle_goalkeeper_minutes(df):
    """
    Handles the goalkeeper minutes. Only one
    goalkeeper per team should be predicted to play.
    Returns dataframe with updated minutes
    """

    # Set ALL Goalkeepers to 0 minutes by default
    df.loc[df["pos_GK"] == 1, "predicted_minutes"] = 0

    # Find eligible GKs
    eligible_gks_mask = (df["pos_GK"] == 1) & (df["status"] != "unavailable")

    # Find the Index of the Best GK per Team AND Round
    best_gk_indices = (
        df[eligible_gks_mask].groupby(["team", "round"])["prob_play"].idxmax()
    )

    # D. Set those specific 'Number 1' keepers to 90 minutes
    df.loc[best_gk_indices.values, "predicted_minutes"] = 90

    # --- Verification Prints ---
    print("\n--- Goalkeeper Selection Check ---")
    playing_gks = df[(df["pos_GK"] == 1) & (df["predicted_minutes"] > 0)]

    # Check counts per round (should be 20 per round)
    print("Playing GKs per round:")
    print(playing_gks.groupby("round")["name"].count())

    return df


def minutes_prediction_pipeline():
    """
    Main pipeline to predict player minutes using a two-stage model.
    Returns two DataFrames - one with all features and one with only identifier features.
    """
    last_round = get_last_completed_round()
    print(f"--- Predicting for GW{last_round + 1} ---")

    rows_to_predict = get_rows_to_predict(
        last_round, is_minutes_model=True, return_identifiers=True
    )
    classifier, regressor, feature_order = load_models()

    identifiers = rows_to_predict[
        [
            "name",
            "team",
            "value",
            "round",
            "pos_GK",
            "status",
        ]
    ].copy()

    # Ensure the feature order from training is enforced.
    # Will also drop the columns that should be dropped
    X = rows_to_predict[feature_order].copy()

    # Make predictions
    print("Running Two-Stage Model...")
    prob_playing = classifier.predict_proba(X)[:, 1]
    raw_minutes = regressor.predict(X)

    # Combine to get final minutes prediction
    results = identifiers.copy()
    results["prob_play"] = prob_playing.round(2)
    results["raw_minutes"] = np.round(raw_minutes, 1)
    expected_minutes = (prob_playing * raw_minutes).round(0)

    # Set minutes to 90 for "nailed" players, and
    # to 0 for unavailable players
    results["predicted_minutes"] = np.round(expected_minutes).astype(int).clip(0, 90)
    is_nailed = (results["prob_play"] >= 0.95) & (results["raw_minutes"] >= 85)
    results.loc[is_nailed, "predicted_minutes"] = 90
    results.loc[results["status"] == "unavailable", "predicted_minutes"] = 0

    # Handle goalkeeper minutes
    results = handle_goalkeeper_minutes(results)

    X["predicted_minutes"] = results["predicted_minutes"]

    return results, X


if __name__ == "__main__":
    predictions, df_with_predictions = minutes_prediction_pipeline()

    # 8. Save
    predictions.sort_values(
        by=["round", "predicted_minutes", "name"], ascending=[True, False, True]
    ).to_csv("data/prediction_data/combined_minutes_predictions.csv", index=False)

    df_with_predictions.sort_values(
        by=["round", "predicted_minutes"], ascending=[True, False]
    ).to_csv("data/prediction_data/df_with_minutes_pred.csv", index=False)
