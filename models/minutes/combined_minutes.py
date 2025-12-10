import pandas as pd
import numpy as np
import joblib
from utils.processing.get_last_completed_round import get_last_completed_round
from utils.processing.get_prediction_data import get_rows_to_predict


# --- 1. Model Loading ---
def load_models():
    """Loads models and the feature list used during training."""
    try:
        classifier = joblib.load(
            "data/saved_models/minutes/minutes_classifier_model.pkl"
        )
        regressor = joblib.load(
            "data/saved_models/minutes/minutes_regression_model.pkl"
        )
        feature_order = joblib.load("data/feature_order/minutes_feature_order.pkl")
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

    # Set those specific 'Number 1' keepers to 90 minutes
    df.loc[best_gk_indices.values, "predicted_minutes"] = 90

    return df


def minutes_prediction_pipeline():
    """
    Main pipeline to predict player minutes using a two-stage model.
    Returns two DataFrames - one with all features and one with only identifier features.
    Also returns the feature order used.
    """
    last_round = get_last_completed_round()
    print(f"--- Predicting for GW{last_round + 1} ---")

    rows_to_predict = get_rows_to_predict(last_round, is_minutes_model=True)
    classifier, regressor, feature_order = load_models()

    identifiers = rows_to_predict[
        [
            "name",
            "team",
            "value",
            "round",
            "pos_GK",
            "pos_DEF",
            "pos_MID",
            "pos_FWD",
            "status",
            "opponent_team",
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

    # Add back identifiers to the full dataframe.
    # This dataframe will be used to make final predictions in
    # the main pipeline
    cols_to_add = [col for col in identifiers.columns if col not in X.columns] + [
        "predicted_minutes"
    ]
    X = pd.concat([X, results[cols_to_add]], axis=1)

    # 8. Save
    results.sort_values(
        by=["round", "predicted_minutes", "name"], ascending=[True, False, True]
    ).to_csv("data/prediction_data/combined_minutes_predictions.csv", index=False)

    X.sort_values(by=["round", "predicted_minutes"], ascending=[True, False]).to_csv(
        "data/prediction_data/df_with_minutes_pred.csv", index=False
    )

    return results, X


minutes_prediction_pipeline()
