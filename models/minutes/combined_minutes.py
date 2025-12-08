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


def debug_specific_player(target_player, full_df):
    """Debug function to check predictions for a specific player."""
    player_row = full_df[full_df["name"] == target_player]

    if not player_row.empty:
        print(f"\n--- DEBUGGING {target_player} ---")

        # 1. Check the inputs that drive "Probability"
        important_cols = [
            "ewma_minutes",
            "rolling_avg_minutes",
            "minutes_consistency",
            "ewma_points",
            "starts",
            "value",
            "team",
        ]
        # Filter to only existing columns
        cols_to_show = [c for c in important_cols if c in player_row.columns]
        print(player_row[cols_to_show].iloc[0])


if __name__ == "__main__":
    last_round = get_last_completed_round()

    print(f"--- Predicting for GW{last_round + 1} ---")

    # 1. Get Data with identifiers
    full_df = get_rows_to_predict(
        last_round, is_minutes_model=True, return_identifiers=True
    )

    # save dataframe for inspection
    full_df.to_csv("data/prediction_data/full_rows_to_predict.csv", index=False)

    # 2. Split: Identifiers vs Features
    classifier, regressor, feature_order = load_models()

    # Identifiers
    identifiers = full_df[
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
        ]
    ].copy()

    # This automatically drops the "columns_to_drop" that get_rows_to_predict usually drops
    X = full_df[feature_order].copy()

    # save features dataframe for inspection
    X.to_csv("data/prediction_data/features_rows_to_predict.csv", index=False)

    # 3. Predict
    print("Running Two-Stage Model...")
    prob_playing = classifier.predict_proba(X)[:, 1]
    raw_minutes = regressor.predict(X)

    # 4. Combine
    results = identifiers.copy()
    results["prob_play"] = prob_playing.round(2)
    results["raw_minutes"] = np.round(raw_minutes, 1)

    # 5. Set to 90 for nailed players
    expected_minutes = prob_playing * raw_minutes
    results["predicted_minutes"] = np.round(expected_minutes).astype(int).clip(0, 90)
    is_nailed = (results["prob_play"] >= 0.95) & (results["raw_minutes"] >= 85)
    results.loc[is_nailed, "predicted_minutes"] = 90

    # Set minutes to 0 for unavailable players
    results.loc[results["status"] == "unavailable", "predicted_minutes"] = 0

    # 6. Set all goalkeepers to 90 minutes if they are predicted to play
    if "pos_GK" in results.columns:

        # Set ALL Goalkeepers to 0 minutes by default
        results.loc[results["pos_GK"] == 1, "predicted_minutes"] = 0

        # Find eligible GKs
        eligible_gks_mask = (results["pos_GK"] == 1) & (
            results["status"] != "unavailable"
        )

        # Find the Index of the Best GK per Team AND Round
        best_gk_indices = (
            results[eligible_gks_mask].groupby(["team", "round"])["prob_play"].idxmax()
        )

        # D. Set those specific 'Number 1' keepers to 90 minutes
        results.loc[best_gk_indices.values, "predicted_minutes"] = 90

        # --- Verification Prints ---
        print("\n--- Goalkeeper Selection Check ---")
        playing_gks = results[
            (results["pos_GK"] == 1) & (results["predicted_minutes"] > 0)
        ]

        # Check counts per round (should be 20 per round)
        print("Playing GKs per round:")
        print(playing_gks.groupby("round")["name"].count())

        # Show a sample from the next round
        next_round = results["round"].min()
        print(f"\nSample GKs for Round {next_round}:")
        print(
            f"Numbers predicted to play: {len(playing_gks[playing_gks['round'] == next_round])}"
        )
        print(
            playing_gks[playing_gks["round"] == next_round][
                ["name", "team", "prob_play", "predicted_minutes"]
            ].sort_values("team")
        )

    # 7. Add position column
    def get_position(row):
        if row["pos_GK"] == 1:
            return "GK"
        elif row["pos_DEF"] == 1:
            return "DEF"
        elif row["pos_MID"] == 1:
            return "MID"
        elif row["pos_FWD"] == 1:
            return "FWD"
        else:
            return "UNK"

    results["position"] = results.apply(get_position, axis=1)

    # Remove OHE columns
    results = results.drop(columns=["pos_GK", "pos_DEF", "pos_MID", "pos_FWD"])

    # 8. Save
    results.sort_values(
        by=["round", "predicted_minutes", "name"], ascending=[True, False, True]
    ).to_csv("data/prediction_data/combined_minutes_predictions.csv", index=False)

    print("Predictions saved.")

    # 9. Debug specific players
    debug_players = ["B.Fernandes"]
    for player in debug_players:
        debug_specific_player(player, full_df)
