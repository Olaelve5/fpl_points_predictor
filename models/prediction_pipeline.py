from utils.get_last_completed_round import (
    get_last_completed_round,
    get_last_completed_round_local,
)
from fpl_api.fetch_updated_player_data import fetch_all_players_data
from utils.get_prediction_data import get_rows_to_predict
from models.minutes.combined_minutes import minutes_prediction_pipeline
import pandas as pd
import joblib
import numpy as np
from utils.team_id_name_map import team_id_name_map


def prediction_pipeline():
    """
    Orchestrates the full two-stage prediction pipeline:
    1. Predicts minutes.
    2. Uses those predictions as a feature to predict points.
    """

    last_completed_round_api = get_last_completed_round()
    last_completed_round_local = get_last_completed_round_local()

    # Fetch new player data if the local data is outdated
    if last_completed_round_api > last_completed_round_local:
        fetch_all_players_data()

    # Get the rows to predict
    rows_to_predict = get_rows_to_predict(last_completed_round_api)

    # Save to temporary CSV for inspection
    rows_to_predict.to_csv("data/prediction_data/rows_to_predict.csv", index=False)

    predicted_minutes = minutes_prediction_pipeline(
        rows_to_predict=rows_to_predict, last_completed_round=last_completed_round_api
    )

    # Ensure no duplicate 'minutes_next' column
    rows_to_predict.drop(columns=["minutes_next"], inplace=True, errors="ignore")

    # Merge predicted minutes back into the original rows_to_predict DataFrame
    rows_with_mins = pd.merge(
        rows_to_predict,
        predicted_minutes[["name", "team", "position", "predicted_minutes"]],
        on=["name", "team"],
        how="left",
    )

    # Rename predicted_minutes to minutes_next for the points model
    rows_with_mins.rename(columns={"predicted_minutes": "minutes_next"}, inplace=True)

    output_columns = [
        "name",
        "team",
        "position",
        "value",
        "round",
        "minutes_next",
        "opponent_team",
    ]
    identifiers_df = rows_with_mins[output_columns].copy()

    # Remove name and team columns to prepare for prediction
    features_for_prediction = rows_with_mins.drop(
        columns=["name", "team", "opponent_team"], errors="ignore"
    )

    # Load the model and feature order
    try:
        points_model = joblib.load("data/saved_models/stacking_model.pkl")
        points_feature_order = joblib.load("data/saved_models/feature_order.pkl")
    except FileNotFoundError as e:
        print(
            f"Error: {e}. Please ensure the points model and its feature order are saved."
        )
        return

    # Reorder columns to match the training feature order
    features_for_prediction = features_for_prediction[points_feature_order]

    # Predict using the points model
    predicted_points_log = points_model.predict(features_for_prediction)
    predicted_points = np.expm1(predicted_points_log)
    predicted_points = predicted_points.round(1).clip(min=0)

    # Use the identifiers from the minutes prediction df
    final_df = identifiers_df[
        ["name", "team", "position", "value", "round", "opponent_team", "minutes_next"]
    ].copy()
    final_df["predicted_points"] = predicted_points

    # Map opponent_team IDs to names
    final_df["opponent_team"] = final_df["opponent_team"].map(team_id_name_map())

    # Increment round by 1 to reflect the upcoming round
    final_df["round"] = final_df["round"] + 1

    # For each player, shift opponent_team to the next round's opponent
    final_df["opponent_team"] = final_df.groupby("name")["opponent_team"].shift(-1)

    # Rename opponent_team column for clarity
    final_df.rename(columns={"opponent_team": "next_opponent"}, inplace=True)

    # Divide value by 10 to convert to standard FPL format
    final_df["value"] = final_df["value"] / 10.0

    # Rename predicted_minutes column for clarity
    final_df.rename(columns={"minutes_next": "predicted_minutes"}, inplace=True)

    final_df.sort_values(
        by=["round", "predicted_points"], ascending=[True, False], inplace=True
    )

    return final_df


if __name__ == "__main__":
    final_predictions_df = prediction_pipeline()
    print(final_predictions_df.head())

    # save to csv
    final_predictions_df.to_csv(
        "data/prediction_data/predicted_points_with_minutes.csv", index=False
    )
