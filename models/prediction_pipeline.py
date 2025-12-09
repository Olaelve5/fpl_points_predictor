from utils.get_last_completed_round import (
    get_last_completed_round,
    get_last_completed_round_local,
)
from fpl_api.fetch_updated_player_data import fetch_all_players_data
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
    It also fetches updated data if the local data is outdated. 
    """

    last_completed_round_api = get_last_completed_round()
    last_completed_round_local = get_last_completed_round_local()

    # Fetch new player data if the local data is outdated
    if last_completed_round_api > last_completed_round_local:
        fetch_all_players_data()

    _, df_with_mins = minutes_prediction_pipeline()



    

    
    
    

    
    # # Map opponent_team IDs to names
    # final_df["opponent_team"] = final_df["opponent_team"].map(team_id_name_map())

    # # Increment round by 1 to reflect the upcoming round
    # final_df["round"] = final_df["round"] + 1

    # # For each player, shift opponent_team to the next round's opponent
    # final_df["opponent_team"] = final_df.groupby("name")["opponent_team"].shift(-1)

    # # Rename opponent_team column for clarity
    # final_df.rename(columns={"opponent_team": "next_opponent"}, inplace=True)

    # # Divide value by 10 to convert to standard FPL format
    # final_df["value"] = final_df["value"] / 10.0

    # # Rename predicted_minutes column for clarity
    # final_df.rename(columns={"minutes_next": "predicted_minutes"}, inplace=True)

    # final_df.sort_values(
    #     by=["round", "predicted_points"], ascending=[True, False], inplace=True
    # )

    # return final_df, features_for_prediction, identifiers_df, points_model


if __name__ == "__main__":
    final_predictions_df, features_for_prediction, identifiers_df, points_model = prediction_pipeline()
    print(final_predictions_df.head())

    # save to csv
    final_predictions_df.to_csv(
        "data/prediction_data/predicted_points_with_minutes.csv", index=False
    )
