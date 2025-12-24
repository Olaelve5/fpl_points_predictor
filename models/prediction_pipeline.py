from utils.processing.get_last_completed_round import (
    get_last_completed_round,
    get_last_completed_round_local,
)
from fpl_api.fetch_updated_player_data import fetch_all_players_data
from models.minutes.combined_minutes import minutes_prediction_pipeline
import joblib
from utils.processing.team_id_name_map import team_id_name_map
from models.points.gradient_boost.get_columns_to_drop import get_columns_to_drop
from utils.processing.dynamic_team_data import save_ratings_to_csv


def prediction_pipeline(points_model):
    """
    Orchestrates the full two-stage prediction pipeline:
    1. Predicts minutes.
    2. Uses those predictions as a feature to predict points.
    It also fetches updated data if the local data is outdated.
    """

    last_completed_round_api = get_last_completed_round()
    last_completed_round_local = get_last_completed_round_local()

    # Fetch new player data if the local data is outdated
    # Also update team ratings CSV
    if last_completed_round_api > last_completed_round_local:
        fetch_all_players_data()
        save_ratings_to_csv()

    _, df_with_mins = minutes_prediction_pipeline()

    # Load and use points feature order
    points_feature_order = joblib.load("data/feature_order/points_feature_order.pkl")
    rows_to_predict = df_with_mins[points_feature_order]

    print("Predicting future points...")

    # Filter to selected features only and make predictions
    points_predictions = points_model.predict(
        rows_to_predict.drop(columns=get_columns_to_drop())
    ).round(1)

    final_df = df_with_mins.copy()
    final_df["predicted_points"] = points_predictions

    # Map opponent_team IDs to names
    final_df["opponent_team"] = final_df["opponent_team"].map(team_id_name_map())

    # Rename opponent_team column for clarity
    final_df.rename(columns={"opponent_team": "next_opponent"}, inplace=True)

    # Divide value by 10 to convert to standard FPL format
    final_df["value"] = final_df["value"] / 10.0

    final_df.sort_values(
        by=["round", "predicted_points"], ascending=[True, False], inplace=True
    )

    final_df.to_csv("data/prediction_data/final_predictions.csv", index=False)

    print("Success! Predictions saved ✅")

    return final_df


if __name__ == "__main__":
    # Load points prediction model and make predictions
    points_model = joblib.load("data/saved_models/points/voting_model.pkl")
    prediction_pipeline(points_model)
