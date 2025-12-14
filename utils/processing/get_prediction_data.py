import pandas as pd
from utils.processing.load_csv_to_df import load_csv_to_df
from utils.processing.process_features import process_features
from utils.processing.feature_engineering import (
    add_fixture_difficulty_rating,
    do_feature_engineering,
)


def get_rows_to_predict(last_completed_round, is_minutes_model=False):
    """Function to process raw data for making predictions."""

    raw_df = load_csv_to_df("data/players_data/players_20-21_to_25-26.csv")
    full_df = do_feature_engineering(raw_df, drop_targets=False)

    this_season_df = load_csv_to_df("data/players_data/merged_gw_25_26.csv")

    # Split the dataframe
    current_gw_rows = full_df[
        (full_df["round"] == last_completed_round) & (full_df["season"] == "25_26")
    ].copy()

    # Save for debugging
    current_gw_rows.to_csv("data/debugging/current_gw.csv")

    future_gw_rows = this_season_df[
        this_season_df["round"] > last_completed_round
    ].copy()

    future_gw_rows.to_csv("data/debugging/future_gw.csv")

    if future_gw_rows.empty:
        print(
            "⚠️ Warning: No future fixtures found in main CSV. Predictions will be empty."
        )
        return None

    # Impute player data
    rows_to_predict = impute_values_into_future_gws(current_gw_rows, future_gw_rows)

    # Recalculate fixture difficulty ratings based on opponent_team
    # - which is the next fixture for prediction rows
    rows_to_predict["next_fixture"] = rows_to_predict["opponent_team"]
    rows_to_predict["next_is_home"] = rows_to_predict["was_home"]

    rows_to_predict = add_fixture_difficulty_rating(
        rows_to_predict,
        "/Users/ola/Documents/FPL_Price_Predictor/data/team_data/updated_teams_25_26.csv",
    )

    # Recalculate ratios
    rows_to_predict["next_fixture_atk_def_ratio"] = (
        rows_to_predict["self_team_attack_rating"]
        / rows_to_predict["next_fixture_defense_rating"]
    ).round(2)

    rows_to_predict["next_fixture_def_atk_ratio"] = (
        rows_to_predict["self_team_defense_rating"]
        / rows_to_predict["next_fixture_attack_rating"]
    ).round(2)

    # Drop rows with NaN
    rows_to_predict = rows_to_predict.dropna(subset=["next_fixture_attack_rating"])

    processed_df, features, _ = process_features(
        rows_to_predict, is_training=False, is_minutes_model=is_minutes_model
    )

    return processed_df, features


def impute_values_into_future_gws(current_gw_rows, future_gw_rows):
    fixture_specific_cols = [
        "round",
        "opponent_team",
        "was_home",
        "kickoff_time",
    ]

    player_key = ["name", "team"]

    player_snapshot_cols = [
        col for col in current_gw_rows.columns if col not in fixture_specific_cols
    ]

    player_snapshot = current_gw_rows[player_snapshot_cols]
    future_fixtures = future_gw_rows[player_key + fixture_specific_cols]

    rows_to_predict = pd.merge(
        future_fixtures,
        player_snapshot,
        on=player_key,
        how="left",  # Use a left join to keep all future fixtures
    )

    return rows_to_predict
