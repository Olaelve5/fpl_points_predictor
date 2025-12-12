import pandas as pd
from utils.processing.load_csv_to_df import load_csv_to_df
from utils.processing.process_features import process_features
from utils.processing.feature_engineering import add_columns
from utils.processing.feature_engineering import add_fixture_difficulty_rating
from utils.processing.get_train_test_data import get_historic_stats_map


def get_rows_to_predict(last_completed_round, is_minutes_model=False):
    """Function to process raw data for making predictions."""

    raw_df = load_csv_to_df("data/players_data/players_22-23_to_25-26.csv")
    historic_points_map, positional_avg_map = get_historic_stats_map(raw_df)

    this_season = raw_df[raw_df["season"] == "25_26"].copy()

    full_df = add_columns(
        this_season,
        "data/team_data/teams_25_26.csv",
        historic_points_map,
        positional_avg_map,
    )

    # Split the dataframe
    current_gw_rows = full_df[full_df["round"] == last_completed_round]
    future_gw_rows = full_df[full_df["round"] > last_completed_round]

    # Impute player data
    rows_to_predict = impute_values_into_future_gws(current_gw_rows, future_gw_rows)

    # Recalculate fixture difficulty ratings based on opponent_team
    # - which is the next fixture for prediction rows
    rows_to_predict["next_fixture"] = rows_to_predict["opponent_team"]
    rows_to_predict["next_is_home"] = rows_to_predict["was_home"]

    rows_to_predict = add_fixture_difficulty_rating(
        rows_to_predict,
        "/Users/ola/Documents/FPL_Price_Predictor/data/team_data/teams_25_26.csv",
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

    # Drop target columns
    if is_minutes_model:
        rows_to_predict.drop(
            columns=["target_score", "minutes_next"], inplace=True, errors="ignore"
        )
    else:
        rows_to_predict.drop(columns=["target_score"], inplace=True, errors="ignore")

    rows_to_predict = process_features(rows_to_predict, is_training=False)

    return rows_to_predict


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
