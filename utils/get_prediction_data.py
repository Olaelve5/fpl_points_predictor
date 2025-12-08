import pandas as pd
from utils.load_csv_to_df import load_csv_to_df
from utils.get_columns_to_drop import get_columns_to_drop
from utils.add_columns import add_columns


def get_rows_to_predict(last_completed_round, is_minutes_model=False, return_identifiers=False):
    """Function to process raw data for making predictions."""

    raw_df = load_csv_to_df(
        "/Users/ola/Documents/FPL_Price_Predictor/data/players_data/merged_gw_25_26.csv"
    )

    processed_df = add_columns(
        raw_df,
        "/Users/ola/Documents/FPL_Price_Predictor/data/team_data/teams_25_26.csv",
    )

    # Split the dataframe into current and future gws
    current_gw_rows = processed_df[processed_df["round"] == last_completed_round]
    future_gw_rows = processed_df[processed_df["round"] > last_completed_round]

    # Impute player data into future gameweeks
    rows_to_predict = impute_values_into_future_gws(current_gw_rows, future_gw_rows)

    # Drop rows with NaN in next_fixture_attack_rating
    rows_to_predict = rows_to_predict.dropna(subset=["next_fixture_attack_rating"])

    # Drop target columns if they exist
    if is_minutes_model:
        rows_to_predict.drop(
            columns=["target_score", "minutes_next"], inplace=True, errors="ignore"
        )
    else:
        rows_to_predict.drop(columns=["target_score"], inplace=True, errors="ignore")

    # Drop unwanted columns
    columns_to_drop = get_columns_to_drop(is_minutes_model)
    
    if return_identifiers:
        pass 
    else:
        # Standard training behavior
        rows_to_predict.drop(columns=columns_to_drop, inplace=True, errors="ignore")

    position_cols = ["pos_DEF", "pos_FWD", "pos_GK", "pos_MID"]
    for col in position_cols:
        rows_to_predict[col] = rows_to_predict[col].fillna(False)
        rows_to_predict[col] = rows_to_predict[col].astype(bool)

    return rows_to_predict


def impute_values_into_future_gws(current_gw_rows, future_gw_rows):
    fixture_specific_cols = [
        "round",
        "opponent_team",
        "was_home",
        "kickoff_time",
        "next_fixture_attack_rating",
        "next_fixture_defense_rating",
        "next_fixture_atk_def_ratio",
        "next_fixture_def_atk_ratio",
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
