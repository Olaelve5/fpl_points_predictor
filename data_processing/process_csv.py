import pandas as pd
from utils.load_csv_to_df import load_csv_to_df
from data_processing.add_columns import add_columns


player_data_base_path = (
    "/Users/ola/Documents/FPL_Price_Predictor/players_data/merged_gw_"
)
team_data_base_path = "/Users/ola/Documents/FPL_Price_Predictor/team_data/teams_"

seasons = [
    # "20_21",
    # "21_22",
    "22_23",
    "23_24",
    "24_25",
]

def process_df(original_df, team_data_file_path: str, is_training=True):

    # Add new columns
    original_df = add_columns(original_df, team_data_file_path)

    columns_to_drop = [
        "name",
        "team",
        "element",
        "fixture",
        "kickoff_time",
        "modified",
        "total_points",
        "xP",
        "expected_goals",
        "expected_assists",
        "expected_goals_conceded",
        "goals_scored",
        "goals_conceded",
        "saves",
        "assists",
        "bonus",
        "bps",
        "clean_sheets",
        # transfers info
        "selected",
        "transfers_balance",
        "transfers_in",
        "transfers_out",
        # fixture info
        "team_a_score",
        "team_h_score",
        "penalties_saved",
        "penalties_missed",
        "opponent_team",
        "was_home",
        "own_goals",
        "next_fixture",
        "minutes",
        # Use round instead of GW
        "GW",
        # Remove fields from new data
        "clearances_blocks_interceptions",
        "defensive_contribution",
        "recoveries",
        "tackles",
    ]

    # Drop columns that are not needed for modeling (ignore errors if they don't exist)
    cleaned_df = original_df.drop(columns=columns_to_drop, errors="ignore")

    # Drop rows where target_score is NaN (last gameweek for each player)
    if is_training:
        cleaned_df.dropna(subset=["target_score"], inplace=True)

    # Change boolean values to integers (0 and 1)
    boolean_columns = ["pos_DEF", "pos_FWD", "pos_GK", "pos_MID", "next_is_home"]
    cleaned_df[boolean_columns] = cleaned_df[boolean_columns].fillna(0)
    cleaned_df[boolean_columns] = cleaned_df[boolean_columns].astype(int)

    return cleaned_df


def combine_csv():
    list_of_dfs = []

    for season in seasons:
        df = load_csv_to_df(player_data_base_path + season + ".csv")
        processed_df = process_df(df, team_data_base_path + season + ".csv")
        list_of_dfs.append(processed_df)

    combined_df = pd.concat(list_of_dfs, ignore_index=True)

    # Remove players with position 'AM' (assistant managers)
    combined_df = combined_df[combined_df["pos_AM"] == 0].copy()
    combined_df.drop(columns=["pos_AM"], inplace=True, errors="ignore")

    print(f"Combined DataFrame shape: {combined_df.shape}")
    print(combined_df.columns)

    min_season = min(seasons).replace("_", "-")
    max_season = max(seasons).replace("_", "-")
    print(f"Data from seasons: {min_season} to {max_season}")

    # save to new CSV
    combined_df.to_csv(
        f"/Users/ola/Documents/FPL_Price_Predictor/players_data/players_{min_season}_to_{max_season}.csv",
        index=False,
    )

    return combined_df


if __name__ == "__main__":
    combine_csv()
