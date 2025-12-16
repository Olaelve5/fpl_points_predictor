import pandas as pd
from utils.processing.load_csv_to_df import load_csv_to_df
from fpl_api.fetch_updated_player_data import fetch_all_players_data
import numpy as np

player_data_base_path = (
    "/Users/ola/Documents/FPL_Price_Predictor/data/players_data/merged_gw_"
)
team_data_base_path = "/Users/ola/Documents/FPL_Price_Predictor/data/team_data/teams_"

seasons = [
    "20_21",
    "21_22",
    "22_23",
    "23_24",
    "24_25",
    "25_26",
]


def combine_csv(pull_latest_data=False):
    print("Combining player data CSVs...")

    # Pull latest data if 25_26 in seasons
    if "25_26" in seasons and pull_latest_data:
        fetch_all_players_data()

    # Process all DataFrames first to build a complete column list
    all_processed_dfs = []
    all_columns = set()

    for season in seasons:
        print(f"Processing season: {season}")
        df = load_csv_to_df(player_data_base_path + season + ".csv")

        if "position" in df.columns:
            df["position"] = df["position"].replace("GKP", "GK")

        # Add the season -> needed to add team data in feature engineering later
        df["season"] = season

        # Remove expected data from 22_23 as it is all 0s any way
        if season == "22_23":
            df.drop(
                columns=[
                    "expected_goals",
                    "expected_assists",
                    "expected_goal_involvements",
                    "expected_goals_conceded",
                ],
                inplace=True,
            )

        all_processed_dfs.append(df)
        all_columns.update(df.columns)

    # Now, harmonize and append
    final_list_of_dfs = []
    final_columns = sorted(list(all_columns))  # Use a sorted list for consistent order

    for processed_df in all_processed_dfs:
        harmonized_df = processed_df.reindex(columns=final_columns, fill_value=np.nan)
        final_list_of_dfs.append(harmonized_df)

    combined_df = pd.concat(final_list_of_dfs, ignore_index=True)

    min_season = min(seasons).replace("_", "-")
    max_season = max(seasons).replace("_", "-")
    print(f"Data from seasons: {min_season} to {max_season}")

    # save to new CSV
    combined_df.to_csv(
        f"/Users/ola/Documents/FPL_Price_Predictor/data/players_data/players_{min_season}_to_{max_season}.csv",
        index=False,
    )

    print("Combined CSV saved successfully. \n")

    return combined_df


if __name__ == "__main__":
    combine_csv(pull_latest_data=True)
