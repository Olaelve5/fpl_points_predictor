import pandas as pd
import numpy as np
from utils.processing.fixture_difficulty import (
    add_fixture_difficulty_rating,
    add_self_team_strength,
)


def prepare_features(df, team_data_file_path=None, history_map=None, pos_avg_map=None):
    """
    Add last season features + fixture difficulty features.
    Season by season.
    """

    # 1. Add History
    if history_map is not None and pos_avg_map is not None:
        history_data = df.apply(
            lambda row: get_history(row, history_map, pos_avg_map),
            axis=1,
            result_type="expand",
        )
        df[["last_season_ppm", "last_season_minutes"]] = history_data
    else:
        df["last_season_ppm"] = 0
        df["last_season_minutes"] = 0

    # 2. One-hot encode positions
    position_dummies = pd.get_dummies(df["position"], prefix="pos", dtype=int)
    df = pd.concat([df, position_dummies], axis=1)
    df.drop("position", axis=1, inplace=True)

    # 3. Add Next Fixture Info
    # Note: shift(-1) here is fine for 'opponent' because that is contained within the season schedule
    df["next_fixture"] = df.groupby("name")["opponent_team"].shift(-1)
    df["next_is_home"] = df.groupby("name")["was_home"].shift(-1).fillna(0).astype(int)

    # 4. Team Strength & Difficulty
    df = add_fixture_difficulty_rating(df, team_data_file_path)
    df = add_self_team_strength(df, team_data_file_path)

    # 5. Ratios
    df["next_fixture_atk_def_ratio"] = (
        df["self_team_attack_rating"] / df["next_fixture_defense_rating"]
    ).round(2)

    df["next_fixture_def_atk_ratio"] = (
        df["self_team_defense_rating"] / df["next_fixture_attack_rating"]
    ).round(2)

    # 6. Status
    if "status" not in df.columns:
        df["status"] = np.where(df["minutes"] > 0, "available", "unavailable")

    print(f"Shape after adding new columns: {df.shape}")

    return df


def add_rolling_features(df, span_size=4):
    """
    Adds rolling features to the dataset. Should be used on the whole dataset at once.
    Also adds target scores.
    """

    # Needed to ensure the order is correct
    df = df.sort_values(by=["name", "season", "kickoff_time"])

    # Define columns to calculate EWMA for with their decimal places
    ewma_columns = {
        "total_points": {"name": "ewma_points", "decimals": 1},
        "minutes": {"name": "ewma_minutes", "decimals": 1},
        "expected_goals": {"name": "ewma_xG", "decimals": 2},
        "expected_assists": {"name": "ewma_xA", "decimals": 2},
        "threat": {"name": "ewma_threat", "decimals": 1},
        "creativity": {"name": "ewma_creativity", "decimals": 1},
        "clean_sheets": {"name": "ewma_cs", "decimals": 2},
        "yellow_cards": {"name": "ewma_yellow_cards", "decimals": 2},
        "goals_conceded": {"name": "ewma_gc", "decimals": 2},
        "defensive_contribution": {"name": "ewma_def_contr", "decimals": 2},
    }

    # Calculate EWMA for each column
    for source_col, config in ewma_columns.items():
        if source_col in df.columns:
            df[config["name"]] = round(
                df.groupby("name")[source_col]
                .ewm(span=span_size, adjust=False)
                .mean()
                .reset_index(level=0, drop=True),
                config["decimals"],
            )
        else:
            # If the source column doesn't exist, create the EWMA column with NaN values
            df[config["name"]] = np.nan

    # Add target scores.
    df["target_score"] = df.groupby("name")["total_points"].shift(-1)
    df["predicted_minutes"] = df.groupby("name")["minutes"].shift(-1)

    return df


def get_history(row, history_map, pos_avg_map):
    key = (row["name"], row["season"])

    # Try to find specific player history
    if key in history_map:
        return history_map[key]["history_pps"], history_map[key]["history_mpg"]

    # If we have no history (First season in data OR new signing),
    if row["position"] in pos_avg_map:
        return pos_avg_map[row["position"]], 70

    return 0, 60
