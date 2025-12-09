import pandas as pd
import numpy as np
from utils.fixture_difficulty import (
    add_fixture_difficulty_rating,
    add_self_team_strength,
)


def add_columns(df, team_data_file_path=None):
    # EWMA configuration
    span_size = 4

    # Define columns to calculate EWMA for with their decimal places
    ewma_columns = {
        "total_points": {"name": "ewma_points", "decimals": 1},
        "minutes": {"name": "ewma_minutes", "decimals": 1},
        "bps": {"name": "ewma_bps", "decimals": 1},
        "expected_goals": {"name": "ewma_xG", "decimals": 2},
        "expected_goals_involvements": {"name": "ewma_xGI", "decimals": 2},
        "expected_assists": {"name": "ewma_xA", "decimals": 2},
        "threat": {"name": "ewma_threat", "decimals": 1},
        "creativity": {"name": "ewma_creativity", "decimals": 1},
        "influence": {"name": "ewma_influence", "decimals": 1},
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

    # One-hot encode positions
    position_dummies = pd.get_dummies(df["position"], prefix="pos")
    df = pd.concat([df, position_dummies], axis=1)
    df.drop("position", axis=1, inplace=True)

    # Add rolling average of minutes played
    df = add_rolling_average_minutes(df, window_size=3)

    # Add next fixture column
    df["next_fixture"] = df.groupby("name")["opponent_team"].shift(-1)
    df["next_is_home"] = df.groupby("name")["was_home"].shift(-1).fillna(0).astype(int)

    # Add fixture difficulty rating columns
    df = add_fixture_difficulty_rating(df, team_data_file_path)

    # Add self team strength columns
    df = add_self_team_strength(df, team_data_file_path)

    # Add attack to defense ratio column
    df["next_fixture_atk_def_ratio"] = (
        df["self_team_attack_rating"] / df["next_fixture_defense_rating"]
    ).round(2)

    # Add defense to attack ratio column
    df["next_fixture_def_atk_ratio"] = (
        df["self_team_defense_rating"] / df["next_fixture_attack_rating"]
    ).round(2)

    # Add target score column
    df["target_score"] = df.groupby(["name", "team"])["total_points"].shift(-1)

    # Add target minutes column
    df["predicted_minutes"] = df.groupby(["name", "team"])["minutes"].shift(-1)

    # If status columns doesn't exist, create it and add data based on minutes
    if "status" not in df.columns:
        df["status"] = np.where(df["minutes"] > 0, "available", "unavailable")

    print(f"Shape after adding new columns: {df.shape}")

    return df


def add_rolling_average_minutes(df, window_size=3):
    """Function to add rolling average of minutes played over a specified window size."""
    df["rolling_avg_minutes"] = (
        df.groupby("name")["minutes"]
        .transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
        .round(1)
    )

    df["minutes_consistency"] = (df["rolling_avg_minutes"] / 90.0).round(2)
    df["value_x_consistency"] = (df["value"] * df["minutes_consistency"]).round(2)

    return df
