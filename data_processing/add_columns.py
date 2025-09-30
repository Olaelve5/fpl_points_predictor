import pandas as pd
import numpy as np


def add_columns(df, team_data_file_path=None):
    # EWMA configuration
    span_size = 4

    # Define columns to calculate EWMA for with their decimal places
    ewma_columns = {
        "total_points": {"name": "ewma_points", "decimals": 1},
        "minutes": {"name": "ewma_minutes", "decimals": 1},
        "bps": {"name": "ewma_bps", "decimals": 1},
        "xP": {"name": "ewma_xP", "decimals": 1},
        "expected_goals": {"name": "ewma_xG", "decimals": 2},
        "expected_assists": {"name": "ewma_xA", "decimals": 2},
        "threat": {"name": "ewma_threat", "decimals": 1},
        "creativity": {"name": "ewma_creativity", "decimals": 1},
        "influence": {"name": "ewma_influence", "decimals": 1},
        "clean_sheets": {"name": "ewma_cs", "decimals": 2},
        "yellow_cards": {"name": "ewma_yellow_cards", "decimals": 2},
        "goals_conceded": {"name": "ewma_gc", "decimals": 2},
    }

    # Calculate EWMA for each column
    for source_col, config in ewma_columns.items():
        df[config["name"]] = round(
            df.groupby("name")[source_col]
            .ewm(span=span_size, adjust=False)
            .mean()
            .reset_index(level=0, drop=True),
            config["decimals"],
        )

    # One-hot encode positions
    position_dummies = pd.get_dummies(df["position"], prefix="pos")
    df = pd.concat([df, position_dummies], axis=1)
    df.drop("position", axis=1, inplace=True)

    # Add next fixture column
    df["next_fixture"] = df.groupby("name")["opponent_team"].shift(-1)
    df["next_is_home"] = df.groupby("name")["was_home"].shift(-1)

    # Add fixture difficulty rating columns
    df = add_fixture_difficulty_rating(df, team_data_file_path)

    # Add target score column
    df["target_score"] = df.groupby("name")["total_points"].shift(-1)

    return df


def add_fixture_difficulty_rating(df, teams_file_path):

    try:
        teams_df = pd.read_csv(teams_file_path, engine="python")
    except FileNotFoundError:
        print("Error: Teams CSV not found.")
        exit()

    attack_home_map = teams_df.set_index("id")["strength_attack_home"].to_dict()
    attack_away_map = teams_df.set_index("id")["strength_attack_away"].to_dict()
    defense_home_map = teams_df.set_index("id")["strength_defence_home"].to_dict()
    defense_away_map = teams_df.set_index("id")["strength_defence_away"].to_dict()

    is_home = df["next_is_home"] == 1.0
    next_opponent = df["next_fixture"]

    # For attack rating: if home, use opponent's away attack; if away, use opponent's home attack
    df["next_fixture_attack_rating"] = np.where(
        is_home, next_opponent.map(attack_away_map), next_opponent.map(attack_home_map)
    )

    # For defense rating: if home, use opponent's away defense; if away, use opponent's home defense
    df["next_fixture_defense_rating"] = np.where(
        is_home,
        next_opponent.map(defense_away_map),
        next_opponent.map(defense_home_map),
    )

    return df
