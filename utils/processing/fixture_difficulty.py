import pandas as pd
import numpy as np
from utils.processing.team_id_name_map import team_id_name_map


def add_fixture_difficulty_rating(df, teams_file_path):
    """Function to add next fixture difficulty ratings to the DataFrame."""

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


def add_self_team_strength(df, teams_file_path):
    """Function to add self team strength ratings to the DataFrame."""

    try:
        teams_df = pd.read_csv(teams_file_path, engine="python")
    except FileNotFoundError:
        print("Error: Teams CSV not found.")
        exit()

    attack_home_map = teams_df.set_index("id")["strength_attack_home"].to_dict()
    attack_away_map = teams_df.set_index("id")["strength_attack_away"].to_dict()
    defense_home_map = teams_df.set_index("id")["strength_defence_home"].to_dict()
    defense_away_map = teams_df.set_index("id")["strength_defence_away"].to_dict()

    team_id_map = team_id_name_map(file_path=teams_file_path)
    name_to_id_map = {v: k for k, v in team_id_map.items()}

    team_name = df["team"]
    team_id = team_name.map(name_to_id_map)

    is_home = df["next_is_home"] == 1.0

    df["self_team_attack_rating"] = np.where(
        is_home, team_id.map(attack_home_map), team_id.map(attack_away_map)
    )

    df["self_team_defense_rating"] = np.where(
        is_home, team_id.map(defense_home_map), team_id.map(defense_away_map)
    )

    return df
