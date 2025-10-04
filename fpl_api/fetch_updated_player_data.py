import requests
import pandas as pd
import numpy as np
import concurrent.futures
from tqdm import tqdm
from utils.team_id_name_map import team_id_name_map


def get_player_details():
    """
    Fetches the main bootstrap data and returns a list of all player IDs.
    """
    print("Fetching all player IDs...")
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"

    position_map = {
        1: "GK",
        2: "DEF",
        3: "MID",
        4: "FWD",
    }

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        players = data.get("elements", [])

        id_name_map = {
            player["id"]: {
                "name": player["web_name"],
                "position": position_map.get(player["element_type"], "Unknown"),
                "team_id": player["team"],
                "status": player["status"],
            }
            for player in players
        }

        print(f"Fetched details for {len(id_name_map)} players.")

        return id_name_map
    except requests.exceptions.RequestException as e:
        print(f"Error fetching player IDs: {e}")
        return []


def fetch_player_data(player_id):
    """
    Fetches detailed data for a specific player by their ID.
    """

    url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for player ID {player_id}: {e}")
        return None

    return response.json()


def map_status(status):
    """Simplifies the status codes into three categories."""
    if status == "a" or status == "d":
        return "available"
    else:
        return "unavailable"


def format_data(raw_player_data, player_id, id_name_map):
    """Formats the raw player data into a structured format suitable for csv."""
    if raw_player_data is None:
        return pd.DataFrame()  # Return empty DataFrame if no data

    upcoming_fixtures = raw_player_data.get("fixtures", [])
    gameweek_history = raw_player_data.get("history", [])

    if not gameweek_history and not upcoming_fixtures:
        return pd.DataFrame()  # Return empty DataFrame if no relevant data

    history_df = pd.DataFrame(gameweek_history)

    if upcoming_fixtures:
        fixtures_df = pd.DataFrame(upcoming_fixtures)

        # Add opponent_team column based on is_home
        fixtures_df["opponent_team"] = np.where(
            fixtures_df["is_home"], fixtures_df["team_a"], fixtures_df["team_h"]
        )

        # Rename columns to match history_df
        rename_map = {"event": "round", "is_home": "was_home"}
        fixtures_df = fixtures_df.rename(columns=rename_map)

        # Keep only columns that exist in both DataFrames
        columns_to_keep = [
            col for col in fixtures_df.columns if col in history_df.columns
        ]
        fixtures_df_filtered = fixtures_df[columns_to_keep]

        player_df = pd.concat([history_df, fixtures_df_filtered], ignore_index=True)
    else:
        player_df = history_df

    # Add player name using the id_name_map
    player_df["name"] = id_name_map.get(player_id, "Unknown Player").get(
        "name", "Unknown Player"
    )

    # Add the player position
    player_df["position"] = id_name_map.get(player_id, "Unknown").get(
        "position", "Unknown"
    )

    # Add the team name
    team_id = id_name_map.get(player_id, {}).get("team_id")
    player_df["team"] = team_id_name_map().get(team_id, "Unknown Team")

    # Add the player status
    player_status = id_name_map.get(player_id, {}).get("status", "u")
    player_df["status"] = map_status(player_status)

    return player_df


def fetch_and_format_player(player_id, id_name_map):
    """
    A single function for one worker to execute: fetch, format, and return a DataFrame.
    """
    raw_data = fetch_player_data(player_id)
    return format_data(raw_data, player_id, id_name_map)


def fetch_all_players_data():
    """
    Fetches and formats data for all players concurrently, then combines into a single DataFrame,
    and saves to a CSV file.
    """
    print("Starting to fetch all player data...")

    id_name_map = get_player_details()
    player_ids = list(id_name_map.keys())
    all_players_data = []

    # Use a ThreadPoolExecutor to fetch data concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # Create a dictionary to map futures to player_ids for easier processing
        future_to_player = {
            executor.submit(fetch_and_format_player, pid, id_name_map): pid
            for pid in player_ids
        }

        # Use tqdm to create a progress bar
        for future in tqdm(
            concurrent.futures.as_completed(future_to_player),
            total=len(player_ids),
            desc="Fetching latest player data",
        ):
            try:
                formatted_df = future.result()
                if not formatted_df.empty:
                    all_players_data.append(formatted_df)
            except Exception as exc:
                player_id = future_to_player[future]
                print(f"Player ID {player_id} generated an exception: {exc}")

    if all_players_data:
        print("\nCombining all player data...")
        combined_df = pd.concat(all_players_data, ignore_index=True)
        combined_df.sort_values(by=["round", "name"], inplace=True)
        combined_df.to_csv("data/players_data/merged_gw_25_26.csv", index=False)
        print("✅ All player data saved to data/players_data/merged_gw_25_26.csv")
    else:
        print("No player data was successfully fetched.")


if __name__ == "__main__":
    fetch_all_players_data()
