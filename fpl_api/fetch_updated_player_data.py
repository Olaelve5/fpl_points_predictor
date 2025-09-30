import requests
import json


def get_all_player_ids():
    """
    Fetches the main bootstrap data and returns a list of all player IDs.
    """
    print("Fetching all player IDs...")
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return [player["id"] for player in data["elements"]]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching player IDs: {e}")
        return []


def fetch_player_data(player_id):
    print("Fetching data for player ID:", player_id)
    url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for player ID {player_id}: {e}")
        return None

    print("Data fetched successfully for player ID:", player_id)
    fixtures = response.json().get("fixtures", [])
    history = response.json().get("history", [])

    return {"fixtures": fixtures, "history": history}


if __name__ == "__main__":
    player_ids = get_all_player_ids()

    all_players_data = []

    test_player = 1  # Test with a single player ID first
    player_data = fetch_player_data(test_player)
    print(json.dumps(player_data, indent=2))

