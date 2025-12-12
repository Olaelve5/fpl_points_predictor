import requests
from utils.processing.load_csv_to_df import load_csv_to_df
import pandas as pd


def get_last_completed_round():
    try:
        fpl_api_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        response = requests.get(fpl_api_url, timeout=10).json()

        last_completed_gw = 0
        for gw_event in response["events"]:
            if gw_event.get("finished", False):
                last_completed_gw = max(last_completed_gw, gw_event["id"])

        if last_completed_gw == 0:
            print("No gameweeks have been completed yet.")
            return 0

        return last_completed_gw

    except requests.exceptions.RequestException as e:
        print(f"Error fetching FPL API data: {e}")
        return 0


def get_last_completed_round_local():
    file_path = "data/players_data/merged_gw_25_26.csv"
    df = load_csv_to_df(file_path)
    if df is None:
        return 0

    latest_completed_round = df.dropna(subset=["total_points"])["round"].max()

    return latest_completed_round if pd.notna(latest_completed_round) else 0
