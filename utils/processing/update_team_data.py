import requests
import json
import pandas as pd

# The official FPL API endpoint
url = "https://fantasy.premierleague.com/api/bootstrap-static/"

try:
    # Send a GET request to the URL
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

    # Parse the JSON response
    data = response.json()

    # Extract the list of teams
    teams_data = data["teams"]

    # Convert the list of teams into a pandas DataFrame
    teams_df = pd.DataFrame(teams_data)

    # Save the DataFrame to a CSV file
    teams_df.to_csv("data/team_data/teams_25_26.csv", index=False)

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
