import pandas as pd
import requests
import io
import numpy as np
from datetime import datetime


def pull_club_elos():
    try:
        url = "http://api.clubelo.com/" + datetime.now().strftime("%Y-%m-%d")
        response = requests.get(url, timeout=10)
        elo_df = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
        prem_elo = elo_df[(elo_df["Country"] == "ENG") & (elo_df["Level"] == 1)].copy()
    except Exception as e:
        print(f"⚠️ Error fetching ClubElo: {e}")
        return

    raw_max = prem_elo["Elo"].max()
    raw_min = prem_elo["Elo"].min()

    new_max = 1320
    new_min = 1110

    # Squashes raw elo's to range used in predictions (with some room for extra ratings)
    # and adds it as a new column to the dataframe
    prem_elo["normalized_elo"] = new_min + (
        ((prem_elo["Elo"] - raw_min) * (new_max - new_min)) / (raw_max - raw_min)
    ).round(0)

    home_advantage = 20

    prem_elo["strength_overall_home"] = prem_elo["normalized_elo"] + home_advantage
    prem_elo["strength_overall_away"] = prem_elo["normalized_elo"] - home_advantage

    # Correct some club names to match FPL naming conventions
    prem_elo["Club"] = prem_elo["Club"].replace(
        {
            "Forest": "Nott'm Forest",
        }
    )

    return prem_elo


def pull_extra_ratings():
    # Pull xG data from fotmob
    try:
        response = requests.get("https://www.fotmob.com/api/leagues?id=47", timeout=10)
        data = response.json()
        xg_data = data["table"][0]["data"]["table"]["xg"]
    except:
        print("⚠️ Error fetching from Fotmob api")
        return

    df = pd.DataFrame(xg_data)
    df = df[["shortName", "xg", "xgConceded"]]
    df.rename(columns={"shortName": "team"}, inplace=True)

    max_xg = df["xg"].max()
    min_xg = df["xg"].min()
    max_xg_conceded = df["xgConceded"].max()
    min_xg_conceded = df["xgConceded"].min()

    max_additional_rating = 40
    min_additional_rating = -40

    # Calculate additional ratings based on xG and xG conceded
    # will be added to offensive and defensive ratings respectively
    df["additional_off_rating"] = min_additional_rating + (
        (df["xg"] - min_xg)
        * (max_additional_rating - min_additional_rating)
        / (max_xg - min_xg)
    ).round(0)

    df["additional_def_rating"] = -1 * (
        min_additional_rating
        + (
            (df["xgConceded"] - min_xg_conceded)
            * (max_additional_rating - min_additional_rating)
            / (max_xg_conceded - min_xg_conceded)
        )
    ).round(0)

    df["team"] = df["team"].replace(
        {
            "Nottm Forest": "Nott'm Forest",
        }
    )

    return df


def combine_ratings():
    elo_df = pull_club_elos()
    extra_ratings_df = pull_extra_ratings()

    if elo_df is None or extra_ratings_df is None:
        print("⚠️ Could not combine ratings due to previous errors.")
        return

    combined_df = pd.merge(
        extra_ratings_df,
        elo_df,
        how="inner",
        left_on="team",
        right_on="Club",
    )

    combined_df["strength_attack_home"] = (
        combined_df["strength_overall_home"] + combined_df["additional_off_rating"]
    ).round(0)
    combined_df["strength_attack_away"] = (
        combined_df["strength_overall_away"] + combined_df["additional_off_rating"]
    ).round(0)

    combined_df["strength_defence_home"] = (
        combined_df["strength_overall_home"] + combined_df["additional_def_rating"]
    ).round(0)
    combined_df["strength_defence_away"] = (
        combined_df["strength_overall_away"] + combined_df["additional_def_rating"]
    ).round(0)

    return combined_df[
        [
            "Club",
            "strength_overall_home",
            "strength_overall_away",
            "strength_attack_home",
            "strength_attack_away",
            "strength_defence_home",
            "strength_defence_away",
        ]
    ]


def save_ratings_to_csv(filepath="data/team_data/updated_teams_25_26.csv"):
    combined_df = combine_ratings()

    # rename some names to match FPL conventions
    combined_df["Club"] = combined_df["Club"].replace(
        {
            "Man United": "Man Utd",
            "Tottenham": "Spurs",
        }
    )

    # Load original team data to preserve other columns
    original_team_data = pd.read_csv("data/team_data/teams_25_26.csv")
    original_columns = original_team_data.columns.tolist()

    # Drop the columns from original that will be replaced
    columns_to_replace = [
        "strength_overall_home",
        "strength_overall_away",
        "strength_attack_home",
        "strength_attack_away",
        "strength_defence_home",
        "strength_defence_away",
    ]

    original_team_data = original_team_data.drop(columns=columns_to_replace)

    # Merge the new ratings into the original team data
    updated_team_data = pd.merge(
        original_team_data, combined_df, how="left", left_on="name", right_on="Club"
    )

    # Drop the duplicate 'Club' column from the merge
    updated_team_data = updated_team_data.drop(columns=["Club"])

    # Restore the original column order
    updated_team_data = updated_team_data[original_columns]

    updated_team_data.to_csv(filepath, index=False)
    print(f"✅ Updated team ratings saved to {filepath}")


if __name__ == "__main__":
    save_ratings_to_csv()
