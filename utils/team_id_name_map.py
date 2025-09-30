import pandas as pd

def team_id_name_map(file_path="/Users/ola/Documents/FPL_Price_Predictor/team_data/teams_25_26.csv"):
    try:
        df = pd.read_csv(file_path)
        team_id_map = pd.Series(df.name.values, index=df.id).to_dict()
        return team_id_map
    except FileNotFoundError:
        print("Error: Team data CSV not found.")
        return {}