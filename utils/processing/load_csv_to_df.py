import pandas as pd


def load_csv_to_df(file_path):
    try:
        # Try reading with error handling for malformed lines
        df = pd.read_csv(file_path, engine="python")

    except FileNotFoundError:
        print("Error: CSV not found.")
        exit()

    return df
