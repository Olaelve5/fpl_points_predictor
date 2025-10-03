import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from utils.load_csv_to_df import load_csv_to_df
from utils.get_columns_to_drop import get_columns_to_drop
from utils.add_columns import add_columns


def get_train_test_data(minutes_training=False, minutes_classifier=False):
    """Function to return processed training and test data."""

    try:
        original_df = load_csv_to_df("data/players_data/players_22-23_to_25-26.csv")
    except FileNotFoundError:
        print(
            "Error: CSV file not found. Please ensure the file exists at the specified path."
        )
        return None, None, None, None

    processed_df = apply_feature_engineering(original_df)

    # Drop target columns
    processed_df.dropna(subset=["target_score", "minutes_next"], inplace=True)
    features = processed_df.drop(columns=["target_score", "minutes_next"])

    # Set the target type based on training type
    if minutes_training:
        if minutes_classifier:
            target = (processed_df["minutes_next"] > 0).astype(int)
        else:
            target = processed_df["minutes_next"]
    else:
        target = processed_df["target_score"]

    # Clip target in case of negative score
    target.clip(lower=0, inplace=True)

    # Drop unwanted columns
    columns_to_drop = get_columns_to_drop(minutes_training)
    features.drop(columns=columns_to_drop, inplace=True, errors="ignore")

    # Split the data into training and testing sets - 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=50
    )

    # Handle skewed target
    if not minutes_training:
        y_train_log = np.log1p(y_train)
        y_test_log = np.log1p(y_test)
    else:
        y_train_log = y_train
        y_test_log = y_test

    # Save feature order for later use in predictions
    feature_order = X_train.columns.tolist()
    joblib.dump(feature_order, "data/saved_models/feature_order.pkl")

    # Save data for inspection
    X_train.to_csv("data/training_data/X_train.csv", index=False)
    X_test.to_csv("data/training_data/X_test.csv", index=False)
    y_train.to_csv("data/training_data/y_train.csv", index=False)
    y_test.to_csv("data/training_data/y_test.csv", index=False)

    return X_train, X_test, y_train_log, y_test_log


def apply_feature_engineering(df):
    """Applies feature engineering in groups based on seasons."""
    base_team_file_path = (
        "/Users/ola/Documents/FPL_Price_Predictor/data/team_data/teams_"
    )
    processed_seasons = []

    for season, season_df in df.groupby("season"):
        full_path = f"{base_team_file_path}{season}.csv"
        processed_season_df = add_columns(season_df, full_path)
        processed_seasons.append(processed_season_df)

    combined_df = pd.concat(processed_seasons, ignore_index=True)

    # Remove players with position 'AM' (assistant managers)
    combined_df["pos_AM"] = combined_df["pos_AM"].fillna(0)
    combined_df = combined_df[combined_df["pos_AM"] == 0].copy()
    combined_df.drop(columns=["pos_AM"], inplace=True, errors="ignore")

    # Ensure 'next_is_home' is integer type
    combined_df["next_is_home"] = combined_df["next_is_home"].fillna(0).astype(int)

    return combined_df
