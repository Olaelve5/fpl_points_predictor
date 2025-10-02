import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from utils.load_csv_to_df import load_csv_to_df


def get_train_test_data():
    try:
        original_df = load_csv_to_df("data/players_data/players_22-23_to_25-26.csv")
    except FileNotFoundError:
        print(
            "Error: CSV file not found. Please ensure the file exists at the specified path."
        )
        return None, None, None, None

    features = original_df.drop(columns=["target_score"])
    target = original_df["target_score"]
    target.clip(lower=0, inplace=True)

    # Drop unimportant features
    columns_to_drop = [
        "threat",
        "ewma_threat",
        "ewma_xA",
        "ewma_xG",
        "influence",
        "tackles",
        "ewma_cs",
        "yellow_cards",
        "recoveries",
        "clearances_blocks_interceptions",
        "ewma_minutes",
        "defensive_contribution",
        "ewma_gc",
        "expected_goal_involvements",
        "ewma_bps",
    ]

    features.drop(columns=columns_to_drop, inplace=True, errors="ignore")

    # Split the data into training and testing sets - 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=50
    )

    # Handle skewed target
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    # Save feature order for later use in predictions
    feature_order = X_train.columns.tolist()
    joblib.dump(feature_order, "data/saved_models/feature_order.pkl")

    return X_train, X_test, y_train_log, y_test_log
