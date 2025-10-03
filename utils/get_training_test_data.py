import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from utils.load_csv_to_df import load_csv_to_df


def get_train_test_data(minutes_training=False, minutes_classifier=False):
    try:
        original_df = load_csv_to_df("data/players_data/players_22-23_to_25-26.csv")
    except FileNotFoundError:
        print(
            "Error: CSV file not found. Please ensure the file exists at the specified path."
        )
        return None, None, None, None

    original_df.dropna(subset=["target_score", "minutes_next"], inplace=True)

    features = original_df.drop(columns=["target_score", "minutes_next"])

    if minutes_training:
        if minutes_classifier:
            target = (original_df["minutes_next"] > 0).astype(int)
        else:
            target = original_df["minutes_next"]
    else:
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
        "defensive_contribution",
        "ewma_gc",
        "expected_goal_involvements",
        "ewma_bps",
    ]

    if minutes_training:
        columns_to_drop.extend(
            [
                "ewma_points",
                "ewma_minutes",
                "ewma_creativity",
                "creativity",
                "ict_index",
                "pos_FWD",
                "pos_DEF",
                "self_team_attack_rating",
                "self_team_defence_rating",
                "ewma_def_contr",
                "next_fixture_def_atk_ratio",
            ]
        )

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

    return X_train, X_test, y_train_log, y_test_log
