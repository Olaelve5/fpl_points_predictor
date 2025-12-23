import pandas as pd
import joblib
from utils.processing.load_csv_to_df import load_csv_to_df
from utils.processing.process_features import process_features
from utils.processing.feature_engineering import do_feature_engineering


def get_train_test_data(minutes_training=False, test_season="25_26"):
    """Function to return processed training and test data."""

    try:
        # Load the Master CSV
        original_df = load_csv_to_df("data/players_data/players_20-21_to_25-26.csv")
    except FileNotFoundError:
        print("Error: CSV file not found.")
        return None, None, None, None, None

    full_df = do_feature_engineering(original_df)

    # We save these columns now because process_features drops them
    meta_cols = ["name", "team", "position", "season", "round", "total_points"]
    existing_meta_cols = [c for c in meta_cols if c in full_df.columns]
    metadata = full_df[existing_meta_cols].copy()

    full_df, features, target = process_features(
        full_df,
        is_training=True,
        is_minutes_model=minutes_training,
    )

    # Use the index to ensure we drop the exact same rows
    metadata = metadata.loc[full_df.index]

    # Use metadata to split train/test based on season
    # Only include seasons prior to test_season in training
    metadata["season_int"] = metadata["season"].str.replace("_", "").astype(int)
    test_season_int = int(test_season.replace("_", ""))
    train_mask = metadata["season_int"] < test_season_int
    test_mask = metadata["season_int"] == test_season_int

    X_train = features.loc[train_mask]
    y_train = target.loc[train_mask]
    X_test = features.loc[test_mask]
    y_test = target.loc[test_mask]

    # Metadata for test set
    test_meta = metadata.loc[test_mask].copy()
    test_meta.drop("season_int", axis=1, inplace=True)

    # Save feature order for later use in prediction
    feature_order = X_train.columns.tolist()
    if minutes_training:
        joblib.dump(feature_order, "data/feature_order/minutes_feature_order.pkl")
    else:
        joblib.dump(feature_order, "data/feature_order/points_feature_order.pkl")

    # Save training sets for inspection
    X_train.to_csv("data/training_data/X_train.csv", index=False)
    y_train.to_csv("data/training_data/y_train.csv", index=False)
    X_test.to_csv("data/training_data/X_test.csv", index=False)
    y_test.to_csv("data/training_data/y_test.csv", index=False)

    return X_train, X_test, y_train, y_test, test_meta
