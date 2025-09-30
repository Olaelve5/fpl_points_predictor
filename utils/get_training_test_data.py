import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from utils.load_csv_to_df import load_csv_to_df


def get_train_test_data():
    try:
        original_df = pd.read_pickle("data/players_data/processed_data_cached.pkl")
    except FileNotFoundError:
        print(
            "Error: Processed data file not found in cache. Running data processing script..."
        )
        original_df = load_csv_to_df("data/players_data/players_22-23_to_24-25.csv")
        original_df.to_pickle("data/players_data/processed_data_cached.pkl")

    features = original_df.drop(columns=["target_score"])
    target = original_df["target_score"]
    target.clip(lower=0, inplace=True)

    # Split the data into training and testing sets - 85% train, 15% test
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.15, random_state=42
    )

    # Handle skewed target
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    # Save feature order for later use in predictions
    feature_order = X_train.columns.tolist()
    joblib.dump(feature_order, "data/saved_models/feature_order.pkl")

    return X_train, X_test, y_train_log, y_test_log
