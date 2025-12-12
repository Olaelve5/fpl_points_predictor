import pandas as pd
import joblib
from utils.load_csv_to_df import load_csv_to_df
from utils.processing.feature_processing import process_features
from utils.processing.add_columns import add_columns


def get_train_test_data(minutes_training=False, minutes_classifier=False):
    """Function to return processed training and test data."""

    try:
        # Load the Master CSV
        original_df = load_csv_to_df("data/players_data/players_20-21_to_25-26.csv")
    except FileNotFoundError:
        print("Error: CSV file not found.")
        return None, None, None, None, None

    full_df = do_feature_engineering(original_df)

    # Filter for minutes training if needed
    if minutes_training and not minutes_classifier:
        full_df = full_df[full_df["predicted_minutes"] >= 5].copy()

    # We save these columns now because process_features drops them
    meta_cols = ["name", "team", "position", "season", "round", "total_points"]
    existing_meta_cols = [c for c in meta_cols if c in full_df.columns]
    metadata = full_df[existing_meta_cols].copy()

    full_df = process_features(
        full_df, is_training=True, is_points_model=not minutes_training
    )

    # Drop rows where targets are NaN
    full_df.dropna(subset=["target_score", "predicted_minutes"], inplace=True)

    # Use the index to ensure we drop the exact same rows
    metadata = metadata.loc[full_df.index]

    # Define features and target
    if minutes_training:
        features = full_df.drop(columns=["target_score", "predicted_minutes"])
        if minutes_classifier:
            target = (full_df["predicted_minutes"] > 1).astype(int)
        else:
            target = full_df["predicted_minutes"]
    else:
        features = full_df.drop(columns=["target_score"])
        target = full_df["target_score"]

    target.clip(lower=0, inplace=True)

    # Use metadata to split train/test based on season
    # The latest season "25_26" is the test set
    train_mask = metadata["season"] != "25_26"
    test_mask = metadata["season"] == "25_26"

    X_train = features.loc[train_mask]
    y_train = target.loc[train_mask]
    X_test = features.loc[test_mask]
    y_test = target.loc[test_mask]

    # Get metadata specifically for the test set (for evaluation later)
    test_meta = metadata.loc[test_mask]

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


def do_feature_engineering(df):
    """Applies feature engineering in groups based on seasons."""

    history_map, pos_avg_map = get_historic_stats_map(df)

    base_team_file_path = (
        "/Users/ola/Documents/FPL_Price_Predictor/data/team_data/teams_"
    )
    processed_seasons = []

    for season, season_df in df.groupby("season"):
        full_path = f"{base_team_file_path}{season}.csv"
        processed_season_df = add_columns(
            season_df, full_path, history_map, pos_avg_map
        )
        processed_seasons.append(processed_season_df)

    combined_df = pd.concat(processed_seasons, ignore_index=True)

    return combined_df


def get_historic_stats_map(df):
    """
    Returns a dictionary mapping (Player, Current_Season) -> Previous_Season_Stats
    """
    # Filter for valid games (where they actually played)
    played_df = df[df["minutes"] > 45].copy()

    # Group by Player and Season to get averages
    season_stats = (
        played_df.groupby(["name", "season"])
        .agg(
            {
                "total_points": "mean",
                "minutes": "mean",
            }
        )
        .reset_index()
    )

    season_stats["total_points"] = season_stats["total_points"].round(2)
    season_stats["minutes"] = season_stats["minutes"].round(0)

    history_map = {}

    # Helper to shift season string: "23_24" -> "24_25"
    # This allows us to look up "23_24" stats using the "24_25" key
    def get_next_season(s_str):
        try:
            start_year = int(s_str.split("_")[0])
            end_year = int(s_str.split("_")[1])
            return f"{start_year+1}_{end_year+1}"
        except:
            return None

    for _, row in season_stats.iterrows():
        next_season = get_next_season(row["season"])
        if next_season:
            # The Key is (Name, Next_Season)
            history_map[(row["name"], next_season)] = {
                "history_pps": row["total_points"],
                "history_mpg": row["minutes"],
            }

    # Calculate Positional Averages (For Imputation of the First Season)
    # This is the fallback for when history_map fails
    position_averages = (
        played_df.groupby("position")["total_points"].mean().round(2).to_dict()
    )

    return history_map, position_averages
