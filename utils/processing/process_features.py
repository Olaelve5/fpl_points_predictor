from utils.processing.get_columns_to_drop import get_columns_to_drop
import pandas as pd


def process_features(df, is_training=False, is_minutes_model=True):
    """
    Centralized logic to clean/format data for both training and prediction.
    Returns full dataframe + features and targets
    """
    df = df.copy()

    # Remove rows where target is NaN
    if is_training:
        subset_cols = ["predicted_minutes"] if is_minutes_model else ["target_score"]
        df.dropna(subset=subset_cols, inplace=True)

    # Remove Assistant Managers (pos_AM)
    if "pos_AM" in df.columns:
        df = df[df["pos_AM"] != 1].copy()
        df.drop(columns=["pos_AM"], inplace=True, errors="ignore")

    # Drop Unwanted Columns
    cols_to_drop = get_columns_to_drop(is_minutes_model=is_minutes_model)
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # Define features and target
    if is_minutes_model:
        target = pd.DataFrame(index=df.index)
        features = df.drop(columns=["target_score", "predicted_minutes"])
        target["classifier_target"] = (df["predicted_minutes"] > 1).astype(int)
        target["regressor_target"] = df["predicted_minutes"]
    else:
        features = df.drop(
            columns=["target_score", "minutes", "ewma_minutes"], errors="ignore"
        )
        target = df["target_score"]

    # Training Specific: Drop Identifiers
    if is_training:
        drop_list = [
            "name",
            "team",
            "opponent_team",
            "status",
            "position",
            "kickoff_time",
        ]
        features.drop(columns=drop_list, inplace=True, errors="ignore")

    if isinstance(target, pd.DataFrame):
        target = target.clip(lower=0)
    elif isinstance(target, pd.Series):
        target = target.clip(lower=0)

    return df, features, target
