from utils.processing.get_columns_to_drop import get_columns_to_drop


def process_features(
    df, is_training=False, is_minutes_model=True, is_minutes_classifier=False
):
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
        df.drop(columns=drop_list, inplace=True, errors="ignore")

    # Define features and target
    if is_minutes_model:
        features = df.drop(columns=["target_score", "predicted_minutes"])
        if is_minutes_classifier:
            target = (df["predicted_minutes"] > 1).astype(int)
        else:
            target = df["predicted_minutes"]
    else:
        features = df.drop(columns=["target_score"])
        target = df["target_score"]

    target.clip(lower=0, inplace=True)

    return df, features, target