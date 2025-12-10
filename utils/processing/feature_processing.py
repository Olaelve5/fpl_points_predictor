from utils.processing.get_columns_to_drop import get_columns_to_drop


def process_features(df, is_training=False):
    """
    Centralized logic to clean/format data for BOTH training and prediction.
    """
    df = df.copy()

    # 1. Remove Assistant Managers (pos_AM)
    if "pos_AM" in df.columns:
        df = df[df["pos_AM"] != 1].copy()
        df.drop(columns=["pos_AM"], inplace=True, errors="ignore")

    # 2. Standardize Position Booleans (Ensure True/False)
    position_cols = ["pos_DEF", "pos_FWD", "pos_GK", "pos_MID"]
    for col in position_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(bool)

    # 3. Drop Unwanted Columns
    # We assume get_columns_to_drop() returns the standard list for all models now
    cols_to_drop = get_columns_to_drop()
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # 4. Training Specific: Drop Identifiers
    # During prediction, we might want to keep these, but for training X/y we must drop them.
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

    return df
