def get_columns_to_drop():
    """
    Returns the list of features to keep based on average rank performance.
    """
    return [
        "minutes",
        "last_active_minutes",
        "transfers_balance",
        "selected",
        "own_goals",
    ]
