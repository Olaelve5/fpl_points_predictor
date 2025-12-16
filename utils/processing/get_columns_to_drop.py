def get_columns_to_drop(is_minutes_model=True):
    """Returns a list of columns to drop."""

    columns_to_drop = [
        "element",
        "modified",
        "season",
        "xP",
        "team_a_score",
        "team_h_score",
        "was_home",
        "next_fixture",
        "fixture",
        "kickoff_time",
        "GW",
        "status_available",
        "status_unavailable",
        "pos_AM",
        "defensive_contribution",
        "clearances_blocks_interceptions",
        "tackles",
        "starts",
        "recoveries",
        "penalties_saved",
        "red_cards",
        "total_points",
        "value_x_consistency",
    ]

    if not is_minutes_model:
        columns_to_drop.append("minutes")
        columns_to_drop.append("ewma_minutes")

    return columns_to_drop
