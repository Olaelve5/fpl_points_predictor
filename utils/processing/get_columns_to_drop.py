def get_columns_to_drop(is_points_model=False):
    """Returns a list of columns to drop."""

    columns_to_drop = [
        # Identification columns
        # Dont drop name and team as they are needed for merging and identification
        # "name",
        # "team",
        "element",
        "modified",
        "season",
        # Stats from previous gameweek
        # "total_points",
        # "goals_scored",
        # "goals_conceded",
        # "saves",
        # "assists",
        # "own_goals",
        # "penalties_saved",
        # "penalties_missed",
        # "tackles",
        # "clearances_blocks_interceptions",
        # "defensive_contribution",
        # "yellow_cards",
        # "recoveries",
        # "threat",
        # "influence",
        # "bonus",
        # "starts",
        # "creativity",
        # "bps",
        # "clean_sheets",
        # Expected stats
        "xP",
        # "expected_goals",
        # "expected_assists",
        # "expected_goals_conceded",
        # "expected_goal_involvements",
        # transfers info
        # "selected",
        # "transfers_balance",
        # "transfers_in",
        # "transfers_out",
        # fixture info
        "team_a_score",
        "team_h_score",
        # "opponent_team",
        "was_home",
        "next_fixture",
        "fixture",
        "kickoff_time",
        # Use round instead of GW
        "GW",
        # EWMA columns
        # "ewma_xA",
        # "ewma_xG",
        # "ewma_gc",
        # "ewma_bps",
        # "ewma_threat",
        # "ewma_cs",
        # "ewma_creativity",
        # Available status
        "status_available",
        "status_unavailable",
        "pos_AM",
        "ewma_def_contr",
        "defensive_contribution",
        "clearances_blocks_interceptions",
        "tackles",
        "starts",
        "recoveries",
        "penalties_saved",
        "red_cards",
        # --- 2. REDUNDANT ECHOES (Corr > 0.90) ---
        "total_points",  # Rely on 'ewma_points' instead
        "ewma_xGI",  # Redundant with xG + xA
        "rolling_avg_minutes",  # Redundant with ewma_minutes or predicted_minutes
        "minutes_consistency",  # Redundant
        "value_x_consistency",  # Redundant
        "ewma_influence",  # Often overlaps with Threat/Creativity
        "ewma_bps",
    ]

    if is_points_model:
        # For points model, drop target variable if present
        columns_to_drop.append("minutes")
        columns_to_drop.append("ewma_minutes")

    return columns_to_drop
