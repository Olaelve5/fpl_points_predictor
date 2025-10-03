def get_columns_to_drop(is_minutes_training=False):
    """Returns a list of columns to drop."""

    columns_to_drop = [
        # Identification columns
        "name",
        "team",
        "element",
        "modified",
        "season",
        # Stats from previous gameweek
        "total_points",
        "goals_scored",
        "goals_conceded",
        "saves",
        "assists",
        "own_goals",
        "penalties_saved",
        "penalties_missed",
        "tackles",
        "clearances_blocks_interceptions",
        "defensive_contribution",
        "yellow_cards",
        "recoveries",
        "threat",
        "influence",
        "bonus",
        "bps",
        "clean_sheets",
        # Expected stats
        "xP",
        "expected_goals",
        "expected_assists",
        "expected_goals_conceded",
        "expected_goal_involvements",
        # transfers info
        "selected",
        "transfers_balance",
        "transfers_in",
        "transfers_out",
        # fixture info
        "team_a_score",
        "team_h_score",
        "opponent_team",
        "was_home",
        "next_fixture",
        "fixture",
        "kickoff_time",
        # Use round instead of GW
        "GW",
        # EWMA columns
        "ewma_xA",
        "ewma_xG",
        "ewma_gc",
        "ewma_bps",
        "ewma_threat",
        "ewma_cs",
    ]

    if is_minutes_training:
        columns_to_drop.extend(
            [
                # EWMA columns
                "ewma_points",
                "ewma_minutes",
                "ewma_creativity",
                "ewma_def_contr",
                # Other columns
                "creativity",
                "ict_index",
                "pos_FWD",
                "pos_DEF",
                "self_team_attack_rating",
                "self_team_defence_rating",
                "next_fixture_def_atk_ratio",
            ]
        )
    return columns_to_drop
