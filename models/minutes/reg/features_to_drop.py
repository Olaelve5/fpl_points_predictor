def reg_features_to_drop():
    # Drops features found to be useless for regression task
    # Found with the feature importance analysis in features_search
    return [
        "goals_conceded",
        "assists",
        "bonus",
        "goals_scored",
        "penalties_missed",
        "own_goals",
        "clean_sheets",
        "yellow_cards",
        "next_is_home",
        "expected_goal_involvements",
        "next_fixture_defense_rating",
        "next_fixture_def_atk_ratio",
        "next_fixture_atk_def_ratio",
        "next_fixture_attack_rating",
        "transfers_balance",
        "expected_goals",
        "transfers_out",
        "ewma_points",
    ]
