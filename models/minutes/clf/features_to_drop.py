def clf_features_to_drop():
    """
    Returns a list of features to drop based on prior analysis.
    """
    features_to_drop = [
        "goals_scored",
        "creativity",
        "assists",
        "bonus",
        "saves",
        "penalties_missed",
        "own_goals",
        "clean_sheets",
        "goals_conceded",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "ewma_xG",
        "ewma_xA",
        "ewma_threat",
        "ewma_creativity",
        "ewma_gc",
        "ewma_cs",
        "next_fixture_defense_rating",
        "next_fixture_attack_rating",
        "next_fixture_def_atk_ratio",
    ]

    return features_to_drop
