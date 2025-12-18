def clf_features_to_drop():
    """
    Returns a list of features to drop based on prior analysis.
    """
    features_to_drop = [
        "goals_scored",
        "assists",
        "bonus",
        "saves",
        "penalties_missed",
        "own_goals",
        "clean_sheets",
        "goals_conceded",
        "expected_goals",
        "expected_assists",
        "ewma_xG",
        "ewma_xA",
        "ewma_threat",
        "ewma_creativity",
        "ewma_gc",
        "ewma_cs",
        "next_fixture_defense_rating",
        "next_fixture_attack_rating",
    ]

    return features_to_drop
