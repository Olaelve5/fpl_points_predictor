def get_selected_features():
    """
    Returns the list of features to keep based on average rank performance.
    """
    return [
        "predicted_minutes",
        "ewma_ict",
        "ewma_points",
        "ict_index",
        "next_fixture_atk_def_ratio",
        "value",
        "next_fixture_def_atk_ratio",
        "selected",
        "ewma_threat",
        "next_fixture_defense_rating",
        "next_is_home",
        "last_season_ppm",
        # "influence",
        # "pos_GK",
        # "self_team_defense_rating",
        # "next_fixture_attack_rating",
        # "bps",
        # "self_team_attack_rating",
        # "transfers_in",
        # "saves",
        # "pos_DEF",
        # "pos_MID",
        # "ewma_creativity",
        # "ewma_gc",
        # "creativity",
        # "transfers_out",
    ]
