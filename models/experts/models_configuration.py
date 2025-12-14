from xgboost import XGBRFRegressor


def get_gk_model():
    # GK: Small sample size.
    # We use a constrained depth (10) to prevent memorizing specific matches.
    return XGBRFRegressor(
        objective="reg:squarederror",
        n_estimators=500,  # 500 trees is usually sufficient for RF
        learning_rate=1.0,  # RF always uses lr=1.0
        max_depth=10,  # Deeper than boosting, but still limited
        min_child_weight=3,  # Prevents leaves with too few samples
        random_state=42,
        n_jobs=-1,
        subsample=0.8,  # Train on 80% of data per tree (Bagging)
        colsample_bynode=0.8,  # 80% of features per node split
    )


def get_def_model():
    # DEF: Similar to GK, needs robustness.
    return XGBRFRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=1.0,
        max_depth=10,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
        subsample=0.8,
        colsample_bynode=0.8,
    )


def get_fwd_model():
    # FWD: High noise (goals are rare).
    # We use a slightly shallower depth (8) and higher child weight (5)
    # to force the model to find broader patterns.
    return XGBRFRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=1.0,
        max_depth=8,  # Shallower to prevent overfitting noise
        min_child_weight=5,  # Requires more players to form a rule
        reg_alpha=1.0,  # L1 Regularization
        reg_lambda=2.0,  # L2 Regularization
        random_state=42,
        n_jobs=-1,
        subsample=0.7,  # More aggressive randomness
        colsample_bynode=0.8,
    )


def get_mid_model():
    # MID: Largest dataset, can handle complexity.
    # We allow deeper trees (12) here.
    return XGBRFRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=1.0,
        max_depth=12,  # Deeper trees allowed for complex midfield roles
        min_child_weight=2,
        random_state=42,
        n_jobs=-1,
        subsample=0.8,
        colsample_bynode=0.8,
    )
