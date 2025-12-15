from xgboost import XGBRFRegressor


def get_gk_model():
    # GK: Small sample size.
    # We use a constrained depth (10) to prevent memorizing specific matches.
    return XGBRFRegressor(
        n_estimators=400,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.8,
        min_child_weight=10,
        # --- REGULARIZATION ---
        reg_lambda=1.2,  # L2 (Ridge): Good for reducing variance/noise
        # ----------------------
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )


def get_def_model():
    # DEF: Similar to GK, needs robustness.
    return XGBRFRegressor(
        n_estimators=400,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.8,
        min_child_weight=10,
        # --- REGULARIZATION ---
        reg_lambda=1.2,  # L2 (Ridge): Good for reducing variance/noise
        # ----------------------
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )


def get_fwd_model():
    # FWD: High noise (goals are rare).
    # We use a slightly shallower depth (8) and higher child weight (5)
    # to force the model to find broader patterns.
    return XGBRFRegressor(
        n_estimators=400,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.8,
        min_child_weight=10,
        # --- REGULARIZATION ---
        reg_lambda=1.2,  # L2 (Ridge): Good for reducing variance/noise
        # ----------------------
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )


def get_mid_model():
    # MID: Largest dataset, can handle complexity.
    # We allow deeper trees (12) here.
    return XGBRFRegressor(
        n_estimators=400,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.8,
        min_child_weight=10,
        # --- REGULARIZATION ---
        reg_lambda=1.2,  # L2 (Ridge): Good for reducing variance/noise
        # ----------------------
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )
