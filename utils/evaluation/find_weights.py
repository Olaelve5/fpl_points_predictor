import pandas as pd
import numpy as np
from itertools import product
from utils.processing.get_train_test_data import get_train_test_data
from models.points.voting_model import train_voting_model
from sklearn.metrics import ndcg_score


# --- Helper: Rolling NDCG Calculation ---
def calculate_rolling_ndcg(metadata, y_actual, y_pred, window_size=5):
    df = metadata.copy()
    df["actual"] = y_actual
    df["pred"] = y_pred

    # Sort for rolling calculation
    df.sort_values(by=["name", "round"], inplace=True)

    # Calculate Rolling Sums
    df["roll_act"] = df.groupby("name")["actual"].transform(
        lambda x: x.rolling(window_size, min_periods=window_size).sum()
    )
    df["roll_pred"] = df.groupby("name")["pred"].transform(
        lambda x: x.rolling(window_size, min_periods=window_size).sum()
    )
    df.dropna(subset=["roll_act", "roll_pred"], inplace=True)

    gw_scores = []
    for _, group in df.groupby("round"):
        y_true = np.asarray([group["roll_act"].values])
        y_score = np.asarray([group["roll_pred"].values])
        if y_true.shape[1] > 1:
            gw_scores.append(ndcg_score(y_true, y_score, k=10))

    return np.mean(gw_scores) if gw_scores else 0


# --- Main Optimization Function ---
def optimize_voting_weights(window_size=5):
    # 1. Generate Weight Combinations (Steps of 0.1)
    # We want w1 + w2 + w3 = 1.0
    steps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    combinations = [p for p in product(steps, repeat=3) if sum(p) == 1.0]

    # Optional: Filter out unlikely combos (e.g., 0 weight for XGBoost) to save time
    combinations = [c for c in combinations if c[0] >= 0.1]

    print(
        f"--- Starting Optimization: Testing {len(combinations)} Weight Combinations ---"
    )

    results = []

    # Pre-load data to speed up loop
    test_seasons = ["23_24", "24_25", "25_26"]
    data_cache = {}

    for season in test_seasons:
        print(f"Loading data for {season}...")
        x_tr, x_te, y_tr, y_te, meta = get_train_test_data(
            test_season=season, minutes_training=False
        )
        data_cache[season] = (x_tr, x_te, y_tr, y_te, meta)

    # 2. Loop Through Combinations
    for i, weights in enumerate(combinations):
        weight_str = f"{weights[0]}-{weights[1]}-{weights[2]}"  # XGB-RF-Linear
        scores = []

        print(f"[{i+1}/{len(combinations)}] Testing weights: {weight_str} ...", end=" ")

        for season in test_seasons:
            x_train, x_test, y_train, y_test, test_meta = data_cache[season]

            # Train with specific weights
            # UPDATED: Returns only 'model' based on your instruction (no external imputer)
            model = train_voting_model(x_train, y_train, weights=list(weights))

            # Predict
            # UPDATED: Pass raw X_test directly.
            # XGB/RF handle NaNs natively. Ridge (wrapper) handles them internally.
            preds = model.predict(x_test)

            # Evaluate
            score = calculate_rolling_ndcg(test_meta, y_test.values, preds, window_size)
            scores.append(score)

        avg_score = np.mean(scores).round(5)
        print(f"Score: {avg_score}")

        results.append(
            {
                "weights": str(weights),
                "xgb_w": weights[0],
                "rf_w": weights[1],
                "ridge_w": weights[2],
                "avg_score": avg_score,
            }
        )

    # 3. Save Results
    results_df = pd.DataFrame(results)
    results_df.sort_values(by="avg_score", ascending=False, inplace=True)

    output_path = "data/model_evaluation/weight_optimization_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\n--- OPTIMIZATION COMPLETE ---")
    print("Top 3 Weight Combinations:")
    print(results_df.head(3))
    print(f"Full results saved to {output_path}")


if __name__ == "__main__":
    optimize_voting_weights()
