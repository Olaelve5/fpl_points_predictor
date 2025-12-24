import pandas as pd
import numpy as np
from itertools import product
from utils.processing.get_train_test_data import get_train_test_data
from models.points.voting.model import train_voting_model
from models.evaluate_model import calculate_rolling_ndcg
from models.points.gradient_boost.get_columns_to_drop import get_columns_to_drop


def optimize_voting_weights(window_size=5):
    # Generate combinations of weights that sum to 1.0
    steps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    combinations = [p for p in product(steps, repeat=3) if np.isclose(sum(p), 1.0)]
    combinations = [c for c in combinations if c[0] >= 0.1]

    print(
        f"--- Starting Optimization: Testing {len(combinations)} Weight Combinations ---"
    )

    # Store scores for every combo across all seasons
    combo_scores = {c: [] for c in combinations}
    test_seasons = ["23_24", "24_25", "25_26"]

    for season in test_seasons:
        print(f"\nProcessing Season: {season}...")

        x_train, x_test, y_train, y_test, meta = get_train_test_data(
            test_season=season, minutes_training=False
        )

        # Keep only selected features
        x_train = x_train.drop(columns=get_columns_to_drop(), errors="ignore")
        x_test = x_test.drop(columns=get_columns_to_drop(), errors="ignore")

        # Pass dummy weights because they don't affect fitting, only prediction
        print("  Training Ensemble (Once)...")
        model = train_voting_model(x_train, y_train, weights=[0.33, 0.33, 0.33])

        # Access the fitted estimators directly
        pred_xgb = model.named_estimators_["xgb"].predict(x_test)
        pred_rf = model.named_estimators_["rf"].predict(x_test)
        pred_linear = model.named_estimators_["linear"].predict(x_test)

        # Stack into a matrix: Shape (n_samples, 3)
        base_preds = np.column_stack([pred_xgb, pred_rf, pred_linear])

        # D. Inner Loop: Weights (The Fast Part - Pure Math)
        print(f"  Calculating scores for {len(combinations)} combinations...")

        y_trainue = y_test.values

        for weights in combinations:
            # Vectorized Weighted Average: (Matrix) dot (Vector)
            final_pred = base_preds @ np.array(weights)

            # Evaluate
            score = calculate_rolling_ndcg(meta, y_trainue, final_pred, window_size)
            combo_scores[weights].append(score)

    # Aggregate & Save Results
    print("\n--- Aggregating Results ---")
    final_results = []

    for weights, scores in combo_scores.items():
        avg_score = np.mean(scores)
        final_results.append(
            {
                "weights": str(weights),
                "xgb_w": weights[0],
                "rf_w": weights[1],
                "ridge_w": weights[2],
                "avg_score": round(avg_score, 5),
                "scores_by_season": str([round(s, 4) for s in scores]),
            }
        )

    results_df = pd.DataFrame(final_results)
    results_df.sort_values(by="avg_score", ascending=False, inplace=True)

    output_path = "data/model_evaluation/weight_optimization_results.csv"
    results_df.to_csv(output_path, index=False)

    print("--- OPTIMIZATION COMPLETE ---")
    print("Top 3 Weight Combinations:")
    print(results_df.head(3)[["xgb_w", "rf_w", "ridge_w", "avg_score"]])


if __name__ == "__main__":
    optimize_voting_weights()
