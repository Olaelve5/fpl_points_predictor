import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_selection import RFE
from models.points.gradient_boost.model import model_params
from utils.processing.get_train_test_data import get_train_test_data


def run_robust_rfe():
    """
    Iterates over multiple seasons, performing RFE on each to rank features.
    Aggregates rankings to identify the most stable and important features.
    Saves the final rankings to a CSV file.
    """

    seasons = ["23_24", "24_25", "25_26"]

    feature_rankings = {}

    print(f"--- Starting Robust RFE across {len(seasons)} seasons ---")

    for season in seasons:
        print(f"\nProcessing Season {season}...")

        X_train, _, y_train, _, _ = get_train_test_data(
            test_season=season, minutes_training=False
        )

        model = xgb.XGBRegressor(**model_params)

        # Run RFE to rank all features
        selector = RFE(estimator=model, n_features_to_select=1, step=1)
        selector.fit(X_train, y_train)

        # Store rankings
        for feature, rank in zip(X_train.columns, selector.ranking_):
            if feature not in feature_rankings:
                feature_rankings[feature] = []
            feature_rankings[feature].append(rank)

    # --- Aggregation ---
    print("\n--- Calculating Average Rankings ---")
    final_stats = []

    for feature, ranks in feature_rankings.items():
        avg_rank = np.mean(ranks)
        std_rank = np.std(ranks)
        final_stats.append(
            {
                "feature": feature,
                "avg_rank": avg_rank,
                "std_rank": std_rank,
                "ranks_per_season": ranks,
            }
        )

    df = pd.DataFrame(final_stats)
    df.sort_values(by="avg_rank", ascending=True, inplace=True)

    # Save
    output_path = "data/model_evaluation/robust_feature_ranking.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved robust rankings to {output_path}")
    print("\nTOP 10 MOST STABLE FEATURES:")
    print(df.head(10)[["feature", "avg_rank", "std_rank"]])


if __name__ == "__main__":
    run_robust_rfe()
