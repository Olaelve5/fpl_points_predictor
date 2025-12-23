import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import ndcg_score
from models.points.gradient_boost.model import model_params
from utils.processing.get_train_test_data import get_train_test_data


def calculate_rolling_ndcg(metadata, y_actual, y_pred, window_size=5):
    """
    Recalculates NDCG for the permuted predictions.
    """
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
    # Note: Ensure your metadata column is named correctly ('round' vs 'gameweek')
    group_col = "round" if "round" in df.columns else "gameweek"

    for _, group in df.groupby(group_col):
        y_true = np.asarray([group["roll_act"].values])
        y_score = np.asarray([group["roll_pred"].values])
        if y_true.shape[1] > 1:
            score = ndcg_score(y_true, y_score, k=10)
            gw_scores.append(score)

    return np.mean(gw_scores) if gw_scores else 0


def run_ndcg_permutation_importance(window_size=5):
    seasons = ["23_24", "24_25", "25_26"]
    feature_impacts = {}

    print(f"--- Starting NDCG Permutation Importance ({len(seasons)} Seasons) ---")

    for season in seasons:
        print(f"\nProcessing Season {season}...")

        # 1. Get Data (Need Test set this time!)
        x_train, x_test, y_train, y_test, test_metadata = get_train_test_data(
            test_season=season, minutes_training=False
        )

        # Ensure we are only using numeric features for training
        # (Assuming get_train_test_data returns some non-numeric cols, filter if needed)
        # x_train = x_train.select_dtypes(include=np.number)
        # x_test = x_test[x_train.columns]

        # 2. Train Model ONCE per season
        model = xgb.XGBRegressor(**model_params)
        model.fit(x_train, y_train, verbose=False)

        # 3. Establish Baseline Score
        baseline_preds = model.predict(x_test)
        baseline_ndcg = calculate_rolling_ndcg(
            test_metadata, y_test.values, baseline_preds, window_size
        )
        print(f"  > Baseline NDCG: {baseline_ndcg:.4f}")

        # 4. Permutation Loop
        # We shuffle one column at a time and see how much NDCG drops
        print(f"  > Testing {len(x_test.columns)} features...")

        for feature in x_test.columns:
            # Save original column
            original_col = x_test[feature].copy()

            # SHUFFLE the feature (break the relationship)
            x_test[feature] = np.random.permutation(x_test[feature].values)

            # Predict & Score
            permuted_preds = model.predict(x_test)
            permuted_ndcg = calculate_rolling_ndcg(
                test_metadata, y_test.values, permuted_preds, window_size
            )

            # Calculate Impact (Higher is better)
            # Impact = How much score DID WE LOSE?
            impact = baseline_ndcg - permuted_ndcg

            if feature not in feature_impacts:
                feature_impacts[feature] = []
            feature_impacts[feature].append(impact)

            # RESTORE original column for next iteration
            x_test[feature] = original_col

    # --- Aggregation ---
    print("\n--- Aggregating Results ---")
    final_stats = []

    for feature, impacts in feature_impacts.items():
        avg_impact = np.mean(impacts)
        std_impact = np.std(impacts)
        final_stats.append(
            {
                "feature": feature,
                "avg_ndcg_drop": avg_impact,  # Positive = Important, Negative = Noise
                "std_ndcg_drop": std_impact,
                "raw_impacts": impacts,
            }
        )

    df = pd.DataFrame(final_stats)
    # Sort by Highest Drop (Most Important)
    df.sort_values(by="avg_ndcg_drop", ascending=False, inplace=True)

    output_path = "data/model_evaluation/ndcg_feature_importance.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved rankings to {output_path}")
    print("\nTOP 20 MOST VITAL FEATURES (Highest NDCG Loss when removed):")
    print(df.head(20)[["feature", "avg_ndcg_drop", "std_ndcg_drop"]])

    print("\nTOP 30 CANDIDATES FOR REMOVAL (Negative or Zero Impact):")
    print(df.tail(30)[["feature", "avg_ndcg_drop", "std_ndcg_drop"]])


if __name__ == "__main__":
    run_ndcg_permutation_importance()
