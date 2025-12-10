from utils.get_training_test_data import get_train_test_data
import joblib
import numpy as np
from sklearn.metrics import ndcg_score


def evaluate_rolling_horizon(model_path, feature_order_path, window_size=3):
    """
    Load model and test data -> make predictions
    and test precision on the top 10 players each gameweek.
    """

    try:
        model = joblib.load(model_path)
        feature_order = joblib.load(feature_order_path)
        print("Model loaded successfully.")
    except:
        print("Failed at loading model!")
        exit()

    _, x_test, _, y_test, test_metadata = get_train_test_data()
    x_test = x_test[feature_order]

    predictions = model.predict(x_test)
    df = test_metadata.copy()

    df["minutes_played"] = x_test["predicted_minutes"]
    df["actual_points"] = y_test.values if hasattr(y_test, "values") else y_test
    df["predicted_points"] = predictions.round(2)
    df["error"] = abs(df["actual_points"] - df["predicted_points"]).round(2)

    df["round"] = df["round"] + 1

    df.sort_values(by=["name", "round"], inplace=True)

    # Calculate rolling sums for actual and predicted points
    # based on the specified window size
    df["rolling_actual"] = (
        df.groupby(["name", "team"])["actual_points"]
        .transform(lambda x: x.rolling(window_size, min_periods=window_size).sum())
        .round(2)
    )

    df["rolling_pred"] = (
        df.groupby(["name", "team"])["predicted_points"]
        .transform(lambda x: x.rolling(window_size, min_periods=window_size).sum())
        .round(2)
    )

    # Drop rows where we don't have a full window (e.g., the first 2 weeks)
    df.dropna(subset=["rolling_actual", "rolling_pred"], inplace=True)

    ndcg_scores = []

    print(f"\n--- Rolling {window_size}-GW Performance (NDCG) ---")

    for gw, group in df.groupby("round"):
        # We only care about the top end of the table, but NDCG handles the full list well.
        # However, to be strict, we can just feed it the whole list.

        y_true = np.asarray([group["rolling_actual"].values])
        y_score = np.asarray([group["rolling_pred"].values])

        # Check if we have enough players to score
        if y_true.shape[1] > 1:
            # k=10: How good is the ranking of the Top 10 players?
            score = ndcg_score(y_true, y_score, k=10)
            ndcg_scores.append(score)

            # Optional: Print "Winner" for this 3-week block
            top_player = group.sort_values("rolling_actual", ascending=False).iloc[0][
                "name"
            ]
            top_pred = group.sort_values("rolling_pred", ascending=False).iloc[0][
                "name"
            ]

            print(
                f"GW {gw} (Window ending): NDCG: {score:.3f} | Best Actual: {top_player} | Best Pred: {top_pred}"
            )

    avg_ndcg = np.mean(ndcg_scores)
    print(f"\nAverage NDCG @ 10 over {window_size}-week windows: {avg_ndcg:.4f}")
    print("Interpretation: 1.0 is perfect ranking. >0.9 is god-tier. >0.7 is solid.")

    return df


predictions = evaluate_rolling_horizon(
    model_path="data/saved_models/points/forest_model.pkl",
    feature_order_path="data/feature_order/points_feature_order.pkl",
)

predictions.sort_values(
    by=["round", "rolling_actual"], ascending=[True, False], inplace=True
)

predictions.drop(columns=["total_points", "season"], inplace=True)

# Save predictions for inspection
predictions.to_csv("data/prediction_data/evaluation.csv", index=False)
