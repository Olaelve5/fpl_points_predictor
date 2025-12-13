from utils.processing.get_train_test_data import get_train_test_data
import numpy as np
from sklearn.metrics import ndcg_score
import pandas as pd
from datetime import datetime
from xgboost import XGBRegressor


def run_backtest(model_name, window_size):
    test_seasons = ["23_24", "24_25", "25_26"]

    all_scores = []
    season_scores = {}

    print(f"--- STARTING WALK-FORWARD VALIDATION ({len(test_seasons)} Seasons) ---")

    for season in test_seasons:
        print(f"\nEvaluating on Season: {season}")

        x_train, x_test, y_train, y_test, test_metadata = get_train_test_data(
            minutes_training=False, test_season=season
        )

        model_params = {
            "objective": "reg:squarederror",
            "n_estimators": 500,
            "learning_rate": 0.01,
            "max_depth": 7, 
            "random_state": 42,
            "n_jobs": -1,
            "reg_alpha": 0.8,  # L1 Regularization
            "reg_lambda": 1.0,  # L2 Regularization
            "colsample_bytree": 0.8,
            "subsample": 0.8,
        }

        model = XGBRegressor(**model_params)

        print("Training model...")
        model.fit(x_train, y_train)
        print("Training complete.")

        predictions = model.predict(x_test)

        y_actual = y_test.values
        y_pred = predictions

        avg_season_score = calculate_rolling_ndcg(
            test_metadata, y_actual, y_pred, window_size
        ).round(4)

        all_scores.append(avg_season_score)
        season_scores[season] = avg_season_score

        print(f"Avg. score for season {season} is: {avg_season_score}")

    avg_score = np.mean(all_scores).round(4)
    season_scores["overall_score"] = avg_score
    print(f"\nAverage NDCG @ 10 over {len(test_seasons)} seasons: {avg_score}")

    save_score(model_name, season_scores, window_size)


def calculate_rolling_ndcg(metadata, y_actual, y_pred, window_size=3):
    """
    Helper to calculate average NDCG for a single dataframe/season
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
    for _, group in df.groupby("round"):
        y_true = np.asarray([group["roll_act"].values])
        y_score = np.asarray([group["roll_pred"].values])
        if y_true.shape[1] > 1:
            gw_scores.append(ndcg_score(y_true, y_score, k=10))

    return np.mean(gw_scores) if gw_scores else 0


def save_score(model_name, model_scores, window_size):
    try:
        scores_df = pd.read_csv("data/model_evaluation/model_scores.csv")
    except:
        print("No dataframe for evaluations found, creating a new one...")
        scores_df = pd.DataFrame(
            columns=[
                "model_name",
                "datetime",
                "window_size",
                "23_24",
                "24_25",
                "25_26",
                "overall_score",
            ]
        )

    new_row = {
        "model_name": model_name,
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "window_size": window_size,
        "23_24": model_scores.get("23_24", None),
        "24_25": model_scores.get("24_25", None),
        "25_26": model_scores.get("25_26", None),
        "overall_score": model_scores.get("overall_score", None),
    }

    scores_df.loc[len(scores_df)] = new_row

    # Sort based on window_size, then avg_score
    scores_df.sort_values(
        by=["window_size", "overall_score"], ascending=[False, False], inplace=True
    )
    scores_df.reset_index(drop=True, inplace=True)

    scores_df.to_csv("data/model_evaluation/model_scores.csv", index=False)
    print(f"✅ Saved scores for {model_name}")


if __name__ == "__main__":
    run_backtest(model_name="XGB_boosting_model", window_size=5)
