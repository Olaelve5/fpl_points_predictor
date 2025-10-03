import numpy as np
from data_processing.process_csv import process_df
from utils.load_csv_to_df import load_csv_to_df
import joblib


def make_predictions(model, raw_df, is_minutes_model=False):
    identifiers = raw_df[["name", "team", "position", "value"]].copy()

    processed_df = process_df(
        raw_df,
        "/Users/ola/Documents/FPL_Price_Predictor/data/team_data/teams_25_26.csv",
        is_training=False,
    )

    # Re-order columns to fit the training data order
    feature_order = joblib.load("data/saved_models/feature_order.pkl")
    processed_df = processed_df.reindex(columns=feature_order, fill_value=0)

    # Save to csv for inspection
    processed_df.to_csv(
        "data/prediction_data/processed_player_data_for_prediction.csv", index=False
    )

    latest_completed_round = processed_df.dropna(subset=["starts"])["round"].max()
    round_to_predict = latest_completed_round + 1
    rows_to_predict = processed_df[processed_df["round"] == round_to_predict - 1]

    print(f"Features shape for prediction: {rows_to_predict.shape}")

    y_pred_log = model.predict(rows_to_predict)
    y_pred = np.expm1(y_pred_log)

    results_df = identifiers.loc[rows_to_predict.index].copy()
    results_df["round_predicted"] = round_to_predict

    if not is_minutes_model:
        results_df["predicted_points"] = y_pred
        results_df.sort_values(by="predicted_points", ascending=False, inplace=True)
        results_df["predicted_points"] = results_df["predicted_points"].round(1)
    else:
        results_df["predicted_minutes"] = np.clip(y_pred, 0, 90)  # Clip to valid range

        # 90 mins threshold
        starter_threshold = 75
        # If predicted minutes are above the threshold, set them to 90
        results_df.loc[
            results_df["predicted_minutes"] > starter_threshold, "predicted_minutes"
        ] = 90

        results_df.sort_values(by="predicted_minutes", ascending=False, inplace=True)
        results_df["predicted_minutes"] = results_df["predicted_minutes"].round(0)

    print(results_df.head(20))

    results_df.to_csv("data/prediction_data/predicted_player_scores.csv", index=False)
    print("Predictions saved to predicted_player_scores.csv")


if __name__ == "__main__":
    model = joblib.load("data/saved_models/voting_model.pkl")

    raw_df = load_csv_to_df(
        "/Users/ola/Documents/FPL_Price_Predictor/data/players_data/merged_gw_25_26.csv"
    )

    make_predictions(model, raw_df)
