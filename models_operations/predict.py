import numpy as np
from utils.load_csv_to_df import load_csv_to_df
import joblib
from utils.get_prediction_data import get_rows_to_predict


def make_predictions(model, is_minutes_model=False):
    """Function to make predictions using the trained model and latest data."""

    raw_df = load_csv_to_df(
        "/Users/ola/Documents/FPL_Price_Predictor/data/players_data/merged_gw_25_26.csv"
    )

    identifiers = raw_df[["name", "team", "position", "value"]].copy()

    # Re-order columns to fit the training data order
    if is_minutes_model:
        feature_order = joblib.load("data/saved_models/minutes_feature_order.pkl")
    else:
        feature_order = joblib.load("data/saved_models/feature_order.pkl")

    rows_to_predict, round_to_predict = get_rows_to_predict(raw_df)
    rows_to_predict = rows_to_predict[feature_order]

    # Save to csv for inspection
    rows_to_predict.to_csv(
        "data/prediction_data/processed_player_data_for_prediction.csv", index=False
    )

    y_pred_log = model.predict(rows_to_predict)

    if is_minutes_model:
        y_pred = y_pred_log  # No transformation for minutes model
    else:
        y_pred = np.expm1(y_pred_log)

    results_df = identifiers.loc[rows_to_predict.index].copy()
    results_df["round_predicted"] = round_to_predict

    if not is_minutes_model:
        results_df["predicted_points"] = y_pred
        results_df.sort_values(by="predicted_points", ascending=False, inplace=True)
        results_df["predicted_points"] = results_df["predicted_points"].round(1)
    else:
        results_df["predicted_minutes"] = np.clip(y_pred, 0, 90)  # Clip to valid range

        results_df.sort_values(by="predicted_minutes", ascending=False, inplace=True)
        results_df["predicted_minutes"] = results_df["predicted_minutes"].round(0)

    print(results_df.head(20))

    results_df.to_csv("data/prediction_data/predicted_player_scores.csv", index=False)
    print("Predictions saved to predicted_player_scores.csv")
