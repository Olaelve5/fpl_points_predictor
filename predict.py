import numpy as np
from data_processing.process_csv import load_csv, process_df
import joblib


def make_predictions(model, raw_df):
    identifiers = raw_df[["name", "team", "position", "value"]].copy()

    processed_df = process_df(
        raw_df, "/Users/ola/Documents/FPL_Price_Predictor/team_data/teams_25_26.csv",
        drop_last_gw=True
    )

    print(processed_df)

    print(f"Max round in data: {processed_df['round'].max()}")

    features = processed_df.drop(columns=["target_score"], errors="ignore")

    round_to_predict = 5

    rows_to_predict = features[features["round"] == round_to_predict - 1]

    print(f"Features shape for prediction: {rows_to_predict.shape}")

    y_pred_log = model.predict(rows_to_predict)
    y_pred = np.expm1(y_pred_log)

    results_df = identifiers.loc[rows_to_predict.index].copy()
    results_df["round_predicted"] = round_to_predict
    results_df["predicted_target_score"] = y_pred
    results_df.sort_values(by="predicted_target_score", ascending=False, inplace=True)
    results_df["predicted_target_score"] = results_df["predicted_target_score"].round(2)
    print(results_df.head(20))

    results_df.to_csv("predicted_player_scores.csv", index=False)
    print("Predictions saved to predicted_player_scores.csv")


if __name__ == "__main__":
    model = joblib.load("saved_models/lgbm_model.pkl")

    raw_df = load_csv(
        "/Users/ola/Documents/FPL_Price_Predictor/players_data/merged_gw_25_26.csv"
    )

    make_predictions(model, raw_df)
