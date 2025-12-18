import pandas as pd
import numpy as np
import joblib
from utils.processing.get_last_completed_round import get_last_completed_round
from utils.processing.get_prediction_data import get_rows_to_predict
import lightgbm as lgb
from models.minutes.reg.features_to_drop import reg_features_to_drop
from models.minutes.clf.features_to_drop import clf_features_to_drop


# --- 1. Model Loading ---
def load_models():
    """Loads models and the feature list used during training."""
    try:
        classifier = joblib.load(
            "data/saved_models/minutes/minutes_classifier_model.pkl"
        )
        regressor = joblib.load(
            "data/saved_models/minutes/minutes_regression_model.pkl"
        )
        return classifier, regressor
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        exit()


def handle_goalkeeper_minutes(df):
    """
    Handles the goalkeeper minutes. Only one
    goalkeeper per team should be predicted to play.
    Returns dataframe with updated minutes
    """

    # Set ALL Goalkeepers to 0 minutes by default
    df.loc[df["pos_GK"] == 1, "predicted_minutes"] = 0

    # Find eligible GKs
    eligible_gks_mask = (df["pos_GK"] == 1) & (df["status"] != "unavailable")

    # Find the Index of the Best GK per Team AND Round
    best_gk_indices = (
        df[eligible_gks_mask].groupby(["team", "round"])["prob_play"].idxmax()
    )

    # Set those specific 'Number 1' keepers to 90 minutes
    df.loc[best_gk_indices.values, "predicted_minutes"] = 90

    return df


def minutes_prediction_pipeline():
    """
    Main pipeline to predict player minutes using a two-stage model.
    Returns two DataFrames - one with all features and one with only identifier features.
    Also returns the feature order used.
    """

    last_round = get_last_completed_round()
    print(f"--- 🚀 Predicting for GW{last_round + 1} --- \n")
    _, rows_to_predict = get_rows_to_predict(last_round, is_minutes_model=True)

    classifier, regressor = load_models()

    identifiers = rows_to_predict[
        [
            "name",
            "team",
            "value",
            "round",
            "pos_GK",
            "pos_DEF",
            "pos_MID",
            "pos_FWD",
            "status",
            "opponent_team",
        ]
    ].copy()

    # Drop useless features before prediction + identifiers
    features_to_drop_clf = clf_features_to_drop() + [
        "name",
        "team",
        "status",
        "opponent_team",
    ]
    features_to_drop_reg = reg_features_to_drop() + [
        "name",
        "team",
        "status",
        "opponent_team",
    ]

    clf_X = rows_to_predict.drop(columns=features_to_drop_clf, errors="ignore")
    reg_X = rows_to_predict.drop(columns=features_to_drop_reg, errors="ignore")

    # Make predictions
    print("\nRunning Two-Stage Minutes Model...")
    prob_playing = classifier.predict_proba(clf_X)[:, 1]
    raw_minutes = regressor.predict(reg_X)

    # Combine to get final minutes prediction
    results = identifiers.copy()
    results["prob_play"] = prob_playing.round(2)
    results["raw_minutes"] = np.round(raw_minutes, 1)

    results["xMins"] = (results["prob_play"] * results["raw_minutes"]).round(1)
    results["predicted_minutes"] = np.round(results["xMins"]).astype(int)

    # RULE A: The "Likely Starter" Snap-to-Grid
    # If a player has > 75% chance to play, assume they play their full duration.
    likely_starter_mask = results["prob_play"] >= 0.75
    results.loc[likely_starter_mask, "predicted_minutes"] = (
        results.loc[likely_starter_mask, "raw_minutes"].round().astype(int)
    )

    # Set minutes to 0 for unavailable players
    results.loc[results["status"] == "unavailable", "predicted_minutes"] = 0

    # Handle goalkeeper minutes
    results = handle_goalkeeper_minutes(results)

    # Add back identifiers to the full dataframe.
    # This dataframe will be used to make final predictions in
    # the main pipeline
    cols_to_add = [
        col for col in identifiers.columns if col not in rows_to_predict.columns
    ] + ["predicted_minutes"]
    X = pd.concat([rows_to_predict, results[cols_to_add]], axis=1)

    # 8. Save
    results.sort_values(
        by=["round", "predicted_minutes", "name"], ascending=[True, False, True]
    ).to_csv("data/prediction_data/combined_minutes_predictions.csv", index=False)

    X.sort_values(by=["round", "predicted_minutes"], ascending=[True, False]).to_csv(
        "data/prediction_data/df_with_minutes_pred.csv", index=False
    )

    print("Minutes successfully predicted ✅ \n")

    return results, X


def pipeline_for_testing(reg_model, clf_model, rows_to_predict):
    """
    Function to run the minutes prediction pipeline for testing.
    Should not be used to predict future GWs.

    :param rows_to_predict: DataFrame containing the rows to predict minutes for.
    """
    classifier = clf_model
    regressor = reg_model

    # Drop useless features before prediction
    features_to_drop_clf = clf_features_to_drop()
    features_to_drop_reg = reg_features_to_drop()
    clf_X = rows_to_predict.drop(columns=features_to_drop_clf, errors="ignore")
    reg_X = rows_to_predict.drop(columns=features_to_drop_reg, errors="ignore")

    # Make predictions
    print("\nRunning Two-Stage Minutes Model for Testing...")
    prob_playing = classifier.predict_proba(clf_X)[:, 1]
    raw_minutes = regressor.predict(reg_X)
    predicted_minutes = np.round(prob_playing * raw_minutes).astype(int).clip(0, 90)

    rows_to_predict["predicted_minutes"] = predicted_minutes

    print("Minutes successfully predicted for testing ✅ \n")

    return rows_to_predict


def train_both_models(training_data, params_clf, params_reg):
    X_train, X_test, y_train, y_test, _ = training_data

    # Drop features before training
    clf_x_train = X_train.drop(columns=clf_features_to_drop(), errors="ignore")
    clf_x_test = X_test.drop(columns=clf_features_to_drop(), errors="ignore")

    reg_x_train = X_train.drop(columns=reg_features_to_drop(), errors="ignore")
    reg_x_test = X_test.drop(columns=reg_features_to_drop(), errors="ignore")

    # ==========================================
    # MODEL 1: THE CLASSIFIER (Probability of Playing)
    # ==========================================
    print("\n--- Training Classifier (Did they play?) ---")

    # 1. Select Targets (Binary)
    y_train_clf = y_train["classifier_target"]
    y_test_clf = y_test["classifier_target"]

    # 2. Train (Uses FULL dataset - needs to see zeros)
    clf_model = lgb.LGBMClassifier(**params_clf)

    clf_model.fit(
        clf_x_train,
        y_train_clf,
        eval_set=[(clf_x_train, y_train_clf), (clf_x_test, y_test_clf)],
        eval_metric="logloss",  # AUC or logloss is best for binary classification
        callbacks=[
            lgb.early_stopping(100, verbose=True),
            lgb.log_evaluation(100),
        ],
    )

    # ==========================================
    # MODEL 2: THE REGRESSOR (Minutes if Playing)
    # ==========================================
    print("\n--- Training Regressor (How long do they play?) ---")

    # 1. Create Masks (Filter for > 5 mins)
    # We must filter BOTH train and test for the learning phase
    mask_train = y_train["regressor_target"] > 5
    mask_test = y_test["regressor_target"] > 5

    # 2. Apply Masks
    X_train_reg = reg_x_train.loc[mask_train]
    y_train_reg = y_train.loc[mask_train, "regressor_target"]

    X_test_reg = reg_x_test.loc[mask_test]
    y_test_reg = y_test.loc[mask_test, "regressor_target"]

    # 3. Train (Uses FILTERED dataset)
    reg_model = lgb.LGBMRegressor(**params_reg)

    reg_model.fit(
        X_train_reg,
        y_train_reg,
        eval_set=[(X_train_reg, y_train_reg), (X_test_reg, y_test_reg)],
        eval_metric="mae",
        callbacks=[
            lgb.early_stopping(100, verbose=True),
            lgb.log_evaluation(100),
        ],
    )

    return clf_model, reg_model


if __name__ == "__main__":
    results, X = minutes_prediction_pipeline()
