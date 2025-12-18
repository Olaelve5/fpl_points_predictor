import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.inspection import permutation_importance
from utils.processing.get_train_test_data import get_train_test_data
from models.minutes.reg.minutes_regressor import reg_params


def analyze_regressor_feature_importance():
    """
    Trains the Minutes Regressor and identifies useless features.
    """
    print("--- 🕵️‍♀️ Starting Regressor Feature Audit ---")

    # Get Data
    training_data = get_train_test_data(minutes_training=True, test_season="24_25")
    x_train_full, x_test_full, y_train_full, y_test_full, _ = training_data

    # FILTER: Only audit on players who actually played (> 10 mins)
    # The regressor's job is to predict duration, not appearance.
    TRAIN_THRESHOLD = 10

    train_mask = y_train_full["regressor_target"] > TRAIN_THRESHOLD
    x_train = x_train_full.loc[train_mask]
    y_train = y_train_full.loc[train_mask, "regressor_target"]

    test_mask = y_test_full["regressor_target"] > TRAIN_THRESHOLD
    x_test = x_test_full.loc[test_mask]
    y_test = y_test_full.loc[test_mask, "regressor_target"]

    print(
        f"Auditing on {len(x_train)} training samples (Players > {TRAIN_THRESHOLD} mins)"
    )

    # Train Model
    model = LGBMRegressor(**reg_params)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # Native Feature Importance
    feature_imp = pd.DataFrame(
        {
            "feature": x_train.columns,
            "importance_gain": model.booster_.feature_importance(
                importance_type="gain"
            ),
            "importance_split": model.booster_.feature_importance(
                importance_type="split"
            ),
        }
    )

    # Permutation Importance (The Truth Test)
    print("Calculating Permutation Importance (this takes a minute)...")
    perm_result = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    feature_imp["permutation_importance"] = perm_result.importances_mean

    # Sort by Permutation Importance (Most critical first)
    feature_imp = feature_imp.sort_values(by="permutation_importance", ascending=False)

    # --- VISUALIZATION ---
    plt.figure(figsize=(12, 8))
    sns.barplot(x="permutation_importance", y="feature", data=feature_imp.head(30))
    plt.title("Feature Importance (Impact on MAE)")
    plt.xlabel("MAE Reduction (Higher is Better)")
    plt.tight_layout()
    plt.show()

    # --- THE VERDICT ---
    # Useless = features that reduce error by less than 0.01 minutes (essentially zero)
    useless_features = feature_imp[feature_imp["permutation_importance"] <= 0.001]

    print("\n--- ✂️ CUT CANDIDATES ---")
    print("These features contribute almost nothing to minutes prediction:")
    print(useless_features[["feature", "permutation_importance"]])

    print("\n--- ✅ KEEPERS ---")
    print("Top 10 Drivers of Minutes:")
    print(feature_imp[["feature", "permutation_importance"]].head(10))

    return feature_imp


if __name__ == "__main__":
    analyze_regressor_feature_importance()
