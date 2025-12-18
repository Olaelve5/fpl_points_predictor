import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.inspection import permutation_importance
from models.minutes.clf.minutes_classifier import clf_params
from utils.processing.get_train_test_data import get_train_test_data


def analyze_feature_importance():
    """
    Trains a model and identifies useless features using Permutation Importance.
    """
    print("--- 🕵️‍♀️ Starting Feature Audit ---")

    # Get Data
    x_train, x_test, y_train_full, y_test_full, _ = get_train_test_data(
        minutes_training=True,
    )

    y_train = y_train_full["classifier_target"]
    y_test = y_test_full["classifier_target"]

    model = lgb.LGBMClassifier(**clf_params)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_test, y_test)],
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
        scoring="neg_log_loss",
        n_jobs=-1,
    )

    feature_imp["permutation_importance"] = perm_result.importances_mean

    # Sort by Permutation Importance (Most critical first)
    feature_imp = feature_imp.sort_values(by="permutation_importance", ascending=False)

    # --- VISUALIZATION ---
    plt.figure(figsize=(12, 8))
    sns.barplot(x="permutation_importance", y="feature", data=feature_imp.head(30))
    plt.title("Feature Importance (Impact on LogLoss)")
    plt.xlabel("LogLoss Reduction (Higher is Better)")
    plt.tight_layout()
    plt.show()

    # --- THE VERDICT ---
    # Define "Useless" as features that improve LogLoss by less than 0.0001 (or are negative)
    useless_features = feature_imp[feature_imp["permutation_importance"] <= 0.00001]

    print("\n--- ✂️ CUT CANDIDATES ---")
    print("These features contribute almost nothing (or add noise):")
    print(useless_features[["feature", "permutation_importance", "importance_split"]])

    print("\n--- ✅ KEEPERS ---")
    print("Top 10 Drivers of your model:")
    print(feature_imp[["feature", "permutation_importance"]].head(10))

    return feature_imp


if __name__ == "__main__":
    results = analyze_feature_importance()
