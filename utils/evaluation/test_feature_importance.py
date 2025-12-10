import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns
from utils.processing.get_training_test_data import get_train_test_data
import joblib


def plot_feature_importance(model, feature_names):
    # Get importance
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        # For XGBoost sometimes it's a dict
        importance = model.get_score(importance_type="gain")
        print("Check model type for importance extraction")
        return

    # Create DataFrame
    feat_imp = pd.DataFrame({"feature": feature_names, "importance": importance})
    feat_imp = feat_imp.sort_values("importance", ascending=False)

    # Plot Top 20
    plt.figure(figsize=(10, 8))
    sns.barplot(x="importance", y="feature", data=feat_imp.head(20))
    plt.title("Top 20 Features")
    plt.show()

    # Identify useless features (e.g., absolute zero importance)
    useless = feat_imp[feat_imp["importance"] == 0]["feature"].tolist()
    print(f"Completely useless features (Importance = 0): {useless}")

    return feat_imp


# Usage:

X_train, X_test, y_train, y_test, _ = get_train_test_data()
model = joblib.load("data/saved_models/points/forest_model.pkl")

features = plot_feature_importance(model, X_train.columns)
