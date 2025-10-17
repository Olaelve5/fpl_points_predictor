import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def plot_permutations(model, X_test, y_test):
    """Plot permutation feature importance for the model."""
    importances = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    feature_importances = importances.importances_mean

    sorted_idx = feature_importances.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align="center")
    plt.yticks(range(len(sorted_idx)), [X_test.columns[i] for i in sorted_idx])
    plt.xlabel("Mean Decrease in RMSE")
    plt.title("Top 28 Feature Importances (Permutation Importance)")
    plt.show()
