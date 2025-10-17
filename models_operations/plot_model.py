import joblib
from utils.get_training_test_data import get_train_test_data
from plotting import (
    plot_predictions,
    plot_permutations,
    plot_target_distribution,
    plot_tree,
    plot_data,
    plot_shap_values,
)


if __name__ == "__main__":
    model = joblib.load("data/saved_models/stacking_model.pkl")

    X_train, X_test, y_train_log, y_test_log = get_train_test_data(
        minutes_training=False
    )

    # plot_target_distribution(y_test_log)
    # plot_permutations(model, X_test, y_test_log)
    # plot_tree(model)
    # plot_data(original_df)
    plot_shap_values(model, X_train)
