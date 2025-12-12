import shap
import joblib
from utils.processing.get_train_test_data import get_train_test_data
from utils.processing.get_prediction_data import get_rows_to_predict

shap.initjs()


def plot_shap_values(model, X_train, base_model_name="lgbm_huber"):
    """
    Plot SHAP summary values for a base model from a stacking ensemble.

    Args:
        model: Stacking model with named_estimators_
        X_train: Training features
        base_model_name: Name of the base model to explain (default: "lgbm_huber")
    """

    base_model = model.named_estimators_[base_model_name]
    explainer = shap.TreeExplainer(base_model)

    # Use a sample of the training data
    X_sample = X_train[0:2000]
    shap_values = explainer(X_sample)

    shap.summary_plot(shap_values, X_sample, show=True)


if __name__ == "__main__":
    model = joblib.load("data/saved_models/stacking_model.pkl")

    X_train, X_test, y_train_log, y_test_log = get_train_test_data(
        minutes_training=False
    )

    plot_shap_values(model, X_train)
