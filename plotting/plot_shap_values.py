import shap


def plot_shap_values(model, X_train, base_model_name="lgbm_huber"):
    """
    Plot SHAP summary values for a base model from a stacking ensemble.

    Args:
        model: Stacking model with named_estimators_
        X_train: Training features
        base_model_name: Name of the base model to explain (default: "lgbm_huber")
    """
    shap.initjs()

    base_model = model.named_estimators_[base_model_name]
    explainer = shap.TreeExplainer(base_model)

    # Use a sample of the training data
    X_sample = X_train[0:1000]
    shap_values = explainer(X_sample)

    shap.summary_plot(shap_values, X_sample, show=True)
