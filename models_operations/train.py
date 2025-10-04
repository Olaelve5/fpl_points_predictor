import joblib
from models_operations.plot_model import plot_predictions
import numpy as np


def train_model(
    model,
    training_data,
    save_path,
    plot=True,
    with_sample_weights=True,
    is_minutes_model=False,
):
    X_train, X_test, y_train_log, y_test_log = training_data

    if with_sample_weights:
        sample_weights = y_train_log.clip(lower=1, upper=3)
        model.fit(X_train, y_train_log, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train_log)

    # Save the model to a file
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")

    model_preds = model.predict(X_test)
    
    if not is_minutes_model:
        model_preds = np.expm1(model_preds)
        y_test_log = np.expm1(y_test_log)

    # Plot predictions
    if plot:
        plot_predictions(y_test_log, model_preds)

    return model
