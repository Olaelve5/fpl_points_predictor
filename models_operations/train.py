import joblib
from utils.get_training_test_data import get_train_test_data
from models_operations.plot_model import plot_predictions


def train_model(model, save_path, plot=True, with_sample_weights=True):
    X_train, X_test, y_train_log, y_test_log = get_train_test_data()

    if with_sample_weights:
        sample_weights = y_train_log.clip(lower=1, upper=3)
        model.fit(X_train, y_train_log, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train_log)

    # Save the model to a file
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")

    # Plot predictions
    if plot:
        plot_predictions(y_test_log, model.predict(X_test))

    return model
