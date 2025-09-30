import joblib
from utils.get_training_test_data import get_train_test_data


def train_model(model, save_path):
    X_train, X_test, y_train_log, y_test_log = get_train_test_data()

    sample_weights = y_train_log.clip(lower=1, upper=3)
    model.fit(X_train, y_train_log, sample_weight=sample_weights)

    # Save the model to a file
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")

    return model
