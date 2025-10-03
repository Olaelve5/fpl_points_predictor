import pandas as pd
from lightgbm import LGBMClassifier
from utils.get_training_test_data import get_train_test_data
from models_operations.plot_model import plot_predictions


model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
)


if __name__ == "__main__":
    training_data = get_train_test_data(minutes_training=True, minutes_classifier=True)
    X_train, X_test, y_train, y_test = training_data

    trained_model = model.fit(X_train, y_train)

    # Plot predictions vs actual
    plot_predictions(trained_model, X_test, y_test, is_classifier=True)
