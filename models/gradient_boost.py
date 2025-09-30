from lightgbm import LGBMRegressor
from models_operations.train import train_model
from models_operations.test import compare_model_to_baseline
from utils.get_training_test_data import get_train_test_data
import numpy as np


model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    num_leaves=50,
    random_state=42,
    n_jobs=-1,
    alpha=0.8,
    objective="huber",
)


if __name__ == "__main__":
    X_train, X_test, y_train_log, y_test_log = get_train_test_data()

    trained_model = train_model(model, "data/saved_models/lgbm_model.pkl", plot=True)

    model_preds = np.expm1(trained_model.predict(X_test))
    compare_model_to_baseline(model_preds, np.expm1(y_test_log), X_test)

    print("Training complete.")
