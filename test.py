from data_processing.process_csv import load_csv
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)


try:
    original_df = pd.read_pickle("players_data/processed_data.pkl")
except FileNotFoundError:
    print(
        "Error: Processed data file not found in cache. Running data processing script..."
    )
    original_df = load_csv("players_data/players_22-23_to_24-25.csv")
    original_df.to_pickle("players_data/processed_data.pkl")
    exit()


features = original_df.drop(columns=["target_score"])
target = original_df["target_score"]
target.clip(lower=0, inplace=True)

# Split the data into training and testing sets - 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)


def rmsle(y_true, y_pred):
    """Root Mean Squared Logarithmic Error"""
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def test_model(model, X_test, y_test):
    # 1. Make predictions and inverse the log transformation
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    # 2. Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)  # <-- Add this
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # A more direct way to get RMSE
    r2 = r2_score(y_test, y_pred)

    # 3. Print the results clearly
    print(f"--- Model Performance ---")
    print(f"MAE: {mae:.2f} (On average, predictions are off by this many points)")
    print(f"RMSE: {rmse:.2f} (Penalizes larger errors more than MAE)")
    print(f"R^2: {r2:.2f} (Explains the variance in the target variable)")


model = joblib.load("saved_models/lgbm_model.pkl")

test_model(model, X_test, y_test)


# Create and test with a improved baseline model
class BaselineModel:
    def predict(self, X_test):
        # Handle NaN values in ewma_points
        ewma_points = X_test["ewma_points"].fillna(0)  # Fill NaN with 0
        ewma_points = np.maximum(ewma_points, 0)  # Ensure non-negative
        return np.log1p(ewma_points)


baseline_model = BaselineModel()
print("\nBaseline Model Performance:")
test_model(baseline_model, X_test, y_test)


def test_agains_single_sampel(sample, model):
    sample_df = pd.DataFrame([sample])
    target = sample_df["target_score"].values[0]

    prediction_log = model.predict(sample_df)
    prediction = np.expm1(prediction_log)

    return prediction[0], target
