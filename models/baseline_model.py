import numpy as np


# Baseline model that predicts based on ewma_points
class BaselineModel:
    def predict(self, X_test):
        ewma_points = X_test["ewma_points"].fillna(0)  # Fill NaN with 0
        ewma_points = np.maximum(ewma_points, 0)  # Ensure non-negative
        return np.log1p(ewma_points)

    def predict_minutes(self, X_test):
        minutes = X_test["minutes"].fillna(0)  # Fill NaN with 0
        minutes = np.maximum(minutes, 0)  # Ensure non-negative
        return minutes
