from lightgbm import LGBMRegressor
from data_processing.process_csv import load_csv
from sklearn.model_selection import train_test_split
from plot_model import plot_predictions
import joblib
import pandas as pd
import numpy as np


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

# Handle skewed target
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)


def train_model(X_train, y_train_log):
    print("Training model...")
    model = LGBMRegressor(
        boosting_type="gbdt",
        objective="regression",
        n_estimators=500,
        num_leaves=31,
        max_depth=15,
        learning_rate=0.01,
        random_state=42,
    )
    sample_weights = y_train_log.clip(lower=1, upper=5)

    model.fit(X_train, y_train_log, sample_weight=sample_weights)

    # Save the model to a file
    joblib.dump(model, "saved_models/lgbm_model.pkl")
    print("Model saved to lgbm_model.pkl")

    return model


def find_best_parameters():
    from sklearn.model_selection import GridSearchCV

    # Define the parameter grid you want to search
    param_grid = {
        "num_leaves": [31, 40, 50, 60],
        "max_depth": [7, 10, 15, -1],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [200, 500, 1000],
    }

    # Initialize the model
    model = LGBMRegressor(random_state=42, force_row_wise=True)

    # Set up the grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=3,  # 3-fold cross-validation
        n_jobs=-1,
        verbose=2,
    )

    # Fit the grid search to your data (using the log-transformed target)
    grid_search.fit(X_train, y_train_log)

    # Get the best model
    print(f"Best parameters found: {grid_search.best_params_}")


if __name__ == "__main__":
    model = train_model(X_train, y_train_log)
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    plot_predictions(y_test, y_pred)

    # find_best_parameters()
