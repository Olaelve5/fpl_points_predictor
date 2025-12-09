import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os


def plot_shap_explanation(player_name, df_path, model_path, feature_order_path):
    """
    Plots the SHAP waterfall plot for a specific player's prediction.

    Args:
        player_name (str): Name of the player to explain (e.g., "B.Fernandes").
        df_path (str): Path to the CSV containing the player data (rows to predict).
        model_path (str): Path to the saved model (.pkl).
        feature_order_path (str): Path to the saved feature order list (.pkl).
    """

    # 1. Load Data and Resources
    print(f"Loading data from {df_path}...")
    try:
        df = pd.read_csv(df_path)
    except FileNotFoundError:
        print("Error: DataFrame file not found.")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print("Loading model and feature order...")
    model = joblib.load(model_path)
    feature_cols = joblib.load(feature_order_path)

    # 2. Find the Player
    # We look for the exact name. You might want to use str.contains for flexibility.
    player_row = df[df["name"] == player_name]

    if player_row.empty:
        print(f"Error: Player '{player_name}' not found in the DataFrame.")
        return

    # Take the first instance found (usually the upcoming fixture)
    instance = player_row.iloc[[0]]
    print(
        f"Found row for {player_name}. Predicting against: {instance['opponent_team'].values[0] if 'opponent_team' in instance else 'Unknown'}"
    )

    # 3. Prepare Features
    # Ensure we only use the columns the model was trained on, in the correct order
    try:
        X = instance[feature_cols]
    except KeyError as e:
        print(f"Error: Missing columns in DataFrame required by model: {e}")
        return

    # 4. Create Explainer and Calculate SHAP values
    # TreeExplainer is optimized for XGBoost/LightGBM/CatBoost/RandomForest
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # 5. Plot
    plt.figure(figsize=(12, 8))

    # The waterfall plot shows how each feature pushes the prediction from the base value
    shap.plots.waterfall(shap_values[0], max_display=15, show=False)

    # Customizing the plot
    plt.title(
        f"Why did the model predict {model.predict(X)[0]:.2f} for {player_name}?",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()


# --- Example Usage ---
if __name__ == "__main__":
    # Update these paths to match your actual file locations
    BASE_DIR = "/Users/ola/Documents/FPL_Price_Predictor"

    # You can pass the 'final_predictions.csv' or the processed rows you used for prediction
    DATA_PATH = "data/prediction_data/df_with_minutes_pred.csv"
    MODEL_PATH = "data/saved_models/points/forest_model.pkl"
    FEATURES_PATH = "data/feature_order/points_feature_order.pkl"

    # Run for Bruno
    plot_shap_explanation(
        player_name="B.Fernandes",
        df_path=DATA_PATH,
        model_path=MODEL_PATH,
        feature_order_path=FEATURES_PATH,
    )
    plot_shap_explanation(
        player_name="Cunha",
        df_path=DATA_PATH,
        model_path=MODEL_PATH,
        feature_order_path=FEATURES_PATH,
    )
