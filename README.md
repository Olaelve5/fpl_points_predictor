# FPL Points Predictor 🧠⚽

This repository contains a complete machine learning pipeline to predict Fantasy Premier League (FPL) player points for upcoming gameweeks.

The project fetches the latest data from the official FPL API, processes it through an extensive feature engineering pipeline, and uses a two-stage modeling process to first predict **minutes played** and then use those minutes to predict **total points**. The results are displayed in an interactive Streamlit dashboard.

-----

## 🎯 Features

  * **Automated Data Fetching**: Pulls the latest player data directly from the official FPL API to ensure predictions are based on the most current stats.
  * **Comprehensive Feature Engineering**: Generates a rich feature set, including:
      * Exponentially Weighted Moving Averages (EWMA) for key stats (points, xG, creativity, etc.).
      * Opponent fixture difficulty ratings (attack/defense strength).
      * Team strength ratings (self team attack/defense).
      * Rolling averages and consistency metrics.
  * **Two-Stage Prediction Pipeline**: Improves accuracy by treating minutes and points as separate (but related) problems:
    1.  **Minutes Model**: A combined Classifier (`LGBMClassifier`) and Regressor (`LGBMRegressor`) predicts the probability and number of minutes a player will play.
    2.  **Points Model**: A `StackingRegressor` (using LightGBM and HistGradientBoosting) predicts total points, using the *predicted minutes* as a critical input feature.
  * **Model Explainability**: Uses **SHAP** to visualize and debug individual player predictions (see `models/prediction_pipeline.ipynb`).

-----

## 🛠️ Tech Stack & Tools

  * **Core**: Python
  * **Data Manipulation**: Pandas, NumPy
  * **Machine Learning**: Scikit-learn, LightGBM
  * **Hyperparameter Tuning**: Optuna
  * **Web Dashboard**: Streamlit
  * **Data Visualization**: Matplotlib, Altair, SHAP
  * **Utilities**: Requests (for API), Joblib (for model persistence)

-----

## 📊 Data Sources

This project relies on two key data sources:

1.  **Official FPL API**: The endpoint `https://fantasy.premierleague.com/api/bootstrap-static/` is used to fetch live gameweek data, player stats, and team information.
2.  **Historical FPL Data**: The foundational historical data for training is based on the dataset provided by **vaastav/Fantasy-Premier-League**.
      * **Source Repository**: [https://github.com/vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)

-----

## ⚙️ How It Works: The Prediction Pipeline

The core logic is orchestrated by `models/prediction_pipeline.py`.

1.  **Check for Updates**: The app first checks the last completed gameweek from the FPL API and compares it to the latest local data. If stale, it fetches new data.
2.  **Get Prediction Rows**: The `get_prediction_data.py` script loads the latest raw data, applies all feature engineering steps (`add_columns.py`), and imputes player stats for all *future* gameweeks.
3.  **Stage 1: Predict Minutes**: The prepared rows are passed to the `minutes_prediction_pipeline`. This pipeline uses a classifier to predict the *probability* of playing and a regressor to predict the *number* of minutes. These are combined and clipped to a realistic 0-90 range.
4.  **Stage 2: Predict Points**: The *predicted minutes* from Stage 1 are added to the feature set. This complete dataset is then fed into the saved `stacking_model.pkl` to generate the final point predictions.
5.  **Format Output**: The final predictions are cleaned, formatted (e.g., converting value to standard FPL format), and sorted before being saved to CSV and displayed in the app.

-----

## 🚀 How to Run

1.  **Install Dependencies**
    *(Note: A `requirements.txt` file was not provided, but based on the imports, you will need the libraries listed in the "Tech Stack" section.)*

    ```bash
    pip install pandas numpy scikit-learn lightgbm optuna streamlit altair shap
    ```

2.  **Train the Models**
    *(This step is only necessary if the `.pkl` model files are not already present in `data/saved_models/`)*

    ```bash
    # Train the minutes models
    python models/minutes/minutes_classifier.py
    python models/minutes/minutes_pred.py

    # Train the main points model
    python models/stacking_model.py
    ```

3.  **Run the Streamlit App**

    ```bash
    streamlit run app.py
    ```

4.  **Get Predictions**
    Once the app is open in your browser, click the **"🔄 Update Predictions"** button to fetch the latest data and run the full pipeline. You can then filter the results by player name.
