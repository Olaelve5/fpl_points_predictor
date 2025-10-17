import streamlit as st
from utils.load_csv_to_df import load_csv_to_df
import altair as alt
from models.prediction_pipeline import prediction_pipeline
import os

st.set_page_config(page_title="FPL Points Predictor", page_icon="⚽", layout="centered")

st.markdown("# :green[FPL] Points Predictor 🧠")

st.write("")  # spacer

# Add a button to run predictions
if st.button("🔄 Update Predictions"):
    with st.spinner("Generating predictions... This may take a minute"):
        try:
            # Run the prediction pipeline
            final_predictions_df = prediction_pipeline()
            # Save the results
            final_predictions_df.to_csv(
                "data/prediction_data/predicted_points_with_minutes.csv", index=False
            )
            st.success("✅ Predictions updated successfully!")
            # Use the newly generated predictions
            df = final_predictions_df
        except Exception as e:
            st.error(f"Error generating predictions: {e}")
            # Load existing predictions if available
            file_path = "data/prediction_data/predicted_points_with_minutes.csv"
            if os.path.exists(file_path):
                df = load_csv_to_df(file_path)
            else:
                st.error("No prediction data available. Please try again.")
                df = None
else:
    # Load existing predictions
    file_path = "data/prediction_data/predicted_points_with_minutes.csv"
    if os.path.exists(file_path):
        df = load_csv_to_df(file_path)
    else:
        st.warning("No prediction data found. Click 'Update Predictions' to generate.")
        df = None

# Only show filters and table if we have data
if df is not None:
    # Add name filter input
    name_filter = st.text_input("🔎 Filter by player name")

    # Filter dataframe based on input
    if name_filter:
        filtered_df = df[df["name"].str.contains(name_filter, case=False)]
    else:
        filtered_df = df

    st.markdown("#### Predicted Points for next gameweek")

    # Show number of players displayed
    st.caption(f"Showing {len(filtered_df)} players")

    # Display the filtered dataframe
    st.dataframe(filtered_df)
