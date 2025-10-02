import streamlit as st
from utils.load_csv_to_df import load_csv_to_df
import altair as alt

st.set_page_config(page_title="FPL Points Predictor", page_icon="⚽", layout="centered")


st.markdown("# :green[FPL] Points Predictor 🧠")

st.write("")  # spacer
st.write("")  # spacer
st.write("")  # spacer

file_path = "data/prediction_data/predicted_player_scores.csv"

df = load_csv_to_df(file_path)

players = df.sort_values(by="predicted_points", ascending=False)

st.markdown("#### Predicted Points for next gameweek")
st.dataframe(df)

st.write("")  # spacer
st.write("")  # spacer
st.write("")  # spacer


teams_file_path = "data/team_data/teams_25_26.csv"

teams_df = load_csv_to_df(teams_file_path)

teams_df = teams_df.drop(
    columns=[
        "id",
        "code",
        "short_name",
        "points",
        "team_division",
        "unavailable",
        "pulse_id",
        "draw",
        "form",
        "loss",
        "played",
        "position",
        "strength",
        "win",
        "strength_overall_home",
        "strength_overall_away",
    ]
)

st.markdown("#### Team Strengths")

edited_df = st.data_editor(
    teams_df,
    use_container_width=True,
    num_rows="fixed",  # or "dynamic" if you want to allow adding rows
    key="teams_editor",
    hide_index=True,  # <— hides the index
)
