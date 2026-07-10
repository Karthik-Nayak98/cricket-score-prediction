import streamlit as st
import random
import requests
import os

BACKEND_URI = os.getenv("BACKEND_URI")

if not BACKEND_URI:
    raise RuntimeError("BACKEND_URI is not configured")

def send_prediction_request(
    batting_team, bowling_team, overs, balls, runs_scored, wickets_fallen, city
):
    print(BACKEND_URI)
    return requests.post(
        f"{BACKEND_URI}/predict",
        json={
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "overs": overs,
            "balls": balls,
            "runs_scored": runs_scored,
            "wickets_fallen": wickets_fallen,
            "city": city,
        },
    )


with st.container():
    st.title("Cricket Score Predictor")

    teams = [
        "Chennai Super Kings",
        "Delhi Capitals",
        "Gujarat Titans",
        "Kolkata Knight Riders",
        "Lucknow Super Giants",
        "Mumbai Indians",
        "Punjab Kings",
        "Rajasthan Royals",
        "Royal Challengers Bengaluru",
        "Sunrisers Hyderabad",
    ]
    batting_team = st.selectbox("Select the batting team:", teams, key="batting_team")

    bowling_team = st.selectbox(
        "Select the bowling team:",
        random.sample(
            teams,
            10,
        ),
        key="bowling_team",
    )

    if bowling_team == batting_team:
        st.error("Bowling team and batting team cannot be the same.")

    venues = [
        "Mohali",
        "Kolkata",
        "Mumbai",
        "Delhi",
        "Pune",
        "Chennai",
        "Ahmedabad",
        "Lucknow",
        "Jaipur",
        "Raipur",
        "Hyderabad",
        "Bengaluru",
        "Chandigarh",
        "Visakhapatnam",
        "Dharamsala",
        "Ranchi",
    ]
    city = st.selectbox("Select the city", venues, key="city")

    col1, col2 = st.columns([1, 1])
    with col1:
        overs = st.selectbox("Overs", range(21))

    with col2:
        balls = st.selectbox("Balls", range(0, 7))

    col3, col4 = st.columns([1, 1])

    with col3:
        runs_scored = st.text_input("Enter the runs scored", key="runs_scored")
    with col4:
        wickets_fallen = st.selectbox("Wickets lost", range(0, 11))

    is_clicked = st.button("Predict", width="stretch", shortcut="Enter")

    if is_clicked:
        response = send_prediction_request(
            batting_team, bowling_team, overs, balls, runs_scored, wickets_fallen, city
        )
        if response.status_code == 200:
            response = response.json()
            st.success(f"Predicted Score: {response['prediction']}")
        else:
            st.error("Error in prediction request.")
