from fastapi import FastAPI
import pandas as pd
import mlflow
import os

app = FastAPI()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_model():
    model_uri = "models:/cricket-score-prediction@champion"
    model = mlflow.sklearn.load_model(model_uri)

    return model


@app.get("/")
async def root():
    return {"message": "Hello, World!"}


@app.post("/predict")
async def predict(data: dict):

    champion_model = load_model()

    total_balls =  data["overs"] * 6 + data["balls"]
    balls_left = 120 - total_balls 
    wickets_left = 10 - data["wickets_fallen"]
    crr = int(data["runs_scored"]) / (total_balls / 6)

    input_df = pd.DataFrame(
        {
            "current_score": [int(data["runs_scored"])],
            "balls_left": [int(balls_left)],
            "wickets_left": [int(wickets_left)],
            "current_run_rate": [float(crr)],
            "batting_team": [data["batting_team"]],
            "bowling_team": [data["bowling_team"]],
            "city": [data["city"]],
        }
    )

    result = champion_model.predict(input_df)

    prediction = {"prediction": round(result[0])}
    return prediction
