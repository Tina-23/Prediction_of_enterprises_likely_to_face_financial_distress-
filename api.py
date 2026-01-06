from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("logistic_baseline_pipeline.joblib")

class Observation(BaseModel):
    # replace with feature names or accept a list
    features: list

@app.post("/predict")
def predict(obs: Observation):
    X = np.array(obs.features).reshape(1, -1)
    proba = model.predict_proba(X)[0,1]
    pred = int(model.predict(X)[0])
    return {"prediction": pred, "probability": float(proba)}
    