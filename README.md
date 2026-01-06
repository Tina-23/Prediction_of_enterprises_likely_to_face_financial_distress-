# Prediction_of_enterprises_likely_to_face_financial_distress-
A machine learning model to see if it is possible to predict that small or medium enterprises (MNEs) are likely to face financial distress . 
# SME Financial Distress Predictor

Short description
-----------------
Predict whether small or medium enterprises (SMEs) are likely to face financial distress using financial ratios.

Contents
--------
- `Eda.ipynb` - exploratory analysis & modelling notebook
- `data.csv` - dataset (not tracked in repo if sensitive)
- `logistic_baseline_pipeline.joblib` - saved best pipeline (if included)
- `api.py` - simple FastAPI app for predictions (example)
- `requirements.txt` - python dependencies

Quick start
-----------
1. Clone the repo (see below).
2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the notebook `Eda.ipynb` to explore data and train models, or run training script:
```bash

```

Train & tune
------------
The notebook demonstrates:
- preprocessing and cleaning,
- building pipelines (scaler + classifier),
- using StratifiedKFold + GridSearchCV to tune hyperparameters,
- saving best pipeline with `joblib.dump(...)`.

Example: after training, load the model:
```python
import joblib
model = joblib.load("logistic_baseline_pipeline.joblib")
```

Deploy
------
- Example FastAPI app `api.py` included; run with:
  `uvicorn api:app --host 0.0.0.0 --port 8000`
- Optionally add a Dockerfile, build and push to container registry, then deploy to any cloud provider (Cloud Run, Heroku, ECS).


How to contribute
-----------------
- Open an issue for bugs or enhancements.
- Send a PR that includes tests and updated README instructions.

License
-------
No license.
