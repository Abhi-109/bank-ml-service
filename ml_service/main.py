import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

# --- 1. Initialize FastAPI App ---
# This is the main application object
app = FastAPI(
    title="Bank Marketing ML Service",
    description="A service to predict if a customer will subscribe to a term deposit.",
    version="1.0.0"
)

# --- 2. Load The Model ---
# We load the model at startup.
# This way, it's in memory and ready for predictions.
# We only do this ONCE.
model_path = 'ml_service/model.pkl'
try:
    model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    model = None # Set model to None to handle errors gracefully

# --- 3. Define Input Data Model (Pydantic) ---
# This defines the exact structure of the JSON we expect in a POST request.
# FastAPI will automatically validate incoming data against this model.
# These fields MUST match the columns our model was trained on.
# We use 'Field(..., example=...)'' to provide example data for the /docs UI.

class BankData(BaseModel):
    # These are the 16 features our model expects
    age: int = Field(..., example=41)
    balance: int = Field(..., example=2143)
    day: int = Field(..., example=5)
    duration: int = Field(..., example=261)
    campaign: int = Field(..., example=1)
    pdays: int = Field(..., example=-1)
    previous: int = Field(..., example=0)
    
    job: str = Field(..., example="blue-collar")
    marital: str = Field(Example="married")
    education: str = Field(..., example="secondary")
    default: str = Field(..., example="no")
    housing: str = Field(..., example="yes")
    loan: str = Field(..., example="no")
    contact: str = Field(..., example="unknown")
    month: str = Field(..., example="may")
    poutcome: str = Field(..., example="unknown")

    # We can use pydantic's 'model_config' to add an example for the /docs
    class Config:
        json_schema_extra = {
            "example": {
                "age": 41,
                "balance": 1500,
                "day": 5,
                "duration": 200,
                "campaign": 1,
                "pdays": -1,
                "previous": 0,
                "job": "management",
                "marital": "married",
                "education": "tertiary",
                "default": "no",
                "housing": "yes",
                "loan": "no",
                "contact": "cellular",
                "month": "may",
                "poutcome": "unknown"
            }
        }

# --- 4. Define Output Data Model (Pydantic) ---
# This defines the structure of the JSON response we will send.
class PredictionOut(BaseModel):
    prediction: str  # 'yes' or 'no'
    probability_no: float = Field(..., example=0.85)
    probability_yes: float = Field(..., example=0.15)


# --- 5. Create the Prediction Endpoint ---
@app.post("/predict", response_model=PredictionOut)
async def predict(data: BankData):
    """
    Takes customer data as input and returns a 'yes' or 'no' prediction
    with associated probabilities.
    """
    if model is None:
        # This handles the case where the model failed to load at startup
        raise HTTPException(status_code=500, detail="Model not loaded. Please check server logs.")

    # 1. Convert Pydantic model to a pandas DataFrame
    # The model.predict() method expects a DataFrame with the exact column
    # names it was trained on.
    input_df = pd.DataFrame([data.model_dump()])

    # 2. Make prediction
    # model.predict() gives the class (0 or 1)
    pred_raw = model.predict(input_df)[0]
    prediction = 'yes' if pred_raw == 1 else 'no'

    # 3. Get probabilities
    # model.predict_proba() gives probabilities for [class_0, class_1]
    probabilities = model.predict_proba(input_df)[0]
    prob_no = round(probabilities[0], 4)
    prob_yes = round(probabilities[1], 4)

    # 4. Return the response
    return PredictionOut(
        prediction=prediction,
        probability_no=prob_no,
        probability_yes=prob_yes
    )

# --- 6. Add a simple root endpoint ---
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Bank Marketing ML Service!",
        "documentation": "/docs"
    }