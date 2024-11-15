from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint, confloat
import joblib
import numpy as np
import pandas as pd

# Load the model and scaler
model = joblib.load("app/eo_best_model.joblib")
scaler = joblib.load("app/scaler.joblib")  # Load scaler from a .pkl file

# Class names for the prediction
class_names = np.array(["No risk", "Early diabetes risk"])

# Define the FastAPI app
app = FastAPI()

# Define expected input data schema with constrained integers
class PredictionRequest(BaseModel):
    RIDAGEYR: confloat(ge=21.0, le=120.0)  # Example: Age from 0 to 120
    RACE: conint(ge=1, le=4)  # Example: Race category, 1 to 5
    EDUC: conint(ge=1, le=3)  # Example: Education level, 1 to 5
    COUPLE: conint(ge=1, le=3)  # Example: 0 or 1 for couple status
    TOTAL_ACCULTURATION_SCORE_v2: conint(ge=1, le=3)  # Example: Score from 0 to 100
    FAT: conint(ge=1, le=3)  # Example: FAT score from 0 to 100
    POVERTIES: conint(ge=0, le=1)  # Example: Poverty level, 0 to 5
    HTN: conint(ge=0, le=1)  # Example: 0 or 1 for hypertension
    RIAGENDR: conint(ge=1, le=2)  # Example: 1 or 2 for gender
    SMOKER: conint(ge=0, le=1)  # Example: 0 or 1 for smoker status

@app.get('/')
def read_root():
    return {'message': 'Early diabetes model API'}

@app.post('/predict')
def predict(data: PredictionRequest):
    """
    Predict the class of a given set of features.

    Args:
        data (PredictionRequest): A dictionary containing the features to predict.

    Returns:
        dict: A dictionary containing the predicted class.
    """

    # Convert the input data to a DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Apply scaling to the input data
    try:
        input_data_scaled = scaler.transform(input_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in scaling input data: {e}")

    # Perform prediction
    try:
        prediction = model.predict(input_data_scaled)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in model prediction: {e}")

    # Convert the prediction output to an integer if valid
    try:
        prediction_index = int(float(prediction[0]))  # Safely convert to integer
    except (ValueError, TypeError) as e:
        raise HTTPException(
            status_code=500, detail=f"Error converting prediction to integer: {e}"
        )

    # Use the integer index to get the class name
    class_name = class_names[prediction_index]

    return {'predicted_class': class_name}