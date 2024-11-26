from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint, confloat
import joblib
import numpy as np
import pandas as pd

# Load the model and scaler
model = joblib.load("app/adaboost_best_model.pkl")
scaler = joblib.load("app/pipeline.pkl")  # Load scaler from a .pkl file

# Class names for the prediction
class_names = np.array(["No Parkinson disease", "Parkinson disease"])

# Define the FastAPI app
app = FastAPI()

# Define expected input data schema with thresholds for validation
class PredictionRequest(BaseModel):
    UPDRS: confloat(ge=0, le=108)  # Unified Parkinson's Disease Rating Scale (0-108)
    FunctionalAssessment: conint(ge=0, le=100)  # Percent functional capacity (0-100%)
    Tremor: confloat(ge=0, le=4)  # Tremor severity score (0-4)
    MoCA: conint(ge=0, le=30)  # Montreal Cognitive Assessment (0-30)
    PosturalInstability: confloat(ge=0, le=4)  # Postural instability score (0-4)
    Bradykinesia: confloat(ge=0, le=4)  # Bradykinesia severity score (0-4)
    EducationLevel: conint(ge=0, le=20)  # Years of education (0-20)
    Diabetes: conint(ge=0, le=1)  # Binary (0: No, 1: Yes)
    Depression: conint(ge=0, le=1)  # Binary (0: No, 1: Yes)
    Hypertension: conint(ge=0, le=1)  # Binary (0: No, 1: Yes)
    Gender: conint(ge=0, le=1)  # Binary (0: Female, 1: Male)
    BMI: confloat(ge=10, le=60)  # Body Mass Index (10-60)
    Stroke: conint(ge=0, le=1)  # Binary (0: No, 1: Yes)
    SleepDisorders: conint(ge=0, le=1)  # Binary (0: No, 1: Yes)
    DiastolicBP: confloat(ge=40, le=120)  # Diastolic blood pressure (40-120 mmHg)
    Constipation: conint(ge=0, le=1)  # Binary (0: No, 1: Yes)
    Rigidity: confloat(ge=0, le=4)  # Rigidity severity score (0-4)
    CholesterolHDL: confloat(ge=20, le=100)  # HDL cholesterol level (20-100 mg/dL)
    FamilyHistoryParkinsons: conint(ge=0, le=1)  # Binary (0: No, 1: Yes)
    TraumaticBrainInjury: conint(ge=0, le=1)  # Binary (0: No, 1: Yes)

@app.get('/')
def read_root():
    return {'message': 'Parkinson Disease prediction API'}

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
