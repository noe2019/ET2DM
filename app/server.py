from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd

# Define the FastAPI app
app = FastAPI()

# Load the model and scaler with fallback error handling
try:
    model = joblib.load("app/best_diabetes_prediction_model.pkl")
    scaler = joblib.load("app/scaler.pkl")
except ModuleNotFoundError as e:
    raise ImportError(
        f"Module missing while loading model or scaler: {e}. "
        "Ensure compatible numpy and joblib versions are installed."
    )
except Exception as e:
    raise RuntimeError(f"Error loading model or scaler: {e}")

# Class names for the prediction
class_names = np.array(["Non-Diabetic", "Diabetic"])

# Define expected input data schema with constraints
class PredictionRequest(BaseModel):
    Pregnancies: int = Field(..., ge=0, le=20, description="Number of pregnancies, typically between 0 and 20.")
    Glucose: float = Field(..., ge=0, le=300, description="Glucose concentration (mg/dL), usually in the range of 0–300.")
    BloodPressure: float = Field(..., ge=0, le=200, description="Blood pressure (mmHg), typically between 0 and 200.")
    SkinThickness: float = Field(..., ge=0, le=100, description="Skin thickness (mm), generally between 0 and 100.")
    Insulin: float = Field(..., ge=0, le=600, description="Insulin level (µU/mL), typically between 0 and 600.")
    BMI: float = Field(..., ge=0.0, le=70.0, description="Body Mass Index (kg/m^2), usually in the range of 0–70.")
    DiabetesPedigreeFunction: float = Field(..., ge=0.0, le=2.5, description="Diabetes pedigree function, typically between 0.0 and 2.5.")
    Age: int = Field(..., ge=0, le=120, description="Age (years), typically between 0 and 120.")

@app.get('/')
def read_root():
    return {'message': 'Diabetes model API'}

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
