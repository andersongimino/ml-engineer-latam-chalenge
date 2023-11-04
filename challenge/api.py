
import fastapi
from fastapi import UploadFile, File
from pydantic import BaseModel
from typing import List
import uvicorn
from model import DelayModel  # Import the DelayModel from model.py
import pandas as pd

app = fastapi.FastAPI()
# Initialize the model
model = DelayModel()
#save feature based on dataset trained


# Define a Pydantic model for the request body of the prediction endpoint
class FlightData(BaseModel):
    Fecha_I: str
    Vlo_I: str
    Ori_I: str
    Des_I: str
    Emp_I: str
    Fecha_O: str
    Vlo_O: str
    Ori_O: str
    Des_O: str
    Emp_O: str
    DIA: str
    MES: str
    AÑO: str
    DIANOM: str
    TIPOVUELO: str
    OPERA: str
    SIGLAORI: str
    SIGLADES: str


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/train", status_code=200)
async def train_model(file: UploadFile = File(...)) -> dict:
    try:
        # Read the uploaded CSV file into a pandas DataFrame
        dataframe = pd.read_csv(file.file)
        # Preprocess the data
        features, target = model.preprocess(dataframe, 'delay')
        # Train the model
        model.fit(features, target)

        return {
            "status": "Model trained successfully"
        }
    except Exception as e:
        raise fastapi.HTTPException(status_code=400, detail=str(e))


@app.post("/predict", status_code=200)
async def post_predict(flight_data: List[FlightData]) -> dict:
    try:
        # Convert the flight data to a pandas DataFrame
        data = pd.DataFrame([item.dict() for item in flight_data])
        # Preprocess the data
        features, target = model.preprocess(data, 'delay')
        # Get predictions
        predictions = model.predict(features)
        # Return the predictions
        return {"predictions": predictions}
    except Exception as e:
        raise fastapi.HTTPException(status_code=400, detail=str(e))
