
import fastapi
from fastapi import UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from model import DelayModel  # Import the DelayModel from model.py
import pandas as pd

app = fastapi.FastAPI()
# Initialize the model
model = DelayModel()

# Define a Pydantic model for the request body of the prediction endpoint
class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: str


class FlightsData(BaseModel):
    flights: List[Flight]


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
        model.fit(features, target, 'delay')

        return {
            "status": "Model trained successfully"
        }
    except Exception as e:
        raise fastapi.HTTPException(status_code=400, detail=str(e))


@app.post("/predict", status_code=200)
async def post_predict(flights_data: FlightsData) -> dict:
    try:
        # Converter a lista de dados de voo em um DataFrame pandas
        data = pd.DataFrame([flight.dict() for flight in flights_data.flights])
        # Preprocessar os dados
        features, target = model.preprocess(data, 'delay')
        # Get predictions
        predictions = model.predict(features)
        # Return the predictions
        return {"predict": predictions}
    except Exception as e:
        print(e)
        raise fastapi.HTTPException(status_code=400, detail=str(e))
