import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

# Load the model using the run_id
model = mlflow.pyfunc.load_model("mlruns/0/abc701b4b3784e0f8e157077f19a088f/artifacts/iris_model")

app = FastAPI()

# Pydantic model for input data
class InputData(BaseModel):
    feature_0: float
    feature_1: float
    feature_2: float
    feature_3: float

@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert the input data to a DataFrame
        input_data = data.dict()
        input_df = pd.DataFrame([input_data])
        
        # Get the prediction from the model
        prediction = model.predict(input_df)
        
        # Return the prediction as a list (JSON-compatible)
        return {"prediction": prediction.tolist()}
    
    except Exception as e:
        return {"error": str(e)}
