from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import torch
from model import BinaryClassifier

app = FastAPI()

class InputData(BaseModel):
    data: list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

def load_model(model_path = "model.pth"):
    try:
        model = BinaryClassifier()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    except Exception as e:
        raise Exception(f"Inference problem: {str(e)}")

@app.on_event("startup")
async def startup_event():
    global model
    model = load_model()

@app.post("/predict")
async def read_root(input_data: InputData):
    try:
        data = torch.tensor(input_data.data).float().to(device)
        with torch.no_grad():
            output = model(data)
            prediction = torch.sigmoid(output).cpu().numpy()
            return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
