from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import uvicorn
import torch
import pandas as pd
from sqlmodel import Session
from app import BinaryClassifier
from .database.db import init_db, engine, get_db, Request

app = FastAPI()

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
    init_db()
    global model
    model = load_model()

@app.post("/predict")
async def read_root(request: Request, db: Session = Depends(get_db)):
    try:
        application = pd.DataFrame(request.model_dump(), index=[0])
        application = application.drop(columns=["id", "DECISION"])

        prepared_application = pd.get_dummies(
            application, columns=["CODE_GENDER", 
            "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE", 
            "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", 
            "OCCUPATION_TYPE"]).astype(int)
        
        data = torch.tensor(prepared_application.values, dtype=torch.float32).to(device)
        with torch.no_grad():
            output = model(data)
            prediction = torch.sigmoid(output).cpu().item()

        request.DECISION = 1 if prediction > 0.5 else 0

        db.add(request)
        db.commit()
        db.refresh(request)
        
        return {"prediction": prediction}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
