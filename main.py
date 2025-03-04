from fastapi import FastAPI
import uvicorn
app = FastAPI()
@app.get("/hello")
async def read_root():
    return {"message": "Hello World"}
