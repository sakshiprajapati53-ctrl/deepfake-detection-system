# Fast api invoke
from fastapi import FastAPI,UploadFile,File
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, We are making Deepfake detection!"}

@app.post("/Predict")
def predict():
    # abhi idhar kaam hoga to idea batao
    pass