from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: TextInput):
    text_vector = vectorizer.transform([data.text])
    prediction = model.predict(text_vector)

    return {
        "input_text": data.text,
        "prediction": prediction[0]
    }