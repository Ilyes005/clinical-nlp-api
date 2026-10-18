
from fastapi import FastAPI
from pydantic import BaseModel, Field
from model import classify_text

app = FastAPI(
    title="Clinical NLP API",
    description="Educational clinical-style text classification API",
    version="1.0.0"
)


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


@app.get("/")
def home():
    return {
        "message": "Clinical NLP API is running",
        "version": "1.0.0"
    }


@app.post("/predict")
def predict(request: TextRequest):
    return classify_text(request.text)
