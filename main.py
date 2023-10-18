from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

class Message(BaseModel):
    message: str

sentiment_pipeline = pipeline("sentiment-analysis")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "status": 200,
        "message": "Welcome to Riffle Ripple backend"
    }

@app.post("/sentiment")
def push_message(msg: Message):
    """Push message to target channel 
    
    Args:
        msg (Message): Message
        
    Returns:
        dict: Response
    """
    res = sentiment_pipeline([msg.message])

    print(res)
    score, label = res[0]['score'], res[0]['label']
    sentiment = 0 if res[0]['score'] < 0.5 else (score if label == 'POSITIVE' else -score)
    return {
        "status": 200,
        "message": "Message sent",
        "sentiment": sentiment
    }
