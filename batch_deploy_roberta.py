from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Load the model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('./models/twitter-roberta-base-sentiment')
model = RobertaForSequenceClassification.from_pretrained('./models/twitter-roberta-base-sentiment')

# Initialize FastAPI app
app = FastAPI()

# Input and output schemas
class BatchRequest(BaseModel):
    texts: List[str]

class BatchResponse(BaseModel):
    results: List[dict]

# Function to predict sentiment for multiple texts
def predict_batch_sentiment(texts):
    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        results.append({"text": text, "sentiment": sentiment_map[predicted_class]})
    return results

# API endpoint for batch predictions
@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    results = predict_batch_sentiment(request.texts)
    return BatchResponse(results=results)
