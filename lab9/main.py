from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
import numpy as np
from typing import List, Dict, Any

app = FastAPI(
    title="Reddit Classifier API",
    description="A simple API to classify Reddit posts",
    version="1.0.0"
)

class RedditPost(BaseModel):
    title: str
    selftext: str = ""

class PredictionResponse(BaseModel):
    subreddit: str
    confidence: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Reddit Classifier API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(post: RedditPost):
    """
    Predict the subreddit for a given post
    
    This is a simplified mock model that just looks for keywords
    In a real application, this would use your trained model
    """
    combined_text = f"{post.title} {post.selftext}".lower()
    
    # Simple keyword-based classifier for demo purposes
    keywords = {
        "r/technology": ["technology", "tech", "computer", "software", "hardware", "app", "AI"],
        "r/science": ["science", "research", "study", "scientific", "discovery"],
        "r/gaming": ["game", "gaming", "play", "xbox", "playstation", "nintendo", "steam"],
        "r/movies": ["movie", "film", "cinema", "actor", "director", "hollywood"],
        "r/books": ["book", "read", "author", "novel", "literature"]
    }
    
    scores = {}
    for subreddit, words in keywords.items():
        score = 0
        for word in words:
            pattern = r'\b' + re.escape(word) + r'\b'
            matches = re.findall(pattern, combined_text)
            score += len(matches)
        scores[subreddit] = score
    
    # If no keywords match, assign random probabilities
    if sum(scores.values()) == 0:
        subreddit = np.random.choice(list(keywords.keys()))
        confidence = float(np.random.uniform(0.3, 0.7))
    else:
        # Find the subreddit with the highest score
        subreddit = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = float(scores[subreddit] / total)
    
    return {
        "subreddit": subreddit,
        "confidence": confidence
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}