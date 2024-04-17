from fastapi import FastAPI

# Import the necessary modules from transformers
from transformers import pipeline

app = FastAPI()

# Load the sentiment analysis models
sentiment_models = {
    "baseline": "blanchefort/rubert-base-cased-sentiment",
    "medical": "blanchefort/rubert-base-cased-sentiment-med",
    "rusentiment": "blanchefort/rubert-base-cased-sentiment-rusentiment",
    "rureviews": "blanchefort/rubert-base-cased-sentiment-rurewiews",
    "social_media": "blanchefort/rubert-base-cased-sentiment-mokoron",
}

# Define the pipeline for each model
pipelines = {name: pipeline("sentiment-analysis", model) for name, model in sentiment_models.items()}



# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
