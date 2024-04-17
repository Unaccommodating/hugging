from fastapi import FastAPI
from configs import host, port

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

# Create an endpoint for sentiment analysis
@app.post("/analyze-sentiment/{model_name}")
async def analyze_sentiment(model_name: str, text: str):
    # Check if the specified model exists
    if model_name not in pipelines:
        return {"error": f"Model '{model_name}' not found."}

    # Perform sentiment analysis using the specified model
    classifier = pipelines[model_name]
    result = classifier(text)

    # Return the sentiment analysis result
    return {"model": model_name, "text": text, "sentiment": result[0]}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=host, port=port)
