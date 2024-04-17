from fastapi import FastAPI
from sentiment_models_dict import sentiment_models

# Import the necessary modules from transformers
from transformers import pipeline

app = FastAPI()

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

    uvicorn.run(app, host="127.0.0.1", port=8000)
