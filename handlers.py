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