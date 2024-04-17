from fastapi import FastAPI
from sentiment_models_dict import sentiment_models
from configs import host, port

# Import the necessary modules from transformers
from transformers import pipeline

app = FastAPI()

# Define the pipeline for each model
pipelines = {name: pipeline("sentiment-analysis", model) for name, model in sentiment_models.items()}



# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=host, port=port)
