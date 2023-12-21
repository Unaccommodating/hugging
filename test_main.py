import json
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_analyze_sentiment_baseline():
    # Тест с базовой моделью
    text = "This is a positive test."
    response = client.post("/analyze-sentiment/baseline", data=json.dumps({"text": text}))
    assert response.status_code == 200
    assert response.json()["sentiment"]["label"] in ["LABEL_0", "LABEL_1"]

def test_analyze_sentiment_medical():
    # Тест с медицинской моделью
    text = "The medical test results are promising."
    response = client.post("/analyze-sentiment/medical", data=json.dumps({"text": text}))
    assert response.status_code == 200
    assert response.json()["sentiment"]["label"] in ["LABEL_0", "LABEL_1"]

def test_analyze_sentiment_social_media():
    # Тест с моделью для анализа социальных медиа
    text = "Having a great time with friends! #happy"
    response = client.post("/analyze-sentiment/social_media", data=json.dumps({"text": text}))
    assert response.status_code == 200
    assert response.json()["sentiment"]["label"] in ["LABEL_0", "LABEL_1"]

