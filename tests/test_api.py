
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["version"] == "1.0.0"


def test_empty_text():
    response = client.post(
        "/predict",
        json={"text": ""}
    )

    assert response.status_code == 422


def test_missing_text():
    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 422


def test_prediction(monkeypatch):
    def fake_classifier(text):
        return {
            "text": text,
            "label": "symptoms",
            "score": 0.9
        }

    monkeypatch.setattr(
        "api.classify_text",
        fake_classifier
    )

    response = client.post(
        "/predict",
        json={"text": "The patient has a headache."}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["label"] == "symptoms"
    assert data["score"] == 0.9
