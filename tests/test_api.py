from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    body = response.json()

    assert isinstance(body, dict)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict():
    payload = {
        "Age": 40,
        "Sex": "M",
        "ChestPainType": "ATA",
        "RestingBP": 140,
        "Cholesterol": 289,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 172,
        "ExerciseAngina": "N",
        "Oldpeak": 0.0,
        "ST_Slope": "Up",
    }

    response = client.post("/predict/", json=payload)

    assert response.status_code == 200

    body = response.json()

    assert "prediction" in body
    assert "diagnosis" in body

    assert body["prediction"] in [0, 1]
    assert isinstance(body["diagnosis"], str)
