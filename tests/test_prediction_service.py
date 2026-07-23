from app.services.prediction_service import PredictionService
from src.pipeline.prediction_pipeline import CustomData


def test_prediction_service():
    service = PredictionService()

    data = CustomData(
        Age=40,
        Sex="M",
        ChestPainType="ATA",
        RestingBP=140,
        Cholesterol=289,
        FastingBS=0,
        RestingECG="Normal",
        MaxHR=172,
        ExerciseAngina="N",
        Oldpeak=0.0,
        ST_Slope="Up",
    )
    prediction = service.predict(data)

    assert prediction in [0, 1]
