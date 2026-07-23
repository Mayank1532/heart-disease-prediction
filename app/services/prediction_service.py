from src.pipeline.prediction_pipeline import (
    CustomData,
    PredictPipeline,
)


class PredictionService:
    """Service layer for Heart Disease Prediction."""

    def __init__(self):
        self.pipeline = PredictPipeline()

    def predict(self, data: CustomData):
        df = data.get_data_as_dataframe()

        prediction = self.pipeline.predict(df)

        return int(prediction[0])
