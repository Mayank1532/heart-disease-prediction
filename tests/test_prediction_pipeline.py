import pandas as pd

from src.pipeline.prediction_pipeline import CustomData, PredictPipeline


def test_prediction_pipeline_initialization():
    pipeline = PredictPipeline()

    assert pipeline.model is not None
    assert pipeline.preprocessor is not None


def test_custom_data_to_dataframe():
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

    df = data.get_data_as_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 11)


def test_prediction_pipeline():
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
    df = data.get_data_as_dataframe()

    pipeline = PredictPipeline()

    prediction = pipeline.predict(df)

    assert prediction is not None
    assert len(prediction) == 1
