import sys
from dataclasses import dataclass

import pandas as pd

from src.config import (
    DataTransformationConfig,
    ModelTrainerConfig,
)
from src.exception import CustomException
from src.logger import logger
from src.utils import load_pickle


class PredictPipeline:
    def __init__(self):
        try:
            logger.info("Loading trained model...")

            self.model = load_pickle(ModelTrainerConfig().trained_model_path)

            logger.info("Loading preprocessor...")

            self.preprocessor = load_pickle(
                DataTransformationConfig().preprocessor_path
            )

            logger.info("Prediction pipeline initialized successfully.")

        except Exception as e:
            logger.exception("Failed to initialize prediction pipeline.")
            raise CustomException(e, sys) from e

    def predict(
        self,
        features: pd.DataFrame,
    ):
        try:
            logger.info("Starting prediction...")

            transformed_features = self.preprocessor.transform(features)

            predictions = self.model.predict(transformed_features)

            return predictions

        except Exception as e:
            logger.exception("Prediction failed.")
            raise CustomException(e, sys) from e


@dataclass
class CustomData:
    """
    Represents a single patient for prediction.
    """

    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Convert patient data into a pandas DataFrame.
        """

        try:
            custom_data_input_dict = {
                "Age": [self.Age],
                "Sex": [self.Sex],
                "ChestPainType": [self.ChestPainType],
                "RestingBP": [self.RestingBP],
                "Cholesterol": [self.Cholesterol],
                "FastingBS": [self.FastingBS],
                "RestingECG": [self.RestingECG],
                "MaxHR": [self.MaxHR],
                "ExerciseAngina": [self.ExerciseAngina],
                "Oldpeak": [self.Oldpeak],
                "ST_Slope": [self.ST_Slope],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            logger.exception("Failed to create prediction dataframe.")
            raise CustomException(e, sys) from e
