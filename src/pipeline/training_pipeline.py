"""
Training pipeline orchestration.

This module coordinates the complete ML workflow:

Data Ingestion
    ↓
Data Transformation
    ↓
Model Training

Run with:

    uv run python -m src.pipeline.training_pipeline
"""

from __future__ import annotations

import sys

from src.components import (
    DataIngestion,
    DataTransformation,
    ModelTrainer,
)
from src.config import (
    DataTransformationConfig,
    ModelTrainerConfig,
)
from src.exception import CustomException
from src.logger import logger


class TrainingPipeline:
    """
    Orchestrates the complete training workflow.
    """

    def __init__(self) -> None:
        self.ingestion = DataIngestion()

        self.transformation = DataTransformation(
            DataTransformationConfig(),
        )

        self.trainer = ModelTrainer(
            ModelTrainerConfig(),
        )

    def run(self):
        """
        Execute the end-to-end training pipeline.
        """

        try:
            logger.info("=" * 100)
            logger.info("Starting Training Pipeline")
            logger.info("=" * 100)

            raw_data_path = self.ingestion.initiate_data_ingestion()

            logger.info(
                "Raw dataset available at: %s",
                raw_data_path,
            )

            transformation_artifact = self.transformation.initiate_data_transformation()

            trainer_artifact = self.trainer.initiate_model_trainer(
                transformation_artifact,
            )

            logger.info("=" * 100)
            logger.info("Training Pipeline Completed Successfully")
            logger.info("=" * 100)

            return trainer_artifact

        except Exception as e:
            logger.exception("Training pipeline execution failed.")
            raise CustomException(e, sys) from e


def main() -> None:
    """
    CLI entry point.
    """

    pipeline = TrainingPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
