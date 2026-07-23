import sys
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from src.config import ModelTrainerConfig
from src.entity import (
    DataTransformationArtifact,
    ModelEvaluationResult,
    ModelTrainerArtifact,
)
from src.exception import CustomException
from src.logger import logger
from src.mlflow.mlflow_manager import MLflowManager
from src.utils import save_pickle


class ModelTrainer:
    """
    Handles model training, hyperparameter tuning,
    evaluation and persistence.
    """

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    ###########################################################################
    # Models
    ###########################################################################

    def _get_models(self) -> dict[str, Any]:
        """
        Return all candidate classification models.
        """

        return {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
            "Decision Tree": DecisionTreeClassifier(
                random_state=42,
            ),
            "Random Forest": RandomForestClassifier(
                random_state=42,
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                random_state=42,
            ),
        }

    ###########################################################################
    # Hyperparameter Search Space
    ###########################################################################

    def _get_model_params(self) -> dict[str, dict]:
        """
        Hyperparameter search space.
        """

        return {
            "Logistic Regression": {
                "C": [0.01, 0.1, 1, 10],
            },
            "Decision Tree": {
                "max_depth": [5, 10, 20],
                "min_samples_split": [2, 5, 10],
            },
            "Random Forest": {
                "n_estimators": [100, 200],
                "max_depth": [10, 20],
                "min_samples_split": [2, 5],
            },
            "Gradient Boosting": {
                "learning_rate": [0.01, 0.1],
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
            },
        }

    ###########################################################################
    # Hyperparameter Tuning
    ###########################################################################

    def _tune_model(
        self,
        model,
        parameter_grid: dict,
        X_train,
        y_train,
    ):
        """
        Tune a single model using GridSearchCV.
        """

        try:

            if not parameter_grid:
                logger.info(
                    "No hyperparameters defined for %s",
                    model.__class__.__name__,
                )

                model.fit(
                    X_train,
                    y_train,
                )

                return model

            logger.info(
                "Running GridSearchCV for %s",
                model.__class__.__name__,
            )

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=parameter_grid,
                cv=5,
                scoring="f1",
                n_jobs=-1,
            )

            grid_search.fit(
                X_train,
                y_train,
            )

            logger.info(
                "Best Parameters : %s",
                grid_search.best_params_,
            )

            logger.info(
                "Best CV F1 Score : %.4f",
                grid_search.best_score_,
            )

            return grid_search.best_estimator_

        except Exception as e:
            logger.exception(
                "Hyperparameter tuning failed."
            )
            raise CustomException(e, sys) from e

                ###########################################################################
    # Model Evaluation
    ###########################################################################

    def _evaluate_models(
        self,
        models: dict[str, Any],
        X_train,
        y_train,
        X_test,
        y_test,
    ) -> dict[str, ModelEvaluationResult]:
        """
        Tune and evaluate all models.
        """

        try:
            logger.info("Evaluating classification models...")

            evaluation_results: dict[str, ModelEvaluationResult] = {}

            parameter_grids = self._get_model_params()

            for model_name, model in models.items():

                logger.info("=" * 80)
                logger.info("Training %s", model_name)

                tuned_model = self._tune_model(
                    model=model,
                    parameter_grid=parameter_grids[model_name],
                    X_train=X_train,
                    y_train=y_train,
                )

                predictions = tuned_model.predict(X_test)

                accuracy = accuracy_score(
                    y_test,
                    predictions,
                )

                precision = precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )

                recall = recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )

                f1 = f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )

                roc_auc = roc_auc_score(
                    y_test,
                    predictions,
                )

                evaluation_results[model_name] = ModelEvaluationResult(
                    model=tuned_model,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    roc_auc=roc_auc,
                )

                logger.info(
                    "%s | Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | ROC-AUC: %.4f",
                    model_name,
                    accuracy,
                    precision,
                    recall,
                    f1,
                    roc_auc,
                )

            logger.info("=" * 80)
            logger.info("All models evaluated successfully.")

            return evaluation_results

        except Exception as e:
            logger.exception("Model evaluation failed.")
            raise CustomException(e, sys) from e

    ###########################################################################
    # Best Model Selection
    ###########################################################################

    def _select_best_model(
        self,
        evaluation_results: dict[str, ModelEvaluationResult],
    ) -> tuple[str, Any, ModelEvaluationResult]:
        """
        Select the best model based on F1 score.
        """

        try:
            logger.info("Selecting best model...")

            best_model_name = max(
                evaluation_results,
                key=lambda x: evaluation_results[x].f1,
            )

            best_result = evaluation_results[
                best_model_name
            ]

            logger.info(
                "Best Model : %s",
                best_model_name,
            )

            logger.info(
                "Accuracy : %.4f",
                best_result.accuracy,
            )

            logger.info(
                "Precision : %.4f",
                best_result.precision,
            )

            logger.info(
                "Recall : %.4f",
                best_result.recall,
            )

            logger.info(
                "F1 Score : %.4f",
                best_result.f1,
            )

            logger.info(
                "ROC-AUC : %.4f",
                best_result.roc_auc,
            )

            return (
                best_model_name,
                best_result.model,
                best_result,
            )

        except Exception as e:
            logger.exception(
                "Best model selection failed."
            )
            raise CustomException(e, sys) from e

    ###########################################################################
    # Save Model
    ###########################################################################

    def _save_model(
        self,
        model: Any,
    ) -> None:
        """
        Persist the trained model.
        """

        try:
            logger.info(
                "Saving model to %s",
                self.config.trained_model_path,
            )

            save_pickle(
                file_path=self.config.trained_model_path,
                obj=model,
            )

            logger.info(
                "Model saved successfully."
            )

        except Exception as e:
            logger.exception(
                "Failed to save model."
            )
            raise CustomException(e, sys) from e

                ###########################################################################
    # Model Training Pipeline
    ###########################################################################

    def initiate_model_trainer(
        self,
        artifact: DataTransformationArtifact,
    ) -> ModelTrainerArtifact:
        """
        Complete model training pipeline.
        """

        try:
            logger.info("=" * 100)
            logger.info("Starting Model Trainer Pipeline")
            logger.info("=" * 100)

            ############################################################
            # Load Models
            ############################################################

            models = self._get_models()

            ############################################################
            # Evaluate Models
            ############################################################

            evaluation_results = self._evaluate_models(
                models=models,
                X_train=artifact.X_train,
                y_train=artifact.y_train,
                X_test=artifact.X_test,
                y_test=artifact.y_test,
            )

            ############################################################
            # Select Best Model
            ############################################################

            (
                best_model_name,
                best_model,
                best_result,
            ) = self._select_best_model(
                evaluation_results=evaluation_results,
            )

            ############################################################
            # Calculate Training Accuracy
            ############################################################

            train_predictions = best_model.predict(
                artifact.X_train,
            )

            train_accuracy = accuracy_score(
                artifact.y_train,
                train_predictions,
            )

            ############################################################
            # Save Model
            ############################################################

            self._save_model(best_model)

            ############################################################
            # MLflow Logging
            ############################################################

            mlflow_manager = MLflowManager(
                "Heart Disease Prediction"
            )

            with mlflow_manager.start_run(
                run_name=best_model_name,
            ):

                mlflow_manager.log_params(
                    {
                        "model": best_model_name,
                    }
                )

                mlflow_manager.log_metrics(
                    {
                        "train_accuracy": train_accuracy,
                        "test_accuracy": best_result.accuracy,
                        "precision": best_result.precision,
                        "recall": best_result.recall,
                        "f1_score": best_result.f1,
                        "roc_auc": best_result.roc_auc,
                    }
                )

                input_example = pd.DataFrame(
                    artifact.X_train[:5]
                )

                signature = infer_signature(
                    artifact.X_train,
                    train_predictions,
                )

                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    name="model",
                    signature=signature,
                    input_example=input_example,
                )

                run = mlflow.active_run()

                if run is not None:
                    mlflow_manager.register_model(
                        model_uri=f"runs:/{run.info.run_id}/model",
                        model_name="HeartDiseasePrediction",
                    )

            ############################################################
            # Build Artifact
            ############################################################

            trainer_artifact = ModelTrainerArtifact(
                model_path=self.config.trained_model_path,
                model_name=best_model_name,
                train_accuracy=train_accuracy,
                test_accuracy=best_result.accuracy,
            )

            logger.info("=" * 100)
            logger.info("Training completed successfully.")
            logger.info("Best Model      : %s", best_model_name)
            logger.info(
                "Train Accuracy : %.4f",
                train_accuracy,
            )
            logger.info(
                "Test Accuracy  : %.4f",
                best_result.accuracy,
            )
            logger.info(
                "Precision      : %.4f",
                best_result.precision,
            )
            logger.info(
                "Recall         : %.4f",
                best_result.recall,
            )
            logger.info(
                "F1 Score       : %.4f",
                best_result.f1,
            )
            logger.info(
                "ROC-AUC        : %.4f",
                best_result.roc_auc,
            )
            logger.info("=" * 100)

            return trainer_artifact

        except Exception as e:
            logger.exception(
                "Model training pipeline failed."
            )
            raise CustomException(e, sys) from e