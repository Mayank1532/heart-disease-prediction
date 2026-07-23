from dataclasses import dataclass


@dataclass
class ModelTrainerArtifact:
    model_path: str
    model_name: str
    train_accuracy: float
    test_accuracy: float