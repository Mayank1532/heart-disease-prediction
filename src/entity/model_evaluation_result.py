from dataclasses import dataclass
from typing import Any


@dataclass
class ModelEvaluationResult:
    model: Any
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float