from dataclasses import dataclass
from pathlib import Path

from .paths import RAW_DATA_DIR


@dataclass(frozen=True)
class DataIngestionConfig:
    dataset_path: Path = RAW_DATA_DIR / "heart_disease.csv"
    raw_data_dir: Path = RAW_DATA_DIR
    raw_data_path: Path = RAW_DATA_DIR / "heart_disease.csv"
