from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    Age: int = Field(..., ge=1, le=120)
    Sex: str
    ChestPainType: str
    RestingBP: int = Field(..., ge=0)
    Cholesterol: int = Field(..., ge=0)
    FastingBS: int
    RestingECG: str
    MaxHR: int = Field(..., ge=0)
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str
