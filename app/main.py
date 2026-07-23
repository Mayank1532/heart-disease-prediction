from fastapi import FastAPI

from app.routers.prediction import router as prediction_router

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predict laptop prices using a trained ML model.",
    version="1.0.0",
)

app.include_router(prediction_router)


@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API is running!"}


@app.get("/health")
def health():
    return {"status": "healthy"}
