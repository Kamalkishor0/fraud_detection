from fastapi import APIRouter
from app.schema.prediction import PredictionRequest
from app.service.prediction_service import predict_transaction

router = APIRouter(
    prefix = "/api/predict",
    tags = ["Prediction"]
)

@router.post("")
def predict(request: PredictionRequest):
    features = request.model_dump()

    result = predict_transaction(features)

    return result
