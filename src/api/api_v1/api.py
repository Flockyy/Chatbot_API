from fastapi import APIRouter

from src.api.api_v1.endpoints import predictions
from src.api.api_v1.endpoints import satisfactions

api_router = APIRouter()

api_router.include_router(predictions.router, prefix="/prediction", tags=["prediction"])
api_router.include_router(satisfactions.router, prefix="/satisfaction", tags=["satisfaction"])