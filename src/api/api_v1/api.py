from fastapi import APIRouter

from src.api.api_v1.endpoints import chat
from src.api.api_v1.endpoints import answer
from src.api.api_v1.endpoints import question

api_router = APIRouter()

api_router.include_router(chat.router, prefix="/chat", tags=["Chat"])
api_router.include_router(answer.router, prefix="/answer", tags=["Answer"])
api_router.include_router(question.router, prefix="/question", tags=["Question"])