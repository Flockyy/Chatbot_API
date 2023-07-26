from typing import List

from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session

from src.crud.base import CRUDBase
from src.models.question import Question
from src.schemas.question import QuestionCreate, QuestionUpdate


class CRUDQuestion(CRUDBase[Question, QuestionCreate, QuestionUpdate]):
    def create_with_owner(
        self, db: Session, *, obj_in: QuestionCreate, related_chat_id: int
    ) -> Question:
        obj_in_data = jsonable_encoder(obj_in)
        db_obj = self.model(**obj_in_data, related_chat_id=related_chat_id)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def get_by_owner(
        self, db: Session, *, related_pred_id: int, skip: int = 0, limit: int = 100
    ) -> List[Question]:
        return (
            db.query(self.model)
            .filter(Question.related_pred_id == related_pred_id)
            .offset(skip)
            .limit(limit)
            .all()
        )


question = CRUDQuestion(Question)
