from typing import List

from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session

from src.crud.base import CRUDBase
from src.models.answer import Answer
from src.schemas.answer import AnswerCreate, AnswerUpdate


class CRUDAnswer(CRUDBase[Answer, AnswerCreate, AnswerUpdate]):
    def create_with_owner(
        self, db: Session, *, obj_in: AnswerUpdate, related_pred_id: int
    ) -> Answer:
        obj_in_data = jsonable_encoder(obj_in)
        db_obj = self.model(**obj_in_data, related_pred_id=related_pred_id)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def get_by_owner(
        self, db: Session, *, related_pred_id: int, skip: int = 0, limit: int = 100
    ) -> List[Answer]:
        return (
            db.query(self.model)
            .filter(Answer.related_pred_id == related_pred_id)
            .offset(skip)
            .limit(limit)
            .all()
        )


answer = CRUDAnswer(Answer)
