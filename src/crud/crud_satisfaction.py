from typing import List

from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session

from crud.base import CRUDBase
from models.satisfaction import Satisfaction
from schemas.satisfaction import SatisfactionCreate, SatisfactionUpdate


class CRUDSatisfaction(CRUDBase[Satisfaction, SatisfactionCreate, SatisfactionUpdate]):
    def create_with_owner(
        self, db: Session, *, obj_in: SatisfactionCreate, related_pred_id: int
    ) -> Satisfaction:
        obj_in_data = jsonable_encoder(obj_in)
        db_obj = self.model(**obj_in_data, related_pred_id=related_pred_id)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def get_by_owner(
        self, db: Session, *, related_pred_id: int, skip: int = 0, limit: int = 100
    ) -> List[Satisfaction]:
        return (
            db.query(self.model)
            .filter(Satisfaction.related_pred_id == related_pred_id)
            .offset(skip)
            .limit(limit)
            .all()
        )


satisfaction = CRUDSatisfaction(Satisfaction)