from typing import TYPE_CHECKING

from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship

from db.base_class import Base

if TYPE_CHECKING:
    from .prediction import Prediction


class Satisfaction(Base):
    id = Column(Integer, primary_key=True, index=True)
    rate = Column(Integer, index=True)
    related_pred_id = Column(Integer, ForeignKey("pred.id"))
    related_pred = relationship("Prediction", back_populates="satisfactions")
