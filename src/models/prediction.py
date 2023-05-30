from typing import TYPE_CHECKING

from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship

from src.db.base_class import Base

if TYPE_CHECKING:
  from .satisfaction import Satisfaction

class Prediction(Base):
  id = Column(Integer, primary_key=True, index=True)
  text = Column(String, index=True)
  satisfactions = relationship("Satisfaction", back_populates="related_pred")