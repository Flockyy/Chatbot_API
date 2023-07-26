from typing import TYPE_CHECKING

from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship

from src.db.base_class import Base

if TYPE_CHECKING:
    from .question import Question
    from .answer import Answer


class Chat(Base):
    id = Column(Integer, primary_key=True, index=True)
    score = Column(Integer, nullable=True)
    review = Column(String, nullable=True)
    questions = relationship("Question", backref="related_question")
    answers = relationship("Answer", backref="related_answer")