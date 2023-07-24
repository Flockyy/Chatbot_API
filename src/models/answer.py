from typing import TYPE_CHECKING

from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from src.db.base_class import Base

if TYPE_CHECKING:
    from .chat import Chat


class Answer(Base):
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String)
    related_chat_id = Column(Integer, ForeignKey("chat.id"))
    related_chat = relationship("Chat", back_populates="answers")
