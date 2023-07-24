from typing import Optional

from pydantic import BaseModel


# Shared properties
class QuestionBase(BaseModel):
    text: Optional[str] = None


# Properties to receive via API on creation
class QuestionCreate(QuestionBase):
    text: str


# Properties to receive via API on update
class QuestionUpdate(QuestionBase):
    pass


class QuestionInDBBase(QuestionBase):
    id: Optional[int] = None

    class Config:
        orm_mode = True


# Additional properties to return via API


class Question(QuestionInDBBase):
    pass
