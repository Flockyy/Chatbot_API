from typing import Optional

from pydantic import BaseModel


# Shared properties
class AnswerBase(BaseModel):
    text: Optional[str] = None


# Properties to receive via API on creation
class AnswerCreate(AnswerBase):
    text: str

# Properties to receive via API on update
class AnswerUpdate(AnswerBase):
    pass


class AnswerInDBBase(AnswerBase):
    id: Optional[int] = None

    class Config:
        orm_mode = True


# Additional properties to return via API


class Answer(AnswerInDBBase):
    pass
