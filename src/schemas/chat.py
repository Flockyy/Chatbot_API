from typing import Optional

from pydantic import BaseModel


# Shared properties
class ChatBase(BaseModel):
    score: Optional[int] = None
    review: Optional[str] = None


# Properties to receive via API on creation
class ChatCreate(ChatBase):
    score: int
    review: str


# Properties to receive via API on update
class ChatUpdate(ChatBase):
    pass


class ChatInDBBase(ChatBase):
    id: Optional[int] = None

    class Config:
        orm_mode = True


# Additional properties to return via API


class Chat(ChatInDBBase):
    pass
