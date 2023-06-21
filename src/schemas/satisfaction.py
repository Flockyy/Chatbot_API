from typing import Optional

from pydantic import BaseModel


# Shared properties
class SatisfactionBase(BaseModel):
    rate: Optional[int] = None


# Properties to receive via API on creation
class SatisfactionCreate(SatisfactionBase):
    rate: int


# Properties to receive via API on update
class SatisfactionUpdate(SatisfactionBase):
    pass


class SatisfactionInDBBase(SatisfactionBase):
    id: int
    rate: int
    related_pred_id: int

    class Config:
        orm_mode = True


# Additional properties to return via API


class Satisfaction(SatisfactionInDBBase):
    pass
