from sqlalchemy.orm import Session

from src import crud
from src import models

from src.schemas.chat import ChatCreate

from src.tests.utils.utils import random_lower_string


def create_random_prediction(db: Session) -> models.Chat:
    text = random_lower_string()
    pred_in = ChatCreate(text=text)
    return crud.chat.create(db=db, obj_in=pred_in)
