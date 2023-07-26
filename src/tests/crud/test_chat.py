from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session

from src import crud
from src.schemas.chat import ChatCreate, ChatUpdate
from src.tests.utils.utils import random_lower_string


def test_create_chat(db: Session) -> None:
    text = random_lower_string()
    chat_in = ChatCreate(text=text)
    chat = crud.chat.create(db, obj_in=chat_in)
    assert chat.text == text
    assert hasattr(chat, "text")


def test_get_chat(db: Session) -> None:
    text = random_lower_string()
    chat_in = ChatCreate(text=text)
    chat = crud.chat.create(db, obj_in=chat_in)
    chat_2 = crud.chat.get(db, id=chat.id)
    assert chat_2
    assert chat.text == chat.text
    assert jsonable_encoder(chat) == jsonable_encoder(chat_2)


def test_update_chat(db: Session) -> None:
    text = random_lower_string()
    chat_in = ChatCreate(text=text)
    chat = crud.chat.create(db, obj_in=chat_in)
    new_text = random_lower_string()
    chat_in_update = ChatUpdate(text=new_text)
    crud.chat.update(db, db_obj=chat, obj_in=chat_in_update)
    chat_2 = crud.chat.get(db, id=chat.id)
    assert chat_2
    assert chat.text != chat_2.text
