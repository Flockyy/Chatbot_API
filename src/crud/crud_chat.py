from typing import Any, Dict, Optional, Union, List

from sqlalchemy.orm import Session

from src.crud.base import CRUDBase
from src.models.chat import Chat
from src.schemas.chat import ChatCreate, ChatUpdate


class CRUDChat(CRUDBase[Chat, ChatCreate, ChatUpdate]):
    def get_by_id(self, db: Session, *, id: int) -> Optional[Chat]:
        return db.query(Chat).filter(Chat.id == id).first()

    def get_multi_by_owner(
        self, db: Session, *, owner_id: int, skip: int = 0, limit: int = 100
    ) -> List[Chat]:
        return (
            db.query(self.model)
            .filter(Chat.owner_id == owner_id)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def create(self, db: Session, *, obj_in: ChatCreate) -> Chat:
        db_obj = Chat(
            text=obj_in.text,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
        self,
        db: Session,
        *,
        db_obj: Chat,
        obj_in: Union[ChatUpdate, Dict[str, Any]]
    ) -> Chat:
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)
        return super().update(db, db_obj=db_obj, obj_in=update_data)


chat = CRUDChat(Chat)
