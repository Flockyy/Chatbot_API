from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src import crud
from src import schemas
from src.api import deps

router = APIRouter()

@router.get("/", response_model=List[schemas.Answer])
def read_answers(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    Retrieve answers.
    """
    answer = crud.answer.get_multi(db, skip=skip, limit=limit)
    return answer


@router.post("/", response_model=schemas.Answer)
def create_answer(
    *,
    db: Session = Depends(deps.get_db),
    answer_in: schemas.AnswerCreate,
) -> Any:
    """
    Create new answer.
    """
    answer = crud.answer.create_with_chat(db=db, obj_in=answer_in)
    return answer


@router.put("/{id}", response_model=schemas.Answer)
def update_answer(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    answer_in: schemas.AnswerUpdate,
) -> Any:
    """
    Update a answer.
    """
    answer = crud.answer.get(db=db, id=id)
    if not answer:
        raise HTTPException(status_code=404, detail="Answer not found")
    answer = crud.answer.update(db=db, db_obj=answer, obj_in=answer_in)
    return answer


@router.get("/{id}", response_model=schemas.Answer)
def read_answer(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
) -> Any:
    """
    Get answer by ID.
    """
    answer = crud.answer.get(db=db, id=id)
    if not answer:
        raise HTTPException(status_code=404, detail="Answer not found")
    return answer


@router.delete("/{id}", response_model=schemas.Answer)
def delete_answer(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
) -> Any:
    """
    Delete an answer.
    """
    answer = crud.answer.get(db=db, id=id)
    if not answer:
        raise HTTPException(status_code=404, detail="Answer not found")
    answer = crud.answer.remove(db=db, id=id)
    return answer
