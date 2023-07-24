from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src import crud
from src import schemas
from src.api import deps

router = APIRouter()

@router.get("/", response_model=List[schemas.Question])
def read_questions(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    Retrieve questions.
    """
    questions = crud.question.get_multi(db, skip=skip, limit=limit)
    return questions


@router.post("/", response_model=schemas.Question)
def create_question(
    *,
    db: Session = Depends(deps.get_db),
    question_in: schemas.QuestionCreate,
) -> Any:
    """
    Create new question.
    """
    question = crud.question.create_with_chat(db=db, obj_in=question_in)
    return question


@router.put("/{id}", response_model=schemas.Question)
def update_question(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    question_in: schemas.QuestionUpdate,
) -> Any:
    """
    Update a question.
    """
    question = crud.question.get(db=db, id=id)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    question = crud.question.update(db=db, db_obj=question, obj_in=question_in)
    return question


@router.get("/{id}", response_model=schemas.Question)
def read_question(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
) -> Any:
    """
    Get question by ID.
    """
    question = crud.question.get(db=db, id=id)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    return question


@router.delete("/{id}", response_model=schemas.Question)
def delete_question(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
) -> Any:
    """
    Delete an question.
    """
    question = crud.question.get(db=db, id=id)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    question = crud.question.remove(db=db, id=id)
    return question
