from typing import Any, List
import json
import torch
import random

from ai.model import NeuralNet
from ai.nltk_utils import bag_of_words, tokenize, correct

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src import crud
from src import schemas
from src.api import deps

router = APIRouter()

@router.get("/", response_model=List[schemas.Satisfaction])
def read_satisfactions(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    Retrieve satisfactions.
    """
    satifactions = crud.satisfaction.get_multi(db, skip=skip, limit=limit)
    return satifactions


@router.post("/", response_model=schemas.Satisfaction)
def create_satisfaction(
    *,
    db: Session = Depends(deps.get_db),
    sat_in: schemas.SatisfactionCreate,
) -> Any:
    """
    Create new satisfaction.
    """
    sat = crud.satisfaction.create_with_owner(db=db, obj_in=sat_in)
    return sat


@router.put("/{id}", response_model=schemas.Satisfaction)
def update_satisfaction(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    sat_in: schemas.SatisfactionUpdate,
) -> Any:
    """
    Update a satisfaction.
    """
    sat = crud.satisfaction.get(db=db, id=id)
    if not sat:
        raise HTTPException(status_code=404, detail="Satisfaction not found")
    sat = crud.satisfaction.update(db=db, db_obj=sat, obj_in=sat_in)
    return sat


@router.get("/{id}", response_model=schemas.Satisfaction)
def read_satisfaction(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
) -> Any:
    """
    Get satisfaction by ID.
    """
    sat = crud.satisfaction.get(db=db, id=id)
    if not sat:
        raise HTTPException(status_code=404, detail="Satisfaction not found")
    return sat


@router.delete("/{id}", response_model=schemas.Satisfaction)
def delete_satisfaction(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
) -> Any:
    """
    Delete an satisfaction.
    """
    sat = crud.satisfaction.get(db=db, id=id)
    if not sat:
        raise HTTPException(status_code=404, detail="Satisfaction not found")
    sat = crud.satisfaction.remove(db=db, id=id)
    return sat
